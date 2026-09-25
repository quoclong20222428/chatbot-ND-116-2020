"""BM25 retriever for the legal RAG chatbot.

Implements BM25 (Okapi BM25) as a standalone, embedding-model-independent
retrieval algorithm operating on the existing legal chunk corpus in
PostgreSQL.

Overview
--------
BM25 (Best Match 25) is a **lexical retrieval** method.  It ranks documents
by the weighted term-frequency of query words in each document, penalising
very long documents through a length-normalisation term.

Key characteristics:
  - No embedding model required.  BM25 is purely term-based.
  - Deterministic: given the same corpus, tokenisation, k1 and b parameters,
    and the same query, BM25 always returns the same ranked list.
  - Scores are NOT comparable to cosine similarity or other embedding-based
    scores.  Do NOT combine scores from BM25 and HNSW without explicit
    normalisation (e.g. RRF, which is a future work item).

How it works (high level):
  1. Load all legal chunk texts from PostgreSQL at construction time.
  2. Tokenise each chunk into a list of lowercased tokens.
  3. Build a BM25 index (``rank_bm25.BM25Okapi``) over the token lists.
  4. For each query, tokenise the query the same way, call
     ``BM25Okapi.get_scores()`` to obtain a score for every document, then
     return the top-k chunk IDs and scores in descending score order.

The corpus is loaded **once** at ``BM25Retriever.__init__`` time and reused
for every query, just like the embedding model in ``Retriever``.

BM25 Parameters
---------------
k1 (float):
    Term saturation parameter.  Controls how quickly the contribution of
    additional term occurrences levels off.  Higher k1 makes term frequency
    matter more before saturation.  Common range: 1.2 - 2.0.  Default: 1.5.

    Rationale: For Vietnamese legal text, which often repeats key legal
    terms many times in a single clause, k1 = 1.5 is a reasonable starting
    point.  Tuning against the evaluation dataset is recommended before
    treating BM25 as a production component.

b (float):
    Length normalisation parameter.  b = 1.0 fully normalises document
    length; b = 0.0 disables normalisation.  Default: 0.75.

    Rationale: Legal clauses vary significantly in length.  b = 0.75 is
    the BM25 standard default and appropriate for this dataset size (~617
    chunks).

epsilon (float):
    Floor value for BM25 IDF scores.  rank_bm25 uses this to handle terms
    that appear in a majority of documents (which would otherwise receive
    negative IDF weights in some BM25 variants).  Default: 0.25.

These defaults are intentionally conservative.  The corpus size (~617 chunks)
is small enough that exhaustive BM25 scoring over the full corpus is fast
(<1 ms for a typical query) without approximate nearest-neighbour indexing.

Tokenisation
------------
Tokens are produced by simple Unicode whitespace splitting followed by
lowercasing.  No stopword removal or stemming is applied, because:

  - Vietnamese is a monosyllabic, isolating language; stemming is not
    directly applicable.
  - Removing legal stopwords requires a domain-specific stoplist that has
    not been curated for this project.
  - Simplicity produces deterministic, auditable results.

A custom tokeniser can be injected via the ``tokenizer`` parameter if
better tokenisation is needed in future experiments (e.g. using underthesea
or VnCoreNLP).

Embedding-model independence
-----------------------------
BM25 has no relationship to any embedding model.  ``RetrievalResult`` objects
produced by ``BM25Retriever`` carry:
  - ``retrieval_method = "bm25"``
  - ``score_type = "bm25"``
  - ``rank`` starting from 1

The evaluation framework (``scripts/test_retrieval.py``) uses
``result.metadata`` for ground-truth matching, which works identically for
BM25 and HNSW results.

Usage::

    from bm25_retriever import BM25Retriever

    retriever = BM25Retriever(database_url="postgresql://...")
    results = retriever.retrieve(
        "Dieu kien de duoc huong chinh sach ho tro la gi?",
        top_k=5,
    )
    for r in results:
        print(r.chunk_id, r.score, r.text[:200])

    # Reload the corpus after database changes:
    retriever.reload_corpus()

Environment variables
---------------------
DATABASE_URL
    PostgreSQL connection URL.  Loaded from ``.env`` if not set.

What is NOT implemented yet
---------------------------
- Hybrid retrieval (BM25 + HNSW): planned as a separate future step.
- Reciprocal Rank Fusion (RRF): planned as a separate future step.
- Reranking: planned as a separate future step.
- Score normalisation for cross-method comparison: needed only for hybrid
  retrieval and will be implemented when RRF is introduced.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

if __package__.startswith("scripts."):
    from ..retrieval_types import RetrievalResult
    from ..database import connect_postgres
    from ..environment import load_dotenv as _load_env_file, require_database_url
else:
    from retrieval_types import RetrievalResult
    from database import connect_postgres
    from environment import load_dotenv as _load_env_file, require_database_url

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5
MAX_TOP_K = 1000          # guard against unreasonably large requests

# BM25 parameters with documented rationale (see module docstring).
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_EPSILON = 0.25

# SQL to load corpus from the database.
# Only ``text`` and the metadata columns needed for evaluation are fetched.
# Embedding columns are intentionally excluded -- BM25 does not use them.
_CORPUS_SQL = """
SELECT
    chunk_id,
    text,
    document_id,
    document_title,
    document_number,
    source_type,
    document_role,
    authority_level,
    retrieval_priority,
    chapter,
    article,
    clause,
    point,
    content_type
FROM legal_chunks
ORDER BY chunk_id
"""


# ---------------------------------------------------------------------------
# Internal corpus entry
# ---------------------------------------------------------------------------


@dataclass
class _CorpusEntry:
    """Internal record for a single chunk in the BM25 corpus.

    Stores the chunk_id, metadata, and pre-tokenised text so that a
    ``RetrievalResult`` can be constructed after scoring without a second
    database round-trip.
    """

    chunk_id: str
    text: str
    tokens: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------


def default_tokenizer(text: str) -> list[str]:
    """Tokenise Vietnamese legal text into lowercase tokens.

    Strategy: split on Unicode whitespace and punctuation, keeping only
    non-empty tokens.  This is intentionally simple to ensure determinism
    and auditability.  See module docstring for rationale.

    Parameters
    ----------
    text:
        Raw chunk text or query string.

    Returns
    -------
    list[str]
        Lowercased, non-empty token list.
    """
    # Split on whitespace and common punctuation.
    tokens = re.split(
        r"[\s\u200b\u00a0,;:!?()\[\]{}<>\"'\u201c\u201d\u2018\u2019\u2013\u2014/\\|]+",
        text.lower(),
    )
    # Drop empty strings produced by leading/trailing delimiters.
    return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# Helpers (shared patterns from retrieval.py)
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path) -> None:
    """Compatibility wrapper for the shared .env parser."""
    _load_env_file(path)


def _resolve_database_url(database_url: str | None) -> str:
    """Return a usable DATABASE_URL, loading .env if necessary."""
    if database_url:
        return database_url
    root = Path(__file__).resolve().parents[2]
    return require_database_url(
        root / ".env",
        loader=_load_dotenv,
        error_message="DATABASE_URL is required.  Set it in the environment or in .env",
    )


def _connect(database_url: str) -> Any:
    """Open and return a psycopg (v3) connection."""
    return connect_postgres(
        database_url,
        error_message="psycopg is not installed.  Run: pip install psycopg[binary]",
    )


def _import_bm25():
    """Import BM25Okapi from rank_bm25, with a helpful error message."""
    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import]
        return BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "rank_bm25 is not installed.  Run: pip install rank-bm25"
        ) from exc


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class BM25Retriever:
    """BM25 retriever operating on the legal chunk corpus.

    Loads all chunk texts from PostgreSQL at construction time, builds a
    BM25 index, and serves queries without any database round-trip.

    This retriever is **embedding-model-independent** -- it never loads or
    calls any neural network.  It can therefore be initialised and used
    alongside HNSW retrievers for different embedding models without
    interference.

    Parameters
    ----------
    database_url:
        PostgreSQL connection URL.  If ``None``, loaded from the
        ``DATABASE_URL`` environment variable or the ``.env`` file at the
        repository root.
    k1:
        BM25 term saturation parameter.  See module docstring for rationale.
        Default: ``1.5``.
    b:
        BM25 length normalisation parameter.  Default: ``0.75``.
    epsilon:
        BM25 IDF floor value.  Default: ``0.25``.
    tokenizer:
        Callable that converts a string to a list of tokens.  Defaults to
        :func:`default_tokenizer`.  The same tokeniser is applied to both
        corpus documents and queries -- ensure consistency if overriding.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        epsilon: float = DEFAULT_EPSILON,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        # Validate BM25 parameters.
        if not isinstance(k1, (int, float)) or isinstance(k1, bool) or k1 <= 0:
            raise ValueError(f"k1 must be a positive number (got {k1!r})")
        if not isinstance(b, (int, float)) or isinstance(b, bool) or not (0.0 <= b <= 1.0):
            raise ValueError(f"b must be a float in [0, 1] (got {b!r})")
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon < 0:
            raise ValueError(f"epsilon must be a non-negative number (got {epsilon!r})")

        self._database_url = _resolve_database_url(database_url)
        self._k1 = k1
        self._b = b
        self._epsilon = epsilon
        self._tokenizer: Callable[[str], list[str]] = tokenizer or default_tokenizer

        # Import BM25 library eagerly so missing dependency is caught at init.
        self._BM25Okapi = _import_bm25()

        # Internal state -- populated by _load_corpus().
        self._corpus: list[_CorpusEntry] = []
        self._bm25: Any = None  # BM25Okapi instance or None if corpus is empty.

        LOGGER.info(
            "BM25Retriever initialising -- k1=%.2f, b=%.2f, epsilon=%.2f",
            self._k1,
            self._b,
            self._epsilon,
        )

        self._load_corpus()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        """Number of chunks currently indexed in the BM25 corpus."""
        return len(self._corpus)

    @property
    def k1(self) -> float:
        """BM25 term saturation parameter."""
        return self._k1

    @property
    def b(self) -> float:
        """BM25 length normalisation parameter."""
        return self._b

    @property
    def epsilon(self) -> float:
        """BM25 IDF floor value."""
        return self._epsilon

    # ------------------------------------------------------------------
    # Corpus loading
    # ------------------------------------------------------------------

    def _load_corpus(self) -> None:
        """Load all chunks from the database and build the BM25 index.

        This method is called once at construction time.  It can also be
        called manually via :meth:`reload_corpus` if the database contents
        change.

        The corpus is sorted by ``chunk_id`` to ensure deterministic ordering
        across different invocations when the database content is unchanged.
        """
        t0 = time.perf_counter()
        LOGGER.info("Loading BM25 corpus from database...")

        with _connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(_CORPUS_SQL)
                rows = cur.fetchall()

        entries: list[_CorpusEntry] = []
        for row in rows:
            (
                chunk_id, text,
                document_id, document_title, document_number,
                source_type, document_role, authority_level,
                retrieval_priority, chapter, article, clause, point, content_type,
            ) = row
            tokens = self._tokenizer(text or "")
            entries.append(
                _CorpusEntry(
                    chunk_id=chunk_id,
                    text=text or "",
                    tokens=tokens,
                    metadata={
                        "document_id": document_id,
                        "document_title": document_title,
                        "document_number": document_number,
                        "source_type": source_type,
                        "document_role": document_role,
                        "authority_level": authority_level,
                        "retrieval_priority": retrieval_priority,
                        "chapter": chapter,
                        "article": article,
                        "clause": clause,
                        "point": point,
                        "content_type": content_type,
                    },
                )
            )

        self._corpus = entries

        if entries:
            self._bm25 = self._BM25Okapi(
                [e.tokens for e in entries],
                k1=self._k1,
                b=self._b,
                epsilon=self._epsilon,
            )
        else:
            self._bm25 = None

        elapsed_ms = (time.perf_counter() - t0) * 1000
        LOGGER.info(
            "BM25 corpus loaded: %d chunks indexed in %.1f ms",
            len(entries),
            elapsed_ms,
        )

    def reload_corpus(self) -> None:
        """Reload the corpus from the database and rebuild the BM25 index.

        Use this if the ``legal_chunks`` table has been updated since the
        retriever was initialised.  The existing index is replaced after the
        new data is fully loaded.
        """
        LOGGER.info("Reloading BM25 corpus...")
        self._load_corpus()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list:
        """Score *query* against the corpus and return the top-*top_k* chunks.

        Parameters
        ----------
        query:
            A natural-language legal question.  Must be a non-empty,
            non-whitespace string.
        top_k:
            Number of results to return.  Must be a positive integer not
            exceeding MAX_TOP_K (1000).

        Returns
        -------
        list[RetrievalResult]
            Results ordered by descending BM25 score (highest first).
            Rank 1 is the most relevant result.  May be shorter than
            *top_k* if the corpus contains fewer than *top_k* chunks or
            if all BM25 scores are zero (no query terms found in corpus).

        Raises
        ------
        ValueError
            If *query* is empty or whitespace-only, or if *top_k* is out
            of the valid range.

        Note on BM25 scores
        -------------------
        BM25 scores are non-negative and unbounded above.  A score of 0.0
        means no query terms appeared in the document.  The scores are
        NOT directly comparable to cosine similarity scores from HNSW
        retrieval.  ``RetrievalResult.score_type == "bm25"`` identifies
        the provenance of these scores.
        """
        # --- Input validation ---
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty, non-whitespace string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer (got {top_k!r})")
        if top_k > MAX_TOP_K:
            raise ValueError(
                f"top_k={top_k} exceeds the maximum allowed value ({MAX_TOP_K})"
            )

        if not self._corpus:
            LOGGER.warning("BM25 corpus is empty; returning no results.")
            return []

        LOGGER.info("BM25: retrieving top-%d chunks for query: %r", top_k, query[:80])
        t0 = time.perf_counter()

        # Tokenise the query using the same tokeniser as the corpus.
        query_tokens = self._tokenizer(query)

        if not query_tokens:
            LOGGER.warning("Query produced no tokens after tokenisation; returning no results.")
            return []

        # BM25 scores for every document in the corpus.
        import numpy as np  # noqa: PLC0415 -- only imported at query time
        scores = self._bm25.get_scores(query_tokens)

        # Rank by descending score; take top_k candidates.
        # np.argsort returns ascending order, so reverse with [::-1].
        n = min(top_k, len(self._corpus))
        top_indices = scores.argsort()[::-1][:n]

        results: list[RetrievalResult] = []
        rank = 0
        for idx in top_indices:
            bm25_score = float(scores[idx])
            # Chunks with a BM25 score of exactly 0.0 had no matching query
            # terms.  Stop once we reach zero-score results -- they provide
            # no ranking signal and would only add noise.
            if bm25_score <= 0.0:
                break
            rank += 1
            entry = self._corpus[int(idx)]
            results.append(
                RetrievalResult(
                    chunk_id=entry.chunk_id,
                    text=entry.text,
                    score=bm25_score,
                    retrieval_method="bm25",
                    score_type="bm25",
                    rank=rank,
                    metadata=entry.metadata,
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        LOGGER.info(
            "BM25: retrieved %d chunks with score > 0 (%.1f ms)",
            len(results),
            elapsed_ms,
        )
        return results
