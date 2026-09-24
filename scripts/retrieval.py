"""Vector retrieval component for the legal RAG chatbot.

Accepts a Vietnamese legal query, converts it to a 1024-dimensional vector
using the currently selected embedding model, and performs cosine-similarity
search against the model-specific embedding column in ``legal_chunks`` via
pgvector's HNSW index.

The selected model is determined by the ``EMBEDDING_MODEL`` environment
variable (or the ``model_name`` constructor argument).  Retrieval
automatically uses the corresponding embedding column and HNSW index so
that vectors from different models cannot be accidentally mixed.

This module is strictly a **retrieval baseline** — it does not implement hybrid
search, RRF, re-ranking, or generation.

Usage::

    from retrieval import Retriever

    retriever = Retriever(database_url="postgresql://...")
    results = retriever.retrieve(
        "Điều kiện để được hưởng chính sách hỗ trợ là gì?",
        top_k=5,
    )
    for r in results:
        print(r.chunk_id, r.score, r.text[:200])

Environment variables
---------------------
DATABASE_URL
    PostgreSQL connection URL.  Loaded from ``.env`` if not set in the
    environment.
EMBEDDING_MODEL
    Hugging Face model identifier or alias (default: ``BAAI/bge-m3``).

Design notes
------------
* The ``EmbeddingModel`` is loaded **once** at ``Retriever.__init__`` time and
  reused for every query — the model is expensive to load.
* Database connections are **short-lived**: the query embedding is computed
  first (no DB connection held during inference), then a fresh connection is
  opened, the retrieval SQL is executed, and the connection is closed.  This
  preserves the NeonDB / cloud-PostgreSQL timeout-safe pattern established in
  ``index_embeddings.py``.
* Metadata fields returned by the query are limited to columns that actually
  exist in ``legal_chunks`` as defined in ``init.sql``.
* Ranking is based **exclusively** on vector cosine similarity.  Metadata does
  not affect ranking in this baseline.
* Each model's embeddings live in a separate column with its own HNSW index,
  preventing accidental cross-model vector mixing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5
MAX_TOP_K = 1000          # guard against unreasonably large requests
DEFAULT_EF_SEARCH = 40    # pgvector HNSW ef_search default

# ---------------------------------------------------------------------------
# SQL builders (model-specific column)
# ---------------------------------------------------------------------------

# Set HNSW ef_search for this transaction only (connection-scoped, not global).
# set_config(key, value, is_local=true) reverts after the transaction ends,
# which is the correct scope for a single-query retrieval call.
_SET_EF_SEARCH = "SELECT set_config('hnsw.ef_search', %s, true)"


def _build_retrieval_sql(column: str) -> str:
    """Build the retrieval SQL for the given embedding column.

    Cosine *distance* = 1 - cosine_similarity.  We convert to similarity in
    the SELECT so callers receive a score where higher = more relevant.
    """
    return f"""
SELECT
    chunk_id,
    text,
    1 - ({column} <=> %s::vector)  AS similarity,
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
WHERE {column} IS NOT NULL
ORDER BY {column} <=> %s::vector
LIMIT %s
"""


def _build_integrity_sql(column: str) -> str:
    """Build the integrity check SQL for the given embedding column."""
    return f"""
SELECT
    count(*)                                  AS total_chunks,
    count({column})                          AS embedded_chunks,
    count(*) - count({column})               AS missing_embeddings
FROM legal_chunks
"""


_HNSW_INDEX_SQL = """
SELECT indexname
FROM pg_indexes
WHERE tablename = 'legal_chunks'
  AND indexname = %s
"""

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """A single retrieved legal chunk with its similarity score and metadata.

    Attributes
    ----------
    chunk_id:
        Primary key of the chunk in ``legal_chunks``.
    text:
        Full text content of the chunk.
    score:
        Cosine similarity in the range ``[-1, 1]``.  Higher means more
        similar.  In practice, for normalised vectors the range is
        approximately ``[0, 1]``.
    metadata:
        Dictionary of legal metadata columns from ``legal_chunks``.  All
        values come directly from the database; no fields are invented.
        Keys: ``document_id``, ``document_title``, ``document_number``,
        ``source_type``, ``document_role``, ``authority_level``,
        ``retrieval_priority``, ``chapter``, ``article``, ``clause``,
        ``point``, ``content_type``.  Values come directly from the
        database; ``None`` means the field is absent for this chunk.
    """

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path) -> None:
    """Load unset variables from a simple .env file (no extra dependencies)."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _vector_to_pg(vector: list[float]) -> str:
    """Serialise a float list to the pgvector literal format ``'[f1,f2,…]'``."""
    return "[" + ",".join(str(v) for v in vector) + "]"


def _connect(database_url: str) -> Any:
    """Open and return a psycopg (v3) connection."""
    try:
        from psycopg import connect  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is not installed.  Run: pip install psycopg[binary]"
        ) from exc
    return connect(database_url)


def _resolve_database_url(database_url: str | None) -> str:
    """Return a usable DATABASE_URL, loading .env if necessary."""
    if database_url:
        return database_url
    root = Path(__file__).resolve().parents[1]
    _load_dotenv(root / ".env")
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is required.  Set it in the environment or in .env"
        )
    return url


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """Embed a query and retrieve the most similar legal chunks.

    Automatically selects the correct embedding model, embedding column,
    and HNSW index based on the model configuration.

    Parameters
    ----------
    database_url:
        PostgreSQL connection URL.  If ``None``, the value is loaded from the
        ``DATABASE_URL`` environment variable or the ``.env`` file at the
        repository root.
    model_name:
        Hugging Face model identifier or alias.  Defaults to ``BAAI/bge-m3``
        or the ``EMBEDDING_MODEL`` environment variable.
    ef_search:
        HNSW ``ef_search`` value applied per-connection before the retrieval
        query.  Higher values improve recall at the cost of latency.
        Default: 40 (pgvector default).
    """

    def __init__(
        self,
        database_url: str | None = None,
        model_name: str | None = None,
        *,
        ef_search: int = DEFAULT_EF_SEARCH,
    ) -> None:
        # Validate ef_search before any expensive work.
        if isinstance(ef_search, bool) or not isinstance(ef_search, int) or ef_search <= 0:
            raise ValueError(
                f"ef_search must be a positive integer (got {ef_search!r})"
            )

        self._database_url = _resolve_database_url(database_url)
        self._ef_search = ef_search

        # Import EmbeddingModel from the scripts directory.
        import sys  # noqa: PLC0415
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from embedding import EmbeddingModel  # noqa: PLC0415

        LOGGER.info("Loading embedding model...")
        self._embedding_model = EmbeddingModel(model_name=model_name)

        # Extract model-specific column and index names from the config.
        self._embedding_column = self._embedding_model.config.embedding_column
        self._hnsw_index_name = self._embedding_model.config.hnsw_index_name

        # Pre-build SQL with the correct column name.
        self._retrieval_sql = _build_retrieval_sql(self._embedding_column)
        self._integrity_sql = _build_integrity_sql(self._embedding_column)

        # Surface device information so callers can verify GPU/CPU execution.
        _device = self._embedding_model.device
        LOGGER.info("Embedding device: %s", _device)
        if _device == "cuda":
            try:
                import torch  # type: ignore[import]  # noqa: PLC0415
                LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))
            except Exception:
                pass

        LOGGER.info(
            "Retriever ready — model: %s, column: %s, ef_search: %d",
            self._embedding_model.model_name,
            self._embedding_column,
            self._ef_search,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        """Compute device used for query embedding: ``'cuda'`` or ``'cpu'``."""
        return self._embedding_model.device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RetrievalResult]:
        """Embed *query* and return the top-*top_k* most similar legal chunks.

        Parameters
        ----------
        query:
            A natural-language legal question in Vietnamese (or any language
            supported by the selected model).  Must be a non-empty,
            non-whitespace string.
        top_k:
            Number of results to return.  Must be a positive integer not
            exceeding ``MAX_TOP_K`` (``{max_top_k}``).

        Returns
        -------
        list[RetrievalResult]
            Results ordered by descending cosine similarity (most relevant
            first).  May be shorter than *top_k* if the database contains
            fewer than *top_k* embedded chunks.

        Raises
        ------
        ValueError
            If *query* is empty or whitespace-only, or if *top_k* is out of
            the valid range.
        """.format(max_top_k=MAX_TOP_K)

        # --- Input validation ---
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "query must be a non-empty, non-whitespace string"
            )
        # Reject bool first (bool is a subclass of int).
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer (got {top_k!r})"
            )
        if top_k > MAX_TOP_K:
            raise ValueError(
                f"top_k={top_k} exceeds the maximum allowed value ({MAX_TOP_K})"
            )

        LOGGER.info("Retrieving top-%d chunks for query: %r", top_k, query[:80])

        # --- Step 1: Embed query (no DB connection held open during inference) ---
        # Use embed_query() to apply the correct query encoding protocol.
        query_vector = self._embedding_model.embed_query(query)
        query_vector_pg = _vector_to_pg(query_vector)

        # --- Step 2: Short-lived DB connection for retrieval only ---
        results: list[RetrievalResult] = []
        with _connect(self._database_url) as conn:
            with conn.cursor() as cur:
                # Set ef_search for this transaction only.
                cur.execute(_SET_EF_SEARCH, (str(self._ef_search),))

                cur.execute(
                    self._retrieval_sql,
                    (query_vector_pg, query_vector_pg, top_k),
                )
                rows = cur.fetchall()

        for row in rows:
            (
                chunk_id, text, similarity,
                document_id, document_title, document_number,
                source_type, document_role, authority_level,
                retrieval_priority, chapter, article, clause, point, content_type,
            ) = row
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(similarity),
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

        LOGGER.info("Retrieved %d chunks", len(results))
        return results

    # ------------------------------------------------------------------
    # Database integrity check (read-only)
    # ------------------------------------------------------------------

    def verify_database_state(self) -> dict[str, Any]:
        """Run read-only sanity checks and return a result dict.

        Checks:
        - Total chunks and embedded chunks for the model-specific column.
        - Whether the model-specific HNSW index exists.

        This method never modifies the database.
        """
        with _connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(self._integrity_sql)
                row = cur.fetchone()
                total, embedded, missing = row[0], row[1], row[2]

                cur.execute(_HNSW_INDEX_SQL, (self._hnsw_index_name,))
                hnsw_exists = cur.fetchone() is not None

        return {
            "total_chunks": total,
            "embedded_chunks": embedded,
            "missing_embeddings": missing,
            "hnsw_index_exists": hnsw_exists,
        }
