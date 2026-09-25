"""Unit tests for scripts/bm25_retriever.py.

Mocking strategy:
  - psycopg is replaced by MagicMock so no real database is needed.
  - rank_bm25 is replaced by a deterministic fake so the library does not
    need to be installed (though it IS listed in requirements.txt).
  - No embedding model is loaded or called -- BM25 is model-independent.

Run from the repository root::

    conda activate chatbot
    python -m pytest tests/test_bm25_retriever.py -v

To run all tests including regression::

    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Make scripts/ importable regardless of how pytest is invoked.
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Fake rank_bm25 module
# ---------------------------------------------------------------------------

FAKE_CORPUS_TEXTS = [
    "Điều 1. Phạm vi điều chỉnh của Nghị định này",
    "Điều 4. Học phí và sinh hoạt phí được hỗ trợ",
    "Điều 6. Nghĩa vụ hoàn trả sau khi tốt nghiệp",
    "Điều 7. Điều kiện để được hưởng chính sách hỗ trợ",
    "Điều 12. Trách nhiệm của cơ sở đào tạo giáo viên",
]

# Deterministic fake BM25 implementation for testing.
# Returns scores based on simple term overlap.


class _FakeBM25:
    """Deterministic fake BM25Okapi for testing."""

    def __init__(self, corpus_tokens, k1=1.5, b=0.75, epsilon=0.25):
        self._corpus = corpus_tokens
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

    def get_scores(self, query_tokens):
        import numpy as np
        scores = []
        query_set = set(query_tokens)
        for doc_tokens in self._corpus:
            doc_set = set(doc_tokens)
            overlap = len(query_set & doc_set)
            # Give decreasing scores so the order is deterministic.
            scores.append(float(overlap))
        return np.array(scores, dtype=float)


def _make_fake_rank_bm25_module() -> types.ModuleType:
    fake_pkg = types.ModuleType("rank_bm25")
    fake_pkg.BM25Okapi = _FakeBM25
    return fake_pkg


# ---------------------------------------------------------------------------
# Fake psycopg connection
# ---------------------------------------------------------------------------

_FAKE_DB_ROWS = [
    (
        "chunk-001",
        "Điều 1. Phạm vi điều chỉnh của Nghị định này",
        "doc-116-2020",
        "Nghị định 116/2020/NĐ-CP",
        "116/2020/NĐ-CP",
        "core",
        "primary",
        "primary_legal_source",
        100,
        "Chương I",
        "Điều 1",
        "Khoản 1",
        None,
        "legal_text",
    ),
    (
        "chunk-002",
        "Điều 4. Học phí và sinh hoạt phí được hỗ trợ theo Nghị định",
        "doc-116-2020",
        "Nghị định 116/2020/NĐ-CP",
        "116/2020/NĐ-CP",
        "core",
        "primary",
        "primary_legal_source",
        100,
        "Chương II",
        "Điều 4",
        "Khoản 1",
        None,
        "legal_text",
    ),
    (
        "chunk-003",
        "Điều 6. Nghĩa vụ hoàn trả học phí sau khi tốt nghiệp",
        "doc-116-2020",
        "Nghị định 116/2020/NĐ-CP",
        "116/2020/NĐ-CP",
        "core",
        "primary",
        "primary_legal_source",
        100,
        "Chương II",
        "Điều 6",
        "Khoản 1",
        None,
        "legal_text",
    ),
]


def _make_fake_psycopg_connect(rows=None):
    """Return a mock psycopg connection that returns the given rows."""
    if rows is None:
        rows = _FAKE_DB_ROWS

    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows
    mock_cursor.__enter__ = lambda s: mock_cursor
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)

    return mock_conn


# ---------------------------------------------------------------------------
# Helper: from retrievers import bm25 as bm25_retriever with fakes installed
# ---------------------------------------------------------------------------


def _import_bm25_retriever():
    """Import (or re-import) bm25_retriever with rank_bm25 faked."""
    sys.modules.pop("bm25_retriever", None)
    sys.modules.pop("retrieval", None)
    sys.modules["rank_bm25"] = _make_fake_rank_bm25_module()
    # Also remove FlagEmbedding from modules cache to avoid stale import errors.
    sys.modules.pop("FlagEmbedding", None)

    from retrievers import bm25  # noqa: PLC0415
    return bm25


class _BM25TestBase(unittest.TestCase):
    """Base class: installs fakes before each test and cleans up after."""

    def setUp(self):
        sys.modules["rank_bm25"] = _make_fake_rank_bm25_module()
        sys.modules.pop("bm25_retriever", None)
        sys.modules.pop("retrieval", None)

    def tearDown(self):
        sys.modules.pop("rank_bm25", None)
        sys.modules.pop("bm25_retriever", None)
        sys.modules.pop("retrieval", None)


# ---------------------------------------------------------------------------
# 1. Tokeniser
# ---------------------------------------------------------------------------


class TestDefaultTokenizer(_BM25TestBase):
    """Tests for the standalone default_tokenizer function."""

    def _get_tokenizer(self):
        bm25 = _import_bm25_retriever()
        return bm25.default_tokenizer

    def test_splits_on_whitespace(self):
        tok = self._get_tokenizer()
        result = tok("Điều 1 phạm vi")
        self.assertEqual(result, ["điều", "1", "phạm", "vi"])

    def test_lowercases_output(self):
        tok = self._get_tokenizer()
        result = tok("NGhị ĐỊNH")
        self.assertIn("nghị", result)
        self.assertIn("định", result)

    def test_splits_on_punctuation(self):
        tok = self._get_tokenizer()
        result = tok("khoản 1, điều 4: hỗ trợ")
        # After splitting on comma and colon, expect individual words.
        self.assertIn("khoản", result)
        self.assertIn("1", result)
        self.assertIn("hỗ", result)

    def test_empty_string_returns_empty_list(self):
        tok = self._get_tokenizer()
        self.assertEqual(tok(""), [])

    def test_whitespace_only_returns_empty_list(self):
        tok = self._get_tokenizer()
        self.assertEqual(tok("   \t\n  "), [])

    def test_no_empty_tokens_in_output(self):
        tok = self._get_tokenizer()
        result = tok("  a  b  c  ")
        for t in result:
            self.assertTrue(t)  # no empty strings

    def test_deterministic_output(self):
        tok = self._get_tokenizer()
        text = "Điều kiện để được hưởng chính sách hỗ trợ là gì?"
        self.assertEqual(tok(text), tok(text))


# ---------------------------------------------------------------------------
# 2. Parameter validation
# ---------------------------------------------------------------------------


class TestBM25ParameterValidation(_BM25TestBase):
    """BM25 constructor must reject invalid parameter values."""

    def _make_retriever(self, bm25_mod, **kwargs):
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25_mod, "_connect", return_value=fake_conn):
            return bm25_mod.BM25Retriever(
                database_url="postgresql://fake/db", **kwargs
            )

    def test_k1_zero_raises(self):
        bm25 = _import_bm25_retriever()
        with self.assertRaises(ValueError):
            self._make_retriever(bm25, k1=0)

    def test_k1_negative_raises(self):
        bm25 = _import_bm25_retriever()
        with self.assertRaises(ValueError):
            self._make_retriever(bm25, k1=-1.0)

    def test_k1_bool_raises(self):
        bm25 = _import_bm25_retriever()
        with self.assertRaises(ValueError):
            self._make_retriever(bm25, k1=True)

    def test_b_negative_raises(self):
        bm25 = _import_bm25_retriever()
        with self.assertRaises(ValueError):
            self._make_retriever(bm25, b=-0.1)

    def test_b_greater_than_one_raises(self):
        bm25 = _import_bm25_retriever()
        with self.assertRaises(ValueError):
            self._make_retriever(bm25, b=1.1)

    def test_b_bool_raises(self):
        bm25 = _import_bm25_retriever()
        with self.assertRaises(ValueError):
            self._make_retriever(bm25, b=True)

    def test_epsilon_negative_raises(self):
        bm25 = _import_bm25_retriever()
        with self.assertRaises(ValueError):
            self._make_retriever(bm25, epsilon=-0.01)

    def test_epsilon_bool_raises(self):
        bm25 = _import_bm25_retriever()
        with self.assertRaises(ValueError):
            self._make_retriever(bm25, epsilon=True)

    def test_valid_defaults_do_not_raise(self):
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(database_url="postgresql://fake/db")
        self.assertEqual(retriever.k1, bm25.DEFAULT_K1)
        self.assertEqual(retriever.b, bm25.DEFAULT_B)
        self.assertEqual(retriever.epsilon, bm25.DEFAULT_EPSILON)

    def test_b_zero_valid(self):
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(database_url="postgresql://fake/db", b=0.0)
        self.assertEqual(retriever.b, 0.0)

    def test_b_one_valid(self):
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(database_url="postgresql://fake/db", b=1.0)
        self.assertEqual(retriever.b, 1.0)

    def test_epsilon_zero_valid(self):
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(database_url="postgresql://fake/db", epsilon=0.0)
        self.assertEqual(retriever.epsilon, 0.0)


# ---------------------------------------------------------------------------
# 3. Corpus loading
# ---------------------------------------------------------------------------


class TestCorpusLoading(_BM25TestBase):
    """Tests for corpus loading from the database."""

    def test_corpus_size_matches_db_rows(self):
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(database_url="postgresql://fake/db")
        self.assertEqual(retriever.corpus_size, len(_FAKE_DB_ROWS))

    def test_empty_corpus_loads_cleanly(self):
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect(rows=[])
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(database_url="postgresql://fake/db")
        self.assertEqual(retriever.corpus_size, 0)

    def test_reload_corpus_updates_size(self):
        bm25 = _import_bm25_retriever()
        fake_conn_initial = _make_fake_psycopg_connect()
        fake_conn_reload = _make_fake_psycopg_connect(rows=_FAKE_DB_ROWS[:1])
        side_effects = iter([fake_conn_initial, fake_conn_reload])
        with patch.object(bm25, "_connect", side_effect=lambda url: next(side_effects)):
            retriever = bm25.BM25Retriever(database_url="postgresql://fake/db")
            self.assertEqual(retriever.corpus_size, len(_FAKE_DB_ROWS))
            retriever.reload_corpus()
        self.assertEqual(retriever.corpus_size, 1)

    def test_custom_tokenizer_is_used_during_loading(self):
        """A custom tokeniser must be applied when building the BM25 index."""
        bm25 = _import_bm25_retriever()
        captured_texts = []

        def custom_tok(text):
            captured_texts.append(text)
            return text.lower().split()

        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(
                database_url="postgresql://fake/db", tokenizer=custom_tok
            )

        # The custom tokeniser should have been called once per corpus entry.
        self.assertEqual(len(captured_texts), len(_FAKE_DB_ROWS))


# ---------------------------------------------------------------------------
# 4. Query execution
# ---------------------------------------------------------------------------


class TestQueryExecution(_BM25TestBase):
    """Tests for the retrieve() method."""

    def _make_retriever(self, bm25_mod, rows=None):
        fake_conn = _make_fake_psycopg_connect(rows=rows)
        with patch.object(bm25_mod, "_connect", return_value=fake_conn):
            return bm25_mod.BM25Retriever(database_url="postgresql://fake/db")

    def test_retrieve_returns_list(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều 1 phạm vi", top_k=5)
        self.assertIsInstance(results, list)

    def test_retrieve_returns_retrieval_results(self):
        """All returned items must be RetrievalResult instances."""
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)

        # Import RetrievalResult from the (possibly mocked) retrieval module.
        sys.modules.pop("retrieval", None)
        # Inject a fake FlagEmbedding to prevent ImportError in retrieval.py
        import types as _types
        fake_fe = _types.ModuleType("FlagEmbedding")
        fake_fe.BGEM3FlagModel = MagicMock()
        sys.modules.setdefault("FlagEmbedding", fake_fe)
        from retrievers.hnsw import RetrievalResult  # noqa: PLC0415

        results = retriever.retrieve("học phí sinh hoạt", top_k=3)
        for r in results:
            self.assertIsInstance(r, RetrievalResult)

    def test_retrieve_score_is_float(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều 1", top_k=5)
        for r in results:
            self.assertIsInstance(r.score, float)

    def test_retrieve_score_is_positive(self):
        """All returned results must have a positive BM25 score (zero excluded)."""
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều 1", top_k=5)
        for r in results:
            self.assertGreater(r.score, 0.0)

    def test_retrieve_results_ordered_by_descending_score(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("học phí sinh hoạt phí hỗ trợ", top_k=5)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_retrieve_top_k_limits_results(self):
        """retrieve(top_k=1) must return at most 1 result."""
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều 1 phạm vi điều chỉnh", top_k=1)
        self.assertLessEqual(len(results), 1)

    def test_retrieve_top_k_not_exceeded(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều phạm vi hỗ trợ nghĩa vụ", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_retrieve_empty_corpus_returns_empty(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25, rows=[])
        results = retriever.retrieve("Điều 1", top_k=5)
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# 5. Ranking and score semantics
# ---------------------------------------------------------------------------


class TestRankingAndScores(_BM25TestBase):
    """Tests for rank assignment and score semantics."""

    def _make_retriever(self, bm25_mod, rows=None):
        fake_conn = _make_fake_psycopg_connect(rows=rows)
        with patch.object(bm25_mod, "_connect", return_value=fake_conn):
            return bm25_mod.BM25Retriever(database_url="postgresql://fake/db")

    def test_rank_starts_at_one(self):
        """First result must have rank == 1."""
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("học phí sinh hoạt phí", top_k=5)
        if results:
            self.assertEqual(results[0].rank, 1)

    def test_ranks_are_sequential(self):
        """Ranks must be 1, 2, 3, ... without gaps."""
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều học phí nghĩa vụ", top_k=5)
        for expected_rank, r in enumerate(results, start=1):
            self.assertEqual(r.rank, expected_rank)

    def test_retrieval_method_is_bm25(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều 1", top_k=5)
        for r in results:
            self.assertEqual(r.retrieval_method, "bm25")

    def test_score_type_is_bm25(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều 1", top_k=5)
        for r in results:
            self.assertEqual(r.score_type, "bm25")

    def test_chunk_id_is_string(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều học phí", top_k=5)
        for r in results:
            self.assertIsInstance(r.chunk_id, str)
            self.assertTrue(r.chunk_id)  # non-empty

    def test_text_is_string(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều học phí", top_k=5)
        for r in results:
            self.assertIsInstance(r.text, str)

    def test_metadata_contains_expected_keys(self):
        """RetrievalResult.metadata must contain all schema-backed keys."""
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều 1", top_k=1)
        if results:
            meta = results[0].metadata
            expected_keys = {
                "document_id",
                "document_title",
                "document_number",
                "source_type",
                "document_role",
                "authority_level",
                "retrieval_priority",
                "chapter",
                "article",
                "clause",
                "point",
                "content_type",
            }
            self.assertTrue(expected_keys.issubset(set(meta.keys())), meta.keys())


# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation(_BM25TestBase):
    """BM25 retrieve() must reject invalid inputs with ValueError."""

    def _make_retriever(self, bm25_mod):
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25_mod, "_connect", return_value=fake_conn):
            return bm25_mod.BM25Retriever(database_url="postgresql://fake/db")

    def test_empty_string_raises(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        with self.assertRaises(ValueError):
            retriever.retrieve("")

    def test_whitespace_only_raises(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        with self.assertRaises(ValueError):
            retriever.retrieve("   \t\n  ")

    def test_top_k_zero_raises(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        with self.assertRaises(ValueError):
            retriever.retrieve("Điều 1", top_k=0)

    def test_top_k_negative_raises(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        with self.assertRaises(ValueError):
            retriever.retrieve("Điều 1", top_k=-5)

    def test_top_k_bool_raises(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        with self.assertRaises(ValueError):
            retriever.retrieve("Điều 1", top_k=True)

    def test_top_k_float_raises(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        with self.assertRaises(ValueError):
            retriever.retrieve("Điều 1", top_k=1.5)

    def test_top_k_exceeds_max_raises(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        with self.assertRaises(ValueError):
            retriever.retrieve("Điều 1", top_k=bm25.MAX_TOP_K + 1)

    def test_top_k_one_valid(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều 1", top_k=1)
        self.assertIsInstance(results, list)

    def test_top_k_max_valid(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("Điều 1", top_k=bm25.MAX_TOP_K)
        self.assertIsInstance(results, list)

    def test_query_none_raises(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        with self.assertRaises((ValueError, AttributeError, TypeError)):
            retriever.retrieve(None)


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism(_BM25TestBase):
    """BM25 results must be deterministic for the same corpus and query."""

    def test_same_query_same_results(self):
        """Running the same query twice must return identical results."""
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect()

        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(database_url="postgresql://fake/db")

        query = "Điều kiện hưởng học phí"
        results_a = retriever.retrieve(query, top_k=5)
        results_b = retriever.retrieve(query, top_k=5)

        self.assertEqual(len(results_a), len(results_b))
        for ra, rb in zip(results_a, results_b):
            self.assertEqual(ra.chunk_id, rb.chunk_id)
            self.assertAlmostEqual(ra.score, rb.score, places=10)
            self.assertEqual(ra.rank, rb.rank)

    def test_different_queries_may_differ(self):
        """Different queries may produce different results or scores."""
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(database_url="postgresql://fake/db")

        # These queries are designed to match different chunks.
        r1 = retriever.retrieve("điều 1 phạm vi", top_k=3)
        r2 = retriever.retrieve("học phí sinh hoạt hỗ trợ", top_k=3)

        # At minimum, the two queries should not crash.
        self.assertIsInstance(r1, list)
        self.assertIsInstance(r2, list)


# ---------------------------------------------------------------------------
# 8. Compatibility with evaluation framework
# ---------------------------------------------------------------------------


class TestEvaluationCompatibility(_BM25TestBase):
    """Verify BM25 results can be consumed by the evaluation layer.

    The evaluation layer in scripts/test_retrieval.py accesses:
      - result.metadata  (dict with legal hierarchy keys)
      - result.score     (float)
      - result.chunk_id  (str)
      - result.text      (str)

    It does NOT depend on retrieval_method, score_type, or rank.
    """

    def _make_retriever(self, bm25_mod):
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25_mod, "_connect", return_value=fake_conn):
            return bm25_mod.BM25Retriever(database_url="postgresql://fake/db")

    def test_metadata_document_title_accessible(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("điều 1", top_k=3)
        for r in results:
            # Should not raise; value may be None for some chunks.
            _ = r.metadata.get("document_title")

    def test_metadata_article_accessible(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("điều 4", top_k=3)
        for r in results:
            _ = r.metadata.get("article")

    def test_metadata_content_type_accessible(self):
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("điều", top_k=3)
        for r in results:
            ct = r.metadata.get("content_type")
            # content_type should be "legal_text" for all fake rows.
            if ct is not None:
                self.assertIsInstance(ct, str)

    def test_chunk_id_consistent_with_hnsw_convention(self):
        """chunk_id from BM25 must be a non-empty string (same as HNSW)."""
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("điều 1 phạm vi", top_k=5)
        for r in results:
            self.assertIsInstance(r.chunk_id, str)
            self.assertTrue(r.chunk_id)

    def test_rank_one_convention_same_as_hnsw(self):
        """Rank 1 must be the best result (convention shared with HNSW)."""
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("điều 1 phạm vi", top_k=5)
        if results:
            self.assertEqual(results[0].rank, 1)

    def test_scores_do_not_have_score_type_cosine_similarity(self):
        """BM25 must NOT claim cosine_similarity as its score type."""
        bm25 = _import_bm25_retriever()
        retriever = self._make_retriever(bm25)
        results = retriever.retrieve("điều 1", top_k=5)
        for r in results:
            self.assertNotEqual(r.score_type, "cosine_similarity")

    def test_hnsw_result_has_different_score_type(self):
        """Verify that HNSW and BM25 results carry different score_type values.

        This ensures the evaluation layer can distinguish them without
        examining the score numerical value.
        """
        # Import retrieval module with fake FlagEmbedding.
        import types as _types
        fake_fe = _types.ModuleType("FlagEmbedding")
        fake_model_cls = MagicMock()
        fake_model_inst = MagicMock()
        import numpy as np
        fake_model_inst.encode.return_value = {
            "dense_vecs": np.ones((1, 1024), dtype="float32")
        }
        fake_model_cls.return_value = fake_model_inst
        fake_fe.BGEM3FlagModel = fake_model_cls
        sys.modules["FlagEmbedding"] = fake_fe
        sys.modules.pop("retrieval", None)

        from retrievers import hnsw as rv  # noqa: PLC0415

        fake_hnsw_conn = _make_fake_psycopg_connect(rows=[
            (
                "chunk-001",
                "Điều 1. Phạm vi điều chỉnh",
                0.92,
                "doc-116-2020",
                "Nghị định 116/2020/NĐ-CP",
                "116/2020/NĐ-CP",
                "core",
                "primary",
                "primary_legal_source",
                100,
                "Chương I",
                "Điều 1",
                None,
                None,
                "legal_text",
            )
        ])
        with patch.object(rv, "_connect", return_value=fake_hnsw_conn):
            hnsw_retriever = rv.Retriever(database_url="postgresql://fake/db")
            hnsw_results = hnsw_retriever.retrieve("Điều 1", top_k=1)

        bm25 = _import_bm25_retriever()
        fake_bm25_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_bm25_conn):
            bm25_retriever = bm25.BM25Retriever(database_url="postgresql://fake/db")
        bm25_results = bm25_retriever.retrieve("điều 1", top_k=1)

        if hnsw_results and bm25_results:
            self.assertNotEqual(
                hnsw_results[0].score_type,
                bm25_results[0].score_type,
            )
            self.assertEqual(hnsw_results[0].score_type, "cosine_similarity")
            self.assertEqual(bm25_results[0].score_type, "bm25")


# ---------------------------------------------------------------------------
# 9. Properties
# ---------------------------------------------------------------------------


class TestProperties(_BM25TestBase):
    """Tests for BM25Retriever property accessors."""

    def test_corpus_size_property(self):
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(database_url="postgresql://fake/db")
        self.assertEqual(retriever.corpus_size, len(_FAKE_DB_ROWS))

    def test_k1_property(self):
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(
                database_url="postgresql://fake/db", k1=2.0
            )
        self.assertEqual(retriever.k1, 2.0)

    def test_b_property(self):
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(
                database_url="postgresql://fake/db", b=0.5
            )
        self.assertEqual(retriever.b, 0.5)

    def test_epsilon_property(self):
        bm25 = _import_bm25_retriever()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(bm25, "_connect", return_value=fake_conn):
            retriever = bm25.BM25Retriever(
                database_url="postgresql://fake/db", epsilon=0.1
            )
        self.assertEqual(retriever.epsilon, 0.1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
