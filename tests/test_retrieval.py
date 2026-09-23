"""Unit tests for scripts/retrieval.py.

Uses the same mocking strategy as tests/test_embedding.py:
- FlagEmbedding (and therefore BAAI/bge-m3) is replaced by a lightweight
  fake so the tests run without downloading the 2.3 GB model.
- psycopg is replaced by a MagicMock so no real database is needed.

Run from the repository root::

    conda activate chatbot
    python -m pytest tests/test_retrieval.py -v
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
# Shared fake helpers (mirrors test_embedding.py conventions)
# ---------------------------------------------------------------------------

FAKE_DIM = 1024
FAKE_VECTOR = [0.1] * FAKE_DIM


def _fake_encode(
    texts,
    batch_size=32,
    max_length=8192,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False,
):
    import numpy as np  # noqa: PLC0415
    return {"dense_vecs": np.ones((len(texts), FAKE_DIM), dtype="float32")}


def _make_fake_flag_embedding_module() -> types.ModuleType:
    fake_pkg = types.ModuleType("FlagEmbedding")
    fake_model_cls = MagicMock(name="BGEM3FlagModel")
    fake_model_instance = MagicMock(name="bgem3_instance")
    fake_model_instance.encode.side_effect = _fake_encode
    fake_model_cls.return_value = fake_model_instance
    fake_pkg.BGEM3FlagModel = fake_model_cls
    return fake_pkg


def _make_fake_psycopg_connect(rows=None):
    """Return a factory that yields a mock psycopg connection.

    Parameters
    ----------
    rows:
        Rows that ``cursor.fetchall()`` will return.  Defaults to a single
        row with plausible legal_chunks data.
    """
    if rows is None:
        rows = [
            (
                "chunk-001",            # chunk_id
                "Điều 1. Phạm vi điều chỉnh...",  # text
                0.92,                   # similarity
                "doc-116-2020",         # document_id
                "Nghị định 116/2020/NĐ-CP",  # document_title
                "116/2020/NĐ-CP",       # document_number
                "core",                 # source_type
                "primary",              # document_role
                "primary_legal_source", # authority_level
                100,                    # retrieval_priority
                "Chương I",             # chapter
                "Điều 1",              # article
                None,                   # clause
                None,                   # point
                "legal_text",           # content_type
            )
        ]

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
# Fixtures — module reload helpers
# ---------------------------------------------------------------------------


def _import_retrieval():
    """Import (or re-import) the retrieval module with fakes in place."""
    for mod in ("retrieval", "embedding"):
        sys.modules.pop(mod, None)
    import retrieval as rv  # noqa: PLC0415
    return rv


class _RetrievalTestBase(unittest.TestCase):
    """Base class that installs fakes before each test and cleans up after."""

    def setUp(self):
        self.fake_fe = _make_fake_flag_embedding_module()
        sys.modules["FlagEmbedding"] = self.fake_fe
        for mod in ("retrieval", "embedding"):
            sys.modules.pop(mod, None)

    def tearDown(self):
        sys.modules.pop("FlagEmbedding", None)
        for mod in ("retrieval", "embedding"):
            sys.modules.pop(mod, None)


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------


class TestRetrieverInit(_RetrievalTestBase):
    def test_embedding_model_loaded_once(self):
        """Retriever.__init__ should load EmbeddingModel exactly once."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")

        # The fake BGEM3FlagModel constructor should have been called once.
        self.assertEqual(self.fake_fe.BGEM3FlagModel.call_count, 1)

    def test_model_not_reloaded_across_queries(self):
        """Calling retrieve() twice must not reload the model."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")
            retriever.retrieve("Điều 1 là gì?", top_k=3)
            retriever.retrieve("Điều 2 là gì?", top_k=3)

        # Constructor called once, not three times.
        self.assertEqual(self.fake_fe.BGEM3FlagModel.call_count, 1)


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation(_RetrievalTestBase):
    def _make_retriever(self, rv):
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            return rv.Retriever(database_url="postgresql://fake/db")

    def test_empty_string_raises_value_error(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("")

    def test_whitespace_only_raises_value_error(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("   \t\n  ")

    def test_top_k_zero_raises_value_error(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("Điều 1", top_k=0)

    def test_top_k_negative_raises_value_error(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("Điều 1", top_k=-5)

    def test_top_k_exceeds_maximum_raises_value_error(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("Điều 1", top_k=rv.MAX_TOP_K + 1)

    def test_valid_query_does_not_raise(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")
            # Should not raise.
            results = retriever.retrieve("Điều kiện hưởng hỗ trợ là gì?", top_k=5)
            self.assertIsInstance(results, list)


# ---------------------------------------------------------------------------
# 3. Query embedding
# ---------------------------------------------------------------------------


class TestQueryEmbedding(_RetrievalTestBase):
    def test_embed_texts_called_with_raw_query(self):
        """retrieve() must pass the raw query string to embed_texts."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        query = "Điều kiện để được hưởng chính sách hỗ trợ là gì?"

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")
            retriever.retrieve(query, top_k=3)

        model_instance = self.fake_fe.BGEM3FlagModel.return_value
        call_args = model_instance.encode.call_args
        texts_sent = call_args[0][0]  # first positional argument
        self.assertIn(query, texts_sent)

    def test_query_produces_1024d_vector(self):
        """The embedding must be 1024-dimensional before being sent to the DB."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")

        # Capture vectors produced by embed_texts.
        original_embed = retriever._embedding_model.embed_texts

        captured: list[list[float]] = []

        def _capturing_embed(texts, **kwargs):
            result = original_embed(texts, **kwargs)
            captured.extend(result)
            return result

        retriever._embedding_model.embed_texts = _capturing_embed
        fake_conn2 = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn2):
            retriever.retrieve("Điều 1 là gì?", top_k=2)

        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), FAKE_DIM)


# ---------------------------------------------------------------------------
# 4. Database interaction
# ---------------------------------------------------------------------------


class TestDatabaseInteraction(_RetrievalTestBase):
    def _make_retriever(self, rv, fake_conn=None):
        if fake_conn is None:
            fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            return rv.Retriever(database_url="postgresql://fake/db")

    def test_returns_list_of_retrieval_results(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("Điều 1 là gì?", top_k=5)

        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, rv.RetrievalResult) for r in results))

    def test_score_is_numeric(self):
        """Score must be a float (no type constraint on range)."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("Điều 1 là gì?", top_k=5)

        for r in results:
            self.assertIsInstance(r.score, float)

    def test_results_ordered_by_descending_score(self):
        """Results must be ordered highest similarity first."""
        rv = _import_retrieval()
        # Provide multiple rows with decreasing similarity.
        rows = [
            ("c1", "text1", 0.95, "d1", "Title1", "N1", "core", "primary",
             "primary_legal_source", 100, "Chương I", "Điều 1", None, None, "legal_text"),
            ("c2", "text2", 0.80, "d1", "Title1", "N1", "core", "primary",
             "primary_legal_source", 100, "Chương I", "Điều 2", None, None, "legal_text"),
            ("c3", "text3", 0.65, "d1", "Title1", "N1", "core", "primary",
             "primary_legal_source", 100, "Chương I", "Điều 3", None, None, "legal_text"),
        ]
        fake_conn = _make_fake_psycopg_connect(rows=rows)
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("câu hỏi pháp lý", top_k=3)

        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_forwarded_to_db(self):
        """LIMIT parameter passed to the DB cursor must equal top_k."""
        rv = _import_retrieval()
        captured_params: list = []

        def _make_tracking_conn(rows=None):
            """Build a mock connection whose cursor.execute records every call."""
            if rows is None:
                rows = []
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = rows
            mock_cur.fetchone.return_value = None
            mock_cur.__enter__ = lambda s: mock_cur
            mock_cur.__exit__ = MagicMock(return_value=False)

            def _execute(sql, params=None):
                captured_params.append((sql, params))

            mock_cur.execute.side_effect = _execute

            mock_conn = MagicMock()
            mock_conn.cursor.return_value = mock_cur
            mock_conn.__enter__ = lambda s: mock_conn
            mock_conn.__exit__ = MagicMock(return_value=False)
            return mock_conn

        # __init__ gets the first conn (for no internal DB call),
        # retrieve() gets the second.
        conns = iter([_make_tracking_conn(), _make_tracking_conn()])
        with patch.object(rv, "_connect", side_effect=lambda url: next(conns)):
            retriever = rv.Retriever(database_url="postgresql://fake/db")
            retriever.retrieve("câu hỏi", top_k=7)

        retrieval_calls = [
            p for sql, p in captured_params
            if p is not None and "ORDER BY embedding" in sql
        ]
        self.assertTrue(len(retrieval_calls) > 0, "Retrieval SQL was not called")
        self.assertEqual(retrieval_calls[0][2], 7)

    def test_empty_db_returns_empty_list(self):
        """No results from DB should return an empty list (no exception)."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect(rows=[])
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("câu hỏi", top_k=5)

        self.assertEqual(results, [])

    def test_metadata_contains_expected_keys(self):
        """RetrievalResult.metadata must contain all schema-backed keys."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("Điều 1 là gì?", top_k=1)

        self.assertEqual(len(results), 1)
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

    def test_chunk_id_and_text_populated(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("Điều 1 là gì?", top_k=1)

        r = results[0]
        self.assertEqual(r.chunk_id, "chunk-001")
        self.assertIn("Điều 1", r.text)


# ---------------------------------------------------------------------------
# 5. ef_search configuration
# ---------------------------------------------------------------------------


class TestEfSearch(_RetrievalTestBase):
    def test_set_config_called_with_ef_search(self):
        """set_config for hnsw.ef_search should be called before the retrieval SQL."""
        rv = _import_retrieval()
        executed_sqls: list[str] = []

        def _make_tracking_conn():
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = []
            mock_cur.fetchone.return_value = None
            mock_cur.__enter__ = lambda s: mock_cur
            mock_cur.__exit__ = MagicMock(return_value=False)

            def _execute(sql, params=None):
                executed_sqls.append(sql)

            mock_cur.execute.side_effect = _execute

            mock_conn = MagicMock()
            mock_conn.cursor.return_value = mock_cur
            mock_conn.__enter__ = lambda s: mock_conn
            mock_conn.__exit__ = MagicMock(return_value=False)
            return mock_conn

        conns = iter([_make_tracking_conn(), _make_tracking_conn()])
        with patch.object(rv, "_connect", side_effect=lambda url: next(conns)):
            retriever = rv.Retriever(database_url="postgresql://fake/db", ef_search=80)
            retriever.retrieve("câu hỏi", top_k=5)

        self.assertTrue(
            any("set_config" in sql for sql in executed_sqls),
            f"set_config not found in executed SQL: {executed_sqls}",
        )


# ---------------------------------------------------------------------------
# 6. verify_database_state
# ---------------------------------------------------------------------------


class TestVerifyDatabaseState(_RetrievalTestBase):
    def test_returns_expected_keys(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()

        # Mock the integrity check cursor responses.
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.side_effect = [
            (617, 617, 0),                           # _INTEGRITY_SQL
            ("legal_chunks_embedding_hnsw_idx",),    # _HNSW_INDEX_SQL
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")

        with patch.object(rv, "_connect", return_value=mock_conn):
            state = retriever.verify_database_state()

        self.assertEqual(state["total_chunks"], 617)
        self.assertEqual(state["embedded_chunks"], 617)
        self.assertEqual(state["missing_embeddings"], 0)
        self.assertTrue(state["hnsw_index_exists"])

    def test_detects_missing_hnsw_index(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.side_effect = [
            (617, 617, 0),
            None,    # HNSW index not found
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")

        with patch.object(rv, "_connect", return_value=mock_conn):
            state = retriever.verify_database_state()

        self.assertFalse(state["hnsw_index_exists"])


# ---------------------------------------------------------------------------
# 7. _vector_to_pg helper
# ---------------------------------------------------------------------------


class TestVectorToPg(unittest.TestCase):
    def setUp(self):
        self.fake_fe = _make_fake_flag_embedding_module()
        sys.modules["FlagEmbedding"] = self.fake_fe
        for mod in ("retrieval", "embedding"):
            sys.modules.pop(mod, None)

    def tearDown(self):
        sys.modules.pop("FlagEmbedding", None)
        for mod in ("retrieval", "embedding"):
            sys.modules.pop(mod, None)

    def test_pgvector_literal_format(self):
        rv = _import_retrieval()
        result = rv._vector_to_pg([0.1, 0.2, 0.3])
        self.assertTrue(result.startswith("["))
        self.assertTrue(result.endswith("]"))
        self.assertEqual(result.count(","), 2)

    def test_1024d_vector(self):
        rv = _import_retrieval()
        vec = [0.0] * 1024
        result = rv._vector_to_pg(vec)
        self.assertEqual(result.count(","), 1023)


# ---------------------------------------------------------------------------
# 8. ef_search validation
# ---------------------------------------------------------------------------


class TestEfSearchValidation(_RetrievalTestBase):
    """ef_search must be a positive integer; bool/float/string/None/<=0 are rejected."""

    def _make_retriever_with_ef(self, rv, ef_search):
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            return rv.Retriever(database_url="postgresql://fake/db", ef_search=ef_search)

    def test_ef_search_zero_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, 0)

    def test_ef_search_negative_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, -1)

    def test_ef_search_true_raises(self):
        """True is a bool (subclass of int) and must be rejected."""
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, True)

    def test_ef_search_false_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, False)

    def test_ef_search_float_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, 1.5)

    def test_ef_search_string_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, "40")

    def test_ef_search_none_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, None)

    def test_ef_search_one_valid(self):
        rv = _import_retrieval()
        # Should not raise.
        retriever = self._make_retriever_with_ef(rv, 1)
        self.assertEqual(retriever._ef_search, 1)

    def test_ef_search_default_valid(self):
        rv = _import_retrieval()
        retriever = self._make_retriever_with_ef(rv, 40)
        self.assertEqual(retriever._ef_search, 40)

    def test_ef_search_large_valid(self):
        rv = _import_retrieval()
        retriever = self._make_retriever_with_ef(rv, 200)
        self.assertEqual(retriever._ef_search, 200)


# ---------------------------------------------------------------------------
# 9. Extended top_k validation (bool / float / string / None / MAX_TOP_K)
# ---------------------------------------------------------------------------


class TestTopKExtendedValidation(_RetrievalTestBase):
    """Supplement existing top_k tests with bool, float, string, None, and boundary."""

    def _make_retriever(self, rv):
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            return rv.Retriever(database_url="postgresql://fake/db")

    def test_top_k_true_raises(self):
        """True is a bool (int subclass) and must be rejected."""
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k=True)

    def test_top_k_false_raises(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k=False)

    def test_top_k_float_raises(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k=1.5)

    def test_top_k_string_raises(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k="5")

    def test_top_k_none_raises(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k=None)

    def test_top_k_one_valid(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("câu hỏi", top_k=1)
        self.assertIsInstance(results, list)

    def test_top_k_max_valid(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            # Should not raise.
            retriever.retrieve("câu hỏi", top_k=rv.MAX_TOP_K)

    def test_top_k_max_plus_one_raises(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k=rv.MAX_TOP_K + 1)


# ---------------------------------------------------------------------------
# 10. Device selection — mocked, no physical GPU required
# ---------------------------------------------------------------------------


class TestDeviceSelection(_RetrievalTestBase):
    """EmbeddingModel._detect_device() and EmbeddingModel.device property.

    All tests mock torch.cuda.is_available() so that no NVIDIA GPU is needed.
    """

    def _import_embedding(self):
        """Import the embedding module with the fake FlagEmbedding in place."""
        sys.modules.pop("embedding", None)
        import embedding as em  # noqa: PLC0415
        return em

    def test_cuda_available_selects_cuda(self):
        """When torch.cuda.is_available() returns True, device must be 'cuda'."""
        em = self._import_embedding()
        with patch("torch.cuda.is_available", return_value=True):
            device = em.EmbeddingModel._detect_device()
        self.assertEqual(device, "cuda")

    def test_cuda_unavailable_selects_cpu(self):
        """When torch.cuda.is_available() returns False, device must be 'cpu'."""
        em = self._import_embedding()
        with patch("torch.cuda.is_available", return_value=False):
            device = em.EmbeddingModel._detect_device()
        self.assertEqual(device, "cpu")

    def test_torch_import_error_selects_cpu(self):
        """If torch is not importable at all, device must fall back to 'cpu'."""
        em = self._import_embedding()
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _blocking_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        import builtins  # noqa: PLC0415
        with patch.object(builtins, "__import__", side_effect=_blocking_import):
            device = em.EmbeddingModel._detect_device()
        self.assertEqual(device, "cpu")

    def test_device_property_reflects_detected_device(self):
        """EmbeddingModel.device must equal what _detect_device() returned."""
        em = self._import_embedding()
        # The fake BGEM3FlagModel is already in sys.modules (setUp installs it).
        # Patch _detect_device to return 'cpu' explicitly.
        with patch.object(em.EmbeddingModel, "_detect_device", staticmethod(lambda: "cpu")):
            model = em.EmbeddingModel()
        self.assertEqual(model.device, "cpu")

    def test_retriever_device_property_delegates_to_model(self):
        """Retriever.device must return the same value as its embedding model's device."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")
        # The fake model's _detect_device returns 'cpu' (no real CUDA in tests).
        self.assertIn(retriever.device, ("cuda", "cpu"))
        # Must match the embedding model.
        self.assertEqual(retriever.device, retriever._embedding_model.device)


# ---------------------------------------------------------------------------
# 11. Evaluation metric functions
# ---------------------------------------------------------------------------


def _compute_hit_at_k(results_articles, expected, k):
    """Local copy of compute_hit_at_k for isolated metric tests."""
    if not expected:
        return 0.0
    top_k_set = {a.strip() for a in results_articles[:k] if a}
    expected_set = {e.strip() for e in expected}
    return 1.0 if top_k_set & expected_set else 0.0


def _compute_recall_at_k(results_articles, expected, k):
    """Local copy of true compute_recall_at_k for isolated metric tests."""
    if not expected:
        return 0.0
    top_k_set = {a.strip() for a in results_articles[:k] if a}
    expected_set = {e.strip() for e in expected}
    found = top_k_set & expected_set
    return len(found) / len(expected_set)


def _compute_mrr(results_articles, expected):
    """Local copy of compute_mrr for isolated metric tests."""
    if not expected:
        return 0.0
    expected_set = {e.strip() for e in expected}
    for rank, art in enumerate(results_articles, start=1):
        if art and art.strip() in expected_set:
            return 1.0 / rank
    return 0.0


class TestMetrics(unittest.TestCase):
    """Tests for Hit@K, true Recall@K, and MRR metric functions."""

    # --- Hit@K ---

    def test_hit_single_relevant_found(self):
        self.assertEqual(_compute_hit_at_k(["Điều 1", "Điều 2"], ["Điều 1"], k=5), 1.0)

    def test_hit_single_relevant_not_found(self):
        self.assertEqual(_compute_hit_at_k(["Điều 3", "Điều 4"], ["Điều 1"], k=5), 0.0)

    def test_hit_relevant_outside_k(self):
        """If the relevant article is at rank > k, Hit@K must be 0."""
        articles = ["Điều 3", "Điều 4", "Điều 5", "Điều 1"]
        self.assertEqual(_compute_hit_at_k(articles, ["Điều 1"], k=2), 0.0)
        self.assertEqual(_compute_hit_at_k(articles, ["Điều 1"], k=4), 1.0)

    def test_hit_empty_expected_returns_zero(self):
        self.assertEqual(_compute_hit_at_k(["Điều 1"], [], k=5), 0.0)

    def test_hit_multiple_expected_one_found(self):
        """Hit@K is 1.0 if ANY of the expected articles is found."""
        self.assertEqual(
            _compute_hit_at_k(["Điều 3"], ["Điều 3", "Điều 4"], k=5), 1.0
        )

    def test_hit_multiple_expected_none_found(self):
        self.assertEqual(
            _compute_hit_at_k(["Điều 5", "Điều 6"], ["Điều 3", "Điều 4"], k=5), 0.0
        )

    # --- Recall@K (true recall) ---

    def test_recall_single_relevant_found(self):
        self.assertEqual(_compute_recall_at_k(["Điều 1"], ["Điều 1"], k=5), 1.0)

    def test_recall_single_relevant_not_found(self):
        self.assertEqual(_compute_recall_at_k(["Điều 2"], ["Điều 1"], k=5), 0.0)

    def test_recall_two_expected_both_found(self):
        articles = ["Điều 1", "Điều 4", "Điều 5"]
        self.assertAlmostEqual(_compute_recall_at_k(articles, ["Điều 1", "Điều 4"], k=5), 1.0)

    def test_recall_two_expected_one_found(self):
        """Partial recall: 1 of 2 expected → 0.5."""
        articles = ["Điều 1", "Điều 5", "Điều 6"]
        self.assertAlmostEqual(_compute_recall_at_k(articles, ["Điều 1", "Điều 4"], k=5), 0.5)

    def test_recall_two_expected_none_found(self):
        self.assertAlmostEqual(
            _compute_recall_at_k(["Điều 5"], ["Điều 1", "Điều 4"], k=5), 0.0
        )

    def test_recall_empty_expected_returns_zero(self):
        self.assertEqual(_compute_recall_at_k(["Điều 1"], [], k=5), 0.0)

    def test_recall_partial_within_k(self):
        """Only articles within top-k count."""
        # relevant at ranks 1 and 5 (k=3 should find only rank-1)
        articles = ["Điều 1", "Điều 3", "Điều 6", "Điều 7", "Điều 4"]
        self.assertAlmostEqual(
            _compute_recall_at_k(articles, ["Điều 1", "Điều 4"], k=3), 0.5
        )

    # --- MRR ---

    def test_mrr_relevant_at_rank_1(self):
        self.assertAlmostEqual(_compute_mrr(["Điều 1", "Điều 2"], ["Điều 1"]), 1.0)

    def test_mrr_relevant_at_rank_2(self):
        self.assertAlmostEqual(_compute_mrr(["Điều 2", "Điều 1"], ["Điều 1"]), 0.5)

    def test_mrr_relevant_at_rank_3(self):
        self.assertAlmostEqual(
            _compute_mrr(["Điều 2", "Điều 3", "Điều 1"], ["Điều 1"]), 1 / 3
        )

    def test_mrr_no_relevant(self):
        self.assertEqual(_compute_mrr(["Điều 2", "Điều 3"], ["Điều 1"]), 0.0)

    def test_mrr_empty_expected_returns_zero(self):
        self.assertEqual(_compute_mrr(["Điều 1"], []), 0.0)

    def test_mrr_uses_first_relevant_rank(self):
        """MRR ranks the FIRST relevant hit, not the last."""
        articles = ["Điều 5", "Điều 1", "Điều 1"]
        self.assertAlmostEqual(_compute_mrr(articles, ["Điều 1"]), 0.5)

    def test_recall_equals_hit_for_single_expected(self):
        """For a single expected article, Recall@K and Hit@K must agree."""
        for k in (1, 3, 5):
            for articles in (
                ["Điều 1", "Điều 2"],
                ["Điều 3", "Điều 4"],
            ):
                h = _compute_hit_at_k(articles, ["Điều 1"], k)
                r = _compute_recall_at_k(articles, ["Điều 1"], k)
                self.assertAlmostEqual(h, r, msg=f"k={k}, articles={articles}")


# ---------------------------------------------------------------------------
# 12. Hierarchical Ground Truth Matching
# ---------------------------------------------------------------------------

# Import the hierarchical helpers and dataclasses from the evaluation script.
_SCRIPTS_DIR_FOR_EVAL = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR_FOR_EVAL) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR_FOR_EVAL))


def _import_test_retrieval():
    """Import scripts/test_retrieval.py, avoiding name clash with this file.

    The module MUST be registered in ``sys.modules`` under its name before
    ``exec_module`` is called.  If it is not, the ``dataclasses`` machinery
    fails to resolve ``cls.__module__`` for string annotations and raises
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``.
    """
    import importlib  # noqa: PLC0415
    import importlib.util  # noqa: PLC0415

    mod_name = "test_retrieval_script"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        _SCRIPTS_DIR_FOR_EVAL / "test_retrieval.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    # Register BEFORE exec_module so that dataclasses can resolve cls.__module__.
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return mod


class _FakeResult:
    """Minimal stand-in for RetrievalResult used in metric tests."""

    def __init__(self, score: float, metadata: dict):
        self.score = score
        self.metadata = metadata
        self.chunk_id = "fake-chunk"
        self.text = "fake text"


class TestExpectedSection(unittest.TestCase):
    """Validate ExpectedSection construction and invariants."""

    def setUp(self):
        self.tr = _import_test_retrieval()

    def test_valid_section_no_clauses(self):
        s = self.tr.ExpectedSection("Nghị định 116/2020/NĐ-CP", "Điều 4")
        self.assertEqual(s.document_title, "Nghị định 116/2020/NĐ-CP")
        self.assertEqual(s.article, "Điều 4")
        self.assertEqual(s.clauses, [])

    def test_valid_section_with_clauses(self):
        s = self.tr.ExpectedSection("Nghị định 116/2020/NĐ-CP", "Điều 4", ["Khoản 1"])
        self.assertEqual(s.clauses, ["Khoản 1"])

    def test_empty_document_title_raises(self):
        with self.assertRaises(ValueError):
            self.tr.ExpectedSection("", "Điều 4")

    def test_whitespace_document_title_raises(self):
        with self.assertRaises(ValueError):
            self.tr.ExpectedSection("   ", "Điều 4")

    def test_empty_article_raises(self):
        with self.assertRaises(ValueError):
            self.tr.ExpectedSection("Nghị định 116/2020/NĐ-CP", "")

    def test_whitespace_article_raises(self):
        with self.assertRaises(ValueError):
            self.tr.ExpectedSection("Nghị định 116/2020/NĐ-CP", "  ")


class TestMatchesSection(unittest.TestCase):
    """Unit tests for _matches_section hierarchical matching."""

    _ND116 = "Nghị định 116/2020/NĐ-CP"
    _ND60 = "Nghị định 60/2025/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _section(self, doc=None, art="Điều 4", clauses=None):
        doc = doc or self._ND116
        return self.tr.ExpectedSection(doc, art, clauses or [])

    def _meta(self, doc=None, art="Điều 4", clause="Khoản 1"):
        return {
            "document_title": doc or self._ND116,
            "article": art,
            "clause": clause,
            "content_type": "legal_text",
        }

    # --- Full match ---

    def test_doc_article_clause_all_match(self):
        s = self._section(clauses=["Khoản 1"])
        self.assertTrue(self.tr._matches_section(self._meta(), s))

    # --- Document mismatch ---

    def test_different_document_does_not_match(self):
        """Same article/clause but different document must NOT match."""
        s = self._section(doc=self._ND116, clauses=["Khoản 1"])
        meta = self._meta(doc=self._ND60)   # different doc, same article+clause
        self.assertFalse(self.tr._matches_section(meta, s))

    def test_document_mismatch_ignores_article_and_clause(self):
        s = self._section(doc=self._ND116, art="Điều 1", clauses=["Khoản 1"])
        meta = self._meta(doc=self._ND60, art="Điều 1", clause="Khoản 1")
        self.assertFalse(self.tr._matches_section(meta, s))

    # --- Article mismatch ---

    def test_different_article_does_not_match(self):
        s = self._section(art="Điều 4", clauses=["Khoản 1"])
        meta = self._meta(art="Điều 6", clause="Khoản 1")
        self.assertFalse(self.tr._matches_section(meta, s))

    # --- Clause mismatch when clauses specified ---

    def test_different_clause_does_not_match(self):
        s = self._section(clauses=["Khoản 1"])
        meta = self._meta(clause="Khoản 2")
        self.assertFalse(self.tr._matches_section(meta, s))

    # --- Article-level GT (empty clauses) ---

    def test_empty_clauses_any_clause_accepted(self):
        s = self._section(clauses=[])
        meta = self._meta(clause="Khoản 99")
        self.assertTrue(self.tr._matches_section(meta, s))

    def test_empty_clauses_no_clause_in_meta_accepted(self):
        s = self._section(clauses=[])
        meta = self._meta(clause=None)  # chunk has no clause
        self.assertTrue(self.tr._matches_section(meta, s))

    # --- Case / whitespace insensitivity ---

    def test_case_insensitive_document(self):
        s = self.tr.ExpectedSection(self._ND116.upper(), "Điều 4", ["Khoản 1"])
        self.assertTrue(self.tr._matches_section(self._meta(), s))

    def test_whitespace_stripped_article(self):
        s = self.tr.ExpectedSection(self._ND116, "  Điều 4  ", ["Khoản 1"])
        meta = self._meta(art="Điều 4 ")
        self.assertTrue(self.tr._matches_section(meta, s))

    # --- Multiple clauses in section ---

    def test_one_of_multiple_clauses_matches(self):
        s = self._section(clauses=["Khoản 1", "Khoản 2"])
        meta_k2 = self._meta(clause="Khoản 2")
        self.assertTrue(self.tr._matches_section(meta_k2, s))

    def test_none_of_multiple_clauses_matches(self):
        s = self._section(clauses=["Khoản 1", "Khoản 2"])
        meta_k3 = self._meta(clause="Khoản 3")
        self.assertFalse(self.tr._matches_section(meta_k3, s))


class TestIsGroundTruthMatch(unittest.TestCase):
    """Unit tests for _is_ground_truth_match (OR over multiple sections)."""

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _section(self, art, clauses=None):
        return self.tr.ExpectedSection(self._ND116, art, clauses or [])

    def _meta(self, art, clause="Khoản 1"):
        return {
            "document_title": self._ND116,
            "article": art,
            "clause": clause,
            "content_type": "legal_text",
        }

    def test_empty_sections_returns_false(self):
        self.assertFalse(self.tr._is_ground_truth_match(self._meta("Điều 4"), []))

    def test_single_section_matches(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        self.assertTrue(self.tr._is_ground_truth_match(self._meta("Điều 4"), sections))

    def test_single_section_no_match(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        self.assertFalse(
            self.tr._is_ground_truth_match(self._meta("Điều 6", "Khoản 1"), sections)
        )

    def test_first_section_matches(self):
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        self.assertTrue(
            self.tr._is_ground_truth_match(self._meta("Điều 1", "Khoản 1"), sections)
        )

    def test_second_section_matches(self):
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        self.assertTrue(
            self.tr._is_ground_truth_match(self._meta("Điều 7", "Khoản 2"), sections)
        )

    def test_no_section_matches(self):
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        self.assertFalse(
            self.tr._is_ground_truth_match(self._meta("Điều 9", "Khoản 3"), sections)
        )

    def test_regression_same_article_different_document(self):
        """Chunk from Điều 4 of ND60 must NOT match a section expecting ND116/Điều 4."""
        section = self.tr.ExpectedSection(
            "Nghị định 116/2020/NĐ-CP", "Điều 4", ["Khoản 1"]
        )
        meta_wrong_doc = {
            "document_title": "Nghị định 60/2025/NĐ-CP",
            "article": "Điều 4",
            "clause": "Khoản 1",
        }
        self.assertFalse(self.tr._is_ground_truth_match(meta_wrong_doc, [section]))


class TestHierarchicalMetrics(unittest.TestCase):
    """Tests for compute_hit_at_k, compute_recall_at_k, compute_mrr using
    the hierarchical _FakeResult objects (full metadata path)."""

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _section(self, art, clauses=None):
        return self.tr.ExpectedSection(self._ND116, art, clauses or [])

    def _result(self, art, clause="Khoản 1", doc=None, score=0.9):
        return _FakeResult(
            score=score,
            metadata={
                "document_title": doc or self._ND116,
                "article": art,
                "clause": clause,
                "content_type": "legal_text",
            },
        )

    # --- Hit@K ---

    def test_hit_at_k_match_in_top_k(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [self._result("Điều 4", "Khoản 1")]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=3), 1.0)

    def test_hit_at_k_no_match(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [self._result("Điều 6", "Khoản 1")]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=3), 0.0)

    def test_hit_at_k_match_outside_k(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [
            self._result("Điều 6"),
            self._result("Điều 6"),
            self._result("Điều 6"),
            self._result("Điều 4", "Khoản 1"),  # rank 4
        ]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=3), 0.0)
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=4), 1.0)

    def test_hit_at_k_empty_sections(self):
        results = [self._result("Điều 4")]
        self.assertEqual(self.tr.compute_hit_at_k(results, [], k=5), 0.0)

    def test_hit_at_k_wrong_document_is_no_match(self):
        """A result from a different document must not count as a hit."""
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [self._result("Điều 4", "Khoản 1", doc="Nghị định 60/2025/NĐ-CP")]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=3), 0.0)

    # --- Recall@K ---

    def test_recall_at_k_single_section_found(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [self._result("Điều 4", "Khoản 1")]
        self.assertAlmostEqual(self.tr.compute_recall_at_k(results, sections, k=5), 1.0)

    def test_recall_at_k_two_sections_one_found(self):
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        results = [self._result("Điều 1", "Khoản 1")]  # only Điều 1 found
        self.assertAlmostEqual(self.tr.compute_recall_at_k(results, sections, k=5), 0.5)

    def test_recall_at_k_two_sections_both_found(self):
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        results = [
            self._result("Điều 1", "Khoản 1"),
            self._result("Điều 7", "Khoản 2"),
        ]
        self.assertAlmostEqual(self.tr.compute_recall_at_k(results, sections, k=5), 1.0)

    def test_recall_at_k_section_counted_once_for_duplicate_matches(self):
        """Two results satisfying the same section count as 1 found (not 2)."""
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [
            self._result("Điều 4", "Khoản 1", score=0.9),
            self._result("Điều 4", "Khoản 1", score=0.8),  # same section, duplicate
        ]
        self.assertAlmostEqual(self.tr.compute_recall_at_k(results, sections, k=5), 1.0)

    def test_recall_at_k_empty_sections(self):
        results = [self._result("Điều 4")]
        self.assertEqual(self.tr.compute_recall_at_k(results, [], k=5), 0.0)

    # --- MRR ---

    def test_mrr_first_result_matches(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [self._result("Điều 4", "Khoản 1")]
        self.assertAlmostEqual(self.tr.compute_mrr(results, sections), 1.0)

    def test_mrr_second_result_matches(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [
            self._result("Điều 6"),
            self._result("Điều 4", "Khoản 1"),
        ]
        self.assertAlmostEqual(self.tr.compute_mrr(results, sections), 0.5)

    def test_mrr_no_match(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [self._result("Điều 9")]
        self.assertAlmostEqual(self.tr.compute_mrr(results, sections), 0.0)

    def test_mrr_empty_sections(self):
        results = [self._result("Điều 4")]
        self.assertEqual(self.tr.compute_mrr(results, []), 0.0)


# ---------------------------------------------------------------------------
# 13. ExpectedSection — points field
# ---------------------------------------------------------------------------


class TestExpectedSectionPoints(unittest.TestCase):
    """Validate the new ``points`` field on ExpectedSection."""

    def setUp(self):
        self.tr = _import_test_retrieval()

    def test_points_defaults_to_empty_list(self):
        s = self.tr.ExpectedSection("Nghị định 116/2020/NĐ-CP", "Điều 1")
        self.assertEqual(s.points, [])

    def test_valid_section_with_points(self):
        s = self.tr.ExpectedSection(
            "Nghị định 116/2020/NĐ-CP", "Điều 1", ["Khoản 2"], ["Điểm a"]
        )
        self.assertEqual(s.points, ["Điểm a"])

    def test_valid_section_clauses_and_points(self):
        s = self.tr.ExpectedSection(
            "Nghị định 116/2020/NĐ-CP", "Điều 1",
            clauses=["Khoản 2"],
            points=["Điểm a", "Điểm b"],
        )
        self.assertEqual(len(s.points), 2)

    def test_empty_document_title_still_raises(self):
        with self.assertRaises(ValueError):
            self.tr.ExpectedSection("", "Điều 1", [], ["Điểm a"])

    def test_empty_article_still_raises(self):
        with self.assertRaises(ValueError):
            self.tr.ExpectedSection("Nghị định 116/2020/NĐ-CP", "", [], ["Điểm a"])


# ---------------------------------------------------------------------------
# 14. _matches_section — point-level matching
# ---------------------------------------------------------------------------


class TestMatchesSectionPoints(unittest.TestCase):
    """Tests for the point level of _matches_section (tests 7–9 from spec)."""

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _meta(self, doc=None, art="Điều 1", clause="Khoản 2", point=None,
               content_type="legal_text"):
        return {
            "document_title": doc or self._ND116,
            "article": art,
            "clause": clause,
            "point": point,
            "content_type": content_type,
        }

    def _section(self, art="Điều 1", clauses=None, points=None):
        return self.tr.ExpectedSection(
            self._ND116, art, clauses or [], points or []
        )

    # --- Test 7: exact point match ---
    def test_exact_point_match(self):
        """A chunk whose point equals the expected point must match."""
        s = self._section(clauses=["Khoản 2"], points=["Điểm a"])
        meta = self._meta(clause="Khoản 2", point="Điểm a")
        self.assertTrue(self.tr._matches_section(meta, s))

    # --- Test 8: wrong point does not match ---
    def test_wrong_point_does_not_match(self):
        """A chunk with a different point must NOT match."""
        s = self._section(clauses=["Khoản 2"], points=["Điểm a"])
        meta = self._meta(clause="Khoản 2", point="Điểm b")
        self.assertFalse(self.tr._matches_section(meta, s))

    # --- Test 9: empty points accepts any point ---
    def test_empty_points_accepts_any_point(self):
        """When points=[], any point value (or no point) is acceptable."""
        s = self._section(clauses=["Khoản 2"], points=[])
        meta_point_a = self._meta(clause="Khoản 2", point="Điểm a")
        meta_point_z = self._meta(clause="Khoản 2", point="Điểm z")
        self.assertTrue(self.tr._matches_section(meta_point_a, s))
        self.assertTrue(self.tr._matches_section(meta_point_z, s))

    def test_empty_points_accepts_no_point_in_meta(self):
        """When points=[], a chunk with no point (None) still matches."""
        s = self._section(clauses=["Khoản 2"], points=[])
        meta = self._meta(clause="Khoản 2", point=None)
        self.assertTrue(self.tr._matches_section(meta, s))

    def test_non_empty_points_with_null_chunk_point_does_not_match(self):
        """If points is non-empty but the chunk has no point, it must NOT match."""
        s = self._section(clauses=["Khoản 2"], points=["Điểm a"])
        meta = self._meta(clause="Khoản 2", point=None)
        self.assertFalse(self.tr._matches_section(meta, s))

    def test_point_case_insensitive(self):
        """Point comparison must be case-insensitive."""
        s = self._section(clauses=["Khoản 2"], points=["điểm a"])
        meta = self._meta(clause="Khoản 2", point="Điểm A")
        self.assertTrue(self.tr._matches_section(meta, s))

    def test_one_of_multiple_points_matches(self):
        """If points has multiple entries, matching any one is sufficient."""
        s = self._section(clauses=["Khoản 2"], points=["Điểm a", "Điểm b"])
        meta_b = self._meta(clause="Khoản 2", point="Điểm b")
        self.assertTrue(self.tr._matches_section(meta_b, s))

    def test_none_of_multiple_points_matches(self):
        """A point value not in the expected list must NOT match."""
        s = self._section(clauses=["Khoản 2"], points=["Điểm a", "Điểm b"])
        meta_c = self._meta(clause="Khoản 2", point="Điểm c")
        self.assertFalse(self.tr._matches_section(meta_c, s))


# ---------------------------------------------------------------------------
# 15. _is_legal_source — source classification
# ---------------------------------------------------------------------------


class TestIsLegalSource(unittest.TestCase):
    """Unit tests for _is_legal_source classification helper.

    Verified from actual dataset: all 3 discriminator rules agree on all 617
    chunks.  The cleanest discriminator is content_type == 'legal_text'.
    """

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _meta(self, content_type, source_type=None, document_role=None, authority_level=None):
        return {
            "content_type": content_type,
            "source_type": source_type or "core",
            "document_role": document_role or "primary",
            "authority_level": authority_level or "primary_legal_source",
        }

    def test_core_primary_is_legal(self):
        meta = self._meta("legal_text", "core", "primary", "primary_legal_source")
        self.assertTrue(self.tr._is_legal_source(meta))

    def test_core_amendment_is_legal(self):
        meta = self._meta("legal_text", "core", "amendment", "amending_legal_source")
        self.assertTrue(self.tr._is_legal_source(meta))

    def test_reference_supporting_is_legal(self):
        """Luật Giáo dục 2019 (reference/supporting/supporting_legal_source)
        must be classified as a legal source — it is a fully-structured
        legal document."""
        meta = self._meta("legal_text", "reference", "supporting", "supporting_legal_source")
        self.assertTrue(self.tr._is_legal_source(meta))

    def test_qa_is_not_legal(self):
        meta = self._meta("qa", "qa", "qa", "reference_qa")
        self.assertFalse(self.tr._is_legal_source(meta))

    def test_missing_content_type_is_not_legal(self):
        """A chunk with no content_type must not be classified as legal."""
        meta = {"content_type": None, "source_type": "core", "document_role": "primary"}
        self.assertFalse(self.tr._is_legal_source(meta))

    def test_empty_content_type_is_not_legal(self):
        meta = {"content_type": "", "source_type": "core"}
        self.assertFalse(self.tr._is_legal_source(meta))

    def test_whitespace_content_type_is_not_legal(self):
        meta = {"content_type": "   ", "source_type": "core"}
        self.assertFalse(self.tr._is_legal_source(meta))

    def test_unknown_content_type_is_not_legal(self):
        """An unrecognised content_type string must not default to legal."""
        meta = {"content_type": "unknown_type", "source_type": "core"}
        self.assertFalse(self.tr._is_legal_source(meta))


# ---------------------------------------------------------------------------
# 16–20. QA source isolation
# ---------------------------------------------------------------------------


class TestQASourceIsolation(unittest.TestCase):
    """Tests verifying that QA chunks are NEVER treated as structural legal
    matches, even when their text explicitly mentions a legal article/clause.

    Covers requirement tests 16–20 from the task spec.
    """

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _qa_meta(self, text_mentions_article=True):
        """Build a QA-chunk metadata dict.  The text field mentions a legal
        article/clause, but the structural metadata fields are all None."""
        return {
            "document_title": "Hỏi đáp Nghị định 116/2020/NĐ-CP",
            "document_id": "hoi-dap-nghi-dinh-116-2020-nd-cp",
            "source_type": "qa",
            "document_role": "qa",
            "authority_level": "reference_qa",
            "content_type": "qa",
            "article": None,
            "clause": None,
            "point": None,
            "chapter": None,
        }

    def _legal_meta(self, art="Điều 4", clause="Khoản 1", point=None):
        return {
            "document_title": self._ND116,
            "source_type": "core",
            "document_role": "primary",
            "authority_level": "primary_legal_source",
            "content_type": "legal_text",
            "article": art,
            "clause": clause,
            "point": point,
        }

    def _section(self, art="Điều 4", clauses=None, points=None):
        return self.tr.ExpectedSection(
            self._ND116, art, clauses or [], points or []
        )

    # --- Test 16: QA text mentioning article/clause is NOT a structural match ---
    def test_qa_text_mentioning_article_does_not_match(self):
        """A QA chunk whose text says 'Khoản 1 Điều 4 Nghị định 116...' must
        NOT match an ExpectedSection for Nghị định 116 / Điều 4 / Khoản 1."""
        section = self._section(clauses=["Khoản 1"])
        qa_meta = self._qa_meta()
        self.assertFalse(self.tr._matches_section(qa_meta, section))

    def test_qa_does_not_match_via_is_ground_truth_match(self):
        sections = [self._section(clauses=["Khoản 1"])]
        qa_meta = self._qa_meta()
        self.assertFalse(self.tr._is_ground_truth_match(qa_meta, sections))

    # --- Test 17: QA results are evaluated through text content, not structure ---
    def test_qa_chunk_missing_legal_metadata_does_not_raise(self):
        """_matches_section must return False (not raise) for QA chunks."""
        section = self._section()
        qa_meta = {"content_type": "qa", "article": None, "clause": None, "point": None}
        result = self.tr._matches_section(qa_meta, section)
        self.assertFalse(result)

    # --- Test 18: Missing legal metadata on QA result does not cause structural matching ---
    def test_qa_with_null_structural_fields_is_not_legal_match(self):
        """QA chunk with all structural fields as None must not match any section."""
        section = self._section(art="Điều 1", clauses=[], points=[])
        qa_meta = {
            "content_type": "qa",
            "document_title": None,
            "article": None,
            "clause": None,
            "point": None,
        }
        self.assertFalse(self.tr._matches_section(qa_meta, section))

    # --- Test 19: Legal and QA chunks with similar text are distinguishable ---
    def test_legal_chunk_matches_while_qa_chunk_does_not(self):
        """Given identical document_title / article / clause values, a legal
        chunk (content_type='legal_text') matches while a QA chunk
        (content_type='qa') does not."""
        section = self._section(art="Điều 4", clauses=["Khoản 1"])
        legal_meta = self._legal_meta(art="Điều 4", clause="Khoản 1")
        qa_meta = {
            "document_title": self._ND116,  # same title as the expected doc
            "article": "Điều 4",            # same article
            "clause": "Khoản 1",            # same clause
            "content_type": "qa",           # but it is a QA chunk
        }
        self.assertTrue(self.tr._matches_section(legal_meta, section),
                        "Legal chunk should match")
        self.assertFalse(self.tr._matches_section(qa_meta, section),
                         "QA chunk must NOT match even with matching structural values")

    # --- Test 20: Legal metadata never inferred from text fields ---
    def test_legal_metadata_not_inferred_from_qa_text(self):
        """Even if a QA chunk's text contains a legal reference, structural
        matching must never infer metadata from text."""
        section = self._section(art="Điều 4", clauses=["Khoản 1"])
        # Simulate a QA answer that textually quotes the article/clause.
        qa_with_legal_text = {
            "content_type": "qa",
            "document_title": "Hỏi đáp Nghị định 116/2020/NĐ-CP",
            "article": None,
            "clause": None,
            "point": None,
            "text": "Tại Khoản 1, Điều 4 Nghị định 116/2020/NĐ-CP quy định...",
        }
        self.assertFalse(self.tr._matches_section(qa_with_legal_text, section))

    # --- QA chunk in Hit@K / Recall@K / MRR does not count as structural hit ---
    def test_qa_chunk_does_not_count_as_hit(self):
        sections = [self._section(art="Điều 4", clauses=["Khoản 1"])]
        # A QA result at rank 1 — must not be a hit.
        qa_result = _FakeResult(
            score=0.99,
            metadata=self._qa_meta(),
        )
        self.assertEqual(self.tr.compute_hit_at_k([qa_result], sections, k=3), 0.0)

    def test_qa_chunk_does_not_contribute_to_recall(self):
        sections = [self._section(art="Điều 4", clauses=["Khoản 1"])]
        qa_result = _FakeResult(score=0.99, metadata=self._qa_meta())
        self.assertEqual(self.tr.compute_recall_at_k([qa_result], sections, k=5), 0.0)

    def test_qa_chunk_does_not_contribute_to_mrr(self):
        sections = [self._section(art="Điều 4", clauses=["Khoản 1"])]
        qa_result = _FakeResult(score=0.99, metadata=self._qa_meta())
        self.assertEqual(self.tr.compute_mrr([qa_result], sections), 0.0)


# ---------------------------------------------------------------------------
# 21. reference/supporting source treated as legal (Luật Giáo dục 2019)
# ---------------------------------------------------------------------------


class TestReferenceSourceAsLegal(unittest.TestCase):
    """Verify that reference/supporting chunks (Luật Giáo dục 2019) are
    classified as legal sources and participate in structural matching.

    Dataset fact: Luật Giáo dục 2019 has source_type='reference',
    document_role='supporting', authority_level='supporting_legal_source',
    content_type='legal_text'.  All 461 of its chunks have full structural
    metadata.
    """

    _LGD = "Luật Giáo dục 2019"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _lgd_meta(self, art="Điều 66", clause="Khoản 1", point=None):
        return {
            "document_title": self._LGD,
            "source_type": "reference",
            "document_role": "supporting",
            "authority_level": "supporting_legal_source",
            "content_type": "legal_text",
            "chapter": "Chương IV",
            "article": art,
            "clause": clause,
            "point": point,
        }

    def test_reference_supporting_is_classified_as_legal(self):
        meta = self._lgd_meta()
        self.assertTrue(self.tr._is_legal_source(meta))

    def test_reference_supporting_matches_expected_section(self):
        """A reference/supporting chunk must satisfy an ExpectedSection
        targeting the same document/article/clause."""
        section = self.tr.ExpectedSection(self._LGD, "Điều 66", ["Khoản 1"])
        meta = self._lgd_meta(art="Điều 66", clause="Khoản 1")
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_reference_supporting_wrong_article_does_not_match(self):
        section = self.tr.ExpectedSection(self._LGD, "Điều 66", ["Khoản 1"])
        meta = self._lgd_meta(art="Điều 67", clause="Khoản 1")
        self.assertFalse(self.tr._matches_section(meta, section))

    def test_reference_supporting_counts_in_hit_at_k(self):
        """A legal reference chunk at rank 1 must count as a hit."""
        section = self.tr.ExpectedSection(self._LGD, "Điều 66", ["Khoản 1"])
        result = _FakeResult(score=0.85, metadata=self._lgd_meta())
        self.assertEqual(self.tr.compute_hit_at_k([result], [section], k=3), 1.0)


# ---------------------------------------------------------------------------
# 22. Multiple expected sections — OR semantics + Recall@K counting
# ---------------------------------------------------------------------------


class TestMultipleSectionsOrSemantics(unittest.TestCase):
    """Tests 12–14 from the spec: multiple sections use OR; each counted once."""

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _section(self, art, clauses=None, points=None):
        return self.tr.ExpectedSection(self._ND116, art, clauses or [], points or [])

    def _result(self, art, clause=None, point=None, score=0.9, content_type="legal_text"):
        return _FakeResult(score=score, metadata={
            "document_title": self._ND116,
            "article": art,
            "clause": clause,
            "point": point,
            "content_type": content_type,
        })

    # --- Test 12: multiple expected sections use OR semantics ---
    def test_or_semantics_first_section_hit(self):
        """Hit@K is 1.0 if ANY section matches, not ALL."""
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        # Only Điều 1 / Khoản 1 is in results.
        results = [self._result("Điều 1", "Khoản 1")]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=5), 1.0)

    def test_or_semantics_second_section_hit(self):
        """Matching the second section is also a hit."""
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        results = [self._result("Điều 7", "Khoản 2")]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=5), 1.0)

    def test_or_semantics_neither_section_hit(self):
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        results = [self._result("Điều 9", "Khoản 3")]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=5), 0.0)

    # --- Test 13: each section counted once for Recall@K ---
    def test_recall_each_section_counted_once(self):
        """Three retrieval results that all satisfy the same section must
        still count as only 1 found (not 3), giving Recall@K = 0.5."""
        sections = [
            self._section("Điều 4", ["Khoản 1"]),
            self._section("Điều 6", ["Khoản 2"]),
        ]
        # Three results for Điều 4 / Khoản 1 — same section, duplicated.
        results = [
            self._result("Điều 4", "Khoản 1", score=0.95),
            self._result("Điều 4", "Khoản 1", score=0.90),
            self._result("Điều 4", "Khoản 1", score=0.85),
        ]
        # Điều 6 not found → Recall = 1/2 = 0.5
        self.assertAlmostEqual(
            self.tr.compute_recall_at_k(results, sections, k=5), 0.5
        )

    def test_recall_both_sections_found(self):
        sections = [
            self._section("Điều 4", ["Khoản 1"]),
            self._section("Điều 6", ["Khoản 2"]),
        ]
        results = [
            self._result("Điều 4", "Khoản 1"),
            self._result("Điều 6", "Khoản 2"),
        ]
        self.assertAlmostEqual(
            self.tr.compute_recall_at_k(results, sections, k=5), 1.0
        )

    # --- Test 14: MRR uses rank of FIRST structurally matching result ---
    def test_mrr_uses_first_match_rank(self):
        """MRR must use the rank of the first matching result, not the last."""
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [
            self._result("Điều 9", score=0.99),  # rank 1 — no match
            self._result("Điều 4", "Khoản 1", score=0.85),  # rank 2 — match → MRR = 0.5
            self._result("Điều 4", "Khoản 1", score=0.80),  # rank 3 — also match
        ]
        self.assertAlmostEqual(self.tr.compute_mrr(results, sections), 0.5)

    def test_mrr_qa_at_rank_1_does_not_count(self):
        """A QA result at rank 1 must be skipped; MRR uses the legal match at rank 2."""
        sections = [self._section("Điều 4", ["Khoản 1"])]
        qa_result = _FakeResult(
            score=0.99,
            metadata={
                "document_title": "Hỏi đáp Nghị định 116/2020/NĐ-CP",
                "article": None,
                "clause": None,
                "content_type": "qa",
            },
        )
        legal_result = _FakeResult(
            score=0.85,
            metadata={
                "document_title": self._ND116,
                "article": "Điều 4",
                "clause": "Khoản 1",
                "content_type": "legal_text",
            },
        )
        # QA at rank 1 must not count. Legal at rank 2 → MRR = 0.5.
        mrr = self.tr.compute_mrr([qa_result, legal_result], sections)
        self.assertAlmostEqual(mrr, 0.5)


# ---------------------------------------------------------------------------
# 23. NULL/empty metadata — preserved internally, ignored in matching
# ---------------------------------------------------------------------------


class TestNullMetadataHandling(unittest.TestCase):
    """Verify that NULL/empty metadata values are correctly handled:
    - Preserved in the raw metadata dict (not stripped out).
    - Treated as absent information during matching.
    """

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def test_null_clause_not_stripped_from_metadata(self):
        """A chunk returned from DB with clause=None keeps clause=None in metadata."""
        rv = _import_retrieval()
        row = (
            "chunk-x", "text", 0.9,
            "doc-id", self._ND116, "116/2020/NĐ-CP",
            "core", "primary", "primary_legal_source", 100,
            "Chương I",   # chapter
            "Điều 1",     # article
            None,         # clause — None preserved
            None,         # point
            "legal_text",
        )
        fake_conn = _make_fake_psycopg_connect(rows=[row])
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")
            results = retriever.retrieve("test query", top_k=1)

        meta = results[0].metadata
        self.assertIn("clause", meta)
        self.assertIsNone(meta["clause"])
        self.assertIn("chapter", meta)
        self.assertEqual(meta["chapter"], "Chương I")

    def test_legal_match_ignores_null_clause_when_clauses_unconstrained(self):
        """A legal chunk with clause=None can still match a section with
        empty clauses (article-level ground truth)."""
        section = self.tr.ExpectedSection(self._ND116, "Điều 1")  # no clauses
        meta = {
            "document_title": self._ND116,
            "article": "Điều 1",
            "clause": None,   # NULL preserved
            "point": None,
            "content_type": "legal_text",
        }
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_legal_match_null_clause_fails_when_clause_constrained(self):
        """A legal chunk with clause=None does NOT match when a specific
        clause is required."""
        section = self.tr.ExpectedSection(self._ND116, "Điều 1", ["Khoản 1"])
        meta = {
            "document_title": self._ND116,
            "article": "Điều 1",
            "clause": None,
            "point": None,
            "content_type": "legal_text",
        }
        self.assertFalse(self.tr._matches_section(meta, section))




# ---------------------------------------------------------------------------
# 24. Full hierarchical structural matching (all four levels: doc/art/clause/point)
# ---------------------------------------------------------------------------


class TestHierarchicalStructuralMatching(unittest.TestCase):
    """Validate four-level hierarchical matching: Document → Article → Clause → Point.

    Chapter is available in the metadata but is NOT a required matching level
    in ExpectedSection.  Tests here verify that fact explicitly.
    """

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    # ------------------------------------------------------------------ helpers

    def _meta(
        self,
        doc=None,
        chapter=None,
        art="Điều 4",
        clause="Khoản 1",
        point=None,
        content_type="legal_text",
    ):
        return {
            "document_title": doc or self._ND116,
            "chapter": chapter,
            "article": art,
            "clause": clause,
            "point": point,
            "content_type": content_type,
        }

    def _section(self, doc=None, art="Điều 4", clauses=None, points=None):
        return self.tr.ExpectedSection(
            doc or self._ND116, art, clauses or [], points or []
        )

    # ------------------------------------------------------------------ 4.1 full match

    def test_full_structural_match_all_levels(self):
        """Chunk with doc/chapter/article/clause/point matches section at all levels."""
        meta = self._meta(chapter="Chương II", art="Điều 4", clause="Khoản 1", point="Điểm a")
        section = self._section(art="Điều 4", clauses=["Khoản 1"], points=["Điểm a"])
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_full_structural_match_no_point_constraint(self):
        """Chunk with all metadata matches section that does not constrain point."""
        meta = self._meta(chapter="Chương II", art="Điều 4", clause="Khoản 1", point="Điểm a")
        section = self._section(art="Điều 4", clauses=["Khoản 1"])  # no point constraint
        self.assertTrue(self.tr._matches_section(meta, section))

    # ------------------------------------------------------------------ 4.2 chapter not required

    def test_chapter_present_but_not_required(self):
        """Chapter in metadata must NOT prevent a match when ExpectedSection omits chapter."""
        meta = self._meta(chapter="Chương II", art="Điều 4", clause="Khoản 1")
        section = self._section(art="Điều 4", clauses=["Khoản 1"])
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_chapter_absent_still_matches(self):
        """A chunk with no chapter (None) must still match when chapter is not required."""
        meta = self._meta(chapter=None, art="Điều 4", clause="Khoản 1")
        section = self._section(art="Điều 4", clauses=["Khoản 1"])
        self.assertTrue(self.tr._matches_section(meta, section))

    # ------------------------------------------------------------------ 4.3 point matching

    def test_matching_point(self):
        """Retrieved Điểm a satisfies expected Điểm a."""
        meta = self._meta(art="Điều 4", clause="Khoản 1", point="Điểm a")
        section = self._section(art="Điều 4", clauses=["Khoản 1"], points=["Điểm a"])
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_different_point_does_not_match(self):
        """Retrieved Điểm b does NOT satisfy expected Điểm a."""
        meta = self._meta(art="Điều 4", clause="Khoản 1", point="Điểm b")
        section = self._section(art="Điều 4", clauses=["Khoản 1"], points=["Điểm a"])
        self.assertFalse(self.tr._matches_section(meta, section))

    def test_missing_retrieved_point_does_not_match_point_constraint(self):
        """Retrieved point=None does NOT satisfy a non-empty points constraint."""
        meta = self._meta(art="Điều 4", clause="Khoản 1", point=None)
        section = self._section(art="Điều 4", clauses=["Khoản 1"], points=["Điểm a"])
        self.assertFalse(self.tr._matches_section(meta, section))

    def test_empty_point_in_meta_does_not_match_point_constraint(self):
        """Retrieved point='' (empty string) does NOT satisfy a non-empty points constraint."""
        meta = self._meta(art="Điều 4", clause="Khoản 1", point="")
        section = self._section(art="Điều 4", clauses=["Khoản 1"], points=["Điểm a"])
        self.assertFalse(self.tr._matches_section(meta, section))

    # ------------------------------------------------------------------ 4.4 article-only GT

    def test_article_only_gt_matches_chunk_with_clause(self):
        """Article-only ExpectedSection (no clauses) matches chunk that also has a clause."""
        meta = self._meta(art="Điều 5", clause="Khoản 1")
        section = self._section(art="Điều 5", clauses=[])  # article-only
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_article_only_gt_matches_chunk_with_point(self):
        """Article-only GT matches a chunk that has clause and point."""
        meta = self._meta(art="Điều 5", clause="Khoản 1", point="Điểm a")
        section = self._section(art="Điều 5", clauses=[])
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_article_only_gt_matches_chunk_with_no_clause(self):
        """Article-only GT matches a chunk with no clause at all."""
        meta = self._meta(art="Điều 5", clause=None)
        section = self._section(art="Điều 5", clauses=[])
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_article_only_gt_rejects_wrong_article(self):
        """Article-only GT must NOT match a chunk from a different article."""
        meta = self._meta(art="Điều 6", clause="Khoản 1")
        section = self._section(art="Điều 5", clauses=[])
        self.assertFalse(self.tr._matches_section(meta, section))

    # ------------------------------------------------------------------ QA isolation

    def test_qa_source_never_matches_legal_section(self):
        """A chunk with content_type='qa' never satisfies a structural ExpectedSection,
        even when its structural fields are artificially populated."""
        meta = self._meta(art="Điều 4", clause="Khoản 1", point="Điểm a", content_type="qa")
        section = self._section(art="Điều 4", clauses=["Khoản 1"], points=["Điểm a"])
        self.assertFalse(self.tr._matches_section(meta, section))

    # ------------------------------------------------------------------ supporting/reference legal

    def test_reference_supporting_legal_participates_in_matching(self):
        """Luật Giáo dục 2019 (reference/supporting) is legal_text → eligible for matching."""
        meta = {
            "document_title": "Luật Giáo dục 2019",
            "source_type": "reference",
            "document_role": "supporting",
            "chapter": "Chương IV",
            "article": "Điều 66",
            "clause": "Khoản 1",
            "point": None,
            "content_type": "legal_text",
        }
        section = self.tr.ExpectedSection("Luật Giáo dục 2019", "Điều 66", ["Khoản 1"])
        self.assertTrue(self.tr._matches_section(meta, section))


# ---------------------------------------------------------------------------
# 25. NULL / empty metadata robustness during matching
# ---------------------------------------------------------------------------


class TestMetadataCompletenessAndRobustness(unittest.TestCase):
    """Verify that NULL and empty metadata fields do not cause exceptions, and
    that article/clause matching still works when other fields are absent."""

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _meta(self, **overrides):
        base = {
            "document_title": self._ND116,
            "chapter": None,
            "article": "Điều 4",
            "clause": "Khoản 1",
            "point": None,
            "content_type": "legal_text",
        }
        base.update(overrides)
        return base

    def _section(self, art="Điều 4", clauses=None, points=None):
        return self.tr.ExpectedSection(self._ND116, art, clauses or [], points or [])

    # --- chapter=None, point=None ---

    def test_null_chapter_no_exception(self):
        """chapter=None must not raise when matching article/clause."""
        meta = self._meta(chapter=None)
        section = self._section(art="Điều 4", clauses=["Khoản 1"])
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_null_point_no_exception_unconstrained(self):
        """point=None must not raise when section has no point constraint."""
        meta = self._meta(point=None)
        section = self._section(art="Điều 4", clauses=["Khoản 1"])
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_null_point_no_exception_constrained(self):
        """point=None returns False (not exception) when section requires a point."""
        meta = self._meta(point=None)
        section = self._section(art="Điều 4", clauses=["Khoản 1"], points=["Điểm a"])
        self.assertFalse(self.tr._matches_section(meta, section))

    # --- chapter="", point="" ---

    def test_empty_string_chapter_treated_as_absent(self):
        """chapter='' must behave identically to chapter=None."""
        meta = self._meta(chapter="")
        section = self._section(art="Điều 4", clauses=["Khoản 1"])
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_empty_string_point_treated_as_absent_unconstrained(self):
        """point='' must not raise when section has no point constraint."""
        meta = self._meta(point="")
        section = self._section(art="Điều 4", clauses=["Khoản 1"])
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_empty_string_point_treated_as_absent_constrained(self):
        """point='' returns False when section requires a specific point."""
        meta = self._meta(point="")
        section = self._section(art="Điều 4", clauses=["Khoản 1"], points=["Điểm a"])
        self.assertFalse(self.tr._matches_section(meta, section))

    # --- both chapter and point None simultaneously ---

    def test_chapter_and_point_both_null_article_clause_still_match(self):
        meta = self._meta(chapter=None, point=None)
        section = self._section(art="Điều 4", clauses=["Khoản 1"])
        self.assertTrue(self.tr._matches_section(meta, section))

    # --- is_legal_source with edge-case content_type ---

    def test_whitespace_content_type_is_not_legal_source(self):
        meta = self._meta()
        meta["content_type"] = "  "
        self.assertFalse(self.tr._is_legal_source(meta))

    def test_none_content_type_is_not_legal_source(self):
        meta = self._meta()
        meta["content_type"] = None
        self.assertFalse(self.tr._is_legal_source(meta))


# ---------------------------------------------------------------------------
# 26. Evaluation metric semantics regression
# ---------------------------------------------------------------------------


class TestEvaluationMetricsRegression(unittest.TestCase):
    """Regression tests for Hit@K, Recall@K, and MRR evaluation semantics using
    the real hierarchical matching logic via _FakeResult objects."""

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _section(self, art, clauses=None):
        return self.tr.ExpectedSection(self._ND116, art, clauses or [])

    def _result(self, art, clause=None, score=0.9):
        return _FakeResult(score=score, metadata={
            "document_title": self._ND116,
            "article": art,
            "clause": clause,
            "point": None,
            "content_type": "legal_text",
        })

    # Hit@K

    def test_hit_at_k_is_1_when_first_result_matches(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [self._result("Điều 4", "Khoản 1")]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=3), 1.0)

    def test_hit_at_k_is_0_when_no_result_matches(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [self._result("Điều 9", "Khoản 1")]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=3), 0.0)

    def test_hit_at_k_is_0_when_match_is_outside_k(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [
            self._result("Điều 9"),
            self._result("Điều 9"),
            self._result("Điều 9"),
            self._result("Điều 4", "Khoản 1"),  # rank 4 — outside k=3
        ]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=3), 0.0)
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=4), 1.0)

    # Recall@K

    def test_recall_at_k_duplicate_matches_counted_once(self):
        """Retrieval results for the same section are deduplicated: count once."""
        sections = [
            self._section("Điều 4", ["Khoản 1"]),
            self._section("Điều 6", ["Khoản 2"]),
        ]
        results = [
            self._result("Điều 4", "Khoản 1", score=0.95),
            self._result("Điều 4", "Khoản 1", score=0.90),  # same section: duplicate
            self._result("Điều 4", "Khoản 1", score=0.85),  # same section: duplicate
        ]
        # Only Section A found; Section B not found → Recall = 1/2 = 0.5.
        self.assertAlmostEqual(self.tr.compute_recall_at_k(results, sections, k=5), 0.5)

    def test_recall_at_k_all_sections_found(self):
        sections = [
            self._section("Điều 4", ["Khoản 1"]),
            self._section("Điều 6", ["Khoản 2"]),
        ]
        results = [
            self._result("Điều 4", "Khoản 1"),
            self._result("Điều 6", "Khoản 2"),
        ]
        self.assertAlmostEqual(self.tr.compute_recall_at_k(results, sections, k=5), 1.0)

    def test_recall_at_k_no_section_found(self):
        sections = [
            self._section("Điều 4", ["Khoản 1"]),
            self._section("Điều 6", ["Khoản 2"]),
        ]
        results = [self._result("Điều 9")]
        self.assertAlmostEqual(self.tr.compute_recall_at_k(results, sections, k=5), 0.0)

    # MRR

    def test_mrr_first_rank_match(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [self._result("Điều 4", "Khoản 1")]
        self.assertAlmostEqual(self.tr.compute_mrr(results, sections), 1.0)

    def test_mrr_third_rank_match(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [
            self._result("Điều 9"),        # rank 1 — no match
            self._result("Điều 5"),        # rank 2 — no match
            self._result("Điều 4", "Khoản 1"),  # rank 3 — match → MRR = 1/3
        ]
        self.assertAlmostEqual(self.tr.compute_mrr(results, sections), 1 / 3)

    def test_mrr_uses_rank_of_first_match_not_last(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [
            self._result("Điều 9"),            # rank 1
            self._result("Điều 4", "Khoản 1"),  # rank 2 → MRR = 0.5
            self._result("Điều 4", "Khoản 1"),  # rank 3 (also matches, but rank 2 wins)
        ]
        self.assertAlmostEqual(self.tr.compute_mrr(results, sections), 0.5)

    def test_mrr_no_match_returns_zero(self):
        sections = [self._section("Điều 4", ["Khoản 1"])]
        results = [self._result("Điều 9"), self._result("Điều 5")]
        self.assertAlmostEqual(self.tr.compute_mrr(results, sections), 0.0)


# ---------------------------------------------------------------------------
# 27. Structural query evaluation logic regression
# ---------------------------------------------------------------------------


class TestStructuralQueryRegression(unittest.TestCase):
    """Regression test for the evaluation logic applied to a clause-level
    structural query: "Khoản 1 Điều 4 Nghị định 116/2020 quy định gì?"

    This test does NOT assert anything about live retrieval ranking.
    It validates that the evaluation functions correctly compute GT rank,
    Hit@3, and MRR for a synthetic ordered result set.
    """

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _result(self, art, clause=None, score=0.9):
        return _FakeResult(score=score, metadata={
            "document_title": self._ND116,
            "article": art,
            "clause": clause,
            "point": None,
            "content_type": "legal_text",
        })

    def test_gt_found_at_rank_3(self):
        """With the GT match at rank 3, GT rank should be 3."""
        sections = [self.tr.ExpectedSection(self._ND116, "Điều 4", ["Khoản 1"])]
        results = [
            self._result("Điều 4", "Khoản 2", score=0.95),   # rank 1 — wrong clause
            self._result("Điều 5", None, score=0.90),         # rank 2 — wrong article
            self._result("Điều 4", "Khoản 1", score=0.85),   # rank 3 — GT match
        ]
        gt_rank = None
        for rank_idx, r in enumerate(results, start=1):
            if self.tr._is_ground_truth_match(r.metadata, sections):
                gt_rank = rank_idx
                break
        self.assertEqual(gt_rank, 3)

    def test_hit_at_3_is_1_when_gt_at_rank_3(self):
        sections = [self.tr.ExpectedSection(self._ND116, "Điều 4", ["Khoản 1"])]
        results = [
            self._result("Điều 4", "Khoản 2", score=0.95),
            self._result("Điều 5", None, score=0.90),
            self._result("Điều 4", "Khoản 1", score=0.85),
        ]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=3), 1.0)

    def test_hit_at_2_is_0_when_gt_at_rank_3(self):
        sections = [self.tr.ExpectedSection(self._ND116, "Điều 4", ["Khoản 1"])]
        results = [
            self._result("Điều 4", "Khoản 2", score=0.95),
            self._result("Điều 5", None, score=0.90),
            self._result("Điều 4", "Khoản 1", score=0.85),
        ]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=2), 0.0)

    def test_mrr_equals_one_third_when_gt_at_rank_3(self):
        sections = [self.tr.ExpectedSection(self._ND116, "Điều 4", ["Khoản 1"])]
        results = [
            self._result("Điều 4", "Khoản 2", score=0.95),
            self._result("Điều 5", None, score=0.90),
            self._result("Điều 4", "Khoản 1", score=0.85),
        ]
        self.assertAlmostEqual(self.tr.compute_mrr(results, sections), 1 / 3)


# ---------------------------------------------------------------------------
# 28. Multi-section evaluation query regression
# ---------------------------------------------------------------------------


class TestMultiSectionQueryRegression(unittest.TestCase):
    """Hit@K vs. Recall@K semantics for queries with multiple expected sections.

    Hit@K = 1 whenever ANY section is found in top-K.
    Recall@K = (# sections found) / (# total sections).
    """

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def _section(self, art, clauses=None):
        return self.tr.ExpectedSection(self._ND116, art, clauses or [])

    def _result(self, art, clause=None, score=0.9):
        return _FakeResult(score=score, metadata={
            "document_title": self._ND116,
            "article": art,
            "clause": clause,
            "content_type": "legal_text",
        })

    def test_both_sections_found_hit_1_recall_1(self):
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        results = [
            self._result("Điều 1", "Khoản 1"),
            self._result("Điều 7", "Khoản 2"),
        ]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=5), 1.0)
        self.assertAlmostEqual(self.tr.compute_recall_at_k(results, sections, k=5), 1.0)

    def test_only_first_section_found_hit_1_recall_half(self):
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        results = [self._result("Điều 1", "Khoản 1")]  # only Section A
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=5), 1.0)
        self.assertAlmostEqual(self.tr.compute_recall_at_k(results, sections, k=5), 0.5)

    def test_only_second_section_found_hit_1_recall_half(self):
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        results = [self._result("Điều 7", "Khoản 2")]  # only Section B
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=5), 1.0)
        self.assertAlmostEqual(self.tr.compute_recall_at_k(results, sections, k=5), 0.5)

    def test_neither_section_found_hit_0_recall_0(self):
        sections = [
            self._section("Điều 1", ["Khoản 1"]),
            self._section("Điều 7", ["Khoản 2"]),
        ]
        results = [self._result("Điều 9")]
        self.assertEqual(self.tr.compute_hit_at_k(results, sections, k=5), 0.0)
        self.assertAlmostEqual(self.tr.compute_recall_at_k(results, sections, k=5), 0.0)


# ---------------------------------------------------------------------------
# 29. Document identity matters in structural matching
# ---------------------------------------------------------------------------


class TestDocumentIdentityMatters(unittest.TestCase):
    """Verify that the same article in a different legal document does NOT
    satisfy the ground truth for the original document."""

    _ND116 = "Nghị định 116/2020/NĐ-CP"
    _ND60 = "Nghị định 60/2025/NĐ-CP"

    def setUp(self):
        self.tr = _import_test_retrieval()

    def test_same_article_different_document_no_match(self):
        """ND60/Điều 4 must NOT satisfy a section targeting ND116/Điều 4."""
        section = self.tr.ExpectedSection(self._ND116, "Điều 4", ["Khoản 1"])
        meta = {
            "document_title": self._ND60,
            "article": "Điều 4",
            "clause": "Khoản 1",
            "content_type": "legal_text",
        }
        self.assertFalse(self.tr._matches_section(meta, section))

    def test_correct_document_matches(self):
        """ND116/Điều 4 must satisfy a section targeting ND116/Điều 4."""
        section = self.tr.ExpectedSection(self._ND116, "Điều 4", ["Khoản 1"])
        meta = {
            "document_title": self._ND116,
            "article": "Điều 4",
            "clause": "Khoản 1",
            "content_type": "legal_text",
        }
        self.assertTrue(self.tr._matches_section(meta, section))

    def test_same_article_in_lgd_does_not_match_nd116_gt(self):
        """Luật Giáo dục 2019 / Điều 4 must NOT match a ND116/Điều 4 expectation."""
        section = self.tr.ExpectedSection(self._ND116, "Điều 4", [])
        meta = {
            "document_title": "Luật Giáo dục 2019",
            "article": "Điều 4",
            "clause": "Khoản 1",
            "content_type": "legal_text",
        }
        self.assertFalse(self.tr._matches_section(meta, section))

    def test_hit_at_k_wrong_document_is_zero(self):
        """compute_hit_at_k returns 0.0 when only a result from the wrong document
        is retrieved, even if the article/clause match."""
        section = self.tr.ExpectedSection(self._ND116, "Điều 4", ["Khoản 1"])
        result = _FakeResult(score=0.99, metadata={
            "document_title": self._ND60,
            "article": "Điều 4",
            "clause": "Khoản 1",
            "content_type": "legal_text",
        })
        self.assertEqual(self.tr.compute_hit_at_k([result], [section], k=3), 0.0)


# ---------------------------------------------------------------------------
# 30. Embedding representation contract
# ---------------------------------------------------------------------------


class TestEmbeddingRepresentationContract(unittest.TestCase):
    """Verify that _get_embedding_representation_description() returns the
    expected deterministic documentation string, and that render_full_report()
    produces a report containing the required EMBEDDING REPRESENTATION section.

    This test class validates the reporting contract, NOT the embedding
    implementation.  It does NOT import or call build_embedding_text().
    """

    def setUp(self):
        self.tr = _import_test_retrieval()

    # ------------------------------------------------------------------ description helper

    def test_description_is_non_empty_string(self):
        desc = self.tr._get_embedding_representation_description()
        self.assertIsInstance(desc, str)
        self.assertTrue(len(desc) > 0)

    def test_description_mentions_metadata_aware(self):
        desc = self.tr._get_embedding_representation_description()
        self.assertIn("Metadata-aware", desc)

    def test_description_lists_all_english_labels(self):
        desc = self.tr._get_embedding_representation_description()
        for label in ("[Document]", "[Chapter]", "[Article]", "[Clause]", "[Point]", "[Content]"):
            self.assertIn(label, desc, msg=f"Expected label {label!r} in description")

    def test_description_mentions_document_title(self):
        desc = self.tr._get_embedding_representation_description()
        self.assertIn("document_title", desc)

    def test_description_mentions_qa_original_text(self):
        desc = self.tr._get_embedding_representation_description()
        # Must describe QA/non-legal sources using original text only
        desc_lower = desc.lower()
        self.assertTrue(
            "qa" in desc_lower or "non-legal" in desc_lower,
            "Description must reference QA/non-legal sources"
        )
        self.assertIn("original text", desc.lower())

    def test_description_mentions_query_embedding(self):
        desc = self.tr._get_embedding_representation_description()
        self.assertIn("Query embedding", desc)

    def test_description_states_no_query_transformation(self):
        desc = self.tr._get_embedding_representation_description()
        self.assertIn("Query transformation", desc)
        self.assertIn("None", desc)

    # ------------------------------------------------------------------ render_full_report section

    def _make_minimal_report(self):
        """Build a minimal render_full_report() call with no query records."""
        from datetime import datetime  # noqa: PLC0415
        start = datetime(2026, 9, 23, 8, 0, 0)
        end = datetime(2026, 9, 23, 8, 1, 0)
        return self.tr.render_full_report(
            start_time=start,
            end_time=end,
            model_name="BAAI/bge-m3",
            model_revision="N/A",
            embedding_dim=1024,
            device="cpu",
            gpu_name="N/A",
            cuda_version="N/A",
            pytorch_version="N/A",
            database_name="testdb",
            state={
                "total_chunks": 617,
                "embedded_chunks": 617,
                "missing_embeddings": 0,
                "hnsw_index_exists": True,
            },
            ef_search=40,
            top_k=5,
            cli_args_str="(test)",
            query_records=[],
            metrics={"evaluated_queries": 0, "total_queries": 0},
            retrieval_errors=[],
            uncaught_exception=None,
        )

    def test_report_contains_embedding_representation_section(self):
        report = self._make_minimal_report()
        self.assertIn("EMBEDDING REPRESENTATION", report)

    def test_report_states_metadata_aware(self):
        report = self._make_minimal_report()
        self.assertIn("Metadata-aware", report)

    def test_report_lists_document_title_field(self):
        report = self._make_minimal_report()
        self.assertIn("document_title", report)

    def test_report_lists_chapter_field(self):
        report = self._make_minimal_report()
        self.assertIn("chapter", report)

    def test_report_lists_all_english_labels(self):
        report = self._make_minimal_report()
        for label in ("[Document]", "[Chapter]", "[Article]", "[Clause]", "[Point]", "[Content]"):
            self.assertIn(label, report, msg=f"Expected label {label!r} in report")

    def test_report_states_document_number_not_included(self):
        report = self._make_minimal_report()
        # Both the field name and its exclusion must be documented
        self.assertIn("Document number", report)
        self.assertIn("Not included", report)

    def test_report_states_null_empty_field_omitted(self):
        report = self._make_minimal_report()
        self.assertIn("Null/empty metadata", report)
        self.assertIn("Field omitted", report)

    def test_report_states_qa_original_text_only(self):
        report = self._make_minimal_report()
        self.assertIn("QA/non-legal chunks", report)
        self.assertIn("Original text only", report)

    def test_report_states_query_embedding_original_natural_language(self):
        report = self._make_minimal_report()
        self.assertIn("Query embedding", report)
        self.assertIn("Original natural-language query", report)

    def test_report_states_no_query_transformation(self):
        report = self._make_minimal_report()
        self.assertIn("Query transformation", report)
        # Check that "None" appears after the "Query transformation" label
        idx = report.index("Query transformation")
        self.assertIn("None", report[idx:idx + 50])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
