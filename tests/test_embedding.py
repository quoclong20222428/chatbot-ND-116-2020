"""Unit tests for the embedding, model registry, and indexing components.

These tests use unittest.mock to patch model loading so that the real models
(1+ GB each) are never downloaded during CI or local test runs.

Run from the repository root::

    python -m pytest tests/test_embedding.py -v
"""

from __future__ import annotations

import importlib
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
# Helpers
# ---------------------------------------------------------------------------

FAKE_DIM = 1024


def _fake_encode(texts, batch_size=32, max_length=8192,
                 return_dense=True, return_sparse=False, return_colbert_vecs=False):
    """Return a plausible BGEM3FlagModel.encode() result dict."""
    import numpy as np  # noqa: PLC0415
    return {"dense_vecs": np.zeros((len(texts), FAKE_DIM), dtype="float32")}


def _make_fake_flag_embedding_module() -> types.ModuleType:
    """Build a minimal fake FlagEmbedding package."""
    fake_pkg = types.ModuleType("FlagEmbedding")
    fake_model_cls = MagicMock(name="BGEM3FlagModel")
    fake_model_instance = MagicMock(name="bgem3_instance")
    fake_model_instance.encode.side_effect = _fake_encode
    fake_model_cls.return_value = fake_model_instance
    fake_pkg.BGEM3FlagModel = fake_model_cls
    return fake_pkg


def _fake_st_encode(texts, batch_size=32, normalize_embeddings=True, **kwargs):
    """Return a plausible SentenceTransformer.encode() result."""
    import numpy as np  # noqa: PLC0415
    return np.zeros((len(texts), FAKE_DIM), dtype="float32")


def _make_fake_sentence_transformers_module() -> types.ModuleType:
    """Build a minimal fake sentence_transformers package."""
    fake_pkg = types.ModuleType("sentence_transformers")
    fake_model_cls = MagicMock(name="SentenceTransformer")
    fake_model_instance = MagicMock(name="st_instance")
    fake_model_instance.encode.side_effect = _fake_st_encode
    fake_model_instance.get_sentence_embedding_dimension.return_value = FAKE_DIM
    fake_model_cls.return_value = fake_model_instance
    fake_pkg.SentenceTransformer = fake_model_cls
    return fake_pkg


def _fake_deepx_encode(texts, truncate_dim=1024, normalize=True, batch_size=None, **kwargs):
    """Return a plausible DeepXEmbed.encode() result."""
    import numpy as np  # noqa: PLC0415
    n = len(texts) if hasattr(texts, "__len__") else 1
    return np.zeros((n, truncate_dim), dtype="float32")


def _make_fake_deepx_module() -> types.ModuleType:
    """Build a minimal fake deepx_embed package with a DeepXEmbed class."""
    fake_pkg = types.ModuleType("deepx_embed")

    fake_model_cls = MagicMock(name="DeepXEmbed")
    fake_model_instance = MagicMock(name="deepx_instance")
    fake_model_instance.encode.side_effect = _fake_deepx_encode
    fake_model_cls.from_pretrained.return_value = fake_model_instance
    fake_pkg.DeepXEmbed = fake_model_cls
    return fake_pkg


def _cleanup_modules():
    """Remove embedding-related modules from sys.modules for clean reimport."""
    for mod in list(sys.modules):
        if mod in ("embedding", "model_registry", "index_embeddings"):
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# Model Registry tests
# ---------------------------------------------------------------------------


class TestModelRegistry(unittest.TestCase):
    """Tests for model_registry.resolve_model_config and model metadata."""

    def setUp(self):
        _cleanup_modules()

    def tearDown(self):
        _cleanup_modules()

    def _import(self):
        import model_registry as mr  # noqa: PLC0415
        return mr

    def test_default_model_is_bge_m3(self):
        mr = self._import()
        config = mr.resolve_model_config(None)
        self.assertEqual(config.model_id, "BAAI/bge-m3")
        self.assertEqual(config.alias, "bge-m3")

    def test_resolve_by_alias(self):
        mr = self._import()
        config = mr.resolve_model_config("jina-v3")
        self.assertEqual(config.model_id, "jinaai/jina-embeddings-v3-hf")
        self.assertEqual(config.backend, "jina")

    def test_resolve_by_model_id(self):
        mr = self._import()
        config = mr.resolve_model_config("mainguyen9/vietlegal-e5")
        self.assertEqual(config.alias, "vietlegal-e5")
        self.assertEqual(config.query_prefix, "query: ")
        self.assertEqual(config.document_prefix, "passage: ")

    def test_unsupported_model_raises_valueerror(self):
        mr = self._import()
        with self.assertRaises(ValueError) as ctx:
            mr.resolve_model_config("totally/fake-model")
        self.assertIn("Unsupported embedding model", str(ctx.exception))
        # Error message should list supported models.
        self.assertIn("bge-m3", str(ctx.exception))
        self.assertIn("jina-v3", str(ctx.exception))

    def test_empty_string_resolves_to_default(self):
        mr = self._import()
        config = mr.resolve_model_config("")
        self.assertEqual(config.alias, "bge-m3")

    def test_all_models_have_1024_dimension(self):
        mr = self._import()
        for alias in mr.SUPPORTED_ALIASES:
            config = mr.resolve_model_config(alias)
            self.assertEqual(
                config.dimension, 1024,
                f"Model {alias} has dimension {config.dimension}, expected 1024",
            )

    def test_all_models_have_unique_columns(self):
        mr = self._import()
        columns = set()
        for alias in mr.SUPPORTED_ALIASES:
            config = mr.resolve_model_config(alias)
            self.assertNotIn(
                config.embedding_column, columns,
                f"Duplicate embedding column: {config.embedding_column}",
            )
            columns.add(config.embedding_column)

    def test_bge_m3_backwards_compatible_column(self):
        """The BGE-M3 model should use the original 'embedding' column name."""
        mr = self._import()
        config = mr.resolve_model_config("bge-m3")
        self.assertEqual(config.embedding_column, "embedding")
        self.assertEqual(config.hnsw_index_name, "legal_chunks_embedding_hnsw_idx")

    def test_other_models_have_prefixed_columns(self):
        mr = self._import()
        for alias in mr.SUPPORTED_ALIASES:
            if alias == "bge-m3":
                continue
            config = mr.resolve_model_config(alias)
            self.assertTrue(
                config.embedding_column.startswith("embedding_"),
                f"Model {alias} column should start with 'embedding_': {config.embedding_column}",
            )

    def test_six_supported_models(self):
        mr = self._import()
        self.assertEqual(len(mr.SUPPORTED_ALIASES), 6)

    def test_case_insensitive_resolve(self):
        mr = self._import()
        config = mr.resolve_model_config("BGE-M3")
        self.assertEqual(config.model_id, "BAAI/bge-m3")

    def test_e5_model_has_prefixes(self):
        mr = self._import()
        config = mr.resolve_model_config("vietlegal-e5")
        self.assertEqual(config.query_prefix, "query: ")
        self.assertEqual(config.document_prefix, "passage: ")

    def test_jina_model_has_prompt_names(self):
        mr = self._import()
        config = mr.resolve_model_config("jina-v3")
        self.assertEqual(config.query_prompt_name, "retrieval.query")
        self.assertEqual(config.document_prompt_name, "retrieval.passage")
        self.assertTrue(config.trust_remote_code)

    def test_bge_model_has_no_prefixes(self):
        mr = self._import()
        config = mr.resolve_model_config("bge-m3")
        self.assertEqual(config.query_prefix, "")
        self.assertEqual(config.document_prefix, "")


# ---------------------------------------------------------------------------
# EmbeddingModel tests (BGE backend)
# ---------------------------------------------------------------------------


class TestEmbeddingModelInit(unittest.TestCase):
    def setUp(self):
        self.fake_fe = _make_fake_flag_embedding_module()
        sys.modules["FlagEmbedding"] = self.fake_fe
        _cleanup_modules()

    def tearDown(self):
        sys.modules.pop("FlagEmbedding", None)
        _cleanup_modules()

    def _import(self):
        import embedding as em  # noqa: PLC0415
        return em

    def test_default_model_name(self):
        em = self._import()
        model = em.EmbeddingModel()
        self.assertEqual(model.model_name, "BAAI/bge-m3")

    def test_custom_model_name_bge(self):
        em = self._import()
        model = em.EmbeddingModel(model_name="BAAI/bge-m3")
        self.assertEqual(model.model_name, "BAAI/bge-m3")

    def test_embedding_dim_constant(self):
        em = self._import()
        model = em.EmbeddingModel()
        self.assertEqual(model.embedding_dim, 1024)

    def test_get_dimension_method(self):
        em = self._import()
        model = em.EmbeddingModel()
        self.assertEqual(model.get_dimension(), 1024)

    def test_get_model_name_method(self):
        em = self._import()
        model = em.EmbeddingModel()
        self.assertEqual(model.get_model_name(), "BAAI/bge-m3")

    def test_config_property_available(self):
        em = self._import()
        model = em.EmbeddingModel()
        self.assertEqual(model.config.alias, "bge-m3")
        self.assertEqual(model.config.backend, "bge")


# ---------------------------------------------------------------------------
# Query vs Document embedding tests (BGE backend)
# ---------------------------------------------------------------------------


class TestEmbedQueryAndDocuments(unittest.TestCase):
    def setUp(self):
        self.fake_fe = _make_fake_flag_embedding_module()
        sys.modules["FlagEmbedding"] = self.fake_fe
        _cleanup_modules()

    def tearDown(self):
        sys.modules.pop("FlagEmbedding", None)
        _cleanup_modules()

    def _make_model(self):
        import embedding as em  # noqa: PLC0415
        return em.EmbeddingModel()

    def test_embed_query_returns_single_vector(self):
        model = self._make_model()
        result = model.embed_query("Điều 1. Phạm vi điều chỉnh")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), FAKE_DIM)
        self.assertIsInstance(result[0], float)

    def test_embed_documents_returns_list_of_vectors(self):
        model = self._make_model()
        result = model.embed_documents(["Điều 1", "Khoản 2"])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for vec in result:
            self.assertIsInstance(vec, list)
            self.assertEqual(len(vec), FAKE_DIM)

    def test_embed_texts_backwards_compat(self):
        """embed_texts is a backwards-compatible alias for embed_documents."""
        model = self._make_model()
        result = model.embed_texts(["Điều 1", "Khoản 2"])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for vec in result:
            self.assertEqual(len(vec), FAKE_DIM)

    def test_embed_documents_empty_input(self):
        model = self._make_model()
        result = model.embed_documents([])
        self.assertEqual(result, [])

    def test_embed_documents_multiple_texts(self):
        model = self._make_model()
        texts = [f"Văn bản pháp luật {i}" for i in range(10)]
        result = model.embed_documents(texts)
        self.assertEqual(len(result), 10)


# ---------------------------------------------------------------------------
# SentenceTransformer backend tests
# ---------------------------------------------------------------------------


class TestSentenceTransformerBackend(unittest.TestCase):
    def setUp(self):
        self.fake_st = _make_fake_sentence_transformers_module()
        sys.modules["sentence_transformers"] = self.fake_st
        _cleanup_modules()

    def tearDown(self):
        sys.modules.pop("sentence_transformers", None)
        _cleanup_modules()

    def _make_model(self, model_name):
        import embedding as em  # noqa: PLC0415
        return em.EmbeddingModel(model_name=model_name)

    def test_vnlegal_lal_loads_correctly(self):
        model = self._make_model("vnlegal-lal")
        self.assertEqual(model.model_name, "darklethelong/vnlegal-lal")
        self.assertEqual(model.config.backend, "sentence_transformer")

    def test_vietlegal_harrier_loads_correctly(self):
        model = self._make_model("vietlegal-harrier")
        self.assertEqual(model.model_name, "mainguyen9/vietlegal-harrier-0.6b")

    def test_e5_embed_query_prepends_prefix(self):
        model = self._make_model("vietlegal-e5")
        model.embed_query("test query")
        # The encode call should have received the prefixed text.
        call_args = self.fake_st.SentenceTransformer.return_value.encode.call_args
        texts_sent = call_args[0][0]  # first positional arg
        self.assertEqual(texts_sent[0], "query: test query")

    def test_e5_embed_documents_prepends_prefix(self):
        model = self._make_model("vietlegal-e5")
        model.embed_documents(["doc1", "doc2"])
        call_args = self.fake_st.SentenceTransformer.return_value.encode.call_args
        texts_sent = call_args[0][0]
        self.assertEqual(texts_sent[0], "passage: doc1")
        self.assertEqual(texts_sent[1], "passage: doc2")

    def test_no_prefix_for_generic_models(self):
        model = self._make_model("vnlegal-lal")
        model.embed_query("test query")
        call_args = self.fake_st.SentenceTransformer.return_value.encode.call_args
        texts_sent = call_args[0][0]
        self.assertEqual(texts_sent[0], "test query")

    def test_embed_query_returns_correct_dimension(self):
        model = self._make_model("vietlegal-harrier")
        result = model.embed_query("test")
        self.assertEqual(len(result), FAKE_DIM)

    def test_embed_documents_returns_correct_dimension(self):
        model = self._make_model("vietlegal-harrier")
        result = model.embed_documents(["a", "b", "c"])
        self.assertEqual(len(result), 3)
        for vec in result:
            self.assertEqual(len(vec), FAKE_DIM)


# ---------------------------------------------------------------------------
# Jina backend tests
# ---------------------------------------------------------------------------


class TestJinaBackend(unittest.TestCase):
    def setUp(self):
        self.fake_st = _make_fake_sentence_transformers_module()
        sys.modules["sentence_transformers"] = self.fake_st
        _cleanup_modules()

    def tearDown(self):
        sys.modules.pop("sentence_transformers", None)
        _cleanup_modules()

    def test_jina_loads_with_trust_remote_code(self):
        import embedding as em  # noqa: PLC0415
        model = em.EmbeddingModel(model_name="jina-v3")
        self.assertEqual(model.model_name, "jinaai/jina-embeddings-v3-hf")
        call_kwargs = self.fake_st.SentenceTransformer.call_args
        self.assertTrue(call_kwargs[1].get("trust_remote_code", False))

    def test_jina_embed_query_returns_vector(self):
        import embedding as em  # noqa: PLC0415
        model = em.EmbeddingModel(model_name="jina-v3")
        result = model.embed_query("test query")
        self.assertEqual(len(result), FAKE_DIM)

    def test_jina_embed_documents_returns_vectors(self):
        import embedding as em  # noqa: PLC0415
        model = em.EmbeddingModel(model_name="jina-v3")
        result = model.embed_documents(["doc1", "doc2"])
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Dimension validation tests
# ---------------------------------------------------------------------------


class TestDimensionValidation(unittest.TestCase):
    def setUp(self):
        self.fake_fe = _make_fake_flag_embedding_module()
        sys.modules["FlagEmbedding"] = self.fake_fe
        _cleanup_modules()

    def tearDown(self):
        sys.modules.pop("FlagEmbedding", None)
        _cleanup_modules()

    def test_correct_dimension_passes(self):
        import embedding as em  # noqa: PLC0415
        model = em.EmbeddingModel()
        # Should not raise.
        result = model.embed_query("test")
        self.assertEqual(len(result), 1024)

    def test_wrong_dimension_raises_valueerror(self):
        """If the model returns wrong-dimension vectors, validation catches it."""
        import numpy as np  # noqa: PLC0415
        import embedding as em  # noqa: PLC0415

        model = em.EmbeddingModel()

        # Patch the backend to return 768-dimensional vectors.
        def bad_encode(texts, **kwargs):
            return {"dense_vecs": np.zeros((len(texts), 768), dtype="float32")}

        model._backend._model.encode.side_effect = bad_encode

        with self.assertRaises(ValueError) as ctx:
            model.embed_query("test")
        self.assertIn("Dimension mismatch", str(ctx.exception))
        self.assertIn("768", str(ctx.exception))


# ---------------------------------------------------------------------------
# Model isolation tests
# ---------------------------------------------------------------------------


class TestModelIsolation(unittest.TestCase):
    """Verify that different models map to different database columns."""

    def setUp(self):
        _cleanup_modules()

    def tearDown(self):
        _cleanup_modules()

    def test_different_models_use_different_columns(self):
        import model_registry as mr  # noqa: PLC0415
        columns = {}
        for alias in mr.SUPPORTED_ALIASES:
            config = mr.resolve_model_config(alias)
            columns[alias] = config.embedding_column

        # All columns must be unique.
        self.assertEqual(len(columns), len(set(columns.values())))

    def test_different_models_use_different_indexes(self):
        import model_registry as mr  # noqa: PLC0415
        indexes = {}
        for alias in mr.SUPPORTED_ALIASES:
            config = mr.resolve_model_config(alias)
            indexes[alias] = config.hnsw_index_name

        self.assertEqual(len(indexes), len(set(indexes.values())))


# ---------------------------------------------------------------------------
# Missing library tests
# ---------------------------------------------------------------------------


class TestEmbedTextsMissingFlagEmbedding(unittest.TestCase):
    """Verify that a clear ImportError is raised if FlagEmbedding is not installed."""

    def setUp(self):
        for mod in list(sys.modules):
            if mod == "embedding" or mod.startswith("FlagEmbedding"):
                del sys.modules[mod]

    def tearDown(self):
        for mod in list(sys.modules):
            if mod == "embedding":
                del sys.modules[mod]

    def test_import_error_raised(self):
        """Block FlagEmbedding at the import level using a custom import hook."""
        _real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _blocking_import(name, *args, **kwargs):
            if name == "FlagEmbedding" or name.startswith("FlagEmbedding."):
                raise ModuleNotFoundError(f"No module named '{name}'")
            return _real_import(name, *args, **kwargs)

        import builtins  # noqa: PLC0415
        with patch.object(builtins, "__import__", side_effect=_blocking_import):
            # Force reload so _load_model() runs inside the patch context.
            import embedding as em  # noqa: PLC0415
            with self.assertRaises(ImportError):
                em.EmbeddingModel()


# ---------------------------------------------------------------------------
# Idempotency and SQL logic tests (index_embeddings.py)
# ---------------------------------------------------------------------------


class TestIndexEmbeddingsLogic(unittest.TestCase):
    """Test core logic of index_embeddings.py without a real DB or model."""

    def setUp(self):
        self.fake_fe = _make_fake_flag_embedding_module()
        sys.modules["FlagEmbedding"] = self.fake_fe
        _cleanup_modules()

    def tearDown(self):
        sys.modules.pop("FlagEmbedding", None)
        _cleanup_modules()

    def _import(self):
        import embedding  # noqa: PLC0415  (needed to resolve import inside index_embeddings)
        import index_embeddings as ie  # noqa: PLC0415
        return ie

    def test_vector_to_pg_format(self):
        ie = self._import()
        result = ie._vector_to_pg([0.1, 0.2, 0.3])
        self.assertTrue(result.startswith("["))
        self.assertTrue(result.endswith("]"))
        self.assertIn(",", result)

    def test_vector_to_pg_correct_length(self):
        ie = self._import()
        vec = [0.0] * 1024
        result = ie._vector_to_pg(vec)
        # Should have 1024 comma-separated values.
        self.assertEqual(result.count(","), 1023)

    def test_index_chunks_succeeds(self):
        """index_chunks should commit embeddings and return correct stats."""
        ie = self._import()
        import embedding as em  # noqa: PLC0415

        model = em.EmbeddingModel()

        # Build a mock connection + cursor that connect_db returns.
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # 8-tuple rows: (chunk_id, text, content_type, document_title,
        #                chapter, article, clause, point)
        chunks = [
            (
                "chunk-1",
                "Điều 1 nội dung",
                "legal_text",
                "Nghị định 116/2020/NĐ-CP",
                "Chương I",
                "Điều 1",
                "Khoản 1",
                None,
            ),
            (
                "chunk-2",
                "Điều 2 nội dung",
                "qa",
                None,
                None,
                None,
                None,
                None,
            ),
        ]
        with patch.object(ie, "connect_db", return_value=mock_conn):
            stats = ie.index_chunks(
                "postgresql://fake/db", model, chunks,
                batch_size=32, embedding_column="embedding",
            )

        self.assertEqual(stats.succeeded, 2)
        self.assertEqual(stats.failed, 0)
        mock_conn.commit.assert_called()

    def test_select_unembedded_uses_model_column(self):
        ie = self._import()
        sql = ie._select_unembedded("embedding_jina_v3")
        self.assertIn("embedding_jina_v3 IS NULL", sql)

    def test_select_unembedded_default_column(self):
        ie = self._import()
        sql = ie._select_unembedded("embedding")
        self.assertIn("embedding IS NULL", sql)

    def test_update_embedding_uses_model_column(self):
        ie = self._import()
        sql = ie._update_embedding("embedding_vietlegal_e5")
        self.assertIn("embedding_vietlegal_e5 =", sql)

    def test_verify_with_model_column(self):
        """verify() should check the correct model-specific column."""
        ie = self._import()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            (618, 500),   # verify counts
            ("chunk-1", 1024),  # verify dimension
            ("legal_chunks_embedding_hnsw_idx",),  # check index
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = ie.verify(
            mock_conn,
            embedding_column="embedding",
            hnsw_index_name="legal_chunks_embedding_hnsw_idx",
        )

        self.assertEqual(result["total_chunks"], 618)
        self.assertEqual(result["embedded_chunks"], 500)
        self.assertEqual(result["missing_embeddings"], 118)
        self.assertTrue(result["embedding_dim_ok"])
        self.assertTrue(result["hnsw_index_exists"])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# build_embedding_text tests (unchanged — model-independent)
# ---------------------------------------------------------------------------


class TestBuildEmbeddingText(unittest.TestCase):
    """Tests for embedding.build_embedding_text().

    These tests do NOT load any embedding model; they only verify the string
    construction logic.  The fake FlagEmbedding module is injected via
    sys.modules so that importing ``embedding`` succeeds without the real model.
    """

    _ND116 = "Nghị định 116/2020/NĐ-CP"

    def setUp(self):
        fake_fe = _make_fake_flag_embedding_module()
        sys.modules["FlagEmbedding"] = fake_fe
        sys.modules.pop("embedding", None)

    def tearDown(self):
        sys.modules.pop("FlagEmbedding", None)
        sys.modules.pop("embedding", None)

    def _fn(self):
        """Return the build_embedding_text function."""
        import embedding as em  # noqa: PLC0415
        return em.build_embedding_text

    # --- Legal chunk with full metadata ---

    def test_legal_full_metadata_contains_all_labels(self):
        """All populated fields appear in the correct label order."""
        fn = self._fn()
        result = fn(
            text="Mức hỗ trợ tiền đóng học phí...",
            metadata={
                "content_type": "legal_text",
                "document_title": self._ND116,
                "chapter": "Chương II",
                "article": "Điều 4",
                "clause": "Khoản 1",
                "point": "Điểm a",
            },
        )
        self.assertIn("[Document]", result)
        self.assertIn("[Chapter]", result)
        self.assertIn("[Article]", result)
        self.assertIn("[Clause]", result)
        self.assertIn("[Point]", result)
        self.assertIn("[Content]", result)
        self.assertIn(self._ND116, result)
        self.assertIn("Chương II", result)
        self.assertIn("Điều 4", result)
        self.assertIn("Khoản 1", result)
        self.assertIn("Điểm a", result)
        self.assertIn("Mức hỗ trợ tiền đóng học phí...", result)

    def test_legal_full_metadata_label_order(self):
        """Labels appear in Document → Chapter → Article → Clause → Point → Content order."""
        fn = self._fn()
        result = fn(
            text="nội dung",
            metadata={
                "content_type": "legal_text",
                "document_title": self._ND116,
                "chapter": "Chương I",
                "article": "Điều 1",
                "clause": "Khoản 1",
                "point": "Điểm a",
            },
        )
        positions = {
            label: result.index(label)
            for label in ("[Document]", "[Chapter]", "[Article]", "[Clause]", "[Point]", "[Content]")
        }
        self.assertLess(positions["[Document]"], positions["[Chapter]"])
        self.assertLess(positions["[Chapter]"], positions["[Article]"])
        self.assertLess(positions["[Article]"], positions["[Clause]"])
        self.assertLess(positions["[Clause]"], positions["[Point]"])
        self.assertLess(positions["[Point]"], positions["[Content]"])

    # --- Legal chunk with NULL fields ---

    def test_legal_null_chapter_and_point_omitted(self):
        """NULL chapter and point produce no label; no placeholder strings."""
        fn = self._fn()
        result = fn(
            text="nội dung",
            metadata={
                "content_type": "legal_text",
                "document_title": self._ND116,
                "chapter": None,
                "article": "Điều 4",
                "clause": "Khoản 1",
                "point": None,
            },
        )
        self.assertNotIn("[Chapter]", result)
        self.assertNotIn("[Point]", result)
        self.assertIn("[Document]", result)
        self.assertIn("[Article]", result)
        self.assertIn("[Clause]", result)
        self.assertIn("[Content]", result)
        # No placeholder strings.
        for placeholder in ("None", "NULL", "N/A", "null"):
            self.assertNotIn(placeholder, result)

    def test_legal_null_all_structural_only_document_and_content(self):
        """When only document_title is non-null, only [Document] and [Content] appear."""
        fn = self._fn()
        result = fn(
            text="some text",
            metadata={
                "content_type": "legal_text",
                "document_title": self._ND116,
                "chapter": None,
                "article": None,
                "clause": None,
                "point": None,
            },
        )
        self.assertIn("[Document]", result)
        self.assertIn("[Content]", result)
        self.assertNotIn("[Chapter]", result)
        self.assertNotIn("[Article]", result)
        self.assertNotIn("[Clause]", result)
        self.assertNotIn("[Point]", result)

    # --- Legal chunk with empty-string metadata fields ---

    def test_legal_empty_string_field_omitted(self):
        """Empty-string metadata fields are treated the same as NULL."""
        fn = self._fn()
        result = fn(
            text="text",
            metadata={
                "content_type": "legal_text",
                "document_title": self._ND116,
                "chapter": "",
                "article": "Điều 4",
                "clause": "   ",  # whitespace-only
                "point": None,
            },
        )
        self.assertNotIn("[Chapter]", result)
        self.assertNotIn("[Clause]", result)
        self.assertIn("[Article]", result)

    # --- QA chunk ---

    def test_qa_chunk_returns_text_unchanged(self):
        """QA chunks are returned unchanged — no structural labels added."""
        fn = self._fn()
        qa_text = "Câu hỏi: Sinh viên sư phạm được hỗ trợ bao nhiêu?\nGiải đáp: Theo khoản 1 Điều 4 Nghị định 116..."
        result = fn(
            text=qa_text,
            metadata={"content_type": "qa"},
        )
        self.assertEqual(result, qa_text)
        for label in ("[Document]", "[Chapter]", "[Article]", "[Clause]", "[Point]", "[Content]"):
            self.assertNotIn(label, result)

    def test_qa_with_legal_text_mention_generates_no_structural_labels(self):
        """A QA chunk whose text mentions 'Điều 4 Khoản 1' must not produce structural labels."""
        fn = self._fn()
        text = "Theo Điều 4 Khoản 1 Nghị định 116 thì ..."
        result = fn(text=text, metadata={"content_type": "qa"})
        self.assertEqual(result, text)
        self.assertNotIn("[Article]", result)
        self.assertNotIn("[Clause]", result)

    # --- reference/supporting legal source (Luật Giáo dục 2019) ---

    def test_reference_supporting_legal_source_gets_prefix(self):
        """reference/supporting with content_type='legal_text' must get the full prefix."""
        fn = self._fn()
        result = fn(
            text="Nội dung điều khoản",
            metadata={
                "content_type": "legal_text",
                "source_type": "reference",       # extra key, must be ignored for prefix
                "document_role": "supporting",    # extra key, must be ignored for prefix
                "document_title": "Luật Giáo dục 2019",
                "chapter": "Chương I",
                "article": "Điều 5",
                "clause": "Khoản 2",
                "point": None,
            },
        )
        self.assertIn("[Document]", result)
        self.assertIn("Luật Giáo dục 2019", result)
        self.assertIn("[Article]", result)
        self.assertIn("Điều 5", result)
        self.assertNotIn("[Point]", result)

    # --- Determinism ---

    def test_determinism_same_input_same_output(self):
        """Calling build_embedding_text twice with identical args produces identical output."""
        fn = self._fn()
        meta = {
            "content_type": "legal_text",
            "document_title": self._ND116,
            "chapter": "Chương II",
            "article": "Điều 4",
            "clause": "Khoản 1",
            "point": "Điểm a",
        }
        result1 = fn(text="nội dung", metadata=meta)
        result2 = fn(text="nội dung", metadata=meta)
        self.assertEqual(result1, result2)

    # --- Original text preserved verbatim ---

    def test_original_text_preserved_in_content_section(self):
        """The original chunk text is included verbatim after [Content]."""
        fn = self._fn()
        original = "Mức hỗ trợ tiền đóng học phí **quan trọng** \n\n- Bullet 1\n- Bullet 2"
        result = fn(
            text=original,
            metadata={
                "content_type": "legal_text",
                "document_title": self._ND116,
                "chapter": None,
                "article": "Điều 4",
                "clause": None,
                "point": None,
            },
        )
        # The original text must appear after [Content]
        self.assertIn(original, result)
        content_pos = result.index("[Content]")
        text_pos = result.index(original)
        self.assertGreater(text_pos, content_pos)

    # --- Unknown content_type ---

    def test_unknown_content_type_returns_text_unchanged(self):
        """An unrecognised content_type is treated like 'qa': text returned unchanged."""
        fn = self._fn()
        text = "some unrecognised chunk"
        result = fn(text=text, metadata={"content_type": "unknown_type"})
        self.assertEqual(result, text)

    def test_missing_content_type_returns_text_unchanged(self):
        """Missing content_type key is treated as non-legal: text returned unchanged."""
        fn = self._fn()
        text = "orphan chunk"
        result = fn(text=text, metadata={})
        self.assertEqual(result, text)

    # --- No placeholder strings in any legal output ---

    def test_no_placeholder_strings_in_legal_output(self):
        """'None', 'NULL', 'N/A' must never appear in the output for a legal chunk."""
        fn = self._fn()
        result = fn(
            text="nội dung",
            metadata={
                "content_type": "legal_text",
                "document_title": self._ND116,
                "chapter": None,
                "article": "Điều 4",
                "clause": None,
                "point": None,
            },
        )
        for placeholder in ("None", "NULL", "N/A", "null", "none"):
            self.assertNotIn(placeholder, result)

    # --- Integration: build_embedding_text output fed to embed_texts ---

    def test_build_embedding_text_output_is_embeddable(self):
        """Output of build_embedding_text can be passed to embed_documents without error."""
        import embedding as em  # noqa: PLC0415
        fn = em.build_embedding_text
        model = em.EmbeddingModel()
        emb_input = fn(
            text="Mức hỗ trợ tiền đóng học phí...",
            metadata={
                "content_type": "legal_text",
                "document_title": self._ND116,
                "chapter": "Chương II",
                "article": "Điều 4",
                "clause": "Khoản 1",
                "point": None,
            },
        )
        vectors = model.embed_documents([emb_input])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), FAKE_DIM)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# DeepX backend tests
# ---------------------------------------------------------------------------


class TestDeepXRegistry(unittest.TestCase):
    """Verify DeepX model_registry entries without loading any real model."""

    def setUp(self):
        _cleanup_modules()

    def tearDown(self):
        _cleanup_modules()

    def _import(self):
        import model_registry as mr  # noqa: PLC0415
        return mr

    def test_deepx_resolves_by_alias(self):
        mr = self._import()
        config = mr.resolve_model_config("deepx")
        self.assertEqual(config.alias, "deepx")

    def test_deepx_resolves_by_model_id(self):
        mr = self._import()
        config = mr.resolve_model_config("dxtech-asia/deepx-embedding-v1")
        self.assertEqual(config.alias, "deepx")

    def test_deepx_backend_is_deepx(self):
        mr = self._import()
        config = mr.resolve_model_config("deepx")
        self.assertEqual(config.backend, "deepx")

    def test_deepx_dimension_is_1024(self):
        mr = self._import()
        config = mr.resolve_model_config("deepx")
        self.assertEqual(config.dimension, 1024)

    def test_deepx_max_seq_length_is_8192(self):
        mr = self._import()
        config = mr.resolve_model_config("deepx")
        self.assertEqual(config.max_seq_length, 8192)

    def test_deepx_trust_remote_code_is_true(self):
        mr = self._import()
        config = mr.resolve_model_config("deepx")
        self.assertTrue(config.trust_remote_code)

    def test_deepx_embedding_column(self):
        mr = self._import()
        config = mr.resolve_model_config("deepx")
        self.assertEqual(config.embedding_column, "embedding_deepx")

    def test_deepx_hnsw_index_name(self):
        mr = self._import()
        config = mr.resolve_model_config("deepx")
        self.assertEqual(config.hnsw_index_name, "legal_chunks_embedding_deepx_hnsw_idx")

    def test_deepx_normalize_is_true(self):
        mr = self._import()
        config = mr.resolve_model_config("deepx")
        self.assertTrue(config.normalize)

    # --- Verify other models still use their original backends ---

    def test_bge_m3_backend_unchanged(self):
        mr = self._import()
        self.assertEqual(mr.resolve_model_config("bge-m3").backend, "bge")

    def test_vnlegal_lal_backend_unchanged(self):
        mr = self._import()
        self.assertEqual(mr.resolve_model_config("vnlegal-lal").backend, "sentence_transformer")

    def test_vietlegal_harrier_backend_unchanged(self):
        mr = self._import()
        self.assertEqual(mr.resolve_model_config("vietlegal-harrier").backend, "sentence_transformer")

    def test_vietlegal_e5_backend_unchanged(self):
        mr = self._import()
        self.assertEqual(mr.resolve_model_config("vietlegal-e5").backend, "sentence_transformer")

    def test_jina_v3_backend_unchanged(self):
        mr = self._import()
        self.assertEqual(mr.resolve_model_config("jina-v3").backend, "jina")


class TestDeepXBackendFactory(unittest.TestCase):
    """Verify that the backend factory creates a _DeepXBackend for deepx configs."""

    def setUp(self):
        self.fake_deepx = _make_fake_deepx_module()
        sys.modules["deepx_embed"] = self.fake_deepx
        _cleanup_modules()

    def tearDown(self):
        sys.modules.pop("deepx_embed", None)
        _cleanup_modules()

    def test_factory_creates_deepx_backend(self):
        import embedding as em  # noqa: PLC0415
        import model_registry as mr  # noqa: PLC0415
        config = mr.resolve_model_config("deepx")
        backend = em._create_backend(config)
        self.assertIsInstance(backend, em._DeepXBackend)

    def test_factory_does_not_create_st_backend_for_deepx(self):
        import embedding as em  # noqa: PLC0415
        import model_registry as mr  # noqa: PLC0415
        config = mr.resolve_model_config("deepx")
        backend = em._create_backend(config)
        self.assertNotIsInstance(backend, em._SentenceTransformerBackend)

    def test_deepx_embedding_model_uses_deepx_backend(self):
        import embedding as em  # noqa: PLC0415
        model = em.EmbeddingModel(model_name="deepx")
        self.assertIsInstance(model._backend, em._DeepXBackend)

    def test_from_pretrained_called_with_model_id(self):
        import embedding as em  # noqa: PLC0415
        em.EmbeddingModel(model_name="deepx")
        call_args = self.fake_deepx.DeepXEmbed.from_pretrained.call_args
        self.assertEqual(call_args[0][0], "dxtech-asia/deepx-embedding-v1")


class TestDeepXBatching(unittest.TestCase):
    """Verify batching logic: ordering, count, and batch_size handling."""

    def setUp(self):
        self.fake_deepx = _make_fake_deepx_module()
        sys.modules["deepx_embed"] = self.fake_deepx
        _cleanup_modules()

    def tearDown(self):
        sys.modules.pop("deepx_embed", None)
        _cleanup_modules()

    def _make_model(self):
        import embedding as em  # noqa: PLC0415
        return em.EmbeddingModel(model_name="deepx")

    def test_output_count_matches_input_count(self):
        model = self._make_model()
        texts = [f"document {i}" for i in range(17)]
        result = model.embed_documents(texts, batch_size=5)
        self.assertEqual(len(result), 17)

    def test_single_text_batch(self):
        model = self._make_model()
        result = model.embed_documents(["single doc"], batch_size=10)
        self.assertEqual(len(result), 1)

    def test_batch_size_larger_than_input(self):
        model = self._make_model()
        texts = ["a", "b", "c"]
        result = model.embed_documents(texts, batch_size=100)
        self.assertEqual(len(result), 3)

    def test_ordering_preserved_across_batches(self):
        """Each input position maps to the correct output vector."""
        import numpy as np  # noqa: PLC0415

        call_order: list[str] = []

        def ordered_encode(texts, truncate_dim=1024, **kwargs):
            call_order.extend(texts)
            n = len(texts) if hasattr(texts, "__len__") else 1
            return np.zeros((n, truncate_dim), dtype="float32")

        self.fake_deepx.DeepXEmbed.from_pretrained.return_value.encode.side_effect = ordered_encode

        model = self._make_model()
        texts = [f"text-{i}" for i in range(10)]
        model.embed_documents(texts, batch_size=3)

        self.assertEqual(call_order, texts)

    def test_embed_query_returns_single_vector(self):
        model = self._make_model()
        result = model.embed_query("test query")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), FAKE_DIM)
        self.assertIsInstance(result[0], float)


class TestDeepXDimension(unittest.TestCase):
    """Verify DeepX always returns exactly 1024-dimensional vectors."""

    def setUp(self):
        self.fake_deepx = _make_fake_deepx_module()
        sys.modules["deepx_embed"] = self.fake_deepx
        _cleanup_modules()

    def tearDown(self):
        sys.modules.pop("deepx_embed", None)
        _cleanup_modules()

    def _make_model(self):
        import embedding as em  # noqa: PLC0415
        return em.EmbeddingModel(model_name="deepx")

    def test_embed_query_returns_1024d(self):
        model = self._make_model()
        result = model.embed_query("test")
        self.assertEqual(len(result), 1024)

    def test_embed_documents_returns_1024d_each(self):
        model = self._make_model()
        result = model.embed_documents(["a", "b", "c"])
        for vec in result:
            self.assertEqual(len(vec), 1024)

    def test_truncate_dim_forwarded_to_encode(self):
        """The backend must request truncate_dim=1024 from the model."""
        import embedding as em  # noqa: PLC0415
        model = em.EmbeddingModel(model_name="deepx")
        model.embed_documents(["test text"])
        encode_call = self.fake_deepx.DeepXEmbed.from_pretrained.return_value.encode.call_args
        # truncate_dim should be 1024, not 1536 or None.
        kwargs = encode_call[1] if encode_call[1] else {}
        args = encode_call[0] if encode_call[0] else ()
        # truncate_dim may be positional or keyword depending on the fake.
        self.assertIn("truncate_dim", kwargs)
        self.assertEqual(kwargs["truncate_dim"], 1024)

    def test_wrong_dimension_caught_by_validation(self):
        """If the backend returns wrong-dim vectors, EmbeddingModel catches it."""
        import numpy as np  # noqa: PLC0415
        import embedding as em  # noqa: PLC0415

        def bad_encode(texts, truncate_dim=1024, **kwargs):
            return np.zeros((len(texts), 768), dtype="float32")

        self.fake_deepx.DeepXEmbed.from_pretrained.return_value.encode.side_effect = bad_encode

        model = em.EmbeddingModel(model_name="deepx")
        with self.assertRaises(ValueError) as ctx:
            model.embed_documents(["test"])
        self.assertIn("Dimension mismatch", str(ctx.exception))


class TestDeepXEmptyInput(unittest.TestCase):
    """Verify that empty input is handled correctly."""

    def setUp(self):
        self.fake_deepx = _make_fake_deepx_module()
        sys.modules["deepx_embed"] = self.fake_deepx
        _cleanup_modules()

    def tearDown(self):
        sys.modules.pop("deepx_embed", None)
        _cleanup_modules()

    def test_embed_documents_empty_list_returns_empty(self):
        import embedding as em  # noqa: PLC0415
        model = em.EmbeddingModel(model_name="deepx")
        result = model.embed_documents([])
        self.assertEqual(result, [])

    def test_encode_not_called_for_empty_input(self):
        import embedding as em  # noqa: PLC0415
        model = em.EmbeddingModel(model_name="deepx")
        model.embed_documents([])
        encode_mock = self.fake_deepx.DeepXEmbed.from_pretrained.return_value.encode
        encode_mock.assert_not_called()


class TestDeepXMissingLibrary(unittest.TestCase):
    """Verify a clear ImportError is raised when deepx_embed is not installed."""

    def setUp(self):
        # Ensure deepx_embed is NOT in sys.modules.
        sys.modules.pop("deepx_embed", None)
        _cleanup_modules()

    def tearDown(self):
        sys.modules.pop("deepx_embed", None)
        _cleanup_modules()

    def test_import_error_raised_with_install_hint(self):
        import builtins  # noqa: PLC0415
        _real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "deepx_embed" or name.startswith("deepx_embed."):
                raise ModuleNotFoundError(f"No module named '{name}'")
            return _real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_blocking_import):
            import embedding as em  # noqa: PLC0415
            import model_registry as mr  # noqa: PLC0415
            config = mr.resolve_model_config("deepx")
            with self.assertRaises(ImportError) as ctx:
                em._create_backend(config)
        self.assertIn("deepx_embed", str(ctx.exception))
        self.assertIn("pip install", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
