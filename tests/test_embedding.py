"""Unit tests for the embedding and indexing components.

These tests use unittest.mock to patch BAAI/bge-m3 so that the real 1 GB
model is never downloaded during CI or local test runs.

Run from the repository root::

    conda activate chatbot
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


# ---------------------------------------------------------------------------
# EmbeddingModel tests
# ---------------------------------------------------------------------------


class TestEmbeddingModelInit(unittest.TestCase):
    def setUp(self):
        self.fake_fe = _make_fake_flag_embedding_module()
        sys.modules["FlagEmbedding"] = self.fake_fe
        # Force reimport so the patched module is used.
        if "embedding" in sys.modules:
            del sys.modules["embedding"]

    def tearDown(self):
        del sys.modules["FlagEmbedding"]
        if "embedding" in sys.modules:
            del sys.modules["embedding"]

    def _import(self):
        import embedding as em  # noqa: PLC0415
        return em

    def test_default_model_name(self):
        em = self._import()
        model = em.EmbeddingModel()
        self.assertEqual(model.model_name, "BAAI/bge-m3")

    def test_custom_model_name(self):
        em = self._import()
        model = em.EmbeddingModel(model_name="custom/model")
        self.assertEqual(model.model_name, "custom/model")

    def test_embedding_dim_constant(self):
        em = self._import()
        model = em.EmbeddingModel()
        self.assertEqual(model.embedding_dim, 1024)


class TestEmbedTexts(unittest.TestCase):
    def setUp(self):
        self.fake_fe = _make_fake_flag_embedding_module()
        sys.modules["FlagEmbedding"] = self.fake_fe
        if "embedding" in sys.modules:
            del sys.modules["embedding"]

    def tearDown(self):
        del sys.modules["FlagEmbedding"]
        if "embedding" in sys.modules:
            del sys.modules["embedding"]

    def _make_model(self):
        import embedding as em  # noqa: PLC0415
        return em.EmbeddingModel()

    def test_returns_list_of_lists(self):
        model = self._make_model()
        result = model.embed_texts(["Điều 1. Phạm vi điều chỉnh"])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], list)

    def test_correct_output_dimension(self):
        model = self._make_model()
        result = model.embed_texts(["Điều 1", "Khoản 2"])
        self.assertEqual(len(result), 2)
        for vec in result:
            self.assertEqual(len(vec), FAKE_DIM)

    def test_empty_input_returns_empty(self):
        model = self._make_model()
        result = model.embed_texts([])
        self.assertEqual(result, [])

    def test_multiple_texts(self):
        model = self._make_model()
        texts = [f"Văn bản pháp luật {i}" for i in range(10)]
        result = model.embed_texts(texts)
        self.assertEqual(len(result), 10)

    def test_vietnamese_text_passes_through(self):
        """Verify that Vietnamese diacritics are not stripped (model receives original text)."""
        model = self._make_model()
        original = "Nghị định số 116/2020/NĐ-CP của Chính phủ"
        model.embed_texts([original])
        call_args = self.fake_fe.BGEM3FlagModel.return_value.encode.call_args
        texts_sent = call_args[0][0]  # first positional arg
        self.assertIn(original, texts_sent)


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
        for mod in ("embedding", "index_embeddings"):
            sys.modules.pop(mod, None)

    def tearDown(self):
        del sys.modules["FlagEmbedding"]
        for mod in ("embedding", "index_embeddings"):
            sys.modules.pop(mod, None)

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
            stats = ie.index_chunks("postgresql://fake/db", model, chunks, batch_size=32)

        self.assertEqual(stats.succeeded, 2)
        self.assertEqual(stats.failed, 0)
        mock_conn.commit.assert_called()

    def test_index_chunks_idempotency_skip_null(self):
        """SELECT_UNEMBEDDED filters chunks WHERE embedding IS NULL."""
        ie = self._import()
        # The SQL constant must contain the IS NULL guard.
        self.assertIn("IS NULL", ie.SELECT_UNEMBEDDED)

    def test_rebuild_query_has_no_null_filter(self):
        ie = self._import()
        self.assertNotIn("IS NULL", ie.SELECT_ALL)

    def test_verify_detects_missing(self):
        """verify() should reflect counts returned by the mock DB cursor."""
        ie = self._import()

        mock_cursor = MagicMock()
        # VERIFY_COUNTS returns (total, embedded).
        # VERIFY_DIMENSION returns (chunk_id, dim).
        # CHECK_HNSW_INDEX returns a row (index exists).
        mock_cursor.fetchone.side_effect = [
            (618, 500),   # VERIFY_COUNTS: 118 missing
            ("chunk-1", 1024),  # VERIFY_DIMENSION
            ("legal_chunks_embedding_hnsw_idx",),  # CHECK_HNSW_INDEX
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = ie.verify(mock_conn)

        self.assertEqual(result["total_chunks"], 618)
        self.assertEqual(result["embedded_chunks"], 500)
        self.assertEqual(result["missing_embeddings"], 118)
        self.assertTrue(result["embedding_dim_ok"])
        self.assertTrue(result["hnsw_index_exists"])

    def test_verify_detects_wrong_dimension(self):
        ie = self._import()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            (618, 618),
            ("chunk-1", 768),   # wrong dim
            ("legal_chunks_embedding_hnsw_idx",),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = ie.verify(mock_conn)
        self.assertFalse(result["embedding_dim_ok"])
        self.assertEqual(result["embedding_dim"], 768)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# build_embedding_text tests
# ---------------------------------------------------------------------------


class TestBuildEmbeddingText(unittest.TestCase):
    """Tests for embedding.build_embedding_text().

    These tests do NOT load BGE-M3; they only verify the string construction
    logic.  The fake FlagEmbedding module is injected via sys.modules so that
    importing ``embedding`` succeeds without the real model.
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
        """Output of build_embedding_text can be passed to embed_texts without error."""
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
        vectors = model.embed_texts([emb_input])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), FAKE_DIM)


if __name__ == "__main__":
    unittest.main()
