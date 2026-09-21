"""Embedding component for the legal RAG chatbot.

Provides a thin wrapper around BAAI/bge-m3 that converts text into dense
1024-dimensional vectors suitable for cosine-similarity search via pgvector.

This module is intentionally kept free of database logic so that the same
``EmbeddingModel`` class can be reused by both the indexing stage
(``index_embeddings.py``) and the future retrieval stage (query embedding).

Usage::

    from embedding import EmbeddingModel

    model = EmbeddingModel()                      # loads BAAI/bge-m3 once
    vectors = model.embed_texts(["Điều 1 ..."])   # list[list[float]]

Environment variables
---------------------
EMBEDDING_MODEL
    Hugging Face model identifier.  Defaults to ``BAAI/bge-m3``.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # Dense output dimension of BAAI/bge-m3.


# ---------------------------------------------------------------------------
# EmbeddingModel
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Load BAAI/bge-m3 once and embed arbitrary text batches.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier.  Falls back to the ``EMBEDDING_MODEL``
        environment variable, then to ``BAAI/bge-m3``.
    use_fp16:
        Use 16-bit floating point for inference when a CUDA GPU is available.
        Ignored on CPU; FlagEmbedding automatically uses fp32 on CPU.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        use_fp16: bool = True,
    ) -> None:
        self._model_name = (
            model_name
            or os.environ.get("EMBEDDING_MODEL", "").strip()
            or DEFAULT_MODEL
        )
        self._use_fp16 = use_fp16
        self._model = self._load_model()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of strings and return a list of float vectors.

        Parameters
        ----------
        texts:
            Texts to embed.  Vietnamese content is handled transparently by
            the multilingual bge-m3 model; do not pre-translate or strip
            Vietnamese diacritics.
        batch_size:
            Number of texts to process in a single model forward pass.
            This parameter is passed through to FlagEmbedding so that callers
            can control GPU/CPU memory usage.

        Returns
        -------
        list[list[float]]
            One 1024-dimensional float vector per input text, in the same
            order.  Vectors are L2-normalised by FlagEmbedding, making cosine
            similarity equivalent to dot-product similarity.
        """
        if not texts:
            return []

        result = self._model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=8192,   # bge-m3 supports up to 8192 tokens
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        # FlagEmbedding returns a dict when return_dense=True.
        dense = result["dense_vecs"]
        # Convert numpy float32 array rows to plain Python lists.
        return [row.tolist() for row in dense]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Import FlagEmbedding and initialise the model.

        The import is deferred so that modules that merely import this file
        (e.g. tests that mock the model) do not trigger the full model load.
        """
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is not installed.  "
                "Run: pip install FlagEmbedding"
            ) from exc

        device = self._detect_device()
        use_fp16 = self._use_fp16 and (device == "cuda")

        LOGGER.info("Embedding model: %s", self._model_name)
        LOGGER.info("Device: %s", device)
        if device == "cuda":
            try:
                import torch  # type: ignore[import]
                gpu_name = torch.cuda.get_device_name(0)
                LOGGER.info("GPU: %s", gpu_name)
            except Exception:
                pass
            if use_fp16:
                LOGGER.info("Using fp16 inference")
        else:
            LOGGER.info("CUDA unavailable; using CPU")

        model = BGEM3FlagModel(
            self._model_name,
            use_fp16=use_fp16,
            device=device,
        )
        return model


    @staticmethod
    def _detect_device() -> str:
        """Return 'cuda' if a CUDA GPU is available, otherwise 'cpu'."""
        try:
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
