"""Embedding component for the legal RAG chatbot.

Provides a unified interface for multiple Hugging Face embedding models
behind a stable API.  The rest of the system (retrieval, evaluation)
interacts only with :class:`EmbeddingModel` and does not need to know
which backend is currently selected.

Supported backends
------------------
``bge``
    BAAI/bge-m3 via FlagEmbedding (BGEM3FlagModel).
``sentence_transformer``
    Generic models via the ``sentence-transformers`` library.  Handles
    query/passage prefixes (E5), ``trust_remote_code``, and Matryoshka
    dimension truncation via model configuration.
``jina``
    jinaai/jina-embeddings-v3 via ``sentence-transformers`` with
    task-specific LoRA adapters.
``deepx``
    dxtech-asia/deepx-embedding-v1 via the ``deepx_embed`` package
    (``DeepXEmbed.from_pretrained``).  This model uses a fully custom
    Gated DeltaNet-2 linear-attention architecture that is NOT registered
    in the ``transformers`` AutoModel registry.  Attempting to load it
    through the generic ``SentenceTransformer`` constructor therefore
    fails with an "architecture not recognised" error, regardless of
    ``trust_remote_code``.  The dedicated ``_DeepXBackend`` bypasses
    ``SentenceTransformer`` entirely and uses the official
    ``deepx_embed.DeepXEmbed`` API.  Matryoshka truncation to 1024
    dimensions is requested directly from the model via
    ``DeepXEmbed.encode(truncate_dim=1024)``.

    Dependency: ``pip install git+https://github.com/dx-tech-ai/deepx-embed.git``
    (does not change the installed ``transformers`` or
    ``sentence-transformers`` versions).

The module also exposes :func:`build_embedding_text`, which constructs the
string sent to the model for each chunk.  This function is
**model-independent** — it formats the metadata-aware representation
regardless of which backend is active.

Usage::

    from embedding import EmbeddingModel, build_embedding_text

    model = EmbeddingModel()                              # loads configured model
    query_vec = model.embed_query("Điều 1 ...")           # list[float]
    doc_vecs = model.embed_documents(["text1", "text2"])  # list[list[float]]

    # Backwards-compatible alias:
    doc_vecs = model.embed_texts(["text1", "text2"])      # same as embed_documents

Environment variables
---------------------
EMBEDDING_MODEL
    Hugging Face model identifier or alias.  Defaults to ``BAAI/bge-m3``.
"""

from __future__ import annotations

import inspect
import logging
import os
from abc import ABC, abstractmethod
from typing import Sequence

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # Project-wide required dimension.

# Ordered sequence of (label, metadata_key) pairs for legal-document chunks.
# Only fields whose values are non-NULL and non-empty are included.
_LEGAL_META_FIELDS: tuple[tuple[str, str], ...] = (
    ("[Document]",  "document_title"),
    ("[Chapter]",   "chapter"),
    ("[Article]",   "article"),
    ("[Clause]",    "clause"),
    ("[Point]",     "point"),
)


# ---------------------------------------------------------------------------
# Embedding-input formatter (model-independent)
# ---------------------------------------------------------------------------


def build_embedding_text(text: str, metadata: dict) -> str:
    """Build the string sent to the embedding model for a single chunk.

    For legal-document chunks (``content_type == 'legal_text'``) a structured
    prefix is prepended to the original chunk text.  The prefix encodes
    available structural metadata in a deterministic, hierarchical order::

        [Document]
        <document_title>

        [Chapter]
        <chapter>

        [Article]
        <article>

        [Clause]
        <clause>

        [Point]
        <point>

        [Content]
        <original chunk text>

    Only fields whose values are non-``None`` and non-empty after stripping
    are included.  ``None``, empty strings, and whitespace-only strings are
    silently omitted — no placeholder text (``'N/A'``, ``'NULL'``, ``'None'``)
    is ever inserted.

    For QA chunks (``content_type == 'qa'``) the original ``text`` is
    returned unchanged.  Legal references that appear *inside* QA answer text
    are never converted into structural metadata fields.

    The function is **deterministic**: identical inputs always produce
    identical outputs.

    Parameters
    ----------
    text:
        Original chunk text from ``legal_chunks.text``.  Returned verbatim
        as the ``[Content]`` section (or the full output for QA chunks).
    metadata:
        Dictionary of chunk metadata.  Only the following keys are
        inspected: ``content_type``, ``document_title``, ``chapter``,
        ``article``, ``clause``, ``point``.
        Extra keys are ignored; missing keys are treated as ``None``.

    Returns
    -------
    str
        The final string to pass to ``EmbeddingModel.embed_documents()``.
    """
    content_type = (metadata.get("content_type") or "").strip().lower()
    if content_type != "legal_text":
        # QA chunks and any unrecognised type: return text unchanged.
        return text

    # Build the structured prefix for legal-document chunks.
    parts: list[str] = []
    for label, key in _LEGAL_META_FIELDS:
        value = (metadata.get(key) or "").strip()
        if value:
            parts.append(f"{label}\n{value}")

    # Always append the original content under [Content].
    parts.append(f"[Content]\n{text}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Device detection (shared by all backends)
# ---------------------------------------------------------------------------


def _detect_device() -> str:
    """Return ``'cuda'`` if a CUDA GPU is available, otherwise ``'cpu'``."""
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _log_gpu_info() -> None:
    """Log the GPU name if available."""
    try:
        import torch  # type: ignore[import]
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class _EmbeddingBackend(ABC):
    """Internal interface implemented by each model backend."""

    @abstractmethod
    def embed_query(self, text: str, batch_size: int = 32) -> list[float]:
        """Embed a single query text and return a float vector."""

    @abstractmethod
    def embed_documents(
        self, texts: Sequence[str], batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed a batch of document texts and return a list of float vectors."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Compute device: ``'cuda'`` or ``'cpu'``."""

    @property
    def raw_model(self):
        """Return the underlying model object (for introspection only)."""
        return getattr(self, "_model", None)


# ---------------------------------------------------------------------------
# BGE-M3 backend (FlagEmbedding)
# ---------------------------------------------------------------------------


class _BGEBackend(_EmbeddingBackend):
    """Backend for BAAI/bge-m3 using the FlagEmbedding library.

    BGE-M3 does not distinguish query vs. document encoding — both use the
    same ``encode()`` call.  The model natively produces L2-normalised
    1024-dimensional dense vectors.
    """

    def __init__(self, config, *, use_fp16: bool = True) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is not installed.  "
                "Run: pip install FlagEmbedding"
            ) from exc

        self._config = config
        device = _detect_device()
        self._device = device
        fp16 = use_fp16 and (device == "cuda")

        LOGGER.info("Embedding model: %s", config.model_id)
        LOGGER.info("Backend: FlagEmbedding (BGE)")
        LOGGER.info("Device: %s", device)
        if device == "cuda":
            _log_gpu_info()
            if fp16:
                LOGGER.info("Using fp16 inference")
        else:
            LOGGER.info("CUDA unavailable; using CPU")

        self._model = BGEM3FlagModel(
            config.model_id,
            use_fp16=fp16,
            device=device,
        )

    @property
    def device(self) -> str:
        return self._device

    def embed_query(self, text: str, batch_size: int = 32) -> list[float]:
        vecs = self._encode([text], batch_size)
        return vecs[0]

    def embed_documents(
        self, texts: Sequence[str], batch_size: int = 32,
    ) -> list[list[float]]:
        return self._encode(list(texts), batch_size)

    def _encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        if not texts:
            return []
        result = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=self._config.max_seq_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        # FlagEmbedding returns a dict when return_dense=True.
        dense = result["dense_vecs"]
        # Convert numpy float32 array rows to plain Python lists.
        return [row.tolist() for row in dense]


# ---------------------------------------------------------------------------
# Sentence Transformer backend (generic)
# ---------------------------------------------------------------------------


class _SentenceTransformerBackend(_EmbeddingBackend):
    """Generic backend using the ``sentence-transformers`` library.

    Handles via configuration:

    - Query/document text prefixes (E5 models)
    - ``trust_remote_code`` (DeepX and others)
    - Matryoshka dimension truncation (when native dim > required 1024)
    """

    def __init__(self, config, *, use_fp16: bool = True) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed.  "
                "Run: pip install sentence-transformers"
            ) from exc

        self._config = config
        device = _detect_device()
        self._device = device

        LOGGER.info("Embedding model: %s", config.model_id)
        LOGGER.info("Backend: SentenceTransformer")
        LOGGER.info("Device: %s", device)

        load_kwargs: dict = {}
        if config.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        if device == "cuda":
            _log_gpu_info()
        else:
            LOGGER.info("CUDA unavailable; using CPU")

        self._model = SentenceTransformer(
            config.model_id, device=device, **load_kwargs,
        )

        # Matryoshka truncation: if native dimension > required, truncate.
        native_dim = self._model.get_sentence_embedding_dimension()
        if native_dim and native_dim != config.dimension:
            self._model.truncate_dim = config.dimension
            LOGGER.info(
                "Matryoshka truncation: %d \u2192 %d dimensions",
                native_dim, config.dimension,
            )

        # Respect model's documented max_seq_length.
        self._model.max_seq_length = config.max_seq_length

        if config.query_prefix:
            LOGGER.info("Query prefix: %r", config.query_prefix)
        if config.document_prefix:
            LOGGER.info("Document prefix: %r", config.document_prefix)

    @property
    def device(self) -> str:
        return self._device

    def embed_query(self, text: str, batch_size: int = 32) -> list[float]:
        prefixed = (
            self._config.query_prefix + text
            if self._config.query_prefix
            else text
        )
        result = self._model.encode(
            [prefixed],
            batch_size=batch_size,
            normalize_embeddings=self._config.normalize,
        )
        return result[0].tolist()

    def embed_documents(
        self, texts: Sequence[str], batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            return []
        input_texts = (
            [self._config.document_prefix + t for t in texts]
            if self._config.document_prefix
            else list(texts)
        )
        result = self._model.encode(
            input_texts,
            batch_size=batch_size,
            normalize_embeddings=self._config.normalize,
        )
        return [row.tolist() for row in result]


# ---------------------------------------------------------------------------
# Jina backend (task-specific LoRA via prompt_name / task)
# ---------------------------------------------------------------------------


class _JinaBackend(_EmbeddingBackend):
    """Backend for jinaai/jina-embeddings-v3 with task-specific LoRA adapters.

    Jina v3 selects the correct LoRA adapter via a ``task`` or
    ``prompt_name`` keyword argument to ``encode()``.  This backend detects
    which keyword the model's custom ``encode()`` method accepts and uses it
    transparently.
    """

    def __init__(self, config, *, use_fp16: bool = True) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed.  "
                "Run: pip install sentence-transformers"
            ) from exc

        self._config = config
        device = _detect_device()
        self._device = device

        LOGGER.info("Embedding model: %s", config.model_id)
        LOGGER.info("Backend: Jina (task-specific LoRA)")
        LOGGER.info("Device: %s", device)

        if device == "cuda":
            _log_gpu_info()
        else:
            LOGGER.info("CUDA unavailable; using CPU")

        self._model = SentenceTransformer(
            config.model_id,
            trust_remote_code=True,
            device=device,
        )
        self._model.max_seq_length = config.max_seq_length

        # Detect which keyword argument the encode method uses for tasks.
        self._task_key = self._detect_task_key()
        LOGGER.info(
            "Jina task parameter: %s (query=%s, document=%s)",
            self._task_key,
            config.query_prompt_name,
            config.document_prompt_name,
        )

    @property
    def device(self) -> str:
        return self._device

    def embed_query(self, text: str, batch_size: int = 32) -> list[float]:
        vecs = self._encode([text], batch_size, self._config.query_prompt_name)
        return vecs[0]

    def embed_documents(
        self, texts: Sequence[str], batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(list(texts), batch_size, self._config.document_prompt_name)

    def _encode(
        self, texts: list[str], batch_size: int, task: str | None,
    ) -> list[list[float]]:
        kwargs: dict = {
            "batch_size": batch_size,
            "normalize_embeddings": self._config.normalize,
        }
        if task and self._task_key:
            kwargs[self._task_key] = task

        result = self._model.encode(texts, **kwargs)
        return [row.tolist() for row in result]

    def _detect_task_key(self) -> str | None:
        """Detect which keyword argument the ``encode()`` method accepts."""
        try:
            sig = inspect.signature(self._model.encode)
            # Jina's custom code typically adds 'task'; standard ST uses 'prompt_name'.
            if "task" in sig.parameters:
                return "task"
            if "prompt_name" in sig.parameters:
                return "prompt_name"
        except (ValueError, TypeError):
            pass
        # Default to 'task' (Jina custom code expectation).
        return "task"


# ---------------------------------------------------------------------------
# DeepX backend (deepx_embed package — official DeepXEmbed API)
# ---------------------------------------------------------------------------


class _DeepXBackend(_EmbeddingBackend):
    """Backend for dxtech-asia/deepx-embedding-v1 via the ``deepx_embed`` package.

    Why a dedicated backend?
    ~~~~~~~~~~~~~~~~~~~~~~~~
    DeepX uses a fully custom Gated DeltaNet-2 linear-attention architecture
    (``model_type = "deepx-embedding"``) that is NOT registered in the
    ``transformers`` AutoModel class registry.  Its ``config.json`` contains
    no ``auto_map`` entry, so ``AutoModel.from_pretrained(...,
    trust_remote_code=True)`` — which is what the generic
    ``_SentenceTransformerBackend`` ultimately calls — raises:

        "The checkpoint ... has model type 'deepx-embedding' but
         Transformers does not recognize this architecture."

    The solution is to bypass ``SentenceTransformer`` entirely and load the
    model through its own package (``deepx_embed.DeepXEmbed``).

    Dimension guarantee
    ~~~~~~~~~~~~~~~~~~~
    DeepX natively produces 1536-dimensional vectors.  Matryoshka truncation
    to 1024 dimensions is requested at encode time via
    ``DeepXEmbed.encode(truncate_dim=1024)``.  The final
    ``EmbeddingModel._validate_dimension()`` call acts as a safety net.

    Batching
    ~~~~~~~~
    If the ``DeepXEmbed.encode()`` signature does not accept a ``batch_size``
    parameter, batching is performed manually inside this backend so that
    the project-level ``--batch-size`` CLI option is always respected:
    input ordering is preserved, and the model is loaded only once.

    CUDA
    ~~~~
    The existing ``_detect_device()`` helper determines the device.  The
    device is forwarded to ``DeepXEmbed.from_pretrained`` (when supported).
    """

    def __init__(self, config, *, use_fp16: bool = True) -> None:
        try:
            from deepx_embed import DeepXEmbed  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "deepx_embed is not installed.  "
                "Run: pip install git+https://github.com/dx-tech-ai/deepx-embed.git"
            ) from exc

        self._config = config
        device = _detect_device()
        self._device = device

        LOGGER.info("Embedding model: %s", config.model_id)
        LOGGER.info("Backend: DeepX")
        LOGGER.info("Device: %s", device)
        if device == "cuda":
            _log_gpu_info()
        else:
            LOGGER.info("CUDA unavailable; using CPU")

        # Build keyword arguments accepted by DeepXEmbed.from_pretrained.
        load_kwargs: dict = {}
        try:
            import inspect as _inspect  # noqa: PLC0415
            sig = _inspect.signature(DeepXEmbed.from_pretrained)
            if "device" in sig.parameters:
                load_kwargs["device"] = device
        except (ValueError, TypeError):
            pass

        self._model: DeepXEmbed = DeepXEmbed.from_pretrained(
            config.model_id, **load_kwargs,
        )

        # Detect whether encode() accepts batch_size natively.
        try:
            import inspect as _inspect  # noqa: PLC0415
            sig = _inspect.signature(self._model.encode)
            self._encode_supports_batch_size = "batch_size" in sig.parameters
        except (ValueError, TypeError):
            self._encode_supports_batch_size = False

        LOGGER.info(
            "DeepX Matryoshka truncation: 1536 → %d dimensions",
            config.dimension,
        )
        if config.normalize:
            LOGGER.info("DeepX normalization: enabled")

    @property
    def device(self) -> str:
        return self._device

    def embed_query(self, text: str, batch_size: int = 32) -> list[float]:
        vecs = self._encode_batch([text], batch_size)
        return vecs[0]

    def embed_documents(
        self, texts: Sequence[str], batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            return []
        return self._encode_batch(list(texts), batch_size)

    def _encode_batch(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Encode *texts* in batches, returning one float vector per input.

        Implements manual batching so the project-level ``--batch-size``
        option is respected even when the ``deepx_embed`` encode() API does
        not expose a ``batch_size`` parameter.
        """
        if not texts:
            return []

        # Build keyword arguments for encode().
        encode_kwargs: dict = {
            "truncate_dim": self._config.dimension,
        }
        # Only pass normalize if the API supports it.
        try:
            import inspect as _inspect  # noqa: PLC0415
            sig = _inspect.signature(self._model.encode)
            if "normalize" in sig.parameters:
                encode_kwargs["normalize"] = self._config.normalize
        except (ValueError, TypeError):
            pass

        if self._encode_supports_batch_size:
            # Let the library handle batching natively.
            encode_kwargs["batch_size"] = batch_size
            result = self._model.encode(texts, **encode_kwargs)
            return self._to_list(result)

        # Manual batching: split into chunks of batch_size.
        all_vecs: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            result = self._model.encode(chunk, **encode_kwargs)
            all_vecs.extend(self._to_list(result))
        return all_vecs

    @staticmethod
    def _to_list(result) -> list[list[float]]:
        """Convert numpy arrays or tensors returned by encode() to list[list[float]]."""
        try:
            # numpy array  (shape: N × D)
            if hasattr(result, "tolist"):
                converted = result.tolist()
                # encode() for a single text may return shape (D,) instead of (1, D).
                if converted and not isinstance(converted[0], list):
                    converted = [converted]
                return converted
        except Exception:  # noqa: BLE001
            pass
        # Fallback: assume list[list[float]] or list[float] already.
        if result and not isinstance(result[0], list):
            return [list(result)]
        return [list(row) for row in result]


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def _create_backend(config, *, use_fp16: bool = True) -> _EmbeddingBackend:
    """Instantiate the correct backend for the given model configuration.

    Dispatch table
    --------------
    ``bge``                → :class:`_BGEBackend`
    ``sentence_transformer``→ :class:`_SentenceTransformerBackend`
    ``jina``               → :class:`_JinaBackend`
    ``deepx``              → :class:`_DeepXBackend`
    """
    if config.backend == "bge":
        return _BGEBackend(config, use_fp16=use_fp16)
    if config.backend == "jina":
        return _JinaBackend(config, use_fp16=use_fp16)
    if config.backend == "sentence_transformer":
        return _SentenceTransformerBackend(config, use_fp16=use_fp16)
    if config.backend == "deepx":
        return _DeepXBackend(config, use_fp16=use_fp16)
    raise ValueError(
        f"Unknown backend: {config.backend!r} "
        f"(model: {config.model_id})"
    )


# ---------------------------------------------------------------------------
# Public EmbeddingModel class
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Unified embedding model interface.

    Automatically selects the correct backend (FlagEmbedding, Sentence
    Transformers, Jina, or DeepX) based on the model's entry in the
    :mod:`model_registry`.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier or short alias.  Falls back to the
        ``EMBEDDING_MODEL`` environment variable, then to ``BAAI/bge-m3``.
    use_fp16:
        Use 16-bit floating point for inference when a CUDA GPU is
        available.  Currently applies to the BGE backend only.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        use_fp16: bool = True,
    ) -> None:
        if __package__:
            from .model_registry import resolve_model_config  # noqa: PLC0415
        else:
            from model_registry import resolve_model_config  # noqa: PLC0415

        name = (
            model_name
            or os.environ.get("EMBEDDING_MODEL", "").strip()
            or None
        )
        self._config = resolve_model_config(name)
        LOGGER.info(
            "Resolved model: %s (%s, backend=%s)",
            self._config.alias,
            self._config.model_id,
            self._config.backend,
        )
        self._backend = _create_backend(self._config, use_fp16=use_fp16)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Hugging Face model identifier."""
        return self._config.model_id

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimension (always 1024)."""
        return self._config.dimension

    @property
    def device(self) -> str:
        """Compute device selected at model-load time: ``'cuda'`` or ``'cpu'``."""
        return self._backend.device

    @property
    def config(self):
        """The :class:`model_registry.ModelConfig` for this model."""
        return self._config

    # Expose the underlying model for introspection (e.g. model revision).
    @property
    def _model(self):
        return self._backend.raw_model

    # ------------------------------------------------------------------
    # Public API — query vs. document embedding
    # ------------------------------------------------------------------

    def embed_query(self, text: str, batch_size: int = 32) -> list[float]:
        """Embed a single query text.

        Applies the model-specific query encoding protocol (prefixes,
        task-specific LoRA adapters, etc.) automatically.

        Parameters
        ----------
        text:
            Natural-language query string.

        Returns
        -------
        list[float]
            A 1024-dimensional float vector.
        """
        vec = self._backend.embed_query(text, batch_size=batch_size)
        self._validate_dimension([vec])
        return vec

    def embed_documents(
        self, texts: Sequence[str], batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed a batch of document texts.

        Applies the model-specific document encoding protocol (prefixes,
        task-specific LoRA adapters, etc.) automatically.

        Parameters
        ----------
        texts:
            Document/passage strings to embed.
        batch_size:
            Number of texts per model forward pass.

        Returns
        -------
        list[list[float]]
            One 1024-dimensional float vector per input text, in the same
            order.
        """
        if not texts:
            return []
        vecs = self._backend.embed_documents(list(texts), batch_size=batch_size)
        self._validate_dimension(vecs)
        return vecs

    def embed_texts(
        self, texts: Sequence[str], batch_size: int = 32,
    ) -> list[list[float]]:
        """Backwards-compatible alias for :meth:`embed_documents`.

        Existing code that calls ``model.embed_texts(...)`` continues to work.
        New code should prefer :meth:`embed_documents` (for passages) and
        :meth:`embed_query` (for queries) to ensure the correct model-specific
        encoding protocol is applied.
        """
        return self.embed_documents(texts, batch_size=batch_size)

    # Convenience methods matching the interface contract.

    def get_dimension(self) -> int:
        """Return the embedding dimension (1024)."""
        return self._config.dimension

    def get_model_name(self) -> str:
        """Return the Hugging Face model identifier."""
        return self._config.model_id

    # ------------------------------------------------------------------
    # Dimension validation
    # ------------------------------------------------------------------

    def _validate_dimension(self, vectors: list[list[float]]) -> None:
        """Raise ``ValueError`` if any vector has the wrong dimension."""
        expected = self._config.dimension
        for i, vec in enumerate(vectors):
            if len(vec) != expected:
                raise ValueError(
                    f"Dimension mismatch: expected {expected}, got {len(vec)} "
                    f"from model {self._config.model_id} (vector index {i}). "
                    f"Check model configuration and Matryoshka truncation settings."
                )

    # Keep the static method for backwards compatibility.
    @staticmethod
    def _detect_device() -> str:
        """Return 'cuda' if a CUDA GPU is available, otherwise 'cpu'."""
        return _detect_device()
