"""Centralized embedding model configuration registry.

Every supported model is defined here with its backend type, dimension,
encoding protocol, and database column mapping.  Adding a new model to the
project requires editing **only** this file and (if the model's inference
protocol is not covered by an existing backend) adding a backend class in
``embedding.py``.

Other modules import from this file to resolve model identifiers:

- ``embedding.py``       — backend selection and protocol parameters
- ``index_embeddings.py`` — column name for writing embeddings
- ``retrieval.py``        — column name for reading embeddings
- ``test_retrieval.py``   — model metadata for evaluation reports
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Model configuration data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Complete specification for a single supported embedding model.

    Attributes
    ----------
    model_id : str
        Hugging Face model identifier (e.g. ``"BAAI/bge-m3"``).
    alias : str
        Short human-friendly alias used in CLI, ``.env``, and scripts
        (e.g. ``"bge-m3"``).
    backend : str
        Backend identifier: ``"bge"`` | ``"sentence_transformer"`` | ``"jina"``.
    dimension : int
        Required output embedding dimension (must be 1024).
    max_seq_length : int
        Maximum input sequence length in tokens.
    query_prefix : str
        Text prepended to queries before encoding (e.g. ``"query: "`` for E5).
    document_prefix : str
        Text prepended to documents before encoding (e.g. ``"passage: "``).
    trust_remote_code : bool
        Whether ``trust_remote_code=True`` is needed when loading the model.
    query_prompt_name : str | None
        ``prompt_name`` / ``task`` parameter for query encoding
        (e.g. Jina LoRA adapter ``"retrieval.query"``).
    document_prompt_name : str | None
        ``prompt_name`` / ``task`` parameter for document encoding
        (e.g. Jina LoRA adapter ``"retrieval.passage"``).
    normalize : bool
        Whether to L2-normalise output embeddings.
    embedding_column : str
        PostgreSQL column name for this model's embeddings in ``legal_chunks``.
        Derived automatically from ``alias`` if not explicitly set.
    hnsw_index_name : str
        Name of the HNSW index for this model's embedding column.
        Derived automatically from ``alias`` if not explicitly set.
    """

    model_id: str
    alias: str
    backend: str  # "bge", "sentence_transformer", "jina", "deepx"
    dimension: int = 1024
    max_seq_length: int = 512
    query_prefix: str = ""
    document_prefix: str = ""
    trust_remote_code: bool = False
    query_prompt_name: str | None = None
    document_prompt_name: str | None = None
    normalize: bool = True
    embedding_column: str = ""
    hnsw_index_name: str = ""

    def __post_init__(self) -> None:
        # Derive database column and index names from alias.
        if not self.embedding_column:
            col = (
                "embedding"
                if self.alias == "bge-m3"
                else f"embedding_{self.alias.replace('-', '_')}"
            )
            object.__setattr__(self, "embedding_column", col)
        if not self.hnsw_index_name:
            idx = (
                "legal_chunks_embedding_hnsw_idx"
                if self.alias == "bge-m3"
                else f"legal_chunks_embedding_{self.alias.replace('-', '_')}_hnsw_idx"
            )
            object.__setattr__(self, "hnsw_index_name", idx)


# ---------------------------------------------------------------------------
# Supported models registry
# ---------------------------------------------------------------------------

_MODELS: list[ModelConfig] = [
    # 1. BAAI/bge-m3 — FlagEmbedding (BGEM3FlagModel), native 1024d.
    #    No query/document prefix.  Multilingual, up to 8192 tokens.
    ModelConfig(
        model_id="BAAI/bge-m3",
        alias="bge-m3",
        backend="bge",
        dimension=1024,
        max_seq_length=8192,
    ),
    # 2. darklethelong/vnlegal-lal — SentenceTransformer (decoder-only),
    #    native 1024d, last-token pooling, fine-tuned from vietlegal-harrier.
    ModelConfig(
        model_id="darklethelong/vnlegal-lal",
        alias="vnlegal-lal",
        backend="sentence_transformer",
        dimension=1024,
        max_seq_length=2048,
    ),
    # 3. mainguyen9/vietlegal-harrier-0.6b — SentenceTransformer,
    #    native 1024d, based on microsoft/harrier-oss-v1-0.6b.
    ModelConfig(
        model_id="mainguyen9/vietlegal-harrier-0.6b",
        alias="vietlegal-harrier",
        backend="sentence_transformer",
        dimension=1024,
        max_seq_length=512,
    ),
    # 4. mainguyen9/vietlegal-e5 — SentenceTransformer + E5 prefixes.
    #    Based on intfloat/multilingual-e5-large, requires "query: " /
    #    "passage: " prefixes.  Native 1024d, Matryoshka support.
    ModelConfig(
        model_id="mainguyen9/vietlegal-e5",
        alias="vietlegal-e5",
        backend="sentence_transformer",
        dimension=1024,
        max_seq_length=512,
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    # 5. jinaai/jina-embeddings-v3 — requires trust_remote_code,
    #    uses task-specific LoRA adapters for query / passage encoding.
    #    Native 1024d, Matryoshka support, up to 8192 tokens.
    ModelConfig(
        model_id="jinaai/jina-embeddings-v3-hf",
        alias="jina-v3",
        backend="jina",
        dimension=1024,
        max_seq_length=8192,
        trust_remote_code=True,
        query_prompt_name="retrieval.query",
        document_prompt_name="retrieval.passage",
    ),
    # 6. dxtech-asia/deepx-embedding-v1 — dedicated DeepX backend via
    #    the deepx_embed package (DeepXEmbed.from_pretrained).  The model
    #    uses a custom Gated DeltaNet-2 architecture not registered in the
    #    transformers AutoModel registry, so the generic sentence_transformer
    #    backend cannot be used.  Matryoshka truncation to 1024d is applied
    #    at encode time via DeepXEmbed.encode(truncate_dim=1024).
    #    Requires: pip install git+https://github.com/dx-tech-ai/deepx-embed.git
    ModelConfig(
        model_id="dxtech-asia/deepx-embedding-v1",
        alias="deepx",
        backend="deepx",
        dimension=1024,
        max_seq_length=8192,
        trust_remote_code=True,
    ),
]

# Build lookup dicts: alias → config AND model_id → config.
MODEL_CONFIGS: dict[str, ModelConfig] = {}
for _cfg in _MODELS:
    MODEL_CONFIGS[_cfg.alias] = _cfg
    MODEL_CONFIGS[_cfg.model_id] = _cfg

# Sorted unique lists for display / validation messages.
SUPPORTED_ALIASES: list[str] = sorted({c.alias for c in _MODELS})
SUPPORTED_MODEL_IDS: list[str] = sorted({c.model_id for c in _MODELS})

# Default model (backwards compatible).
DEFAULT_ALIAS = "bge-m3"
DEFAULT_MODEL_ID = "BAAI/bge-m3"

# Project-wide embedding dimension requirement.
EMBEDDING_DIM = 1024


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_model_config(name: str | None = None) -> ModelConfig:
    """Resolve a model name (alias or Hugging Face ID) to its ``ModelConfig``.

    Parameters
    ----------
    name:
        Model alias (e.g. ``"bge-m3"``) or Hugging Face identifier
        (e.g. ``"BAAI/bge-m3"``).  If ``None`` or empty, the default
        model (``BAAI/bge-m3``) is returned.

    Returns
    -------
    ModelConfig

    Raises
    ------
    ValueError
        If the model is not in the supported registry.  The error message
        lists all supported models.
    """
    if not name or not name.strip():
        return MODEL_CONFIGS[DEFAULT_ALIAS]

    key = name.strip()
    if key in MODEL_CONFIGS:
        return MODEL_CONFIGS[key]

    # Case-insensitive fallback.
    key_lower = key.lower()
    for k, cfg in MODEL_CONFIGS.items():
        if k.lower() == key_lower:
            return cfg

    supported = "\n".join(
        f"  \u2022 {cfg.alias:20s}  \u2192  {cfg.model_id}" for cfg in _MODELS
    )
    raise ValueError(
        f"Unsupported embedding model: {name!r}\n\n"
        f"Supported models:\n{supported}\n\n"
        f"Set EMBEDDING_MODEL in .env to one of the above identifiers or aliases."
    )
