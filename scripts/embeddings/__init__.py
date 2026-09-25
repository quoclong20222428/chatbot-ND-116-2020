"""Multi-model embedding infrastructure."""

from .embedding import EMBEDDING_DIM, EmbeddingModel, build_embedding_text
from .model_registry import ModelConfig, resolve_model_config

__all__ = [
    "EMBEDDING_DIM",
    "EmbeddingModel",
    "ModelConfig",
    "build_embedding_text",
    "resolve_model_config",
]
