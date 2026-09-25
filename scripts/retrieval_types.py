"""Shared contracts used by independent retrieval implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """One retrieved legal chunk, independent of retrieval algorithm."""

    chunk_id: str
    text: str
    score: float
    retrieval_method: str = "hnsw"
    score_type: str = "cosine_similarity"
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
