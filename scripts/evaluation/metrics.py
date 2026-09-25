"""Hit, recall, and reciprocal-rank metrics."""

from __future__ import annotations

from typing import Any

from .dataset import ExpectedSection
from .matching import _is_ground_truth_match, _matches_section

# Evaluation metrics (hierarchical)
# ---------------------------------------------------------------------------


def compute_hit_at_k(
    results: list[Any],
    sections: list[ExpectedSection],
    k: int,
) -> float:
    """Binary metric: 1.0 if ANY top-k result is a ground truth match, else 0.0.

    Parameters
    ----------
    results:
        List of ``RetrievalResult`` objects (must have a ``metadata`` dict).
    sections:
        List of ``ExpectedSection`` ground truth entries.
    k:
        Cut-off rank.
    """
    if not sections:
        return 0.0
    for r in results[:k]:
        if _is_ground_truth_match(r.metadata or {}, sections):
            return 1.0
    return 0.0


def compute_recall_at_k(
    results: list[Any],
    sections: list[ExpectedSection],
    k: int,
) -> float:
    """Recall@K: fraction of ExpectedSections satisfied by the top-k results.

    Each section is counted **once**, regardless of how many retrieved chunks
    satisfy it.

    Parameters
    ----------
    results:
        List of ``RetrievalResult`` objects.
    sections:
        List of ``ExpectedSection`` ground truth entries.
    k:
        Cut-off rank.
    """
    if not sections:
        return 0.0
    top_k = results[:k]
    found = sum(
        1
        for section in sections
        if any(_matches_section(r.metadata or {}, section) for r in top_k)
    )
    return found / len(sections)


def compute_mrr(
    results: list[Any],
    sections: list[ExpectedSection],
) -> float:
    """Mean Reciprocal Rank: 1/rank of the first ground truth match, 0 if none.

    Parameters
    ----------
    results:
        List of ``RetrievalResult`` objects.
    sections:
        List of ``ExpectedSection`` ground truth entries.
    """
    if not sections:
        return 0.0
    for rank, r in enumerate(results, start=1):
        if _is_ground_truth_match(r.metadata or {}, sections):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
