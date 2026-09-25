"""Run queries and compute benchmark aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dataset import EvalQuery
from .matching import _is_ground_truth_match
from .metrics import compute_hit_at_k, compute_mrr, compute_recall_at_k
from .reporting import _format_section_label

@dataclass
class QueryEvalRecord:
    index: int
    total: int
    query: EvalQuery
    results: list[Any]
    error: str | None = None
    best_score: float | None = None
    gt_rank: int | None = None
    in_top3: bool | None = None
    in_top5: bool | None = None
    in_top10: bool | None = None
    best_rel_score: float | None = None


def evaluate_query(
    query: EvalQuery,
    index: int,
    total: int,
    retriever: Any,
    top_k: int,
) -> QueryEvalRecord:
    error: str | None = None
    results: list[Any] = []
    try:
        results = retriever.retrieve(query.query, top_k=top_k)
    except Exception as exc:
        error = str(exc)

    best_score = results[0].score if results else None
    gt_rank: int | None = None
    best_rel_score: float | None = None
    in_top3: bool | None = None
    in_top5: bool | None = None
    in_top10: bool | None = None

    has_gt = bool(query.expected_sections) and not query.requires_verification
    if has_gt:
        for rank_idx, r in enumerate(results, start=1):
            if _is_ground_truth_match(r.metadata or {}, query.expected_sections):
                if gt_rank is None:
                    gt_rank = rank_idx
                    best_rel_score = r.score

        in_top3 = bool(gt_rank is not None and gt_rank <= 3)
        in_top5 = bool(gt_rank is not None and gt_rank <= 5)
        in_top10 = bool(gt_rank is not None and gt_rank <= 10)

    return QueryEvalRecord(
        index=index,
        total=total,
        query=query,
        results=results,
        error=error,
        best_score=best_score,
        gt_rank=gt_rank,
        in_top3=in_top3,
        in_top5=in_top5,
        in_top10=in_top10,
        best_rel_score=best_rel_score,
    )


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------


def run_evaluation(
    queries: list[EvalQuery],
    retriever: Any,
    top_k: int,
) -> dict[str, Any]:
    """Run evaluation for a list of queries, stream progress, and return metrics."""
    hit3_scores: list[float] = []
    hit5_scores: list[float] = []
    hit10_scores: list[float] = []
    recall3_scores: list[float] = []
    recall5_scores: list[float] = []
    recall10_scores: list[float] = []
    mrr_scores: list[float] = []
    evaluated_count = 0

    records: list[QueryEvalRecord] = []
    total = len(queries)

    for i, eq in enumerate(queries, start=1):
        rec = evaluate_query(eq, i, total, retriever, top_k)
        records.append(rec)

        # Stream progress to console
        print("-" * 60)
        print(f"[{i:02d}/{total:02d}] {eq.description}")
        print(f"Question: {eq.query}")
        if eq.expected_sections and not eq.requires_verification:
            gt_disp = "; ".join(_format_section_label(s) for s in eq.expected_sections)
            print(f"Expected: {gt_disp}")
        else:
            print("Expected: MANUAL VERIFICATION")

        if rec.error:
            print(f"  [ERROR] Query failed: {rec.error}")
            continue

        if not rec.results:
            print("  (no results returned)")
            continue

        # for rank_idx, r in enumerate(rec.results, start=1):
        #     meta = r.metadata or {}
        #     if not eq.requires_verification and eq.expected_sections:
        #         is_match = _is_ground_truth_match(meta, eq.expected_sections)
        #         match_str = "YES" if is_match else "NO"
        #     else:
        #         match_str = "N/A"
        #     if _is_legal_source(meta):
        #         loc = _format_location(meta)
        #         loc_str = f" ({loc})" if loc else ""
        #         print(f"  Rank {rank_idx:2d}: [{r.score:.6f}] {r.chunk_id}{loc_str} [Match: {match_str}]")
        #     else:
        #         print(f"  Rank {rank_idx:2d}: [{r.score:.6f}] {r.chunk_id} [Source: QA] [Match: {match_str}]")
        #     print(f"          Preview: {_truncate(r.text, 180)}")

        has_gt = bool(eq.expected_sections) and not eq.requires_verification
        if has_gt:
            evaluated_count += 1
            hit3_scores.append(compute_hit_at_k(rec.results, eq.expected_sections, 3))
            hit5_scores.append(compute_hit_at_k(rec.results, eq.expected_sections, 5))
            hit10_scores.append(compute_hit_at_k(rec.results, eq.expected_sections, 10))
            recall3_scores.append(compute_recall_at_k(rec.results, eq.expected_sections, 3))
            recall5_scores.append(compute_recall_at_k(rec.results, eq.expected_sections, 5))
            recall10_scores.append(compute_recall_at_k(rec.results, eq.expected_sections, 10))
            mrr_scores.append(compute_mrr(rec.results, eq.expected_sections))
            if rec.gt_rank is not None:
                print(f"  --> Ground truth found at Rank {rec.gt_rank}")
            else:
                print(f"  --> Ground truth NOT found in top-{top_k}")
        else:
            print("  --> Evaluation: MANUAL VERIFICATION")

    # Compute averages
    valid_top1 = [rec.best_score for rec in records if rec.best_score is not None]
    avg_top1 = (sum(valid_top1) / len(valid_top1)) if valid_top1 else None

    valid_rel = [rec.best_rel_score for rec in records if rec.best_rel_score is not None]
    avg_rel = (sum(valid_rel) / len(valid_rel)) if valid_rel else None

    metrics: dict[str, Any] = {
        "evaluated_queries": evaluated_count,
        "total_queries": total,
        "query_records": records,
        "Average_top1_similarity": avg_top1,
        "Average_best_relevant_similarity": avg_rel,
        "errors": [(r.index, r.query.query, r.error) for r in records if r.error],
    }
    if evaluated_count > 0:
        metrics["Hit@3"] = sum(hit3_scores) / evaluated_count
        metrics["Hit@5"] = sum(hit5_scores) / evaluated_count
        metrics["Hit@10"] = sum(hit10_scores) / evaluated_count
        metrics["Recall@3"] = sum(recall3_scores) / evaluated_count
        metrics["Recall@5"] = sum(recall5_scores) / evaluated_count
        metrics["Recall@10"] = sum(recall10_scores) / evaluated_count
        metrics["MRR"] = sum(mrr_scores) / evaluated_count

    return metrics


