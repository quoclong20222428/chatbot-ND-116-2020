"""Formatting and text report generation for retrieval evaluations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .dataset import ExpectedSection
from .matching import _is_ground_truth_match, _is_legal_source

def _format_location(meta: dict[str, Any]) -> str:
    """Return a compact location string omitting NULL/empty fields."""
    parts = []
    if meta.get("chapter"):
        parts.append(str(meta["chapter"]))
    if meta.get("article"):
        parts.append(str(meta["article"]))
    if meta.get("clause"):
        parts.append(str(meta["clause"]))
    if meta.get("point"):
        parts.append(str(meta["point"]))
    return " / ".join(parts) if parts else ""


def _truncate(text: str, max_chars: int = 200) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars] + "…"


def _format_section_label(section: ExpectedSection) -> str:
    """Return a concise human-readable label for one ExpectedSection."""
    doc = section.document_title
    art = section.article
    parts = [doc, art]
    if section.clauses:
        parts.append(", ".join(section.clauses))
    if section.points:
        parts.append(", ".join(section.points))
    return " / ".join(parts)


def _format_sections_for_display(sections: list[ExpectedSection]) -> str:
    """Return a multi-line string listing all expected sections."""
    if not sections:
        return "MANUAL VERIFICATION"
    return "\n".join(f"  • {_format_section_label(s)}" for s in sections)


# ---------------------------------------------------------------------------
# Embedding representation documentation
# ---------------------------------------------------------------------------


def _get_embedding_representation_description() -> str:
    """Return a deterministic, human-readable description of the current
    metadata-aware document embedding representation.

    This helper is **documentation only**.  It does not construct embeddings,
    call ``build_embedding_text()``, or access the database.

    The actual embedding implementation lives in ``scripts/embedding.py``,
    function ``build_embedding_text()``.  This description reflects that
    contract so that evaluation reports remain self-contained and auditable.

    Document embedding (legal sources)
    -----------------------------------
    Each legal chunk is embedded as::

        [Document]
        <document_title>

        [Chapter]
        <chapter>            # when present and non-empty

        [Article]
        <article>            # when present and non-empty

        [Clause]
        <clause>             # when present and non-empty

        [Point]
        <point>              # when present and non-empty

        [Content]
        <original text>

    Only metadata fields that are non-null and non-empty are included.
    ``document_number`` is intentionally excluded from the representation.

    Document embedding (QA / non-legal sources)
    --------------------------------------------
    The original chunk text is embedded unchanged::

        <original text>

    No structural prefix is added.

    Query embedding
    ---------------
    The original natural-language query is embedded directly.  No metadata
    prefix, query rewriting, query classification, or query transformation
    of any kind is applied.  The query embedding flow is unchanged from the
    pre-metadata-aware baseline.
    """
    return (
        "Metadata-aware document embedding\n"
        "\n"
        "Legal sources:\n"
        "  [Document] <document_title>\n"
        "  [Chapter]  <chapter>            # when present\n"
        "  [Article]  <article>            # when present\n"
        "  [Clause]   <clause>             # when present\n"
        "  [Point]    <point>              # when present\n"
        "  [Content]  <original text>\n"
        "\n"
        "Non-legal / QA sources:\n"
        "  <original text>\n"
        "\n"
        "Query embedding:\n"
        "  Original natural-language query\n"
        "\n"
        "Query transformation:\n"
        "  None"
    )


# ---------------------------------------------------------------------------
# Report Generator (Full untruncated report for .txt file)
# ---------------------------------------------------------------------------


def render_full_report(
    *,
    start_time: datetime,
    end_time: datetime,
    model_name: str,
    model_revision: str,
    embedding_dim: int,
    device: str,
    gpu_name: str,
    cuda_version: str,
    pytorch_version: str,
    database_name: str,
    state: dict[str, Any],
    ef_search: int,
    top_k: int,
    cli_args_str: str,
    query_records: list[Any],
    metrics: dict[str, Any],
    retrieval_errors: list[tuple[int, str, str]],
    uncaught_exception: str | None = None,
) -> str:
    duration_sec = (end_time - start_time).total_seconds()
    duration_str = f"{duration_sec:.2f}s"
    total_chunks = state.get("total_chunks", 0)
    embedded_chunks = state.get("embedded_chunks", 0)
    missing_embeddings = state.get("missing_embeddings", 0)
    coverage_pct = (embedded_chunks / total_chunks * 100) if total_chunks > 0 else 0.0
    coverage_str = f"{coverage_pct:.1f}% ({embedded_chunks} / {total_chunks})"
    hnsw_exists = state.get("hnsw_index_exists", False)
    hnsw_status = "EXISTS" if hnsw_exists else "NOT FOUND"
    hnsw_index_name = "legal_chunks_embedding_hnsw_idx" if hnsw_exists else "N/A"

    lines: list[str] = []

    # ============================================================
    # 1. Thông tin tổng quan của lần chạy
    # ============================================================
    lines.append("=" * 60)
    lines.append("LEGAL RAG CHATBOT - RETRIEVAL EVALUATION")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Duration: {duration_str}")
    lines.append("")
    lines.append(f"Model: {model_name}")
    lines.append(f"Model revision/version (nếu lấy được): {model_revision}")
    lines.append(f"Embedding dimension: {embedding_dim}")
    lines.append(f"Embedding device: {device}")
    lines.append(f"GPU: {gpu_name}")
    lines.append(f"CUDA version (nếu lấy được): {cuda_version}")
    lines.append(f"PyTorch version: {pytorch_version}")
    lines.append("")
    lines.append(f"Database: {database_name}")
    lines.append(f"Total chunks: {total_chunks}")
    lines.append(f"Embedded chunks: {embedded_chunks}")
    lines.append(f"Missing embeddings: {missing_embeddings}")
    lines.append(f"Embedding coverage: {coverage_str}")
    lines.append("")
    lines.append(f"HNSW index: {hnsw_status}")
    lines.append(f"HNSW index name: {hnsw_index_name}")
    lines.append(f"ef_search: {ef_search}")
    lines.append(f"top_k: {top_k}")
    lines.append("")
    lines.append(f"Number of evaluation queries: {len(query_records)}")
    lines.append("")

    # ============================================================
    # 2. Evaluation configuration
    # ============================================================
    lines.append("=" * 60)
    lines.append("EVALUATION CONFIGURATION")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Top-K: {top_k}")
    lines.append(f"ef_search: {ef_search}")
    lines.append(f"Embedding model: {model_name}")
    lines.append(f"Embedding dimension: {embedding_dim}")
    lines.append(f"Device: {device}")
    lines.append(f"Database: {database_name}")
    lines.append(f"Evaluation query count: {len(query_records)}")
    lines.append(f"CLI arguments: {cli_args_str}")
    lines.append("")
    lines.append("Ground Truth matching mode: HIERARCHICAL (legal sources only)")
    lines.append("  Document → Article → Clause → Point")
    lines.append("  Structural matching applies only to chunks with content_type='legal_text'.")
    lines.append("  QA/non-legal chunks never satisfy a structural ground-truth section.")
    lines.append("  Each ExpectedSection is counted once for Recall@K regardless of")
    lines.append("  duplicate matches.  Unspecified levels (clause/point) are unconstrained.")
    lines.append("")
    lines.append(f"Embedding representation: Metadata-aware")
    lines.append("  Document embedding: structural metadata + content (legal sources)")
    lines.append("  Query embedding: original natural-language query (unchanged)")
    lines.append("  Query transformation: None")
    lines.append("")

    # ============================================================
    # 2b. Embedding Representation (dedicated section)
    # ============================================================
    lines.append("=" * 60)
    lines.append("EMBEDDING REPRESENTATION")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Document embedding:")
    lines.append("  Mode: Metadata-aware")
    lines.append("")
    lines.append("Legal metadata fields:")
    lines.append("  - document_title")
    lines.append("  - chapter")
    lines.append("  - article")
    lines.append("  - clause")
    lines.append("  - point")
    lines.append("")
    lines.append("Labels:")
    lines.append("  [Document]")
    lines.append("  [Chapter]")
    lines.append("  [Article]")
    lines.append("  [Clause]")
    lines.append("  [Point]")
    lines.append("  [Content]")
    lines.append("")
    lines.append("Document number:")
    lines.append("  Not included")
    lines.append("")
    lines.append("Null/empty metadata:")
    lines.append("  Field omitted")
    lines.append("")
    lines.append("QA/non-legal chunks:")
    lines.append("  Original text only")
    lines.append("")
    lines.append("Query embedding:")
    lines.append("  Original natural-language query")
    lines.append("")
    lines.append("Query transformation:")
    lines.append("  None")
    lines.append("")

    # ============================================================
    # 3. Query & Retrieval Results & Summary
    # ============================================================
    for rec in query_records:
        lines.append("#" * 64)
        lines.append(f"QUERY {rec.index:02d} / {rec.total:02d}")
        lines.append("#" * 64)
        lines.append("")
        lines.append("Question:")
        lines.append(rec.query.query)
        lines.append("")
        lines.append("Expected / Ground Truth (hierarchical):")
        if rec.query.requires_verification or not rec.query.expected_sections:
            lines.append("  MANUAL VERIFICATION")
        else:
            for s in rec.query.expected_sections:
                lines.append(f"  • {_format_section_label(s)}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("RETRIEVAL RESULTS")
        lines.append("-" * 60)
        lines.append("")

        if rec.error:
            lines.append(f"[ERROR] Query retrieval failed: {rec.error}")
            lines.append("")
        elif not rec.results:
            lines.append("(no results returned)")
            lines.append("")
        else:
            has_gt = bool(rec.query.expected_sections) and not rec.query.requires_verification
            for rank_idx, r in enumerate(rec.results, start=1):
                meta = r.metadata or {}

                if has_gt:
                    is_match = _is_ground_truth_match(meta, rec.query.expected_sections)
                    match_label = "YES" if is_match else "NO"
                else:
                    match_label = "N/A"

                lines.append(f"Rank {rank_idx}")
                lines.append(f"Score: {r.score:.6f}")
                lines.append(f"Chunk ID: {r.chunk_id or 'N/A'}")

                if _is_legal_source(meta):
                    # Legal document: show non-NULL structural fields only.
                    if meta.get("document_title"):
                        lines.append(f"Document: {meta['document_title']}")
                    if meta.get("chapter"):
                        lines.append(f"Chapter: {meta['chapter']}")
                    if meta.get("article"):
                        lines.append(f"Article: {meta['article']}")
                    if meta.get("clause"):
                        lines.append(f"Clause: {meta['clause']}")
                    if meta.get("point"):
                        lines.append(f"Point: {meta['point']}")
                    lines.append(f"Match: {match_label}")
                else:
                    # QA / non-legal source: simplified block.
                    lines.append(f"Source: QA")
                    lines.append(f"Match: {match_label}")

                lines.append("")
                lines.append("Text:")
                # FULL UNTRUNCATED CHUNK TEXT
                lines.append(r.text)
                lines.append("")
                lines.append("-" * 60)
                lines.append("")

        # Query Summary
        lines.append("-" * 60)
        lines.append("QUERY SUMMARY")
        lines.append("-" * 60)
        lines.append("")
        if rec.results:
            lines.append("Top result:")
            lines.append("Rank 1")
            lines.append("")
            lines.append("Best similarity:")
            lines.append(f"{rec.best_score:.6f}")
            lines.append("")
        else:
            lines.append("Top result:")
            lines.append("None")
            lines.append("")
            lines.append("Best similarity:")
            lines.append("N/A")
            lines.append("")

        has_gt = bool(rec.query.expected_sections) and not rec.query.requires_verification
        if not has_gt:
            lines.append("Ground truth found:")
            lines.append("N/A")
            lines.append("")
            lines.append("Ground truth rank:")
            lines.append("N/A")
            lines.append("")
            lines.append("Found in Top-3:")
            lines.append("N/A")
            lines.append("")
            lines.append("Found in Top-5:")
            lines.append("N/A")
            lines.append("")
            lines.append("Found in Top-10:")
            lines.append("N/A")
            lines.append("")
            lines.append("Evaluation:")
            lines.append("MANUAL VERIFICATION")
        else:
            if rec.gt_rank is not None:
                lines.append("Ground truth found:")
                lines.append("YES")
                lines.append("")
                lines.append("Ground truth rank:")
                lines.append(str(rec.gt_rank))
                lines.append("")
                lines.append("Found in Top-3:")
                lines.append("YES" if rec.in_top3 else "NO")
                lines.append("")
                lines.append("Found in Top-5:")
                lines.append("YES" if rec.in_top5 else "NO")
                lines.append("")
                lines.append("Found in Top-10:")
                lines.append("YES" if rec.in_top10 else "NO")
            else:
                lines.append("Ground truth found:")
                lines.append("NO")
                lines.append("")
                lines.append("Ground truth rank:")
                lines.append("N/A")
                lines.append("")
                lines.append("Found in Top-3:")
                lines.append("NO")
                lines.append("")
                lines.append("Found in Top-5:")
                lines.append("NO")
                lines.append("")
                lines.append("Found in Top-10:")
                lines.append("NO")

        lines.append("")

    # ============================================================
    # 4. Aggregate Evaluation Summary
    # ============================================================
    total_q = len(query_records)
    verified_q = metrics.get("evaluated_queries", 0)
    manual_q = total_q - verified_q

    lines.append("=" * 60)
    lines.append("AGGREGATE EVALUATION SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total queries: {total_q}")
    lines.append(f"Queries with verified ground truth: {verified_q}")
    lines.append(f"Queries requiring manual verification: {manual_q}")
    lines.append("")

    if verified_q > 0:
        lines.append(f"Hit@3:  {metrics['Hit@3']:.3f}")
        lines.append(f"Hit@5:  {metrics['Hit@5']:.3f}")
        lines.append(f"Hit@10: {metrics['Hit@10']:.3f}")
        lines.append("")
        lines.append(f"Recall@3:  {metrics['Recall@3']:.3f}")
        lines.append(f"Recall@5:  {metrics['Recall@5']:.3f}")
        lines.append(f"Recall@10: {metrics['Recall@10']:.3f}")
        lines.append("")
        lines.append(f"MRR: {metrics['MRR']:.3f}")
        lines.append("")
    else:
        lines.append("Hit@3: N/A - insufficient verified ground truth")
        lines.append("Hit@5: N/A - insufficient verified ground truth")
        lines.append("Hit@10: N/A - insufficient verified ground truth")
        lines.append("")
        lines.append("Recall@3: N/A - insufficient verified ground truth")
        lines.append("Recall@5: N/A - insufficient verified ground truth")
        lines.append("Recall@10: N/A - insufficient verified ground truth")
        lines.append("")
        lines.append("MRR: N/A - insufficient verified ground truth")
        lines.append("")

    top1_avg = metrics.get("Average_top1_similarity")
    if top1_avg is not None:
        lines.append(f"Average top-1 similarity: {top1_avg:.6f}")
    else:
        lines.append("Average top-1 similarity: N/A")

    rel_avg = metrics.get("Average_best_relevant_similarity")
    if rel_avg is not None:
        lines.append(f"Average best relevant similarity: {rel_avg:.6f}")
    else:
        lines.append("Average best relevant similarity: N/A - insufficient verified ground truth")

    lines.append("")

    # ============================================================
    # 5. Query Result Overview table
    # ============================================================
    lines.append("=" * 60)
    lines.append("QUERY RESULT OVERVIEW")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"{'Query':<5} | {'Ground Truth':<30} | {'GT Rank':<7} | {'Top-3':<5} | {'Top-5':<5} | {'Top-10':<6} | {'Top-1 Score':<11}"
    )
    lines.append(
        f"{'-'*5}-|-{'-'*30}-|-{'-'*7}-|-{'-'*5}-|-{'-'*5}-|-{'-'*6}-|-{'-'*11}"
    )

    for rec in query_records:
        q_label = f"Q{rec.index:02d}"
        if rec.query.requires_verification or not rec.query.expected_sections:
            gt_label = "Manual"
        else:
            # Abbreviated: first section only, max 30 chars
            first = _format_section_label(rec.query.expected_sections[0])
            extra = f" (+{len(rec.query.expected_sections)-1})" if len(rec.query.expected_sections) > 1 else ""
            gt_label = first[:27] + "…" + extra if len(first) > 30 else first + extra
        rank_label = str(rec.gt_rank) if rec.gt_rank is not None else "N/A"
        t3_label = "N/A" if rec.in_top3 is None else ("YES" if rec.in_top3 else "NO")
        t5_label = "N/A" if rec.in_top5 is None else ("YES" if rec.in_top5 else "NO")
        t10_label = "N/A" if rec.in_top10 is None else ("YES" if rec.in_top10 else "NO")
        score_label = f"{rec.best_score:.6f}" if rec.best_score is not None else "N/A"

        lines.append(
            f"{q_label:<5} | {gt_label:<30} | {rank_label:<7} | {t3_label:<5} | {t5_label:<5} | {t10_label:<6} | {score_label:<11}"
        )

    lines.append("")

    # ============================================================
    # 6. Retrieval Diagnostics
    # ============================================================
    lines.append("=" * 60)
    lines.append("RETRIEVAL DIAGNOSTICS")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Database connectivity:")
    lines.append("OK" if total_chunks > 0 else "UNKNOWN")
    lines.append("")
    lines.append("Embedding coverage:")
    lines.append(f"{embedded_chunks} / {total_chunks}")
    lines.append("")
    lines.append("Missing embeddings:")
    lines.append(str(missing_embeddings))
    lines.append("")
    lines.append("Embedding dimension:")
    lines.append(str(embedding_dim))
    lines.append("")
    lines.append("HNSW index:")
    lines.append(hnsw_status)
    lines.append("")
    lines.append("Embedding device:")
    lines.append(device.upper())
    lines.append("")
    lines.append("GPU:")
    lines.append(gpu_name)
    lines.append("")
    lines.append("Retrieval errors:")
    lines.append(str(len(retrieval_errors)))
    if retrieval_errors:
        lines.append("")
        lines.append("Error details:")
        for q_idx, q_txt, q_err in retrieval_errors:
            lines.append(f"- Query {q_idx:02d} ({q_txt[:40]}...): {q_err}")

    if uncaught_exception:
        lines.append("")
        lines.append("=" * 60)
        lines.append("UNCAUGHT EXCEPTION")
        lines.append("=" * 60)
        lines.append("")
        lines.append(uncaught_exception)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
