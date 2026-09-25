"""Live retrieval evaluation script for the legal RAG chatbot.

Runs a built-in set of Vietnamese legal questions through the Retriever and
prints the ranked results so that a human can inspect retrieval quality.
Optionally accepts an ad-hoc query via ``--query``.

Run from the repository root::

    conda activate chatbot
    python scripts/test_retrieval.py
    python scripts/test_retrieval.py --query "Điều kiện hưởng hỗ trợ là gì?"
    python scripts/test_retrieval.py --top-k 10 --ef-search 80

Purpose
-------
This script is NOT an automated pass/fail test.  Its purpose is to let a
developer inspect whether the retrieved legal context is relevant for each
query.  Retrieval quality must be assessed by a human familiar with the
source documents (Nghị định 116/2020/NĐ-CP and related texts).

Ground truth
------------
Ground truth is expressed as a list of ``ExpectedSection`` entries, each
specifying an exact ``document_title``, ``article``, and (optionally) one or
more ``clauses``.  A retrieved chunk is a Ground Truth match only when ALL
specified levels (document, article, and – if non-empty – clause) agree.
This hierarchical structure prevents false positives such as the same article
number appearing in a different document.

Queries marked ``requires_verification`` have no confirmed expected section
and must be assessed manually.

Metrics
-------
Hit@K, Recall@K, and MRR are only calculated for queries that have at least
one verified ``ExpectedSection`` and ``requires_verification=False``.
No metrics are fabricated for queries without ground truth.

Recall@K counts each ExpectedSection once, even when multiple retrieved
chunks satisfy the same section.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from secrets import randbelow

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent

if __package__:
    from .environment import load_dotenv as _load_env_file, require_database_url
    from .evaluation.dataset import EVAL_QUERIES, EvalQuery, ExpectedSection
    from .evaluation.matching import _is_ground_truth_match, _is_legal_source, _matches_section
    from .evaluation.metrics import compute_hit_at_k, compute_mrr, compute_recall_at_k
    from .evaluation.runner import QueryEvalRecord, evaluate_query, run_evaluation
    from .evaluation.reporting import (
        _format_location,
        _format_section_label,
        _format_sections_for_display,
        _get_embedding_representation_description,
        _truncate,
        render_full_report,
    )
else:
    from environment import load_dotenv as _load_env_file, require_database_url
    from evaluation.dataset import EVAL_QUERIES, EvalQuery, ExpectedSection
    from evaluation.matching import _is_ground_truth_match, _is_legal_source, _matches_section
    from evaluation.metrics import compute_hit_at_k, compute_mrr, compute_recall_at_k
    from evaluation.runner import QueryEvalRecord, evaluate_query, run_evaluation
    from evaluation.reporting import (
        _format_location,
        _format_section_label,
        _format_sections_for_display,
        _get_embedding_representation_description,
        _truncate,
        render_full_report,
    )

LOGGER = logging.getLogger("test_retrieval")


# Helpers
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path) -> None:
    _load_env_file(path)


def _get_database_url() -> str:
    return require_database_url(
        ROOT / ".env",
        loader=_load_dotenv,
        error_message="DATABASE_URL is required.  Set it in the environment or in .env",
    )


def _get_database_name(url: str) -> str:
    try:
        parsed = urlparse(url)
        name = parsed.path.lstrip("/")
        return name if name else "PostgreSQL"
    except Exception:
        return "PostgreSQL"


def _get_model_revision(retriever: Any) -> str:
    try:
        model_obj = getattr(retriever, "_embedding_model", None)
        if model_obj is None:
            return "N/A"
        inner = getattr(model_obj, "_model", None)
        if inner is None:
            return "N/A"
        hf_model = getattr(inner, "model", None)
        if hf_model is not None and hasattr(hf_model, "config"):
            cfg = hf_model.config
            commit = getattr(cfg, "_commit_hash", None)
            if commit:
                return str(commit)
            version = getattr(cfg, "transformers_version", None)
            if version:
                return f"transformers-{version}"
    except Exception:
        pass
    return "N/A"


def _get_env_versions() -> tuple[str, str, str]:
    gpu_name = "N/A"
    cuda_ver = "N/A"
    torch_ver = "N/A"
    try:
        import torch  # noqa: PLC0415
        torch_ver = str(torch.__version__)
        if torch.cuda.is_available():
            gpu_name = str(torch.cuda.get_device_name(0))
            if hasattr(torch.version, "cuda") and torch.version.cuda:
                cuda_ver = str(torch.version.cuda)
    except Exception:
        pass
    return gpu_name, cuda_ver, torch_ver


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live retrieval evaluation for the legal RAG chatbot. "
            "Runs built-in legal questions and prints ranked results."
        )
    )
    parser.add_argument(
        "--query",
        default=None,
        metavar="TEXT",
        help="Run a single ad-hoc query instead of the built-in evaluation set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of results to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=40,
        metavar="N",
        help="HNSW ef_search value (default: 40).  Higher = better recall, slower.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Reconfigure stdout to UTF-8 so Vietnamese characters print correctly on
    # Windows terminals that default to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    start_time = datetime.now()
    cli_args_str = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "(default arguments)"

    # logs_dir = ROOT / "logs"
    # logs_dir.mkdir(parents=True, exist_ok=True)
    # timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    # log_path = logs_dir / f"retrieval_{timestamp}.txt"
    # if log_path.exists():
    #     timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S_%f")
    #     log_path = logs_dir / f"retrieval_{timestamp}.txt"

    start_time = datetime.now()
    cli_args_str = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "(default arguments)"

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpu_name, cuda_ver, torch_ver = _get_env_versions()

    try:
        database_url = _get_database_url()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    database_name = _get_database_name(database_url)

    try:
        if __package__:
            from .retrievers.hnsw import Retriever  # noqa: PLC0415
        else:
            from retrievers.hnsw import Retriever  # noqa: PLC0415
    except ImportError as exc:
        print(f"ERROR: Could not import retrieval module: {exc}")
        return 1

    print("\n============================================================")
    print("LEGAL RAG CHATBOT - RETRIEVAL EVALUATION")
    print("============================================================")
    print(f"Top-K:      {args.top_k}")
    print(f"ef_search:  {args.ef_search}")
    print("INFO: Initialising retriever...")

    retriever = Retriever(database_url=database_url, ef_search=args.ef_search)
    model_name = getattr(
        retriever,
        "model_name",
        getattr(retriever._embedding_model, "model_name", "BAAI/bge-m3"),
    )

    # Build a filesystem-safe model name.
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    safe_model_name = "_".join(safe_model_name.split())

    # Filename format:
    # <model_name>_<runtime>_<4-digit-random>.txt
    timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    random_suffix = f"{randbelow(10000):04d}"

    log_path = logs_dir / (
        f"{safe_model_name}_{timestamp}_{random_suffix}.txt"
    )

    # Extremely unlikely collision protection.
    while log_path.exists():
        random_suffix = f"{randbelow(10000):04d}"
        log_path = logs_dir / (
            f"{safe_model_name}_{timestamp}_{random_suffix}.txt"
        )

    embedding_dim = getattr(retriever._embedding_model, "embedding_dim", 1024)
    device = retriever.device
    model_revision = _get_model_revision(retriever)

    print(f"Embedding device: {device}")
    if gpu_name != "N/A":
        print(f"GPU: {gpu_name}")

    print("\n--- Database State ---")
    state = retriever.verify_database_state()
    print(f"Total chunks:       {state['total_chunks']}")
    print(f"Embedded chunks:    {state['embedded_chunks']}")
    print(f"Missing embeddings: {state['missing_embeddings']}")
    print(f"HNSW index exists:  {state['hnsw_index_exists']}")

    if args.query:
        queries = [
            EvalQuery(
                query=args.query,
                description="Ad-hoc query",
                requires_verification=False,
            )
        ]
    else:
        queries = EVAL_QUERIES

    print(f"\nRunning {len(queries)} evaluation queries...\n")

    query_records: list[QueryEvalRecord] = []
    metrics: dict[str, Any] = {}
    retrieval_errors: list[tuple[int, str, str]] = []
    uncaught_exception: str | None = None

    try:
        metrics = run_evaluation(queries, retriever, top_k=args.top_k)
        query_records = metrics.get("query_records", [])
        retrieval_errors = metrics.get("errors", [])
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            uncaught_exception = "Evaluation was interrupted by user (KeyboardInterrupt)."
        else:
            uncaught_exception = traceback.format_exc()
        print(f"\n[EXCEPTION] {exc}")
    finally:
        end_time = datetime.now()
        report = render_full_report(
            start_time=start_time,
            end_time=end_time,
            model_name=model_name,
            model_revision=model_revision,
            embedding_dim=embedding_dim,
            device=device,
            gpu_name=gpu_name,
            cuda_version=cuda_ver,
            pytorch_version=torch_ver,
            database_name=database_name,
            state=state,
            ef_search=args.ef_search,
            top_k=args.top_k,
            cli_args_str=cli_args_str,
            query_records=query_records,
            metrics=metrics,
            retrieval_errors=retrieval_errors,
            uncaught_exception=uncaught_exception,
        )
        log_path.write_text(report, encoding="utf-8")

    # Terminal summary display
    print("\n============================================================")
    print("AGGREGATE EVALUATION SUMMARY")
    print("============================================================")
    print(f"Total queries:                      {len(query_records)}")
    print(f"Queries with verified ground truth: {metrics.get('evaluated_queries', 0)}")
    if metrics.get("evaluated_queries", 0) > 0:
        print(f"Hit@3:     {metrics['Hit@3']:.3f}")
        print(f"Hit@5:     {metrics['Hit@5']:.3f}")
        print(f"Hit@10:    {metrics['Hit@10']:.3f}")
        print(f"Recall@3:  {metrics['Recall@3']:.3f}")
        print(f"Recall@5:  {metrics['Recall@5']:.3f}")
        print(f"Recall@10: {metrics['Recall@10']:.3f}")
        print(f"MRR:       {metrics['MRR']:.3f}")
    if metrics.get("Average_top1_similarity") is not None:
        print(f"Average top-1 similarity:          {metrics['Average_top1_similarity']:.6f}")
    if metrics.get("Average_best_relevant_similarity") is not None:
        print(f"Average best relevant similarity:   {metrics['Average_best_relevant_similarity']:.6f}")

    print("\n============================================================")
    print("QUERY RESULT OVERVIEW")
    print("============================================================")
    print(
        f"{'Query':<5} | {'Ground Truth':<30} | {'GT Rank':<7} | {'Top-3':<5} | {'Top-5':<5} | {'Top-10':<6} | {'Top-1 Score':<11}"
    )
    print(
        f"{'-'*5}-|-{'-'*30}-|-{'-'*7}-|-{'-'*5}-|-{'-'*5}-|-{'-'*6}-|-{'-'*11}"
    )
    for rec in query_records:
        q_label = f"Q{rec.index:02d}"
        if rec.query.requires_verification or not rec.query.expected_sections:
            gt_label = "Manual"
        else:
            first = _format_section_label(rec.query.expected_sections[0])
            extra = f" (+{len(rec.query.expected_sections)-1})" if len(rec.query.expected_sections) > 1 else ""
            gt_label = first[:27] + "…" + extra if len(first) > 30 else first + extra
        rank_label = str(rec.gt_rank) if rec.gt_rank is not None else "N/A"
        t3_label = "N/A" if rec.in_top3 is None else ("YES" if rec.in_top3 else "NO")
        t5_label = "N/A" if rec.in_top5 is None else ("YES" if rec.in_top5 else "NO")
        t10_label = "N/A" if rec.in_top10 is None else ("YES" if rec.in_top10 else "NO")
        score_label = f"{rec.best_score:.6f}" if rec.best_score is not None else "N/A"
        print(
            f"{q_label:<5} | {gt_label:<30} | {rank_label:<7} | {t3_label:<5} | {t5_label:<5} | {t10_label:<6} | {score_label:<11}"
        )

    print("\n============================================================")
    print("Evaluation completed.")
    print(f"Log saved to: logs/{log_path.name}")
    print("============================================================")

    if uncaught_exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
