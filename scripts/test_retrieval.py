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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from secrets import randbelow

# ---------------------------------------------------------------------------
# Make scripts/ importable
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ROOT = SCRIPTS_DIR.parent

LOGGER = logging.getLogger("test_retrieval")

# ---------------------------------------------------------------------------
# Ground Truth data model
# ---------------------------------------------------------------------------


@dataclass
class ExpectedSection:
    """One hierarchical ground truth entry: Document → Article → Clause → Point.

    Parameters
    ----------
    document_title:
        Exact document title as stored in the database (e.g.
        ``"Nghị định 116/2020/NĐ-CP"``).  Must be non-empty.
    article:
        Article identifier (e.g. ``"Điều 4"``).  Must be non-empty.
    clauses:
        Optional list of clause identifiers (e.g. ``["Khoản 1"]``).
        When empty, any clause (or no clause) within the specified article
        is acceptable — the section is treated as article-level ground truth.
        When non-empty, at least one of the listed clauses must match.
    points:
        Optional list of point identifiers (e.g. ``["Điểm a"]``).
        When empty, point matching is unconstrained — any point (or no
        point) within the expected clause/article is acceptable.
        When non-empty, at least one of the listed points must match.
        Note: if the retrieved chunk has no point (``None``) but
        ``points`` is non-empty, it does **not** match.

    Raises
    ------
    ValueError
        If ``document_title`` or ``article`` is blank.
    """

    document_title: str
    article: str
    clauses: list[str] = field(default_factory=list)
    points: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.document_title or not self.document_title.strip():
            raise ValueError(
                "ExpectedSection.document_title must be a non-empty string."
            )
        if not self.article or not self.article.strip():
            raise ValueError(
                "ExpectedSection.article must be a non-empty string."
            )


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------
# Each EvalQuery has an ``expected_sections`` list.  Each section specifies
# the exact Document → Article (→ Clause) path that must appear in the
# retrieved results for the query to be considered a hit.
#
# ``requires_verification=False`` means no confirmed Ground Truth exists;
# Hit@K / Recall@K / MRR are skipped for those queries.
# ---------------------------------------------------------------------------


@dataclass
class EvalQuery:
    query: str
    description: str
    expected_sections: list[ExpectedSection] = field(default_factory=list)
    requires_verification: bool = True


# Shorthand aliases for document titles used in the dataset.
_ND116 = "Nghị định 116/2020/NĐ-CP"
_ND60 = "Nghị định 60/2025/NĐ-CP"
_LGD2019 = "Luật Giáo dục 2019"


EVAL_QUERIES: list[EvalQuery] = [
    # 1. General conceptual — what does the decree regulate?
    # Verified: Điều 1 / Khoản 1 covers phạm vi điều chỉnh.
    EvalQuery(
        query="Nghị định 116/2020/NĐ-CP quy định về vấn đề gì?",
        description="General scope of the decree",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 1", ["Khoản 1"]),
        ],
        requires_verification=False,
    ),
    # 2. Subject / scope — who is covered?
    # Verified: Điều 1 / Khoản 2 addresses đối tượng áp dụng.
    EvalQuery(
        query="Đối tượng áp dụng của Nghị định 116 là ai?",
        description="Subject / scope of application",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 1", ["Khoản 2"]),
        ],
        requires_verification=False,
    ),
    # 3. Eligibility — general conditions to receive support.
    # Verified: Điều 1 (scope) and Điều 7 (conditions).  Each article has its
    # own relevant clause; they are recorded as separate sections.
    EvalQuery(
        query="Điều kiện để được hưởng chính sách hỗ trợ theo Nghị định 116 là gì?",
        description="Eligibility conditions for support policy",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 1", ["Khoản 1"]),
            ExpectedSection(_ND116, "Điều 7", ["Khoản 2"]),
        ],
        requires_verification=False,
    ),
    # 4. Scholarship / living allowance amount.
    # Requires verification — exact clause not yet confirmed against source.
    EvalQuery(
        query="Sinh viên sư phạm được hỗ trợ học phí và sinh hoạt phí là bao nhiêu?",
        description="Scholarship / living expense support amount",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 4", ["Khoản 1"]),
        ],
        requires_verification=False,
    ),
    # 5. Commitment to teaching — obligation after graduation.
    # Requires verification — exact clause not yet confirmed against source.
    EvalQuery(
        query="Sau khi tốt nghiệp, sinh viên sư phạm phải làm gì để không phải hoàn trả học phí?",
        description="Post-graduation teaching commitment obligation",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 6", ["Khoản 2"]),
        ],
        requires_verification=False,
    ),
    # 6. Repayment obligation — when must support be repaid?
    # Requires verification — exact clause not yet confirmed against source.
    EvalQuery(
        query="Sinh viên sư phạm phải hoàn trả học phí và chi phí sinh hoạt trong trường hợp nào?",
        description="Conditions triggering repayment of support",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 6", ["Khoản 1"]),
        ],
        requires_verification=False,
    ),
    # 7. Non-teaching job — consequences of working in another field.
    # Requires verification — exact clause not yet confirmed against source.
    EvalQuery(
        query="Nếu sinh viên sư phạm không làm nghề dạy học sau khi ra trường thì sao?",
        description="Natural language: working outside teaching profession",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 6", ["Khoản 1"]),
        ],
        requires_verification=False,
    ),
    # 8. Specific article reference — Điều 5.
    # Verified: any chunk within Điều 5 of ND116 is acceptable.
    EvalQuery(
        query="Điều 5 của Nghị định 116/2020 quy định điều gì?",
        description="Direct article reference (Điều 5)",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 5", []),
        ],
        requires_verification=False,
    ),
    # 9. Role of training institutions.
    # Requires verification — article/clause not confirmed.
    EvalQuery(
        query="Cơ sở đào tạo giáo viên có trách nhiệm gì trong việc thực hiện Nghị định 116?",
        description="Responsibilities of teacher training institutions",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 12", []),
        ],
        requires_verification=False,
    ),
    # 10. Role of local government / Sở GD&ĐT.
    # Requires verification — article/clause not confirmed.
    EvalQuery(
        query="Trách nhiệm của Sở Giáo dục và Đào tạo trong thực hiện chính sách hỗ trợ là gì?",
        description="Responsibilities of provincial education departments",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 11", []),
        ],
        requires_verification=False,
    ),
    # 11. Amendment decree — Nghị định 60/2025.
    # Requires verification — exact clause in ND60 not confirmed.
    EvalQuery(
        query="Nghị định 60/2025/NĐ-CP sửa đổi những nội dung gì của Nghị định 116/2020?",
        description="Amendment: what Nghị định 60/2025 changes",
        expected_sections=[
            ExpectedSection(_ND60, "Điều 1", []),
        ],
        requires_verification=False,
    ),
    # 12. Education Law context — definition from Luật Giáo dục 2019.
    # Requires verification — clause in LGD2019 not confirmed.
    EvalQuery(
        query="Theo Luật Giáo dục 2019, giáo viên được định nghĩa như thế nào?",
        description="Luật Giáo dục 2019 — definition of teacher",
        expected_sections=[
            ExpectedSection(_LGD2019, "Điều 66", ["Khoản 1"]),
        ],
        requires_verification=False,
    ),
    # 13. Natural language paraphrase — no legal jargon.
    # Requires verification — clause not confirmed.
    EvalQuery(
        query="Nhà nước có hỗ trợ tiền học cho giáo viên tương lai không?",
        description="Natural language: does the state fund future teachers?",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 1", ["Khoản 1"]),
            ExpectedSection(_ND116, "Điều 4", ["Khoản 1"]),
        ],
        requires_verification=False,
    ),
    # 14. Clause-level precision — specific clause.
    # Verified: Khoản 1 Điều 4 of ND116.
    EvalQuery(
        query="Khoản 1 Điều 4 Nghị định 116/2020 quy định gì?",
        description="Clause-level precision query (Khoản 1 Điều 4)",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 4", ["Khoản 1"]),
        ],
        requires_verification=False,
    ),
    # 15. Quota / enrolment target definition.
    # Requires verification — clause not confirmed.
    EvalQuery(
        query="Chỉ tiêu tuyển sinh sư phạm theo nhu cầu xã hội là gì?",
        description="Quota / enrolment target definition",
        expected_sections=[
            ExpectedSection(_ND116, "Điều 3", ["Khoản 1"]),
        ],
        requires_verification=False,
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _get_database_url() -> str:
    _load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is required.  Set it in the environment or in .env"
        )
    return url


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
# Legal source classification
# ---------------------------------------------------------------------------


def _is_legal_source(meta: dict[str, Any]) -> bool:
    """Return True if the chunk originates from a legal-document source.

    Classification is based on ``content_type``, which is the cleanest
    discriminator confirmed by the actual dataset:

    - ``content_type == 'legal_text'`` → legal document (core primary,
      core amendment, and reference/supporting legal documents such as
      Luật Giáo dục 2019 all carry this value).
    - ``content_type == 'qa'`` → QA/supporting text source.

    This intentionally treats ``reference/supporting`` chunks (e.g. Luật
    Giáo dục 2019) as legal sources, because they are formally-structured
    legal texts with full structural metadata (chapter, article, clause,
    point).  Only ``qa`` chunks are excluded from structural matching.

    Parameters
    ----------
    meta:
        The ``metadata`` dict from a ``RetrievalResult``.

    Returns
    -------
    bool
        ``True`` when the chunk is a legal-document source, ``False`` when
        it is a QA or other non-legal supporting text.
    """
    content_type = (meta.get("content_type") or "").strip().lower()
    return content_type == "legal_text"


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
# Hierarchical Ground Truth matching
# ---------------------------------------------------------------------------


def _matches_section(meta: dict[str, Any], section: ExpectedSection) -> bool:
    """Return True if a chunk's metadata satisfies one ``ExpectedSection``.

    Hierarchical matching rules — all must hold:

    1. The chunk must be from a **legal-document source** (``content_type ==
       'legal_text'``).  QA/non-legal chunks never satisfy a structural
       ground-truth section, regardless of textual content.
    2. ``meta["document_title"]`` must equal ``section.document_title``
       (case-insensitive, whitespace-stripped).
    3. ``meta["article"]`` must equal ``section.article``
       (case-insensitive, whitespace-stripped).
    4. If ``section.clauses`` is non-empty, ``meta["clause"]`` must match
       at least one entry (case-insensitive, whitespace-stripped).
       When ``section.clauses`` is empty the clause level is unconstrained.
    5. If ``section.points`` is non-empty, ``meta["point"]`` must match
       at least one entry (case-insensitive, whitespace-stripped).  A chunk
       with no point value (``None`` / empty) does **not** satisfy a
       non-empty ``points`` constraint.
       When ``section.points`` is empty the point level is unconstrained.

    A chunk from the *same article but a different document* does **not**
    satisfy the section.  A QA chunk whose ``text`` mentions an article or
    clause does **not** satisfy the section.
    """
    # --- Legal-source guard: QA chunks never match structural ground truth ---
    if not _is_legal_source(meta):
        return False

    # --- Document level ---
    chunk_doc = (meta.get("document_title") or "").strip().lower()
    expected_doc = section.document_title.strip().lower()
    if chunk_doc != expected_doc:
        return False

    # --- Article level ---
    chunk_art = (meta.get("article") or "").strip().lower()
    expected_art = section.article.strip().lower()
    if chunk_art != expected_art:
        return False

    # --- Clause level (only checked when clauses are explicitly specified) ---
    if section.clauses:
        chunk_clause = (meta.get("clause") or "").strip().lower()
        expected_clauses_lower = {c.strip().lower() for c in section.clauses if c.strip()}
        if chunk_clause not in expected_clauses_lower:
            return False

    # --- Point level (only checked when points are explicitly specified) ---
    if section.points:
        chunk_point = (meta.get("point") or "").strip().lower()
        expected_points_lower = {p.strip().lower() for p in section.points if p.strip()}
        # Empty chunk_point cannot satisfy a non-empty points constraint.
        if not chunk_point or chunk_point not in expected_points_lower:
            return False

    return True


def _is_ground_truth_match(
    meta: dict[str, Any], sections: list[ExpectedSection]
) -> bool:
    """Return True if the chunk satisfies ANY of the expected sections."""
    return any(_matches_section(meta, s) for s in sections)


# ---------------------------------------------------------------------------
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
# Per-query evaluation record
# ---------------------------------------------------------------------------


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
    query_records: list[QueryEvalRecord],
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
        from retrieval import Retriever  # noqa: PLC0415
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
