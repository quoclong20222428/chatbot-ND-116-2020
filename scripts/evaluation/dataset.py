"""Evaluation questions and hierarchical ground-truth data."""

from __future__ import annotations

from dataclasses import dataclass, field

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
