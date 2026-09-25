"""Ground-truth matching rules for retrieval evaluation."""

from __future__ import annotations

from typing import Any

from .dataset import ExpectedSection

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
