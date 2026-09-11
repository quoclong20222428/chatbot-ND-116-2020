"""Create deterministic, structure-aware chunks from the legal Markdown corpus."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_DIR = ROOT / "data" / "markdown"
QA_PATH = ROOT / "data" / "raw" / "qa" / "hoi-dap-nghi-dinh-116-2020-nd-cp.md"
OUTPUT_PATH = ROOT / "data" / "processed" / "legal_chunks.jsonl"

SOURCE_CONFIG = {
    "116_2020_ND-CP.md": ("116_2020_ND-CP", "Nghị định 116/2020/NĐ-CP", "core", "116/2020/NĐ-CP"),
    "60_2025_ND-CP.md": ("60_2025_ND-CP", "Nghị định 60/2025/NĐ-CP", "core", "60/2025/NĐ-CP"),
    "43_2019_QH14_Luat_Giao_duc_2019.md": ("LUAT_GIAO_DUC_2019", "Luật Giáo dục 2019", "reference", "43/2019/QH14"),
}
QA_CONFIG = ("hoi-dap-nghi-dinh-116-2020-nd-cp", "Hỏi đáp Nghị định 116/2020/NĐ-CP", "qa")
KNOWN_DOCUMENTS = {config[0] for config in SOURCE_CONFIG.values()} | {QA_CONFIG[0]}

AUTHORITY_CONFIG = {
    "116_2020_ND-CP": ("primary", "primary_legal_source", 100, "primary"),
    "60_2025_ND-CP": ("amendment", "amending_legal_source", 90, "conditional_amendment"),
    "LUAT_GIAO_DUC_2019": ("supporting", "supporting_legal_source", 60, "on_demand_support"),
    "hoi-dap-nghi-dinh-116-2020-nd-cp": ("qa", "reference_qa", 30, "reference_then_validate"),
}

CHAPTER_RE = re.compile(r"^\s*(?:#+\s*)?(?:\*\*|__)?Chương\s+([IVXLCDM]+)\b", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*(?:#+\s*)?(?:\*\*|__)?Mục\s+([IVXLCDM]+)\b", re.IGNORECASE)
SUBSECTION_RE = re.compile(r"^\s*(?:#+\s*)?(?:\*\*|__)?Tiểu mục\s+([IVXLCDM]+)\b", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^\s*(?:#+\s*)?(?:\*\*|__)?[“\"]?Điều\s+(\d+)\b", re.IGNORECASE)
CLAUSE_RE = re.compile(r"^\s*(\d+)\\?\.\s+\S")
POINT_RE = re.compile(r"^\s*([a-zđ])\)\s+\S", re.IGNORECASE)
FORM_RE = re.compile(r"Mẫu\s+số\s+([0-9]+)", re.IGNORECASE)


@dataclass
class Context:
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    article: str | None = None


def clean_marker(value: str) -> str:
    return re.sub(r"[*_#\"“”]", "", value).strip()


def document_metadata(path: Path) -> tuple[str, str, str, str | None]:
    if path.name in SOURCE_CONFIG:
        return SOURCE_CONFIG[path.name]
    raise ValueError(f"Unsupported legal source: {path}")


def detect_references(text: str, current_document_id: str | None = None) -> list[dict[str, str | None]]:
    patterns = [
        re.compile(r"(?P<provision>(?:Điểm\s+[a-zđ]+\s+)?(?:khoản\s+\d+\s+)?Điều\s+\d+)\s+(?P<law>Luật\s+Giáo\s+dục(?:\s+2019)?)", re.IGNORECASE),
        re.compile(r"(?P<provision>(?:Điểm\s+[a-zđ]+\s+)?(?:khoản\s+\d+\s+)?Điều\s+\d+)\s+(?P<law>Nghị\s+định(?:\s+số)?\s+\d{1,4}/\d{4}/NĐ-CP)", re.IGNORECASE),
        re.compile(r"(?P<provision>(?:Điểm\s+[a-zđ]+\s+)?(?:khoản\s+\d+\s+)?Điều\s+\d+)\s+(?P<law>Nghị\s+định\s+này)", re.IGNORECASE),
        re.compile(r"(?P<provision>(?:Điểm\s+[a-zđ]+\s+)?(?:khoản\s+\d+\s+)?Điều\s+(?:\d+|này))(?:\s+của\s+Luật\s+này|\s+Luật\s+này|\s+Điều\s+này)", re.IGNORECASE),
    ]
    references: list[dict[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            law = match.groupdict().get("law")
            normalized = (law or "").lower()
            if "giáo dục" in normalized:
                resolved = "LUAT_GIAO_DUC_2019"
            elif "116/2020" in normalized:
                resolved = "116_2020_ND-CP"
            elif "60/2025" in normalized:
                resolved = "60_2025_ND-CP"
            elif "nghị định này" in normalized:
                resolved = None
            elif current_document_id == "LUAT_GIAO_DUC_2019":
                resolved = current_document_id
            else:
                resolved = None
            key = (match.group("provision"), resolved)
            if key not in seen:
                seen.add(key)
                references.append({
                    "raw_reference": match.group(0),
                    "resolved_document_id": resolved,
                    "provision": match.group("provision"),
                })
    return references


def slug(value: str) -> str:
    value = value.lower().replace("đ", "d")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value


def make_chunk(base: dict, suffix: str, text: str, **metadata) -> dict:
    role, authority_level, retrieval_priority, retrieval_behavior = AUTHORITY_CONFIG[base["document_id"]]
    relations = []
    if base["document_id"] == "60_2025_ND-CP":
        relations.append({
            "relation_type": "amends",
            "target_document_id": "116_2020_ND-CP",
            "target_provision": None,
            "resolution_status": "document_resolved_provision_unresolved",
        })
    record = {
        "chunk_id": f"{base['document_id'].replace('_', '-')}-{suffix}",
        "document_id": base["document_id"],
        "document_title": base["document_title"],
        "document_number": base.get("document_number"),
        "source_type": base["source_type"],
        "document_role": role,
        "authority_level": authority_level,
        "retrieval_priority": retrieval_priority,
        "retrieval_behavior": retrieval_behavior,
        "document_relations": relations,
        "chapter": metadata.get("chapter"),
        "section": metadata.get("section"),
        "subsection": metadata.get("subsection"),
        "article": metadata.get("article"),
        "clause": metadata.get("clause"),
        "point": metadata.get("point"),
        "form_number": metadata.get("form_number"),
        "content_type": metadata.get("content_type", "legal_text"),
        "text": text,
        "references": detect_references(text, base["document_id"]),
    }
    return record


def ensure_unique_chunk_ids(chunks: list[dict]) -> None:
    counts: dict[str, int] = {}
    for chunk in chunks:
        base_id = chunk["chunk_id"]
        occurrence = counts.get(base_id, 0) + 1
        counts[base_id] = occurrence
        if occurrence > 1:
            chunk["chunk_id"] = f"{base_id}-{occurrence}"


def add_resolved_chunk_ids(chunks: list[dict]) -> None:
    article_chunks: dict[tuple[str, str], str] = {}
    provision_chunks: dict[tuple[str, str, str | None, str | None], list[str]] = {}
    provision_re = re.compile(
        r"(?:(?:Điểm\s+([a-zđ]+)\s+)?(?:khoản\s+(\d+)\s+)?Điều\s+(\d+))",
        re.IGNORECASE,
    )
    for chunk in chunks:
        document_id = chunk["document_id"]
        article = chunk.get("article")
        if not article:
            continue
        if not chunk.get("clause") and not chunk.get("point"):
            article_chunks[(document_id, article)] = chunk["chunk_id"]
        key = (document_id, article, chunk.get("clause"), chunk.get("point"))
        provision_chunks.setdefault(key, []).append(chunk["chunk_id"])

    for chunk in chunks:
        for reference in chunk.get("references", []):
            document_id = reference.get("resolved_document_id")
            provision = reference.get("provision", "")
            if not document_id or not provision:
                continue
            match = provision_re.search(provision)
            if not match or match.group(3) is None:
                continue
            article = f"Điều {match.group(3)}"
            point = f"Điểm {match.group(1)}" if match.group(1) else None
            clause = f"Khoản {match.group(2)}" if match.group(2) else None
            candidates = provision_chunks.get((document_id, article, clause, point), [])
            if len(candidates) == 1:
                reference["resolved_chunk_id"] = candidates[0]
            elif not clause and not point and (document_id, article) in article_chunks:
                reference["resolved_chunk_id"] = article_chunks[(document_id, article)]
            if chunk["document_id"] == "60_2025_ND-CP" and document_id == "116_2020_ND-CP":
                relation = chunk["document_relations"][0]
                relation.update({
                    "target_provision": provision,
                    "target_chunk_id": reference.get("resolved_chunk_id"),
                    "resolution_status": "resolved",
                })


def split_structured(lines: list[str], base: dict) -> list[dict]:
    context = Context()
    articles: list[tuple[int, int, str, Context]] = []
    for index, line in enumerate(lines):
        chapter = CHAPTER_RE.match(line)
        section = SECTION_RE.match(line)
        article = ARTICLE_RE.match(line)
        if chapter:
            context.chapter = f"Chương {chapter.group(1).upper()}"
            context.section = None
            context.subsection = None
        elif section:
            context.section = clean_marker(line)
            context.subsection = None
        elif SUBSECTION_RE.match(line):
            context.subsection = clean_marker(line)
        elif article:
            context.article = f"Điều {article.group(1)}"
            articles.append((index, 0, context.article, Context(context.chapter, context.section, context.subsection, context.article)))
    if not articles:
        text = "\n".join(lines).strip("\n")
        return [make_chunk(base, "document", text)] if text.strip() else []

    structural_positions = [
        index for index, line in enumerate(lines)
        if CHAPTER_RE.match(line) or SECTION_RE.match(line) or SUBSECTION_RE.match(line)
    ]
    adjusted_articles: list[tuple[int, str, Context]] = []
    for position, (article_index, _, article, article_context) in enumerate(articles):
        if position == 0:
            start = 0
        else:
            headings = [index for index in structural_positions if articles[position - 1][0] < index < article_index]
            start = headings[0] if headings else article_index
        adjusted_articles.append((start, article, article_context))

    chunks: list[dict] = []
    for position, (start, article, article_context) in enumerate(adjusted_articles):
        end = adjusted_articles[position + 1][0] if position + 1 < len(adjusted_articles) else len(lines)
        block = lines[start:end]
        clause_positions = [i for i, line in enumerate(block) if CLAUSE_RE.match(line)]
        if not clause_positions:
            text = "\n".join(block).strip("\n")
            chunks.append(make_chunk(base, slug(article), text, **vars(article_context)))
            continue

        prefix = block[:clause_positions[0]]
        clause_positions.append(len(block))
        for clause_index in range(len(clause_positions) - 1):
            clause_start = clause_positions[clause_index]
            clause_end = clause_positions[clause_index + 1]
            clause_match = CLAUSE_RE.match(block[clause_start])
            clause = f"Khoản {clause_match.group(1)}"
            clause_lines = block[clause_start:clause_end]
            point_positions = [i for i, line in enumerate(clause_lines) if POINT_RE.match(line)]
            point_positions.append(len(clause_lines))
            clause_prefix = prefix if clause_index == 0 else []
            if len(point_positions) == 1:
                text = "\n".join(clause_prefix + clause_lines).strip("\n")
                chunks.append(make_chunk(base, f"{slug(article)}-{slug(clause)}", text, **vars(article_context), clause=clause))
                continue
            for point_index in range(len(point_positions) - 1):
                point_start = point_positions[point_index]
                point_end = point_positions[point_index + 1]
                point_match = POINT_RE.match(clause_lines[point_start])
                point = f"Điểm {point_match.group(1)}"
                point_text = clause_lines[point_start:point_end]
                if point_index == 0:
                    point_text = clause_prefix + clause_lines[:point_start] + point_text
                text = "\n".join(point_text).strip("\n")
                chunks.append(make_chunk(base, f"{slug(article)}-{slug(clause)}-{slug(point)}", text, **vars(article_context), clause=clause, point=point))
    return chunks


def parse_qa(path: Path) -> list[dict]:
    base = {
        "document_id": QA_CONFIG[0],
        "document_title": QA_CONFIG[1],
        "source_type": QA_CONFIG[2],
    }
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or re.match(r"\|\s*-", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or not cells[0].isdigit():
            continue
        number, question, answer = cells
        text = f"Câu hỏi: {question}\nGiải đáp: {answer}"
        chunk = make_chunk(base, f"qa-{int(number):03d}", text, content_type="qa")
        chunk.update({"question": question, "answer": answer, "related_provisions": []})
        chunks.append(chunk)
    return chunks


def generate(output: Path = OUTPUT_PATH) -> list[dict]:
    all_chunks: list[dict] = []
    for filename in SOURCE_CONFIG:
        path = MARKDOWN_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)
        document_id, title, source_type, document_number = document_metadata(path)
        base = {
            "document_id": document_id,
            "document_title": title,
            "document_number": document_number,
            "source_type": source_type,
        }
        all_chunks.extend(split_structured(path.read_text(encoding="utf-8").splitlines(), base))
    all_chunks.extend(parse_qa(QA_PATH))
    ensure_unique_chunk_ids(all_chunks)
    add_resolved_chunk_ids(all_chunks)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n")
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    chunks = generate(args.output)
    print(json.dumps({"chunks": len(chunks), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()