"""Validate generated legal chunks and report suspicious source-coverage differences."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from legal_chunker import AUTHORITY_CONFIG, KNOWN_DOCUMENTS, MARKDOWN_DIR, QA_PATH, SOURCE_CONFIG, OUTPUT_PATH


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            chunks.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
    return chunks


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def validate(path: Path) -> dict:
    chunks = load_chunks(path)
    errors: list[str] = []
    ids = [chunk.get("chunk_id") for chunk in chunks]
    if any(not chunk_id for chunk_id in ids):
        errors.append("one or more chunks have no chunk_id")
    if len(ids) != len(set(ids)):
        errors.append("chunk_id values are not unique")
    required = {"chunk_id", "document_id", "source_type", "text", "references"}
    for chunk in chunks:
        missing = required - chunk.keys()
        if missing:
            errors.append(f"{chunk.get('chunk_id')}: missing fields {sorted(missing)}")
        if not normalize(chunk.get("text", "")):
            errors.append(f"{chunk.get('chunk_id')}: empty text")
        if chunk.get("document_id") not in KNOWN_DOCUMENTS:
            errors.append(f"{chunk.get('chunk_id')}: unknown document_id")
        if chunk.get("source_type") not in {"core", "reference", "qa"}:
            errors.append(f"{chunk.get('chunk_id')}: invalid source_type")
        expected_role = AUTHORITY_CONFIG.get(chunk.get("document_id"))
        if not expected_role:
            errors.append(f"{chunk.get('chunk_id')}: missing authority configuration")
        else:
            role, authority, priority, behavior = expected_role
            if chunk.get("document_role") != role:
                errors.append(f"{chunk.get('chunk_id')}: invalid document_role")
            if chunk.get("authority_level") != authority:
                errors.append(f"{chunk.get('chunk_id')}: invalid authority_level")
            if chunk.get("retrieval_priority") != priority:
                errors.append(f"{chunk.get('chunk_id')}: invalid retrieval_priority")
            if chunk.get("retrieval_behavior") != behavior:
                errors.append(f"{chunk.get('chunk_id')}: invalid retrieval_behavior")
        for relation in chunk.get("document_relations", []):
            if relation.get("relation_type") == "amends" and chunk.get("document_id") != "60_2025_ND-CP":
                errors.append(f"{chunk.get('chunk_id')}: invalid amendment relation owner")
            if relation.get("target_document_id") not in KNOWN_DOCUMENTS:
                errors.append(f"{chunk.get('chunk_id')}: invalid relation target")
        for reference in chunk.get("references", []):
            resolved = reference.get("resolved_document_id")
            if resolved is not None and resolved not in KNOWN_DOCUMENTS:
                errors.append(f"{chunk.get('chunk_id')}: invalid resolved reference {resolved}")
    expected_sources = set(SOURCE_CONFIG) | {QA_PATH.name}
    actual_docs = {chunk["document_id"] for chunk in chunks}
    if actual_docs != KNOWN_DOCUMENTS:
        errors.append(f"document coverage mismatch: {sorted(actual_docs)}")
    qa_chunks = [chunk for chunk in chunks if chunk.get("source_type") == "qa"]
    if any("question" not in chunk or "answer" not in chunk for chunk in qa_chunks):
        errors.append("one or more QA chunks lack question or answer fields")

    counts = Counter(chunk["document_id"] for chunk in chunks)
    references = [reference for chunk in chunks for reference in chunk.get("references", [])]
    resolved = [reference for reference in references if reference.get("resolved_document_id")]
    unresolved = [reference for reference in references if not reference.get("resolved_document_id")]
    roles = Counter(chunk.get("document_role") for chunk in chunks)
    amendment_relations = [
        relation
        for chunk in chunks
        for relation in chunk.get("document_relations", [])
        if relation.get("relation_type") == "amends"
    ]
    resolved_amendment_relations = [
        relation for relation in amendment_relations if relation.get("resolution_status") == "resolved"
    ]
    return {
        "valid": not errors,
        "errors": errors,
        "chunks_per_document": dict(counts),
        "detected_references": len(references),
        "resolved_references": len(resolved),
        "unresolved_references": len(unresolved),
        "chunks_per_role": dict(roles),
        "amendment_relations": len(amendment_relations),
        "resolved_amendment_relations": len(resolved_amendment_relations),
        "unresolved_amendment_relations": len(amendment_relations) - len(resolved_amendment_relations),
        "qa_pairs": len(qa_chunks),
        "expected_source_files": sorted(expected_sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    report = validate(args.input)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()