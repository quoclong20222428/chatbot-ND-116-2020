"""Validate generated legal chunks and report suspicious source-coverage differences."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from legal_chunker import KNOWN_DOCUMENTS, MARKDOWN_DIR, QA_PATH, SOURCE_CONFIG, OUTPUT_PATH


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
    return {
        "valid": not errors,
        "errors": errors,
        "chunks_per_document": dict(counts),
        "detected_references": len(references),
        "resolved_references": len(resolved),
        "unresolved_references": len(unresolved),
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