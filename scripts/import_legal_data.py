"""Initialize the PostgreSQL schema and import the processed legal corpus.

Run from the repository root after activating the ``chatbot`` Conda environment:

    conda activate chatbot
    python scripts/import_legal_data.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INIT_SQL = ROOT / "init.sql"
JSONL_PATH = ROOT / "data" / "processed" / "legal_chunks.jsonl"
ENV_PATH = ROOT / ".env"

LOGGER = logging.getLogger("import_legal_data")

# These statements deliberately mirror the document and chunk upserts in init.sql.
# init.sql remains the source of truth for the schema and its psql import behavior.
DOCUMENT_UPSERT = """
INSERT INTO documents (
    document_id, document_number, document_title, source_type, document_role,
    authority_level, retrieval_priority, retrieval_behavior
)
VALUES (%s, NULLIF(%s, ''), %s, %s, %s, %s, %s, %s)
ON CONFLICT (document_id) DO UPDATE SET
    document_number = EXCLUDED.document_number,
    document_title = EXCLUDED.document_title,
    source_type = EXCLUDED.source_type,
    document_role = EXCLUDED.document_role,
    authority_level = EXCLUDED.authority_level,
    retrieval_priority = EXCLUDED.retrieval_priority,
    retrieval_behavior = EXCLUDED.retrieval_behavior,
    updated_at = now()
"""

CHUNK_UPSERT = """
INSERT INTO legal_chunks (
    chunk_id, document_id, document_title, document_number, source_type,
    document_role, authority_level, retrieval_priority, retrieval_behavior,
    document_relations, chapter, section, subsection, article, clause, point,
    form_number, content_type, question, answer, related_provisions, text,
    "references"
)
VALUES (
    %s, %s, %s, NULLIF(%s, ''), %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s,
    %s, %s, %s, %s, COALESCE(NULLIF(%s, ''), 'legal_text'), NULLIF(%s, ''),
    NULLIF(%s, ''), %s::jsonb, %s, %s::jsonb
)
ON CONFLICT (chunk_id) DO UPDATE SET
    document_id = EXCLUDED.document_id,
    document_title = EXCLUDED.document_title,
    document_number = EXCLUDED.document_number,
    source_type = EXCLUDED.source_type,
    document_role = EXCLUDED.document_role,
    authority_level = EXCLUDED.authority_level,
    retrieval_priority = EXCLUDED.retrieval_priority,
    retrieval_behavior = EXCLUDED.retrieval_behavior,
    document_relations = EXCLUDED.document_relations,
    chapter = EXCLUDED.chapter,
    section = EXCLUDED.section,
    subsection = EXCLUDED.subsection,
    article = EXCLUDED.article,
    clause = EXCLUDED.clause,
    point = EXCLUDED.point,
    form_number = EXCLUDED.form_number,
    content_type = EXCLUDED.content_type,
    question = EXCLUDED.question,
    answer = EXCLUDED.answer,
    related_provisions = EXCLUDED.related_provisions,
    text = EXCLUDED.text,
    "references" = EXCLUDED."references"
"""

REFERENCE_REBUILD = """
INSERT INTO legal_chunk_references (
    chunk_id, raw_reference, resolved_document_id, resolved_chunk_id, provision
)
SELECT
    chunk.chunk_id,
    reference->>'raw_reference',
    NULLIF(reference->>'resolved_document_id', ''),
    NULLIF(reference->>'resolved_chunk_id', ''),
    NULLIF(reference->>'provision', '')
FROM legal_chunks AS chunk
CROSS JOIN LATERAL jsonb_array_elements(chunk."references") AS reference
WHERE NULLIF(reference->>'raw_reference', '') IS NOT NULL
ON CONFLICT DO NOTHING
"""

REQUIRED_FIELDS = (
    "chunk_id",
    "document_id",
    "document_title",
    "source_type",
    "document_role",
    "authority_level",
    "retrieval_priority",
    "retrieval_behavior",
    "text",
)
ARRAY_FIELDS = ("document_relations", "related_provisions", "references")


@dataclass
class ImportStats:
    processed_records: int = 0
    input_references: int = 0
    document_ids: set[str] = field(default_factory=set)
    chunk_ids: set[str] = field(default_factory=set)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_dotenv(path: Path) -> None:
    """Load unset variables from a simple local .env file without another dependency."""
    if not path.is_file():
        return

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env entry at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid .env entry at {path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def require_inputs() -> str:
    load_dotenv(ENV_PATH)

    if not INIT_SQL.is_file():
        raise FileNotFoundError(f"Schema file not found: {INIT_SQL}")
    if not JSONL_PATH.is_file():
        raise FileNotFoundError(f"JSONL input file not found: {JSONL_PATH}")

    active_environment = os.environ.get("CONDA_DEFAULT_ENV")
    if active_environment != "chatbot":
        found = active_environment or "no active Conda environment"
        raise RuntimeError(
            f"Expected Conda environment 'chatbot' (found {found}). "
            "Run: conda activate chatbot"
        )

    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(f"DATABASE_URL is required; set it in the environment or {ENV_PATH.name}")
    return database_url


def initialize_schema(database_url: str) -> None:
    """Run init.sql unchanged, stopping psql at the first SQL or copy error."""
    psql = shutil.which("psql")
    if not psql:
        raise RuntimeError("psql was not found on PATH in the chatbot environment")

    LOGGER.info("Initializing schema with %s", INIT_SQL.relative_to(ROOT))
    psql_environment = os.environ.copy()
    # psql errors can contain Vietnamese corpus text.  Explicit UTF-8 avoids
    # Windows decoding its captured output with the active CP1252 code page.
    psql_environment.setdefault("PGCLIENTENCODING", "UTF8")
    try:
        result = subprocess.run(
            [
                psql,
                "--no-psqlrc",
                "--dbname",
                database_url,
                "--set",
                "ON_ERROR_STOP=1",
                "--file",
                str(INIT_SQL),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=psql_environment,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).strip()
        raise RuntimeError(f"Schema initialization failed: {detail}") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not run psql: {exc}") from exc

    if result.stderr.strip():
        LOGGER.info(result.stderr.strip())
    LOGGER.info("Schema initialization completed")


def value_or_empty(record: dict[str, Any], key: str) -> Any:
    value = record.get(key, "")
    return "" if value is None else value


def array_value(record: dict[str, Any], key: str) -> list[Any]:
    value = record.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON array")
    return value


def validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("record must be a JSON object")
    for key in REQUIRED_FIELDS:
        if key == "retrieval_priority":
            continue
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if type(record["retrieval_priority"]) is not int:
        raise ValueError("retrieval_priority must be an integer")
    if not 0 <= record["retrieval_priority"] <= 100:
        raise ValueError("retrieval_priority must be between 0 and 100")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError("text must be non-empty")
    for key in ARRAY_FIELDS:
        array_value(record, key)
    return record


def import_records(connection: Any) -> ImportStats:
    """Stream JSONL records, upsert them, then rebuild normalized references."""
    stats = ImportStats()
    with connection.cursor() as cursor, JSONL_PATH.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = validate_record(json.loads(line))
                chunk_id = record["chunk_id"]
                if chunk_id in stats.chunk_ids:
                    raise ValueError(f"duplicate chunk_id {chunk_id!r} in input")

                document_id = record["document_id"]
                cursor.execute(
                    DOCUMENT_UPSERT,
                    (
                        document_id,
                        value_or_empty(record, "document_number"),
                        record["document_title"],
                        record["source_type"],
                        record["document_role"],
                        record["authority_level"],
                        record["retrieval_priority"],
                        record["retrieval_behavior"],
                    ),
                )
                cursor.execute(
                    CHUNK_UPSERT,
                    (
                        chunk_id,
                        document_id,
                        record["document_title"],
                        value_or_empty(record, "document_number"),
                        record["source_type"],
                        record["document_role"],
                        record["authority_level"],
                        record["retrieval_priority"],
                        record["retrieval_behavior"],
                        json.dumps(array_value(record, "document_relations"), ensure_ascii=False),
                        value_or_empty(record, "chapter"),
                        value_or_empty(record, "section"),
                        value_or_empty(record, "subsection"),
                        value_or_empty(record, "article"),
                        value_or_empty(record, "clause"),
                        value_or_empty(record, "point"),
                        value_or_empty(record, "form_number"),
                        value_or_empty(record, "content_type"),
                        value_or_empty(record, "question"),
                        value_or_empty(record, "answer"),
                        json.dumps(array_value(record, "related_provisions"), ensure_ascii=False),
                        record["text"],
                        json.dumps(array_value(record, "references"), ensure_ascii=False),
                    ),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid JSONL record at line {line_number}: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"Database import failed at JSONL line {line_number}: {exc}") from exc

            stats.processed_records += 1
            stats.input_references += len(array_value(record, "references"))
            stats.document_ids.add(document_id)
            stats.chunk_ids.add(chunk_id)

        if not stats.processed_records:
            raise ValueError("JSONL input contains no records")

        # This is the normalized-reference portion of init.sql. Rebuilding it
        # makes reruns idempotent after the chunk upserts above.
        cursor.execute("DELETE FROM legal_chunk_references")
        cursor.execute(REFERENCE_REBUILD)
    return stats


def database_counts(connection: Any) -> tuple[int, int, int]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                (SELECT count(*) FROM documents),
                (SELECT count(*) FROM legal_chunks),
                (SELECT count(*) FROM legal_chunk_references)
            """
        )
        return cursor.fetchone()


def main() -> int:
    configure_logging()
    connection: Any | None = None
    try:
        database_url = require_inputs()
        initialize_schema(database_url)
        try:
            from psycopg import connect
        except ImportError:
            try:
                from psycopg2 import connect  # type: ignore[no-redef]
            except ImportError as exc:
                raise RuntimeError("Install psycopg or psycopg2 in the chatbot environment") from exc

        LOGGER.info("Importing %s line by line", JSONL_PATH.relative_to(ROOT))
        connection = connect(database_url)
        try:
            stats = import_records(connection)
            documents, chunks, references = database_counts(connection)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

        LOGGER.info(
            "Import complete: %s documents, %s legal chunks, %s references; "
            "%s records successfully imported (%s input references).",
            documents,
            chunks,
            references,
            stats.processed_records,
            stats.input_references,
        )
        return 0
    except Exception as exc:
        LOGGER.error("Import failed: %s", exc)
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    sys.exit(main())
