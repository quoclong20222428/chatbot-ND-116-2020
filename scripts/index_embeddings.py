"""Generate and store embeddings for all legal chunks in the database.

Supports multiple embedding models via the :mod:`model_registry` and
:mod:`embedding` abstraction.  Each model writes to its own dedicated
column in ``legal_chunks``, preventing cross-model vector mixing.

Embedding representation
------------------------
For legal-document chunks (``content_type = 'legal_text'``) the text sent to
the model is a **structured prefix + original chunk text** built by
:func:`embedding.build_embedding_text`.  The prefix encodes available
structural metadata (document, chapter, article, clause, point) so that the
generated vectors capture both semantic content and structural location.

For QA chunks (``content_type = 'qa'``) the original text is used unchanged.

Run from the repository root::

    python scripts/index_embeddings.py
    python scripts/index_embeddings.py --model jina-v3
    python scripts/index_embeddings.py --rebuild
    python scripts/index_embeddings.py --preview --limit 5

By default the script processes only chunks where the model's embedding column
is ``NULL``, so it is safe to interrupt and restart.  To regenerate all
embeddings use ``--rebuild``::

    python scripts/index_embeddings.py --model bge-m3 --rebuild

The script does NOT perform retrieval or LLM generation.  It only reads chunk
metadata + text from the database, generates dense vectors with the selected
embedding model using the metadata-aware representation, and writes those
vectors back to the model-specific column in ``legal_chunks``.

Environment variables
---------------------
DATABASE_URL
    PostgreSQL connection URL (required).  Loaded from ``.env`` if not set in
    the environment.
EMBEDDING_MODEL
    Hugging Face model identifier or alias (default: ``BAAI/bge-m3``).
EMBEDDING_BATCH_SIZE
    Number of chunks to embed per model forward pass (default: 8).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

LOGGER = logging.getLogger("index_embeddings")

# ---------------------------------------------------------------------------
# Ensure scripts/ is importable
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# SQL builders (model-specific column)
# ---------------------------------------------------------------------------


def _select_unembedded(column: str) -> str:
    """SQL: fetch chunks that still need an embedding in the given column."""
    return f"""
SELECT
    chunk_id,
    text,
    content_type,
    document_title,
    chapter,
    article,
    clause,
    point
FROM legal_chunks
WHERE {column} IS NULL
ORDER BY chunk_id
"""


def _select_all() -> str:
    """SQL: fetch ALL chunks (used with --rebuild or --preview)."""
    return """
SELECT
    chunk_id,
    text,
    content_type,
    document_title,
    chapter,
    article,
    clause,
    point
FROM legal_chunks
ORDER BY chunk_id
"""


def _update_embedding(column: str) -> str:
    """SQL: persist a single embedding to the model-specific column."""
    return f"""
UPDATE legal_chunks
SET {column} = %s::vector
WHERE chunk_id = %s
"""


def _verify_counts(column: str) -> str:
    """SQL: count total and embedded chunks for the given column."""
    return f"""
SELECT
    count(*)          AS total_chunks,
    count({column})   AS embedded_chunks
FROM legal_chunks
"""


def _verify_dimension(column: str) -> str:
    """SQL: check the embedding dimension for the given column."""
    return f"""
SELECT chunk_id, vector_dims({column}) AS dim
FROM legal_chunks
WHERE {column} IS NOT NULL
LIMIT 1
"""


def _check_hnsw_index(index_name: str) -> str:
    """SQL: check if the HNSW index exists."""
    return """
SELECT indexname
FROM pg_indexes
WHERE tablename = 'legal_chunks'
  AND indexname = %s
"""


def _ensure_column(column: str) -> str:
    """SQL: add the embedding column if it does not exist."""
    return f"""
ALTER TABLE legal_chunks
    ADD COLUMN IF NOT EXISTS {column} vector(1024)
"""


def _ensure_hnsw_index(column: str, index_name: str) -> str:
    """SQL: create the HNSW index if it does not exist."""
    return f"""
CREATE INDEX IF NOT EXISTS {index_name}
    ON legal_chunks USING hnsw ({column} vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
"""


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class IndexStats:
    total_to_embed: int = 0
    succeeded: int = 0
    failed: int = 0
    failed_chunk_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers shared with import_legal_data.py (duplicated to keep scripts standalone)
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_dotenv(path: Path) -> None:
    """Load unset variables from a simple local .env file without extra deps."""
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


def require_env() -> str:
    """Load configuration and return the database URL."""
    load_dotenv(ENV_PATH)

    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(
            f"DATABASE_URL is required; set it in the environment or {ENV_PATH.name}"
        )
    return database_url



def connect_db(database_url: str) -> Any:
    """Return an open psycopg (v3) connection."""
    try:
        from psycopg import connect
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is not installed.  Run: pip install psycopg[binary]"
        ) from exc
    return connect(database_url)


# ---------------------------------------------------------------------------
# Core indexing logic
# ---------------------------------------------------------------------------


# Row type returned by fetch_chunks:
#   (chunk_id, text, content_type, document_title, chapter, article, clause, point)
_ChunkRow = tuple[str, str, str | None, str | None, str | None, str | None, str | None, str | None]


def fetch_chunks(
    connection: Any,
    rebuild: bool,
    embedding_column: str,
    limit: int | None = None,
) -> list[_ChunkRow]:
    """Return chunk rows for chunks that need embedding.

    Each row is an 8-tuple:
    ``(chunk_id, text, content_type, document_title, chapter, article, clause, point)``

    The metadata columns are used by :func:`embedding.build_embedding_text` to
    construct the structured embedding input for legal-document chunks.

    Parameters
    ----------
    connection:
        Open psycopg (v3) connection.
    rebuild:
        When ``True``, fetch all chunks regardless of existing embeddings.
    embedding_column:
        The model-specific column name to check for ``IS NULL``.
    limit:
        Optional maximum number of rows to return (used by ``--preview``).
    """
    sql = _select_all() if rebuild else _select_unembedded(embedding_column)
    if limit is not None:
        sql = sql.rstrip() + f"\nLIMIT {int(limit)}"
    with connection.cursor() as cursor:
        cursor.execute(sql)
        return cursor.fetchall()


def ensure_model_column(database_url: str, embedding_column: str, index_name: str) -> None:
    """Ensure the model-specific embedding column and HNSW index exist.

    This is idempotent — safe to call on every run.  Uses ``ADD COLUMN IF NOT
    EXISTS`` and ``CREATE INDEX IF NOT EXISTS``.
    """
    with connect_db(database_url) as conn:
        with conn.cursor() as cursor:
            LOGGER.info(
                "Ensuring column '%s' and index '%s' exist ...",
                embedding_column, index_name,
            )
            cursor.execute(_ensure_column(embedding_column))
            cursor.execute(_ensure_hnsw_index(embedding_column, index_name))
        conn.commit()
    LOGGER.info("Database schema ready for model column '%s'.", embedding_column)


def index_chunks(
    database_url: str,
    model: Any,
    chunks: list[_ChunkRow],
    batch_size: int,
    embedding_column: str,
) -> IndexStats:
    """Embed chunks in batches and persist each embedding via a short-lived connection.

    For each chunk the embedding input is built by
    :func:`embedding.build_embedding_text`, which prepends a structured
    metadata prefix for legal-document chunks (``content_type = 'legal_text'``).
    QA chunks are embedded using their original text unchanged.

    Strategy: the connection is opened *only* during the DB write phase, not
    during model inference.  This prevents NeonDB (and any cloud PostgreSQL
    service with aggressive idle-connection timeouts) from dropping the
    connection while the CPU is running a multi-minute embedding forward pass.

    Each batch therefore follows:
        1. Build metadata-aware embedding inputs (no model, no DB)
        2. Embed texts  (model inference — potentially minutes on CPU, no DB needed)
        3. Open fresh connection
        4. UPDATE legal_chunks … for each chunk in batch
        5. COMMIT
        6. Close connection
    """
    from embedding import build_embedding_text  # noqa: PLC0415

    update_sql = _update_embedding(embedding_column)
    stats = IndexStats(total_to_embed=len(chunks))
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_index in range(total_batches):
        batch_start = batch_index * batch_size
        batch_end = batch_start + batch_size
        batch = chunks[batch_start:batch_end]

        batch_num = batch_index + 1
        LOGGER.info("Processing batch %d/%d (%d chunks)", batch_num, total_batches, len(batch))

        chunk_ids = [row[0] for row in batch]

        # --- Step 1: Build metadata-aware embedding inputs ---
        embedding_inputs: list[str] = []
        for row in batch:
            chunk_id, text, content_type, document_title, chapter, article, clause, point = row
            meta = {
                "content_type": content_type,
                "document_title": document_title,
                "chapter": chapter,
                "article": article,
                "clause": clause,
                "point": point,
            }
            embedding_inputs.append(build_embedding_text(text, meta))

        # --- Step 2: Embed (no DB connection held open during inference) ---
        # Use embed_documents() to apply the correct document encoding protocol.
        try:
            vectors = model.embed_documents(embedding_inputs, batch_size=len(embedding_inputs))
        except Exception as exc:
            LOGGER.error(
                "Embedding failed for batch %d/%d: %s — marking %d chunks as failed",
                batch_num,
                total_batches,
                exc,
                len(batch),
            )
            stats.failed += len(batch)
            stats.failed_chunk_ids.extend(chunk_ids)
            continue

        # --- Step 3: Write to DB (fresh short-lived connection) ---
        try:
            with connect_db(database_url) as conn:
                with conn.cursor() as cursor:
                    for chunk_id, vector in zip(chunk_ids, vectors):
                        cursor.execute(update_sql, (_vector_to_pg(vector), chunk_id))
                conn.commit()
            stats.succeeded += len(batch)
        except Exception as exc:
            LOGGER.error(
                "Database write failed for batch %d/%d (chunk_ids %s … %s): %s",
                batch_num,
                total_batches,
                chunk_ids[0],
                chunk_ids[-1],
                exc,
            )
            stats.failed += len(batch)
            stats.failed_chunk_ids.extend(chunk_ids)

    return stats



def _vector_to_pg(vector: list[float]) -> str:
    """Serialise a float list to the pgvector literal format '[f1,f2,…]'."""
    return "[" + ",".join(str(v) for v in vector) + "]"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(
    connection: Any,
    embedding_column: str,
    hnsw_index_name: str,
    expected_total: int | None = None,
) -> dict[str, Any]:
    """Run post-indexing sanity checks and return a result dict."""
    result: dict[str, Any] = {}

    with connection.cursor() as cursor:
        # Chunk counts.
        cursor.execute(_verify_counts(embedding_column))
        row = cursor.fetchone()
        total_chunks, embedded_chunks = row[0], row[1]
        result["total_chunks"] = total_chunks
        result["embedded_chunks"] = embedded_chunks
        result["missing_embeddings"] = total_chunks - embedded_chunks

        # Dimension check.
        cursor.execute(_verify_dimension(embedding_column))
        dim_row = cursor.fetchone()
        if dim_row:
            result["embedding_dim"] = dim_row[1]
            result["embedding_dim_ok"] = dim_row[1] == 1024
        else:
            result["embedding_dim"] = None
            result["embedding_dim_ok"] = False

        # HNSW index presence.
        cursor.execute(_check_hnsw_index(hnsw_index_name), (hnsw_index_name,))
        result["hnsw_index_exists"] = cursor.fetchone() is not None

    if expected_total is not None:
        result["expected_total"] = expected_total

    return result


def log_verification(result: dict[str, Any]) -> None:
    LOGGER.info("--- Verification ---")
    LOGGER.info("Total chunks:         %s", result.get("total_chunks"))
    LOGGER.info("Embedded chunks:      %s", result.get("embedded_chunks"))
    LOGGER.info("Missing embeddings:   %s", result.get("missing_embeddings"))
    LOGGER.info("Embedding dimension:  %s", result.get("embedding_dim"))
    LOGGER.info("Dimension correct:    %s", result.get("embedding_dim_ok"))
    LOGGER.info("HNSW index exists:    %s", result.get("hnsw_index_exists"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from model_registry import SUPPORTED_ALIASES  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description=(
            "Generate embeddings for all legal_chunks rows using the "
            "selected embedding model."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Re-embed ALL chunks, not just those with embedding IS NULL.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help=(
            "Fetch chunks from the DB, print the embedding input for each, "
            "and exit WITHOUT calling the model or writing any embeddings."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of chunks to preview (default: 5, only used with --preview).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("EMBEDDING_BATCH_SIZE", "8")),
        metavar="N",
        help=(
            "Chunks per model forward pass (default: 8 or EMBEDDING_BATCH_SIZE env var). "
            "Smaller values reduce the risk of connection timeouts on cloud-hosted "
            "PostgreSQL services (e.g. NeonDB) where idle connections are dropped."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        metavar="NAME",
        help=(
            "Hugging Face model identifier or alias "
            f"(default: BAAI/bge-m3 or EMBEDDING_MODEL env var). "
            f"Supported aliases: {', '.join(SUPPORTED_ALIASES)}"
        ),
    )
    return parser.parse_args(argv)


def run_preview(database_url: str, limit: int) -> int:
    """Fetch the first *limit* chunks and print their embedding inputs.

    Does NOT load any embedding model.  Does NOT write any embeddings.  Safe
    to run at any time to inspect what the metadata-aware representation will
    look like before committing to a full rebuild.
    """
    from embedding import build_embedding_text  # noqa: PLC0415

    # Use a dummy column for the preview (we fetch ALL chunks anyway).
    with connect_db(database_url) as conn:
        rows = fetch_chunks(conn, rebuild=True, embedding_column="embedding", limit=limit)

    if not rows:
        print("No chunks found in the database.")
        return 0

    total = len(rows)
    print(f"Previewing embedding input for {total} chunk(s):\n")
    sep = "-" * 60

    for i, row in enumerate(rows, 1):
        chunk_id, text, content_type, document_title, chapter, article, clause, point = row
        meta = {
            "content_type": content_type,
            "document_title": document_title,
            "chapter": chapter,
            "article": article,
            "clause": clause,
            "point": point,
        }
        emb_input = build_embedding_text(text, meta)

        print(sep)
        print(f"Chunk {i}/{total}")
        print(f"  chunk_id:      {chunk_id}")
        print(f"  content_type:  {content_type}")
        print(f"  document:      {document_title}")
        print(f"  chapter:       {chapter}")
        print(f"  article:       {article}")
        print(f"  clause:        {clause}")
        print(f"  point:         {point}")
        print()
        print("Embedding input:")
        print(emb_input)
        print()

    print(sep)
    print("Preview complete — no embeddings were generated or written.")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)

    try:
        database_url = require_env()

        # --preview: no model loaded, no embeddings written.
        if args.preview:
            return run_preview(database_url, limit=args.limit)

        batch_size = args.batch_size
        model_name = args.model

        # Resolve the model configuration first (fails early if unsupported).
        from model_registry import resolve_model_config  # noqa: PLC0415
        model_config = resolve_model_config(model_name)
        embedding_column = model_config.embedding_column
        hnsw_index_name = model_config.hnsw_index_name

        LOGGER.info("Embedding model:  %s (%s)", model_config.alias, model_config.model_id)
        LOGGER.info("Backend:          %s", model_config.backend)
        LOGGER.info("Embedding column: %s", embedding_column)
        LOGGER.info("HNSW index:       %s", hnsw_index_name)
        LOGGER.info("Batch size:       %d", batch_size)
        LOGGER.info("Rebuild mode:     %s", args.rebuild)

        # Ensure the model-specific column and HNSW index exist.
        ensure_model_column(database_url, embedding_column, hnsw_index_name)

        from embedding import EmbeddingModel  # noqa: PLC0415

        LOGGER.info("Loading embedding model (this may take a moment on first run) ...")
        model = EmbeddingModel(model_name=model_name)

        # Use a short-lived connection only to fetch the chunk list.
        with connect_db(database_url) as fetch_conn:
            chunks = fetch_chunks(
                fetch_conn, rebuild=args.rebuild, embedding_column=embedding_column,
            )
        LOGGER.info("Total chunks requiring embeddings: %d", len(chunks))

        if not chunks:
            LOGGER.info("Nothing to embed — all chunks already have embeddings.")
            with connect_db(database_url) as verify_conn:
                result = verify(verify_conn, embedding_column, hnsw_index_name)
            log_verification(result)
            return 0

        # Index: index_chunks opens its own fresh connection per batch.
        stats = index_chunks(
            database_url, model, chunks,
            batch_size=batch_size,
            embedding_column=embedding_column,
        )

        LOGGER.info("--- Indexing summary ---")
        LOGGER.info("Successfully indexed: %d", stats.succeeded)
        LOGGER.info("Failed:               %d", stats.failed)
        if stats.failed_chunk_ids:
            LOGGER.warning(
                "Failed chunk IDs: %s",
                ", ".join(stats.failed_chunk_ids[:10])
                + (" ..." if len(stats.failed_chunk_ids) > 10 else ""),
            )

        # Verify with a fresh connection.
        with connect_db(database_url) as verify_conn:
            result = verify(
                verify_conn, embedding_column, hnsw_index_name,
                expected_total=len(chunks),
            )
        log_verification(result)

        if stats.failed > 0:
            LOGGER.error("Indexing completed with %d failures.", stats.failed)
            return 1
        if result.get("missing_embeddings", 1) != 0:
            LOGGER.error(
                "Verification failed: %d chunks are still missing embeddings.",
                result["missing_embeddings"],
            )
            return 1
        if not result.get("embedding_dim_ok"):
            LOGGER.error(
                "Verification failed: embedding dimension is %s, expected 1024.",
                result.get("embedding_dim"),
            )
            return 1
        if not result.get("hnsw_index_exists"):
            LOGGER.warning(
                "HNSW index '%s' does not exist. "
                "Run import_legal_data.py to apply the latest init.sql schema.",
                hnsw_index_name,
            )

        LOGGER.info("Indexing stage complete. Database is ready for retrieval.")
        return 0

    except Exception as exc:
        LOGGER.error("Indexing failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
