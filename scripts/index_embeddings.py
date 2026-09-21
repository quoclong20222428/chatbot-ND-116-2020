"""Generate and store BAAI/bge-m3 embeddings for all legal chunks in the database.

Run from the repository root after activating the ``chatbot`` Conda environment::

    conda activate chatbot
    python scripts/index_embeddings.py

By default the script processes only chunks where ``embedding IS NULL``, so it
is safe to interrupt and restart.  To regenerate all embeddings use
``--rebuild``::

    python scripts/index_embeddings.py --rebuild

The script does NOT perform retrieval or LLM generation.  It only reads chunk
text from the database, generates dense vectors with ``BAAI/bge-m3``, and
writes those vectors back to ``legal_chunks.embedding``.

Environment variables
---------------------
DATABASE_URL
    PostgreSQL connection URL (required).  Loaded from ``.env`` if not set in
    the environment.
EMBEDDING_MODEL
    Hugging Face model identifier (default: ``BAAI/bge-m3``).
EMBEDDING_BATCH_SIZE
    Number of chunks to embed per model forward pass (default: 32).
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
# SQL statements
# ---------------------------------------------------------------------------

# Fetch chunks that still need an embedding.
SELECT_UNEMBEDDED = """
SELECT chunk_id, text
FROM legal_chunks
WHERE embedding IS NULL
ORDER BY chunk_id
"""

# Fetch ALL chunks (used with --rebuild).
SELECT_ALL = """
SELECT chunk_id, text
FROM legal_chunks
ORDER BY chunk_id
"""

# Persist a single embedding.
UPDATE_EMBEDDING = """
UPDATE legal_chunks
SET embedding = %s::vector
WHERE chunk_id = %s
"""

# Verification queries.
VERIFY_COUNTS = """
SELECT
    count(*)          AS total_chunks,
    count(embedding)  AS embedded_chunks
FROM legal_chunks
"""

VERIFY_DIMENSION = """
SELECT chunk_id, vector_dims(embedding) AS dim
FROM legal_chunks
WHERE embedding IS NOT NULL
LIMIT 1
"""

CHECK_HNSW_INDEX = """
SELECT indexname
FROM pg_indexes
WHERE tablename = 'legal_chunks'
  AND indexname = 'legal_chunks_embedding_hnsw_idx'
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

    active_environment = os.environ.get("CONDA_DEFAULT_ENV")
    if active_environment != "chatbot":
        found = active_environment or "no active Conda environment"
        raise RuntimeError(
            f"Expected Conda environment 'chatbot' (found {found}). "
            "Run: conda activate chatbot"
        )

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


def fetch_chunks(connection: Any, rebuild: bool) -> list[tuple[str, str]]:
    """Return (chunk_id, text) pairs for chunks that need embedding."""
    sql = SELECT_ALL if rebuild else SELECT_UNEMBEDDED
    with connection.cursor() as cursor:
        cursor.execute(sql)
        return cursor.fetchall()


def index_chunks(
    database_url: str,
    model: Any,
    chunks: list[tuple[str, str]],
    batch_size: int,
) -> IndexStats:
    """Embed chunks in batches and persist each embedding via a short-lived connection.

    Strategy: the connection is opened *only* during the DB write phase, not
    during model inference.  This prevents NeonDB (and any cloud PostgreSQL
    service with aggressive idle-connection timeouts) from dropping the
    connection while the CPU is running a multi-minute embedding forward pass.

    Each batch therefore follows:
        1. Embed texts  (model inference — potentially minutes on CPU, no DB needed)
        2. Open fresh connection
        3. UPDATE legal_chunks … for each chunk in batch
        4. COMMIT
        5. Close connection
    """
    stats = IndexStats(total_to_embed=len(chunks))
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_index in range(total_batches):
        batch_start = batch_index * batch_size
        batch_end = batch_start + batch_size
        batch = chunks[batch_start:batch_end]

        batch_num = batch_index + 1
        LOGGER.info("Processing batch %d/%d (%d chunks)", batch_num, total_batches, len(batch))

        chunk_ids = [row[0] for row in batch]
        texts = [row[1] for row in batch]

        # --- Step 1: Embed (no DB connection held open during inference) ---
        try:
            vectors = model.embed_texts(texts, batch_size=len(texts))
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

        # --- Step 2: Write to DB (fresh short-lived connection) ---
        try:
            with connect_db(database_url) as conn:
                with conn.cursor() as cursor:
                    for chunk_id, vector in zip(chunk_ids, vectors):
                        cursor.execute(UPDATE_EMBEDDING, (_vector_to_pg(vector), chunk_id))
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


def verify(connection: Any, expected_total: int | None = None) -> dict[str, Any]:
    """Run post-indexing sanity checks and return a result dict."""
    result: dict[str, Any] = {}

    with connection.cursor() as cursor:
        # Chunk counts.
        cursor.execute(VERIFY_COUNTS)
        row = cursor.fetchone()
        total_chunks, embedded_chunks = row[0], row[1]
        result["total_chunks"] = total_chunks
        result["embedded_chunks"] = embedded_chunks
        result["missing_embeddings"] = total_chunks - embedded_chunks

        # Dimension check.
        cursor.execute(VERIFY_DIMENSION)
        dim_row = cursor.fetchone()
        if dim_row:
            result["embedding_dim"] = dim_row[1]
            result["embedding_dim_ok"] = dim_row[1] == 1024
        else:
            result["embedding_dim"] = None
            result["embedding_dim_ok"] = False

        # HNSW index presence.
        cursor.execute(CHECK_HNSW_INDEX)
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
    parser = argparse.ArgumentParser(
        description="Generate BAAI/bge-m3 embeddings for all legal_chunks rows."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Re-embed ALL chunks, not just those with embedding IS NULL.",
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
        help="Hugging Face model identifier (default: BAAI/bge-m3 or EMBEDDING_MODEL env var).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)

    try:
        database_url = require_env()

        batch_size = args.batch_size
        model_name = args.model

        LOGGER.info("Embedding model:  %s", model_name)
        LOGGER.info("Batch size:       %d", batch_size)
        LOGGER.info("Rebuild mode:     %s", args.rebuild)

        # Import embedding module from the same directory.
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from embedding import EmbeddingModel  # noqa: PLC0415

        LOGGER.info("Loading embedding model (this may take a moment on first run) ...")
        model = EmbeddingModel(model_name=model_name)

        # Use a short-lived connection only to fetch the chunk list.
        with connect_db(database_url) as fetch_conn:
            chunks = fetch_chunks(fetch_conn, rebuild=args.rebuild)
        LOGGER.info("Total chunks requiring embeddings: %d", len(chunks))

        if not chunks:
            LOGGER.info("Nothing to embed — all chunks already have embeddings.")
            with connect_db(database_url) as verify_conn:
                result = verify(verify_conn)
            log_verification(result)
            return 0

        # Index: index_chunks opens its own fresh connection per batch.
        stats = index_chunks(database_url, model, chunks, batch_size=batch_size)

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
            result = verify(verify_conn, expected_total=len(chunks))
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
                "HNSW index 'legal_chunks_embedding_hnsw_idx' does not exist. "
                "Run import_legal_data.py to apply the latest init.sql schema."
            )

        LOGGER.info("Indexing stage complete. Database is ready for retrieval.")
        return 0

    except Exception as exc:
        LOGGER.error("Indexing failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
