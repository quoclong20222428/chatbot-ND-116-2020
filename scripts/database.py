"""PostgreSQL connection helper shared by database-backed scripts."""

from __future__ import annotations

from typing import Any


def connect_postgres(
    database_url: str,
    *,
    error_message: str,
    fallback_psycopg2: bool = False,
) -> Any:
    """Open a PostgreSQL connection, preserving caller-specific guidance.

    ``fallback_psycopg2`` is reserved for legacy workflows that historically
    supported either driver. Other callers keep the psycopg 3-only behavior.
    """
    try:
        from psycopg import connect  # type: ignore[import]
    except ImportError as exc:
        if not fallback_psycopg2:
            raise RuntimeError(error_message) from exc
        try:
            from psycopg2 import connect  # type: ignore[import,no-redef]
        except ImportError as fallback_exc:
            raise RuntimeError(error_message) from fallback_exc
    return connect(database_url)
