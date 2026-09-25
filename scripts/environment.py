"""Shared parsing for the repository's simple ``.env`` files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


def load_dotenv(path: Path, *, strict: bool = False) -> None:
    """Load unset environment values, optionally rejecting malformed lines.

    Strict mode preserves the validation used by the data-import and indexing
    commands. Lenient mode preserves the behavior used by retrieval commands.
    """
    if not path.is_file():
        return

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            if strict:
                raise ValueError(f"Invalid .env entry at {path}:{line_number}")
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            if strict:
                raise ValueError(f"Invalid .env entry at {path}:{line_number}")
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def require_database_url(
    env_path: Path,
    *,
    loader: Callable[[Path], None] = load_dotenv,
    error_message: str = "DATABASE_URL is required. Set it in the environment or in .env",
) -> str:
    """Load *env_path* and return ``DATABASE_URL`` or raise a clear error."""
    loader(env_path)
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(error_message)
    return database_url
