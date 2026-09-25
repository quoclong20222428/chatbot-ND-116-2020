"""Database import and embedding-index workflows."""

from importlib import import_module

__all__ = ["import_records", "index_chunks", "initialize_schema", "verify"]


def __getattr__(name: str):
    if name in {"index_chunks", "verify"}:
        return getattr(import_module(".embedding_index", __name__), name)
    if name in {"import_records", "initialize_schema"}:
        return getattr(import_module(".import_data", __name__), name)
    raise AttributeError(name)
