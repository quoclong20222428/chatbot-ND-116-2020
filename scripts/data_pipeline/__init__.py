"""Legal chunk generation and validation."""

from importlib import import_module

__all__ = ["generate", "validate"]


def __getattr__(name: str):
    if name == "generate":
        return import_module(".chunking", __name__).generate
    if name == "validate":
        return import_module(".validation", __name__).validate
    raise AttributeError(name)
