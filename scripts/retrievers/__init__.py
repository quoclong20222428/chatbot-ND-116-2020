"""Independent retrieval implementations and their stable public classes."""

from importlib import import_module

__all__ = ["BM25Retriever", "Retriever", "RetrievalResult"]


def __getattr__(name: str):
    if name == "Retriever":
        return import_module(".hnsw", __name__).Retriever
    if name == "BM25Retriever":
        return import_module(".bm25", __name__).BM25Retriever
    if name == "RetrievalResult":
        if __package__.startswith("scripts."):
            return import_module("..retrieval_types", __name__).RetrievalResult
        return import_module("retrieval_types").RetrievalResult
    raise AttributeError(name)
