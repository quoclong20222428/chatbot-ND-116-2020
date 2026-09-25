"""Compatibility CLI and import path for the embedding indexing workflow."""

if __name__ == "__main__":
    if __package__:
        from .indexing.embedding_index import main
    else:
        from indexing.embedding_index import main

    raise SystemExit(main())

import sys

if __package__:
    from .indexing import embedding_index as _implementation
else:
    from indexing import embedding_index as _implementation

sys.modules[__name__] = _implementation
