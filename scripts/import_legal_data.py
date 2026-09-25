"""Compatibility CLI and import path for legal corpus database import."""

if __name__ == "__main__":
    if __package__:
        from .indexing.import_data import main
    else:
        from indexing.import_data import main

    raise SystemExit(main())

import sys

if __package__:
    from .indexing import import_data as _implementation
else:
    from indexing import import_data as _implementation

sys.modules[__name__] = _implementation
