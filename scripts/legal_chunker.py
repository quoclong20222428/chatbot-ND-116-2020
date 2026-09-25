"""Compatibility CLI and import path for legal chunk generation."""

if __name__ == "__main__":
    if __package__:
        from .data_pipeline.chunking import main
    else:
        from data_pipeline.chunking import main

    main()
else:
    import sys

    if __package__:
        from .data_pipeline import chunking as _implementation
    else:
        from data_pipeline import chunking as _implementation

    sys.modules[__name__] = _implementation
