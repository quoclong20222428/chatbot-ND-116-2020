"""Compatibility CLI and import path for legal chunk validation."""

if __name__ == "__main__":
    if __package__:
        from .data_pipeline.validation import main
    else:
        from data_pipeline.validation import main

    main()
else:
    import sys

    if __package__:
        from .data_pipeline import validation as _implementation
    else:
        from data_pipeline import validation as _implementation

    sys.modules[__name__] = _implementation
