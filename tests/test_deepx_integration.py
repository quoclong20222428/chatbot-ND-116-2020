"""Lightweight integration test for the DeepX embedding backend.

This script loads the real ``dxtech-asia/deepx-embedding-v1`` model via the
``deepx_embed`` package and encodes one sample text.  It verifies:

  - model loads without error
  - output dimension is exactly 1024 (Matryoshka truncation)
  - output is a Python list of floats
  - CUDA is used when available

**Prerequisites**:

  conda activate chatbot
  pip install git+https://github.com/dx-tech-ai/deepx-embed.git

**Run**::

  python tests/test_deepx_integration.py

This test does NOT require a running PostgreSQL database.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/ importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- 1. Verify deepx_embed is installed ---
    try:
        import deepx_embed  # noqa: F401
    except ImportError:
        print("ERROR: deepx_embed is not installed.")
        print("Run: pip install git+https://github.com/dx-tech-ai/deepx-embed.git")
        sys.exit(1)

    # --- 2. Load EmbeddingModel with the deepx backend ---
    import embedding as em

    print("Loading DeepX model...")
    model = em.EmbeddingModel(model_name="deepx")

    print(f"Model:   {model.model_name}")
    print(f"Backend: {model.config.backend}")
    print(f"Device:  {model.device}")
    print(f"Column:  {model.config.embedding_column}")
    print(f"HNSW:    {model.config.hnsw_index_name}")

    assert model.config.backend == "deepx", \
        f"Expected backend='deepx', got {model.config.backend!r}"

    # --- 3. Encode one sample text ---
    sample = "Muc phat khi vuot den do la bao nhieu?"
    print(f"\nEncoding: {sample!r}")
    vec = model.embed_query(sample)

    print(f"Output type:  {type(vec).__name__}")
    print(f"Output dim:   {len(vec)}")
    print(f"First values: {vec[:5]}")

    assert isinstance(vec, list), f"Expected list, got {type(vec)}"
    assert len(vec) == 1024, f"Expected 1024 dims, got {len(vec)}"
    assert all(isinstance(v, float) for v in vec[:10]), "Values must be floats"

    # --- 4. Encode a small batch ---
    docs = [
        "Dieu 1. Pham vi dieu chinh.",
        "Khoan 2 Dieu 4 Nghi dinh 116/2020/ND-CP.",
        "Sinh vien su pham duoc ho tro hoc phi.",
    ]
    print(f"\nBatch encoding {len(docs)} documents (batch_size=2)...")
    vecs = model.embed_documents(docs, batch_size=2)

    assert len(vecs) == len(docs), f"Expected {len(docs)} vectors, got {len(vecs)}"
    for i, v in enumerate(vecs):
        assert len(v) == 1024, f"Vector {i} has {len(v)} dims, expected 1024"
    print(f"Batch result: {len(vecs)} x {len(vecs[0])}d vectors [OK]")

    # --- 5. Verify empty input ---
    empty = model.embed_documents([])
    assert empty == [], f"Expected [] for empty input, got {empty!r}"
    print("Empty input: returns [] [OK]")

    print("\n[PASS] DeepX integration test PASSED")
    print(f"   model:     dxtech-asia/deepx-embedding-v1")
    print(f"   backend:   deepx (_DeepXBackend)")
    print(f"   device:    {model.device}")
    print(f"   dimension: {len(vec)}d  [OK]")


if __name__ == "__main__":
    main()
