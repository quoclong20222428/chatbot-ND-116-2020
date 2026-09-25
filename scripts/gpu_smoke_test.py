"""GPU smoke test for BAAI/bge-m3 on CUDA.

Run before the full indexing to confirm the GPU is operational:

    conda activate chatbot
    python scripts/gpu_smoke_test.py

Expected output:
    PyTorch: <version>+cu128
    CUDA available: True
    GPU: NVIDIA GeForce RTX 3050 Laptop GPU
    Tensor test: cuda:0 — PASS
    Embedding dimension: 1024
    GPU smoke test: PASS
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch  # type: ignore[import]

    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Install a CUDA-enabled PyTorch build.")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 1

    print(f"CUDA runtime:   {torch.version.cuda}")
    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    print(f"GPU:            {gpu_name}")
    print(f"VRAM:           {vram_gb:.1f} GB")

    # --- Basic tensor operation on GPU ---
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"Tensor test:    {y.device} — PASS")

    # --- Load BAAI/bge-m3 and embed two texts ---
    print("\nLoading BAAI/bge-m3 on GPU (model loads from cache)...")
    if __package__:
        from .embeddings.embedding import EmbeddingModel  # noqa: PLC0415
    else:
        from embeddings.embedding import EmbeddingModel  # noqa: PLC0415

    model = EmbeddingModel()

    if model._model is None:
        print("ERROR: Model failed to load.")
        return 1

    texts = [
        "Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng",
        "Nghị định này quy định về chính sách hỗ trợ tiền đóng học phí.",
    ]
    print(f"Embedding {len(texts)} texts...")
    vectors = model.embed_texts(texts)

    if len(vectors) != len(texts):
        print(f"ERROR: Expected {len(texts)} vectors, got {len(vectors)}.")
        return 1
    if len(vectors[0]) != 1024:
        print(f"ERROR: Expected dimension 1024, got {len(vectors[0])}.")
        return 1

    vram_used = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"\nEmbedding dimension: {len(vectors[0])}")
    print(f"VRAM used:           {vram_used:.1f} MB")
    print(f"\nGPU smoke test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
