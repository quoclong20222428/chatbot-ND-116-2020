# Hướng dẫn chuyển đổi mô hình embedding

Tài liệu này mô tả cách chuyển đổi giữa các mô hình embedding trong hệ
thống RAG chatbot luật pháp.  Việc thay đổi mô hình là một **thao tác cấu
hình** — không cần sửa mã nguồn.

---

## Mục lục

1. [Các mô hình được hỗ trợ](#các-mô-hình-được-hỗ-trợ)
2. [Chuyển đổi mô hình](#chuyển-đổi-mô-hình)
3. [Kiến trúc kỹ thuật](#kiến-trúc-kỹ-thuật)
4. [Chạy thí nghiệm so sánh](#chạy-thí-nghiệm-so-sánh)
5. [Câu hỏi thường gặp (FAQ)](#câu-hỏi-thường-gặp)

---

## Các mô hình được hỗ trợ

Tất cả các mô hình đều sinh vector **1024 chiều** và sử dụng **cosine
similarity** để tìm kiếm.

| Alias | Hugging Face ID | Backend | Max tokens | Ghi chú |
|---|---|---|---|---|
| `bge-m3` | `BAAI/bge-m3` | FlagEmbedding | 8192 | Mặc định. Đa ngữ. |
| `vnlegal-lal` | `darklethelong/vnlegal-lal` | SentenceTransformer | 2048 | Fine-tuned cho luật VN. Decoder-only. |
| `vietlegal-harrier` | `mainguyen9/vietlegal-harrier-0.6b` | SentenceTransformer | 512 | Dựa trên Microsoft Harrier 0.6B. |
| `vietlegal-e5` | `mainguyen9/vietlegal-e5` | SentenceTransformer | 512 | E5, yêu cầu prefix `query:`/`passage:`. |
| `jina-v3` | `jinaai/jina-embeddings-v3-hf` | Jina (LoRA) | 8192 | LoRA adapter riêng cho query/passage. |
| `deepx` | `dxtech-asia/deepx-embedding-v1` | DeepX (deepx_embed) | 8192 | GDN-2 linear-attention, Matryoshka 1024d. |

---

## Chuyển đổi mô hình

### Bước 1: Cập nhật `.env`

Thay đổi biến `EMBEDDING_MODEL` trong file `.env`:

```dotenv
# Dùng alias ngắn:
EMBEDDING_MODEL=jina-v3

# Hoặc tên đầy đủ từ Hugging Face:
EMBEDDING_MODEL=jinaai/jina-embeddings-v3-hf
```

### Bước 2: Sinh embedding cho mô hình mới

```powershell
conda activate chatbot
python scripts/index_embeddings.py --model jina-v3
```

Script tự động:
- Tạo cột `embedding_jina_v3` trong bảng `legal_chunks` (nếu chưa có).
- Tạo HNSW index tương ứng.
- Sinh embedding cho các chunk chưa có vector trong cột đó.

### Bước 3: Chạy retrieval / đánh giá

```powershell
conda activate chatbot
# Retrieval sẽ tự động đọc EMBEDDING_MODEL từ .env
python scripts/test_retrieval.py
```

> **Quan trọng:** Không cần sửa bất kỳ file mã nguồn nào.  Hệ thống tự động
> chọn đúng cột vector và HNSW index dựa trên cấu hình mô hình.

---

## Kiến trúc kỹ thuật

### Sơ đồ tổng quan

```
.env (EMBEDDING_MODEL=...)
        │
        ▼
  scripts/embeddings/model_registry.py   ← Registry các model được hỗ trợ
        │
        ▼
   scripts/embeddings/embedding.py       ← Chọn backend phù hợp
   ┌────┼────┬─────┐
   ▼    ▼    ▼     ▼
  BGE  ST  Jina  DeepX  ← 4 backend
   │    │    │     │
   └────┼────┴─────┘
        ▼
  embed_query()        ← Mã hoá query (có prefix/LoRA nếu cần)
  embed_documents()    ← Mã hoá document (có prefix/LoRA nếu cần)
        │
        ▼
  legal_chunks         ← Mỗi mô hình ghi vào cột riêng
  ┌─────────────────────────────────┐
  │ embedding           (bge-m3)   │
  │ embedding_vnlegal_lal          │
  │ embedding_vietlegal_harrier    │
  │ embedding_vietlegal_e5         │
  │ embedding_jina_v3              │
  │ embedding_deepx                │
  └─────────────────────────────────┘
```

### Cách ly vector giữa các mô hình

Mỗi mô hình embedding ghi vector vào **cột riêng** trong bảng `legal_chunks`.
Điều này đảm bảo:

- **Không thể trộn lẫn vector** từ hai mô hình khác nhau trong cùng một thí
  nghiệm retrieval.
- Có thể **so sánh A/B** giữa các mô hình mà không cần xoá embedding cũ.
- Cột `embedding` (không prefix) luôn tương thích ngược với BGE-M3.

### Giao thức mã hoá (encoding protocol)

Mỗi mô hình có giao thức mã hoá riêng, được xử lý tự động:

| Mô hình | Query | Document |
|---|---|---|
| BGE-M3 | Gọi `encode()` trực tiếp | Gọi `encode()` trực tiếp |
| VNLegal-LAL | Gọi `encode()` trực tiếp | Gọi `encode()` trực tiếp |
| VietLegal-Harrier | Gọi `encode()` trực tiếp | Gọi `encode()` trực tiếp |
| VietLegal-E5 | Thêm prefix `"query: "` | Thêm prefix `"passage: "` |
| Jina v3 | LoRA adapter `retrieval.query` | LoRA adapter `retrieval.passage` |
| DeepX | `DeepXEmbed.encode(truncate_dim=1024)` | `DeepXEmbed.encode(truncate_dim=1024)` |

### Kiểm tra chiều (dimension validation)

Hệ thống **kiểm tra chiều vector** sau mỗi lần encode.  Nếu vector không đúng
1024 chiều, sẽ raise `ValueError` ngay lập tức — trước khi ghi vào database.

---

## Chạy thí nghiệm so sánh

### Chạy từng mô hình một

```powershell
conda activate chatbot

# 1. Sinh embedding cho mô hình cần thử
python scripts/index_embeddings.py --model vietlegal-e5

# 2. Chọn cùng model cho retrieval/evaluation trong PowerShell hiện tại
$env:EMBEDDING_MODEL = "vietlegal-e5"

# 3. Chạy đánh giá
python scripts/test_retrieval.py --top-k 10 --ef-search 80
```

### So sánh nhanh hai mô hình

```powershell
conda activate chatbot

# Model A (chỉ khi cột embedding BGE-M3 đã được tạo)
$env:EMBEDDING_MODEL = "bge-m3"
python scripts/test_retrieval.py --top-k 10 --ef-search 80

# Mô hình B
python scripts/index_embeddings.py --model jina-v3
$env:EMBEDDING_MODEL = "jina-v3"
python scripts/test_retrieval.py --top-k 10 --ef-search 80
```

Mỗi lần chạy tự ghi báo cáo riêng vào `logs/`, tên file có model, thời điểm chạy và hậu tố duy nhất. CLI đánh giá hiện tại dùng HNSW; không có tùy chọn `--output-dir` hoặc chọn BM25.

---

## Câu hỏi thường gặp

### Q: Tôi có cần xoá embedding cũ khi chuyển mô hình không?

**Không.**  Mỗi mô hình ghi vào cột riêng.  Embedding cũ vẫn còn nguyên.
Bạn có thể quay lại mô hình cũ bất kỳ lúc nào bằng cách đổi
`EMBEDDING_MODEL` trong `.env`.

### Q: Tôi có thể chạy hai mô hình cùng lúc không?

**Không nên** trên GPU 6GB (RTX 3050).  Mỗi mô hình cần 1–3 GB VRAM.  Hãy
chạy tuần tự: sinh embedding cho mô hình A, đánh giá, rồi chuyển sang mô hình
B.

### Q: Thêm mô hình mới cần sửa file nào?

Chỉ cần sửa **một file**: `scripts/embeddings/model_registry.py`.  Thêm một `ModelConfig`
mới vào danh sách `_MODELS`.  Nếu mô hình mới dùng backend chưa có (không phải
BGE, SentenceTransformer, hay Jina), thì cần thêm backend class trong
`scripts/embeddings/embedding.py`.

### Q: Sao cột BGE-M3 tên là `embedding` mà không phải `embedding_bge_m3`?

Để **tương thích ngược** với schema cũ.  Cột `embedding` đã tồn tại trước khi
refactor.  Tất cả mô hình khác dùng prefix `embedding_<alias>`.

### Q: Tôi cần cài thêm thư viện nào?

- Cho BGE-M3: `pip install FlagEmbedding`
- Cho các mô hình khác (trừ DeepX): `pip install sentence-transformers>=3.0.0`
- Cho DeepX: `pip install git+https://github.com/dx-tech-ai/deepx-embed.git`
- Hoặc: `pip install -r requirements.txt`

### Q: Tại sao DeepX không dùng SentenceTransformer backend?

DeepX sử dụng kiến trúc **Gated DeltaNet-2 (GDN-2)** tùy chỉnh hoàn toàn,
chưa được đăng ký trong `transformers` AutoModel registry.  File `config.json`
của model không có `auto_map`, nên `SentenceTransformer("dxtech-asia/...")`
sẽ báo lỗi:

```
The checkpoint ... has model type 'deepx-embedding' but
Transformers does not recognize this architecture.
```

Giải pháp là dùng package chính thức `deepx_embed` (`DeepXEmbed.from_pretrained`)
bỏ qua hoàn toàn `SentenceTransformer`.  Điều này không thay đổi phiên bản
`transformers` hay `sentence-transformers` đã cài.

### Q: Lệnh `--rebuild` làm gì?

`python scripts/index_embeddings.py --model bge-m3 --rebuild` sẽ tính lại và
ghi đè embedding cho toàn bộ chunks trong cột của model đó. Thao tác này tốn
thời gian và không thay đổi cột embedding của model khác.
