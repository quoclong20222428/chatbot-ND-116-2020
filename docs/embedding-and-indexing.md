# Embedding và Indexing (Vector hóa)

Tài liệu này mô tả giai đoạn sinh vector nhúng (embedding) cho toàn bộ legal chunks và tạo chỉ mục HNSW trên PostgreSQL với pgvector.

---

## Tổng quan

Giai đoạn này sinh vector nhúng cho từng `legal_chunk` và lưu vào cột `embedding vector(1024)` trên bảng `legal_chunks` thông qua extension `pgvector` của PostgreSQL.

### Mô hình nhúng

```text
BAAI/bge-m3
```

- Tải tự động từ Hugging Face khi chạy lần đầu (khoảng 2,3 GB).
- Không yêu cầu xác thực.
- Tạo vector **1024 chiều**, chuẩn hóa L2, phù hợp với **cosine similarity**.
- Hỗ trợ tiếng Việt theo mặc định — không cần dịch hay xử lý đặc biệt.
- Ngữ cảnh tối đa: **8192 tokens** — đảm bảo các điều khoản dài không bị cắt cụt.

### Kiến trúc module

```text
scripts/embeddings/embedding.py          ← implementation: text → vector, được import
scripts/indexing/embedding_index.py      ← implementation indexing, được import
scripts/index_embeddings.py              ← CLI indexing: đọc DB → nhúng → ghi DB
scripts/gpu_smoke_test.py                ← CLI kiểm tra tương thích và bộ nhớ GPU
```

`scripts/embeddings/embedding.py` là implementation được indexing và HNSW retriever import; không chạy module này như một lệnh độc lập. Model mặc định và registry được định nghĩa trong `scripts/embeddings/model_registry.py`.

---

## Kiểm tra môi trường GPU

```powershell
conda activate chatbot

# Kiểm tra PyTorch nhận diện GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Chạy smoke test BAAI/bge-m3 trên GPU
python scripts/gpu_smoke_test.py
```

---

## Chạy Indexing

```powershell
conda activate chatbot
# Mặc định dùng model BAAI/bge-m3 và batch size 8; điều chỉnh batch theo bộ nhớ.
python scripts/index_embeddings.py --batch-size 4
```

Indexing đọc chunks đã import từ PostgreSQL, tạo embedding model-specific và ghi vector/cột cùng HNSW index vào database. Chạy sau database import. Chọn model bằng `--model NAME` hoặc `EMBEDDING_MODEL` trong `.env`; mặc định là `BAAI/bge-m3`. `--preview` chỉ xem trước chunks và không tải model hay ghi embeddings.

Script thực hiện theo thứ tự:

1. Tải/nạp model đã chọn (`BAAI/bge-m3` mặc định)
2. Chỉ lấy các chunk có `embedding IS NULL`
3. Xử lý nhúng theo từng batch
4. Cập nhật embedding vào database
5. Tự động xác minh tính toàn vẹn dữ liệu và HNSW index

---

## Kết quả Indexing (Verified Milestone)

Quá trình nhúng vector toàn bộ văn bản pháp lý đã hoàn tất 100%:

```text
INFO: --- Indexing summary ---
INFO: Successfully indexed: 521
INFO: Failed:               0
INFO: --- Verification ---
INFO: Total chunks:         617
INFO: Embedded chunks:      617
INFO: Missing embeddings:   0
INFO: Embedding dimension:  1024
INFO: Dimension correct:    True
INFO: HNSW index exists:    True
INFO: Indexing stage complete. Database is ready for retrieval.
```

**Chi tiết nghiệm thu:**

- **Độ phủ dữ liệu**: Toàn bộ **617/617** legal chunks đã được vector hóa thành công (tỷ lệ thành công 100%, 0 lỗi).
- **Tính năng tiếp nối (Idempotent / Resume)**: Script tự động phát hiện 96 chunks đã nhúng trước đó và chỉ xử lý 521 chunks còn thiếu mà không gây trùng lặp hay ghi đè sai lệch.
- **Kích thước vector**: 1024 chiều, chuẩn hóa L2 từ `BAAI/bge-m3` (`Dimension correct: True`).
- **Chỉ mục HNSW**: `legal_chunks_embedding_hnsw_idx` với toán tử `vector_cosine_ops` đã tồn tại và sẵn sàng phục vụ truy vấn tương đồng cosine (`HNSW index exists: True`).
- **Phần cứng**: Kiểm thử và chạy thực tế thành công trên GPU NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) với CUDA 12.8.

---

## Tham số cấu hình

### Tham số Mô hình & Tiến trình Indexing

| Tham số CLI | Biến môi trường (`.env`) | Mặc định | Mô tả |
|---|---|:---:|---|
| `--model NAME` | `EMBEDDING_MODEL` | `BAAI/bge-m3` | Định danh mô hình Hugging Face. |
| `--batch-size N` | `EMBEDDING_BATCH_SIZE` | `8` | Số chunks xử lý trong một lượt. Điều chỉnh theo dung lượng VRAM/RAM. |
| `--rebuild` | _(không có)_ | `False` | Nhúng lại **toàn bộ** chunk và cập nhật cột embedding của model đã chọn. Mặc định tắt (chỉ nhúng chunk chưa có vector trong cột đó). |
| _(trong `embedding.py`)_ | `use_fp16` | `True` (khi có CUDA) | Float16 khi chạy trên GPU NVIDIA, tiết kiệm ~50% VRAM. Tự về `fp32` trên CPU. |
| _(trong `embedding.py`)_ | `max_length` | `8192` | Giới hạn chiều dài ngữ cảnh token của BGE-M3. |

### Tham số Chỉ mục Vector HNSW (pgvector)

HNSW (Hierarchical Navigable Small World): cấu trúc chỉ mục giúp tìm kiếm vector tương đồng nhanh hơn trong cơ sở dữ liệu.

Chỉ mục HNSW được khởi tạo trong `init.sql`:

```sql
CREATE INDEX IF NOT EXISTS legal_chunks_embedding_hnsw_idx
    ON legal_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

| Tham số | Giá trị hiện tại | Khoảng khuyến nghị | Ý nghĩa |
|---|:---:|:---:|---|
| **Operator class** | `vector_cosine_ops` | Cosine / L2 / IP | Phép đo Cosine Distance. Tối ưu cho vector đã chuẩn hóa L2 của BGE-M3. |
| **`m`** | `16` | `16` – `64` | Số liên kết tối đa trên mỗi node đồ thị HNSW. Tăng `m` → tăng độ chính xác, tốn thêm RAM. |
| **`ef_construction`** | `64` | `64` – `200` | Kích thước hàng đợi ứng viên khi xây dựng đồ thị index. Tăng → chất lượng đồ thị tốt hơn, thời gian tạo index lâu hơn. |
| **`hnsw.ef_search`** | `40` (mặc định) | `40` – `200` | Kích thước hàng đợi ứng viên lúc **truy vấn runtime**. Có thể tinh chỉnh linh hoạt trong phiên kết nối. |

---

## Hướng dẫn tùy biến tham số

### Tình huống 1: Điều chỉnh theo cấu hình phần cứng (Tránh lỗi OOM / Tăng tốc)

- **Triệu chứng:** Gặp lỗi `CUDA out of memory` hoặc tiến trình chạy chậm.
- **GPU 4GB – 6GB VRAM** (ví dụ RTX 3050 Laptop): Đặt batch size nhỏ từ `4` đến `8`:
  ```powershell
  conda activate chatbot
  python scripts/index_embeddings.py --batch-size 4
  ```
- **GPU >= 8GB – 16GB VRAM**: Tăng batch size lên `16` – `32`:
  ```powershell
  conda activate chatbot
  python scripts/index_embeddings.py --batch-size 16
  ```
- **Cấu hình lâu dài trong `.env`**:
  ```env
  DATABASE_URL=postgresql://...
  EMBEDDING_BATCH_SIZE=4
  ```

### Tình huống 2: Thử nghiệm mô hình Embedding khác

Chọn một model đã đăng ký bằng `--model` hoặc cấu hình `EMBEDDING_MODEL`; model được indexing và retrieval chọn phải giống nhau. Các model hiện được hỗ trợ đều dùng vector 1024 chiều. Indexing tự quản lý cột embedding và HNSW index theo model, nên không cần thay đổi schema thủ công.

```powershell
conda activate chatbot
python scripts/index_embeddings.py --model vietlegal-e5 --rebuild --batch-size 8
```

`--rebuild` tính lại embedding cho toàn bộ chunks trong cột của model được chọn. Dùng cờ này khi chủ động muốn ghi đè các vector hiện có.

---

## Xác minh kết quả Indexing

```powershell
# 1. Kiểm tra tổng số chunk và số chunk đã có vector nhúng
psql --dbname "$env:DATABASE_URL" --command "SELECT count(*) AS total, count(embedding) AS embedded FROM legal_chunks;"

# 2. Kiểm tra số chiều thực tế của vector (kỳ vọng: 1024)
psql --dbname "$env:DATABASE_URL" --command "SELECT chunk_id, vector_dims(embedding) AS dim FROM legal_chunks WHERE embedding IS NOT NULL LIMIT 3;"

# 3. Kiểm tra chỉ mục HNSW đã được kích hoạt
psql --dbname "$env:DATABASE_URL" --command "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'legal_chunks' AND indexname = 'legal_chunks_embedding_hnsw_idx';"
```

---

## Unit Test — Embedding

```powershell
conda activate chatbot
python -m pytest tests/test_embedding.py -v
```

Toàn bộ test suite (**91/91 test cases**) xác minh tính toàn vẹn của logic nhúng, kiểm tra kích thước vector, xử lý batch và cơ chế tương thích schema. Bao gồm các test cho tất cả 6 mô hình: BGE-M3, vnlegal-lal, vietlegal-harrier, vietlegal-e5, Jina v3, và **DeepX**. Không yêu cầu model hay DB thật.

---

## Xem thêm

- [Chuẩn bị dữ liệu](data-preparation.md)
- [Cơ sở dữ liệu và Import](database-and-import.md)
- [Chuyển đổi mô hình Embedding](embedding-model-switching.md)
- [DeepX Embedding v1](deepx-embedding.md)
- [Retrieval](retrieval.md)
- [Lịch sử phát triển Retrieval](retrieval-development-history.md)
- [Quay lại README](../README.md)
