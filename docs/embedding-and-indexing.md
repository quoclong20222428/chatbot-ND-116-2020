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
scripts/embedding.py        ← thành phần nhúng thuần túy: text → vector
scripts/index_embeddings.py ← script chạy indexing: đọc DB → nhúng → ghi DB
scripts/gpu_smoke_test.py   ← script kiểm tra khả năng tương thích và bộ nhớ GPU
```

`embedding.py` được tái sử dụng ở giai đoạn Retrieval để nhúng câu hỏi của người dùng với cùng mô hình, không cần thay đổi.

---

## Kiểm tra môi trường GPU

```powershell
# Kiểm tra PyTorch nhận diện GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Chạy smoke test BAAI/bge-m3 trên GPU
python scripts/gpu_smoke_test.py
```

---

## Chạy Indexing

```powershell
conda activate chatbot
# Khuyến nghị --batch-size 4 cho GPU 6GB VRAM (hoặc mặc định 32 nếu chạy CPU)
python scripts/index_embeddings.py --batch-size 4
```

Script thực hiện theo thứ tự:

1. Tải/nạp `BAAI/bge-m3`
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
| `--batch-size N` | `EMBEDDING_BATCH_SIZE` | `32` | Số chunks xử lý trong một lượt. Điều chỉnh theo dung lượng VRAM/RAM. |
| `--rebuild` | _(không có)_ | `False` | Nhúng lại **toàn bộ 100%** bản ghi. Mặc định tắt (chỉ nhúng chunk có `embedding IS NULL`). |
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
  python scripts/index_embeddings.py --batch-size 4
  ```
- **GPU >= 8GB – 16GB VRAM**: Tăng batch size lên `16` – `32`:
  ```powershell
  python scripts/index_embeddings.py --batch-size 16
  ```
- **Cấu hình lâu dài trong `.env`**:
  ```env
  DATABASE_URL=postgresql://...
  EMBEDDING_BATCH_SIZE=4
  ```

### Tình huống 2: Thử nghiệm mô hình Embedding khác

Khi muốn chuyển sang mô hình embedding khác:

1. **Nếu mô hình mới có số chiều khác 1024** (ví dụ `768` chiều):
   ```powershell
   # Xóa index HNSW cũ
   psql --dbname "$env:DATABASE_URL" --command "DROP INDEX IF EXISTS legal_chunks_embedding_hnsw_idx;"
   # Thay đổi kiểu cột embedding sang số chiều mới
   psql --dbname "$env:DATABASE_URL" --command "ALTER TABLE legal_chunks ALTER COLUMN embedding TYPE vector(768);"
   ```
   Cập nhật hằng số `EMBEDDING_DIM = 768` trong `scripts/embedding.py` và `init.sql`.

2. **Chạy nhúng lại toàn bộ với cờ `--rebuild`:**
   ```powershell
   python scripts/index_embeddings.py --rebuild --model "ten-to-chuc/ten-mo-hinh-moi" --batch-size 8
   ```

3. **Tạo lại chỉ mục HNSW tương ứng:**
   ```powershell
   psql --dbname "$env:DATABASE_URL" --command "CREATE INDEX legal_chunks_embedding_hnsw_idx ON legal_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
   ```

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

Toàn bộ test suite (**30/30 test cases**) xác minh tính toàn vẹn của logic nhúng, kiểm tra kích thước vector, xử lý batch và cơ chế tương thích schema. Không yêu cầu model hay DB thật.

---

## Xem thêm

- [Chuẩn bị dữ liệu](data-preparation.md)
- [Cơ sở dữ liệu và Import](database-and-import.md)
- [Retrieval](retrieval.md)
- [Lịch sử phát triển Retrieval](retrieval-development-history.md)
- [Quay lại README](../README.md)
