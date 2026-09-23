# Xây dựng hệ thống hỏi đáp Nghị định 116/2020/NĐ-CP dựa trên Retrieval-Augmented Generation (RAG)

## 1. Giới thiệu dự án

Đây là dự án xây dựng hệ thống hỏi đáp pháp luật tập trung vào **Nghị định 116/2020/NĐ-CP**. Dữ liệu gồm nghị định chính, các văn bản liên quan, **Nghị định 60/2025/NĐ-CP** trong vai trò văn bản sửa đổi khi có quan hệ áp dụng, và **Luật Giáo dục 2019** để bổ sung định nghĩa hoặc ngữ cảnh pháp lý.

Bộ câu hỏi–trả lời về Nghị định 116 chỉ là nguồn tham khảo, không thay thế và không có mức ưu tiên cao hơn nguồn pháp luật chính thức.

Theo định hướng RAG, hệ thống sẽ nhận câu hỏi, truy xuất các đoạn pháp lý liên quan, ưu tiên nguồn có thẩm quyền, xem xét quan hệ giữa văn bản gốc và văn bản sửa đổi, rồi cung cấp câu trả lời dựa trên nội dung đã truy xuất.

Hiện repository đã hoàn thành trọn vẹn pipeline chuẩn bị, import dữ liệu và giai đoạn Vectorization/Indexing (nhúng vector với `BAAI/bge-m3` và tạo chỉ mục HNSW trên PostgreSQL với pgvector). Các thành phần truy vấn (Retrieval), sinh câu trả lời (Generation), giao diện và tích hợp mô hình ngôn ngữ đang trong lộ trình phát triển tiếp theo.

## 2. Mục tiêu

- Chuẩn hóa văn bản pháp lý thành các đoạn (\`legal chunks\`) có cấu trúc.
- Lưu siêu dữ liệu về văn bản, thẩm quyền, ưu tiên truy xuất và quan hệ văn bản.
- Lưu tham chiếu pháp lý ở dạng dữ liệu có cấu trúc.
- Cung cấp quy trình khởi tạo và import có thể chạy lại an toàn.

## 3. Kiến trúc xử lý dữ liệu hiện tại

\`\`\`text
Tệp Markdown pháp lý và tệp hỏi–đáp
        ↓
scripts/legal_chunker.py
        ↓
data/processed/legal_chunks.jsonl
        ↓
scripts/import_legal_data.py
        ↓
init.sql + PostgreSQL
        ↓
documents, legal_chunks, legal_chunk_references
\`\`\`

\`init.sql\` là nguồn sự thật cho schema và quan hệ cơ sở dữ liệu. Script Python thực thi file này, sau đó đọc JSONL từng dòng để cập nhật dữ liệu và tái tạo bảng tham chiếu.

## 4. Nguồn dữ liệu pháp lý

- \`data/markdown/116_2020_ND-CP.md\`: Nghị định 116/2020/NĐ-CP, nguồn chính.
- \`data/markdown/60_2025_ND-CP.md\`: Nghị định 60/2025/NĐ-CP, văn bản sửa đổi liên quan.
- \`data/markdown/43_2019_QH14_Luat_Giao_duc_2019.md\`: Luật Giáo dục 2019, nguồn hỗ trợ.
- \`data/raw/qa/hoi-dap-nghi-dinh-116-2020-nd-cp.md\`: dữ liệu hỏi–đáp tham khảo.
- Các tệp DOCX trong \`data/raw/legal/\`: nguồn thô tương ứng.

Tệp được import là \`data/processed/legal_chunks.jsonl\`, không phải các tệp nguồn thô.

## 5. Công nghệ sử dụng

- Python trong môi trường Conda \`chatbot\`.
- PostgreSQL với extension \`pg_trgm\`, được tạo bởi \`init.sql\` nếu chưa tồn tại.
- \`psql\` để thực thi file SQL có lệnh PostgreSQL \`\\copy\`.
- \`psycopg\` hoặc \`psycopg2\` để kết nối từ Python.
- Định dạng JSONL, mỗi đoạn dữ liệu nằm trên một dòng.

## 6. Yêu cầu môi trường

Cần có Conda, PostgreSQL đang chạy và có thể kết nối bằng \`DATABASE_URL\`, PostgreSQL client \`psql\` trong \`PATH\` của môi trường \`chatbot\`, cùng một trong hai thư viện Python \`psycopg\` hoặc \`psycopg2\`.

Repository không yêu cầu Docker. Các thư viện Python cần thiết được định nghĩa đầy đủ trong `requirements.txt`. PostgreSQL có thể chạy cục bộ hoặc trên máy chủ từ xa (như NeonDB, Supabase); máy chạy script phải truy cập được máy chủ và tài khoản phải có quyền tạo bảng, index và các extension `pg_trgm`, `pgvector`.

## 7. Cấu trúc thư mục liên quan

```text
init.sql
.env.example
requirements.txt
README.md
scripts/
├── import_legal_data.py
├── legal_chunker.py
├── validate_legal_chunks.py
├── embedding.py
├── index_embeddings.py
└── gpu_smoke_test.py
tests/
└── test_embedding.py
data/
├── markdown/
├── raw/
└── processed/
    └── legal_chunks.jsonl
```

## 8. Hướng dẫn cài đặt

### Bước 1: Sao chép dự án và chuyển vào thư mục

```powershell
git clone <URL_KHO_LUU_TRU>
cd RAG-chatbot
```

Nếu đã có mã nguồn:

```powershell
cd <DUONG_DAN_TOI_THU_MUC_RAG-chatbot>
```

### Bước 2: Tạo và kích hoạt môi trường Conda

Chỉ tạo môi trường nếu môi trường chưa tồn tại:

```powershell
conda create -n chatbot python=3.11 -y
conda activate chatbot
```

Cài `psql` (PostgreSQL client) từ conda-forge:

```powershell
conda install -c conda-forge postgresql -y
```

### Bước 3: Cài đặt các thư viện Python (requirements.txt)

Nếu sử dụng GPU NVIDIA CUDA (ví dụ RTX 3050, 40xx):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Nếu chạy CPU thuần:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
### Bước 4: Kiểm tra môi trường

```powershell
where.exe psql
python --version
```

## 9. Cấu hình biến môi trường

Sao chép tệp mẫu `.env.example` thành `.env` ở thư mục gốc và chỉnh sửa cấu hình:

```powershell
copy .env.example .env
```

Nội dung `.env`:

```env
# Cơ sở dữ liệu (Bắt buộc)
DATABASE_URL=postgresql://<ten_nguoi_dung>:<mat_khau>@<may_chu>:<cong>/<ten_co_so_du_lieu>?sslmode=require

# Cấu hình Indexing & Embedding (Tùy chọn / Option)
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BATCH_SIZE=8
HF_TOKEN=

# Cấu hình LLM sinh câu trả lời (Tùy chọn / Option - Phục vụ giai đoạn sau)
GEMINI_API_KEY=
```

Đây chỉ là mẫu; không đưa thông tin đăng nhập thật vào README hoặc đẩy file `.env` lên git (đã được cấu hình trong `.gitignore`). Script ưu tiên biến `DATABASE_URL` trong môi trường hệ thống, sau đó mới đọc `.env`.

Hoặc cấu hình cho phiên PowerShell hiện tại:

\`\`\`powershell
$env:DATABASE_URL = "postgresql://<ten_nguoi_dung>:<mat_khau>@<may_chu>:<cong>/<ten_co_so_du_lieu>"
\`\`\`

## 10. Khởi tạo database và import dữ liệu

Đảm bảo PostgreSQL đang chạy (cục bộ hoặc từ xa), database trong \`DATABASE_URL\` đã tồn tại, và các tệp sau còn nguyên:

\`\`\`text
init.sql
data/processed/legal_chunks.jsonl
scripts/import_legal_data.py
\`\`\`

Chạy từ thư mục gốc:

\`\`\`powershell
conda activate chatbot
python scripts/import_legal_data.py
\`\`\`

Script thực hiện:

\`\`\`text
Kiểm tra cấu hình môi trường và tệp đầu vào
    ↓
Đọc và thực thi init.sql
    ↓
Khởi tạo/cập nhật schema PostgreSQL
    ↓
Đọc data/processed/legal_chunks.jsonl từng dòng
    ↓
Upsert documents và legal_chunks
    ↓
Tái tạo legal_chunk_references
    ↓
Kiểm tra thống kê documents, chunks và references
\`\`\`

Các khóa chính và \`ON CONFLICT\` giúp chạy lại an toàn, không tạo bản ghi trùng. Phần import Python dùng transaction; khi lỗi, transaction được rollback và script kết thúc với thông báo lỗi.

## 11. Kiểm tra dữ liệu sau khi import

Các lệnh sau tương thích với schema hiện tại và chạy trong PowerShell:

\`\`\`powershell
psql --dbname "$env:DATABASE_URL" --command "SELECT count(*) AS documents FROM documents;"
psql --dbname "$env:DATABASE_URL" --command "SELECT count(*) AS legal_chunks FROM legal_chunks;"
psql --dbname "$env:DATABASE_URL" --command "SELECT count(*) AS reference_count FROM legal_chunk_references;"
\`\`\`

Kiểm tra phân bố theo vai trò và mức thẩm quyền:

\`\`\`powershell
psql --dbname "$env:DATABASE_URL" --command "SELECT document_role, count(*) AS chunks FROM legal_chunks GROUP BY document_role ORDER BY document_role;"
psql --dbname "$env:DATABASE_URL" --command "SELECT authority_level, count(*) AS chunks FROM legal_chunks GROUP BY authority_level ORDER BY authority_level;"
\`\`\`

Kiểm tra danh sách văn bản:

\`\`\`powershell
psql --dbname "$env:DATABASE_URL" --command "SELECT document_id, document_number, document_title FROM documents ORDER BY document_id;"
\`\`\`


## 12. Giai đoạn 2: Nhúng văn bản (Indexing / Vectorization)

Giai đoạn này sinh vector nhúng (embedding) cho từng `legal_chunk` và lưu vào cột `embedding vector(1024)` trên bảng `legal_chunks` thông qua extension `pgvector` của PostgreSQL.

### Mô hình nhúng

```text
BAAI/bge-m3
```

Tải tự động từ Hugging Face khi chạy lần đầu (khoảng 2,3 GB). Không yêu cầu xác thực. Mô hình tạo vector **1024 chiều**, chuẩn hóa L2, phù hợp với **cosine similarity**. Hỗ trợ tiếng Việt theo mặc định — không cần dịch hay xử lý đặc biệt.

### Cài đặt thư viện phụ thuộc

Dự án sử dụng tệp `requirements.txt` duy nhất chứa toàn bộ các gói thư viện cần thiết (`psycopg`, `pgvector`, `FlagEmbedding`, `pytest`).

Cài đặt trong môi trường `chatbot`:

```powershell
conda activate chatbot

# Nếu dùng GPU NVIDIA CUDA (khuyến nghị cho tốc độ cao):
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Hoặc nếu chỉ dùng CPU:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install -r requirements.txt
```

Kiểm tra nhận diện GPU và chạy GPU smoke test:

```powershell
# Kiểm tra PyTorch nhận diện GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Chạy smoke test BAAI/bge-m3 trên GPU
python scripts/gpu_smoke_test.py
```

### Kiến trúc module

```text
scripts/embedding.py        ← thành phần thuần nhúng: text → vector
scripts/index_embeddings.py ← script chạy indexing: đọc DB → nhúng → ghi DB
scripts/gpu_smoke_test.py   ← script kiểm tra khả năng tương thích và bộ nhớ GPU
```

`embedding.py` sẽ được tái sử dụng ở giai đoạn Retrieval để nhúng câu hỏi của người dùng với cùng mô hình, không cần thay đổi.

### Chạy indexing

```powershell
conda activate chatbot
# Khuyến nghị --batch-size 4 cho GPU 6GB VRAM (hoặc mặc định 32 nếu chạy CPU)
python scripts/index_embeddings.py --batch-size 4
```

Script sẽ: (1) tải/nạp `BAAI/bge-m3`, (2) chỉ lấy các chunk có `embedding IS NULL`, (3) xử lý nhúng theo từng batch, (4) cập nhật embedding vào database, (5) tự động xác minh tính toàn vẹn của dữ liệu và HNSW index.

### Kết quả Indexing và Nghiệm thu thực tế (Verified Milestone)

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
- **Kích thước vector (Embedding Dimension)**: 1024 chiều, chuẩn hóa L2 từ mô hình `BAAI/bge-m3` (`Dimension correct: True`).
- **Chỉ mục vector (pgvector HNSW Index)**: `legal_chunks_embedding_hnsw_idx` với toán tử `vector_cosine_ops` đã tồn tại và sẵn sàng phục vụ truy vấn tương đồng cosine (`HNSW index exists: True`).
- **Tăng tốc phần cứng**: Kiểm thử và chạy thực tế thành công trên GPU NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) với CUDA 12.8.

### Chi tiết các thông số cấu hình

Hệ thống hỗ trợ cấu hình thông số linh hoạt qua dòng lệnh (CLI), file môi trường (`.env`), mã nguồn module nhúng và chỉ mục cơ sở dữ liệu:

#### 1. Tham số Mô hình & Tiến trình Indexing

| Tham số CLI | Biến môi trường (`.env`) | Mặc định | Mô tả & Tác động |
|---|---|:---:|---|
| `--model NAME` | `EMBEDDING_MODEL` | `BAAI/bge-m3` | Định danh mô hình Hugging Face. Mô hình sinh vector dense 1024 chiều, hỗ trợ đa ngữ tốt với tiếng Việt pháp lý, ngữ cảnh tối đa 8192 tokens. |
| `--batch-size N` | `EMBEDDING_BATCH_SIZE` | `32` (CLI mặc định) | Số chunks xử lý trong một lượt (forward pass). Điều chỉnh theo dung lượng bộ nhớ VRAM của GPU hoặc RAM hệ thống. |
| `--rebuild` | _(không có)_ | `False` | Cờ kích hoạt nhúng lại **toàn bộ 100%** bản ghi. Mặc định tắt (chỉ nhúng các chunk có `embedding IS NULL` để đảm bảo an toàn và khả năng chạy tiếp nối khi bị gián đoạn). |
| _(trong `embedding.py`)_ | `use_fp16` | `True` (khi có CUDA) | Tự động sử dụng Float16 khi chạy trên GPU NVIDIA, tiết kiệm ~50% VRAM và tăng tốc inference đáng kể mà không ảnh hưởng chất lượng vector. Tự về `fp32` trên CPU. |
| _(trong `embedding.py`)_ | `max_length` | `8192` | Giới hạn chiều dài ngữ cảnh token của BGE-M3. Đảm bảo các điều khoản dài của văn bản quy phạm pháp luật không bị cắt cụt (truncate). |

#### 2. Tham số Chỉ mục Vector HNSW (pgvector)

Chỉ mục HNSW được khởi tạo trong `init.sql`:
```sql
CREATE INDEX IF NOT EXISTS legal_chunks_embedding_hnsw_idx
    ON legal_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

| Tham số | Giá trị hiện tại | Khoảng khuyến nghị | Ý nghĩa & Hướng dẫn tinh chỉnh |
|---|:---:|:---:|---|
| **Operator class** | `vector_cosine_ops` | Cosine / L2 / IP | Phép đo khoảng cách Cosine Distance ($1 - \text{cosine similarity}$). Tối ưu nhất cho các vector đã chuẩn hóa L2 của BGE-M3. |
| **`m`** | `16` | `16` – `64` | Số liên kết tối đa trên mỗi node đồ thị HNSW. Tăng `m` giúp tăng độ chính xác tìm kiếm (recall) đối với dữ liệu lớn, đánh đổi bằng việc tốn thêm RAM/dung lượng đĩa. |
| **`ef_construction`** | `64` | `64` – `200` | Kích thước hàng đợi ứng viên khi xây dựng đồ thị index. Tăng giá trị này giúp chất lượng đồ thị tốt hơn, thời gian tạo index ban đầu sẽ dài hơn một chút. |
| **`hnsw.ef_search`** | `40` (mặc định) | `40` – `200` | Kích thước hàng đợi ứng viên lúc **truy vấn (Retrieval runtime)**. Có thể tinh chỉnh linh hoạt trong phiên kết nối (ví dụ: `SET hnsw.ef_search = 100;`) để nâng cao Recall khi truy xuất. |

---

### Hướng dẫn tùy biến và thay đổi tham số

#### Tình huống 1: Điều chỉnh theo cấu hình phần cứng (Tránh lỗi OOM / Tăng tốc)
- **Triệu chứng:** Gặp lỗi `CUDA out of memory` hoặc tiến trình chạy chậm do chưa tận dụng hết tài nguyên.
- **Cách điều chỉnh:**
  - **GPU 4GB – 6GB VRAM** (ví dụ RTX 3050 Laptop): Đặt batch size nhỏ từ `4` đến `8`:
    ```powershell
    python scripts/index_embeddings.py --batch-size 4
    ```
  - **GPU >= 8GB – 16GB VRAM**: Tăng batch size lên `16` – `32` để tối ưu thông lượng:
    ```powershell
    python scripts/index_embeddings.py --batch-size 16
    ```
  - **Cấu hình lâu dài trong `.env`**:
    ```env
    DATABASE_URL=postgresql://...
    EMBEDDING_BATCH_SIZE=4
    ```

#### Tình huống 2: Thử nghiệm hoặc thay đổi mô hình Embedding khác
Khi muốn chuyển sang một mô hình embedding khác (ví dụ mô hình tiếng Việt chuyên dụng hoặc mô hình nhẹ hơn):
1. **Kiểm tra số chiều vector của mô hình mới:**
   - Nếu mô hình mới có số chiều **khác 1024** (ví dụ `768` chiều):
     ```powershell
     # Xóa index HNSW cũ
     psql --dbname "$env:DATABASE_URL" --command "DROP INDEX IF EXISTS legal_chunks_embedding_hnsw_idx;"
     # Thay đổi kiểu cột embedding sang số chiều mới (ví dụ 768)
     psql --dbname "$env:DATABASE_URL" --command "ALTER TABLE legal_chunks ALTER COLUMN embedding TYPE vector(768);"
     ```
   - Cập nhật hằng số `EMBEDDING_DIM = 768` trong `scripts/embedding.py` và `init.sql`.
2. **Chạy nhúng lại toàn bộ với cờ `--rebuild`:**
   ```powershell
   python scripts/index_embeddings.py --rebuild --model "ten-to-chuc/ten-mo-hinh-moi" --batch-size 8
   ```
3. **Tạo lại chỉ mục HNSW tương ứng:**
   ```powershell
   psql --dbname "$env:DATABASE_URL" --command "CREATE INDEX legal_chunks_embedding_hnsw_idx ON legal_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
   ```

#### Tình huống 3: Nâng cao độ chính xác truy xuất (Retrieval Tuning)
Khi bước sang giai đoạn Retrieval, nếu muốn ưu tiên độ chính xác tuyệt đối (Recall) thay vì tốc độ phản hồi cực đoan:
- **Lúc tạo index:** Nâng `m` và `ef_construction`:
  ```sql
  DROP INDEX IF EXISTS legal_chunks_embedding_hnsw_idx;
  CREATE INDEX legal_chunks_embedding_hnsw_idx
      ON legal_chunks USING hnsw (embedding vector_cosine_ops)
      WITH (m = 32, ef_construction = 128);
  ```
- **Lúc truy vấn runtime:** Tăng `ef_search` trước khi query câu hỏi của người dùng:
  ```sql
  SET hnsw.ef_search = 100;
  -- Thực hiện truy vấn SELECT ... ORDER BY embedding <=> $query_vector LIMIT 5;
  ```

---

### Xác minh kết quả indexing và chỉ mục vector

Kiểm tra số lượng bản ghi đã nhúng, số chiều vector và sự hiện diện của chỉ mục HNSW:

```powershell
# 1. Kiểm tra tổng số chunk và số chunk đã có vector nhúng
psql --dbname "$env:DATABASE_URL" --command "SELECT count(*) AS total, count(embedding) AS embedded FROM legal_chunks;"

# 2. Kiểm tra số chiều thực tế của vector (kỳ vọng: 1024)
psql --dbname "$env:DATABASE_URL" --command "SELECT chunk_id, vector_dims(embedding) AS dim FROM legal_chunks WHERE embedding IS NOT NULL LIMIT 3;"

# 3. Kiểm tra chỉ mục HNSW đã được kích hoạt
psql --dbname "$env:DATABASE_URL" --command "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'legal_chunks' AND indexname = 'legal_chunks_embedding_hnsw_idx';"
```

### Quan hệ dữ liệu

```text
documents
    └── legal_chunks
            ├── text               ← nội dung pháp lý gốc (không thay đổi)
            ├── [metadata fields]  ← article, clause, authority_level, …
            ├── search_vector      ← full-text index hiện có
            └── embedding          ← ĐÃ HOÀN TẤT: vector 1024 chiều (BAAI/bge-m3)
                                           ↓
                                    legal_chunks_embedding_hnsw_idx
                                    (HNSW, vector_cosine_ops)
```

### Chạy unit test

```powershell
conda activate chatbot
python -m pytest tests/test_embedding.py -v
```

Toàn bộ test suite (16/16 test cases) xác minh tính toàn vẹn của logic nhúng, kiểm tra kích thước vector, xử lý batch và cơ chế tương thích schema.

---

## 13. Giai đoạn 3: Retrieval (Vector Baseline)

Giai đoạn này triển khai nền tảng truy xuất vector: nhận câu hỏi tự nhiên, chuyển thành vector 1024 chiều bằng `BAAI/bge-m3`, và tìm kiếm cosine similarity qua chỉ mục HNSW trên PostgreSQL + pgvector.

> **Phạm vi của milestone này:** chỉ Vector Retrieval thuần túy.  
> Hybrid Retrieval, RRF, Re-ranking, LLM Generation và Frontend là các bước tiếp theo.

### Kiến trúc Retrieval

```text
Câu hỏi pháp lý (tiếng Việt)
          ↓
       BGE-M3
    (BAAI/bge-m3)
          ↓
   Vector 1024 chiều
          ↓
  PostgreSQL + pgvector
   (cosine distance <=>)
          ↓
    HNSW index search
          ↓
   Top-K chunks (có điểm similarity)
```

### Module mới

```text
scripts/
├── retrieval.py       ← module retrieval chính: Retriever + RetrievalResult
├── test_retrieval.py  ← script đánh giá thực tế với 15 câu hỏi pháp lý
tests/
└── test_retrieval.py  ← bộ unit test (38 test cases, không cần DB thật)
```

### Chạy retrieval đánh giá (yêu cầu DATABASE_URL)

```powershell
# Chạy toàn bộ 15 câu hỏi đánh giá mặc định (top-5)
python scripts/test_retrieval.py

# Tuỳ chỉnh top-K và ef_search
python scripts/test_retrieval.py --top-k 10 --ef-search 80

# Câu hỏi tuỳ chọn (ad-hoc)
python scripts/test_retrieval.py --query "Điều kiện để được hưởng chính sách hỗ trợ là gì?"
```

### Sử dụng trong code

```python
from scripts.retrieval import Retriever

retriever = Retriever()   # tải BGE-M3 một lần
results = retriever.retrieve(
    "Điều kiện để được hưởng chính sách hỗ trợ là gì?",
    top_k=5,
)
for r in results:
    print(r.chunk_id, f"{r.score:.4f}", r.metadata["article"])
    print(r.text[:200])
```

### Cấu trúc kết quả

```python
@dataclass
class RetrievalResult:
    chunk_id: str          # khoá chính trong legal_chunks
    text:     str          # nội dung pháp lý
    score:    float        # cosine similarity (giá trị lớn hơn = liên quan hơn)
    metadata: dict         # document_id, document_title, document_number,
                           # source_type, document_role, authority_level,
                           # retrieval_priority, article, clause, point,
                           # content_type
```

### Các tham số cấu hình

| Tham số | Mặc định | Mô tả |
|---|:---:|---|
| `top_k` | `5` | Số chunk trả về. Phải là số nguyên dương ≤ 1000. |
| `ef_search` | `40` | HNSW ef_search — tăng để nâng recall, giảm tốc độ. Phạm vi khuyến nghị: 40–200. |

`ef_search` được thiết lập mỗi kết nối bằng `SELECT set_config('hnsw.ef_search', ..., true)` (transaction-scoped, không ảnh hưởng phiên khác).

### Chiến lược kết nối (NeonDB-safe)

Kế thừa đúng pattern từ `index_embeddings.py`:

```text
1. embed_texts([query])   ← không có DB connection
2. with connect(url) as conn:   ← kết nối ngắn hạn
3.     set_config ef_search
4.     SELECT ... ORDER BY embedding <=> vector LIMIT k
5. # connection tự đóng
```

Embedding model (`BGE-M3`) được nạp **một lần** khi khởi tạo `Retriever`, không nạp lại theo từng truy vấn.

### Chạy unit test

```powershell
python -m pytest tests/ -v
```

Kết quả kỳ vọng: **38/38 passed** (16 test embedding + 22 test retrieval), không cần DB thật hay tải model.

### Xác minh trạng thái database (read-only)

```python
retriever = Retriever()
state = retriever.verify_database_state()
print(state)
# {'total_chunks': 617, 'embedded_chunks': 617,
#  'missing_embeddings': 0, 'hnsw_index_exists': True}
```

Phương thức này chỉ đọc — **không sửa đổi embedding hay index**.

### Giới hạn đã biết và bước tiếp theo

- Milestone này chỉ là **Vector Retrieval baseline**. Chất lượng truy xuất cần được đánh giá thủ công bằng `test_retrieval.py`.
- **Chưa triển khai**: Hybrid Retrieval (kết hợp BM25 / Full-text search), RRF, Cross-encoder Re-ranking, LLM Generation.
- Bước tiếp theo khuyến nghị: đánh giá chất lượng retrieval baseline → Hybrid Retrieval → Re-ranking.

---

## 14. Tiến trình dự án và Trạng thái phát triển

| Giai đoạn | Nội dung | Trạng thái | Ghi chú |
|---|---|:---:|---|
| **1. Data Preparation & Import** | Thu thập, chuẩn hóa markdown, chunking và nạp vào PostgreSQL | ✅ Hoàn thành | 617 chunks pháp lý, quan hệ tham chiếu đầy đủ |
| **2. Indexing / Vectorization** | Nhúng vector với `BAAI/bge-m3` (1024 dims), tạo HNSW index | ✅ Hoàn thành | 617/617 chunks đã nhúng (0 chunk lỗi), chạy trên GPU RTX 3050 |
| **3. Retrieval – Vector Baseline** | BGE-M3 query embedding + cosine vector search (HNSW) | ✅ Hoàn thành | `retrieval.py` + 22 unit tests (38 total). Hybrid search & re-ranking là bước tiếp theo. |
| **4. Retrieval – Hybrid & Re-ranking** | Full-text search + vector fusion (RRF), cross-encoder re-ranking | 🔄 Kế hoạch tiếp theo | Cơ sở `search_vector` và HNSW index đã sẵn sàng |
| **5. Generation & Synthesis** | Tích hợp LLM sinh câu trả lời trích dẫn điều khoản chính xác | ⏳ Dự kiến | Ưu tiên văn bản có hiệu lực và quan hệ sửa đổi bổ sung |
| **6. Interface & Evaluation** | UI người dùng hỏi đáp và bộ dữ liệu đánh giá chất lượng RAG | ⏳ Dự kiến | |
