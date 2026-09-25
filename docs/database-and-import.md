# Cơ sở dữ liệu và Import dữ liệu

Tài liệu này mô tả schema cơ sở dữ liệu PostgreSQL, quy trình import dữ liệu và các lệnh xác minh sau khi import.

---

## Schema cơ sở dữ liệu

Schema PostgreSQL được định nghĩa đầy đủ trong `init.sql`. File này là nguồn sự thật cho toàn bộ cấu trúc bảng, quan hệ, index và extension.

Các extension PostgreSQL được tạo tự động nếu chưa tồn tại:
- `pg_trgm` — hỗ trợ tìm kiếm trigram (trigram search)
- `pgvector` — hỗ trợ lưu trữ và tìm kiếm vector nhúng

**Quan hệ dữ liệu hiện tại:**

```text
documents
    └── legal_chunks
            ├── text               ← nội dung pháp lý gốc (không thay đổi)
            ├── [metadata fields]  ← article, clause, authority_level, …
            ├── search_vector      ← full-text index (đã có sẵn trong schema)
            └── embedding          ← vector 1024 chiều (BAAI/bge-m3)
                                           ↓
                                    legal_chunks_embedding_hnsw_idx
                                    (HNSW, vector_cosine_ops)

legal_chunk_references             ← tham chiếu pháp lý giữa các văn bản
```

---

## Quy trình Import dữ liệu

### Điều kiện tiên quyết

Trước khi chạy import, đảm bảo:

```text
init.sql
data/processed/legal_chunks.jsonl
scripts/import_legal_data.py
```

đều còn nguyên, và PostgreSQL đang chạy với `DATABASE_URL` được cấu hình.

### Chạy import

Lệnh này import chính tệp JSONL đã được tạo và validation ở bước chuẩn bị dữ liệu:
`data/processed/legal_chunks.jsonl` → PostgreSQL. Import chạy `init.sql` để khởi tạo schema, sau đó upsert documents/chunks và tái tạo references; vì vậy lệnh này thay đổi database. Chạy chunking và validation trước, đồng thời bảo đảm PostgreSQL, `psql` và `DATABASE_URL` sẵn sàng.

```powershell
conda activate chatbot
python scripts/import_legal_data.py
```

### Luồng thực thi

```text
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
```

**Tính an toàn khi chạy lại:** Các khóa chính và `ON CONFLICT` đảm bảo chạy lại không tạo bản ghi trùng. Phần import Python dùng transaction; khi lỗi, transaction được rollback và script kết thúc với thông báo lỗi.

---

## Xác minh dữ liệu sau khi import

Các lệnh sau chạy trong PowerShell:

```powershell
# Số lượng văn bản
psql --dbname "$env:DATABASE_URL" --command "SELECT count(*) AS documents FROM documents;"

# Số lượng chunks
psql --dbname "$env:DATABASE_URL" --command "SELECT count(*) AS legal_chunks FROM legal_chunks;"

# Số lượng tham chiếu
psql --dbname "$env:DATABASE_URL" --command "SELECT count(*) AS reference_count FROM legal_chunk_references;"
```

Kiểm tra phân bố theo vai trò và mức thẩm quyền:

```powershell
psql --dbname "$env:DATABASE_URL" --command "SELECT document_role, count(*) AS chunks FROM legal_chunks GROUP BY document_role ORDER BY document_role;"
psql --dbname "$env:DATABASE_URL" --command "SELECT authority_level, count(*) AS chunks FROM legal_chunks GROUP BY authority_level ORDER BY authority_level;"
```

Kiểm tra danh sách văn bản:

```powershell
psql --dbname "$env:DATABASE_URL" --command "SELECT document_id, document_number, document_title FROM documents ORDER BY document_id;"
```

---

## Xem thêm

- [Cài đặt môi trường](setup.md)
- [Chuẩn bị dữ liệu](data-preparation.md)
- [Embedding và Indexing](embedding-and-indexing.md)
- [Quay lại README](../README.md)
