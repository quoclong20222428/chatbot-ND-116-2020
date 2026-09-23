# Cài đặt và Cấu hình môi trường

Tài liệu này hướng dẫn cài đặt môi trường và cấu hình cần thiết để chạy hệ thống RAG Chatbot Nghị định 116/2020.

---

## Yêu cầu môi trường

| Thành phần | Yêu cầu |
|---|---|
| **Python** | Quản lý qua môi trường Conda `chatbot` |
| **PostgreSQL** | Đang chạy (cục bộ hoặc từ xa), tài khoản có quyền tạo bảng, index và extension |
| **psql** | PostgreSQL client có trong `PATH` của môi trường `chatbot` |
| **Python library** | `psycopg` hoặc `psycopg2` (kết nối Python → PostgreSQL) |
| **Extension PostgreSQL** | `pg_trgm`, `pgvector` (được tạo tự động bởi `init.sql` nếu chưa tồn tại) |

Repository không yêu cầu Docker. PostgreSQL có thể chạy cục bộ hoặc trên máy chủ từ xa (như NeonDB, Supabase); máy chạy script phải truy cập được máy chủ.

---

## Hướng dẫn cài đặt

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

Chỉ tạo môi trường nếu chưa tồn tại:

```powershell
conda create -n chatbot python=3.11 -y
conda activate chatbot
```

Cài `psql` (PostgreSQL client) từ conda-forge:

```powershell
conda install -c conda-forge postgresql -y
```

### Bước 3: Cài đặt các thư viện Python

Các thư viện cần thiết được định nghĩa đầy đủ trong `requirements.txt`.

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

---

## Cấu hình biến môi trường

Sao chép tệp mẫu `.env.example` thành `.env` ở thư mục gốc:

```powershell
copy .env.example .env
```

Chỉnh sửa `.env` theo cấu hình thực tế:

```env
# Cơ sở dữ liệu (Bắt buộc)
DATABASE_URL=postgresql://<ten_nguoi_dung>:<mat_khau>@<may_chu>:<cong>/<ten_co_so_du_lieu>?sslmode=require

# Cấu hình Indexing & Embedding (Tùy chọn)
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BATCH_SIZE=8
HF_TOKEN=

# Cấu hình LLM sinh câu trả lời (Tùy chọn — phục vụ giai đoạn sau)
GEMINI_API_KEY=
```

> Không đưa thông tin đăng nhập thật vào README hoặc đẩy file `.env` lên git (đã được cấu hình trong `.gitignore`).

Script ưu tiên biến `DATABASE_URL` trong môi trường hệ thống, sau đó mới đọc `.env`. Có thể cấu hình cho phiên PowerShell hiện tại:

```powershell
$env:DATABASE_URL = "postgresql://<ten_nguoi_dung>:<mat_khau>@<may_chu>:<cong>/<ten_co_so_du_lieu>"
```

---

## Xem thêm

- [Chuẩn bị dữ liệu](data-preparation.md)
- [Cơ sở dữ liệu và Import](database-and-import.md)
- [Embedding và Indexing](embedding-and-indexing.md)
- [Retrieval](retrieval.md)
- [Quay lại README](../README.md)
