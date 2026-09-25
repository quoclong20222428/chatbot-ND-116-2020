# Chuẩn bị dữ liệu pháp lý

Tài liệu này mô tả nguồn dữ liệu pháp lý, kiến trúc pipeline xử lý và quy trình chunking (tách đoạn) cho hệ thống RAG Chatbot Nghị định 116/2020.

---

## Nguồn dữ liệu pháp lý

Hệ thống sử dụng các văn bản pháp lý sau:

| Tệp | Vai trò |
|---|---|
| `data/markdown/116_2020_ND-CP.md` | Nghị định 116/2020/NĐ-CP — nguồn pháp lý chính |
| `data/markdown/60_2025_ND-CP.md` | Nghị định 60/2025/NĐ-CP — văn bản sửa đổi liên quan |
| `data/markdown/43_2019_QH14_Luat_Giao_duc_2019.md` | Luật Giáo dục 2019 — nguồn hỗ trợ định nghĩa và ngữ cảnh pháp lý |
| `data/raw/qa/hoi-dap-nghi-dinh-116-2020-nd-cp.md` | Dữ liệu hỏi–đáp tham khảo (không thay thế nguồn pháp lý chính thức) |
| `data/raw/legal/` | Các tệp DOCX nguồn thô tương ứng |

Tệp được import vào cơ sở dữ liệu là `data/processed/legal_chunks.jsonl` — không phải các tệp nguồn thô.

Bộ câu hỏi–trả lời về Nghị định 116 chỉ là nguồn tham khảo, không có mức ưu tiên cao hơn nguồn pháp luật chính thức.

---

## Kiến trúc pipeline xử lý

```text
Tệp Markdown pháp lý và tệp hỏi–đáp
        ↓
scripts/legal_chunker.py
        ↓
data/processed/legal_chunks.jsonl
        ↓
scripts/validate_legal_chunks.py
        ↓
scripts/import_legal_data.py
        ↓
init.sql + PostgreSQL
        ↓
documents, legal_chunks, legal_chunk_references
```

Chunking và validation là các bước xử lý tệp; chúng không kết nối hoặc ghi vào database. Import là bước riêng: CLI `scripts/import_legal_data.py` thực thi `init.sql`, sau đó đọc `data/processed/legal_chunks.jsonl` từng dòng để cập nhật dữ liệu và tái tạo bảng tham chiếu.

---

## Chunking — Tách đoạn văn bản pháp lý

Script `scripts/legal_chunker.py` chuyển đổi các tệp Markdown thành các đoạn pháp lý có cấu trúc (`legal chunks`) theo định dạng JSONL.

**Nguyên tắc tách đoạn:**

- Tách theo đơn vị cấu trúc pháp lý: Điều (Article) → Khoản (Clause) → Điểm (Point).
- Mỗi chunk được gắn siêu dữ liệu (metadata) bao gồm tên văn bản, chương, điều, khoản, điểm, vai trò văn bản (`document_role`), mức thẩm quyền (`authority_level`), và loại nội dung (`content_type`).
- Tham chiếu pháp lý giữa các văn bản được lưu ở dạng dữ liệu có cấu trúc trong bảng `legal_chunk_references`.

**Kết quả chunking hiện tại:**

- **617 legal chunks** từ toàn bộ bộ văn bản pháp lý.
- Định dạng đầu ra: JSONL (`data/processed/legal_chunks.jsonl`), mỗi đoạn nằm trên một dòng.

Chạy từ thư mục gốc sau khi kích hoạt môi trường Conda:

```powershell
conda activate chatbot
python scripts/legal_chunker.py
```

Lệnh này chỉ tạo/cập nhật tệp JSONL; không sửa database. Có thể truyền `--output PATH` để chọn tệp đầu ra khác.

## Validation — Kiểm tra JSONL

Sau khi chunking, kiểm tra tệp kết quả trước khi import:

```powershell
conda activate chatbot
python scripts/validate_legal_chunks.py
```

Validator đọc `data/processed/legal_chunks.jsonl` mặc định và in báo cáo JSON về tính hợp lệ, tham chiếu và số lượng chunk. Nó không sửa tệp hay database. Dùng `--input PATH` để kiểm tra tệp khác.

---

## Script liên quan

```text
scripts/
├── legal_chunker.py        ← chuyển đổi Markdown → legal_chunks.jsonl
└── validate_legal_chunks.py  ← kiểm tra tính hợp lệ của JSONL
```

---

## Xem thêm

- [Cài đặt môi trường](setup.md)
- [Cơ sở dữ liệu và Import](database-and-import.md)
- [Embedding và Indexing](embedding-and-indexing.md)
- [Quay lại README](../README.md)
