# Xây dựng hệ thống hỏi đáp Nghị định 116/2020/NĐ-CP dựa trên RAG

## Giới thiệu

Đây là dự án xây dựng hệ thống hỏi đáp pháp luật tập trung vào **Nghị định 116/2020/NĐ-CP**. Dữ liệu gồm nghị định chính, **Nghị định 60/2025/NĐ-CP** (văn bản sửa đổi), và **Luật Giáo dục 2019** để bổ sung định nghĩa và ngữ cảnh pháp lý. Bộ câu hỏi–trả lời về Nghị định 116 được dùng làm tham khảo, không thay thế nguồn pháp lý chính thức.

Theo định hướng RAG (Retrieval-Augmented Generation), hệ thống nhận câu hỏi, truy xuất các đoạn pháp lý liên quan, rồi cung cấp câu trả lời dựa trên nội dung đã truy xuất.

---

## Kiến trúc tổng quát

```text
Tệp Markdown pháp lý
        ↓
legal_chunker.py  →  legal_chunks.jsonl
        ↓
import_legal_data.py  →  PostgreSQL (documents, legal_chunks)
        ↓
index_embeddings.py   →  Embedding BAAI/bge-m3 → HNSW index
        ↓
retrieval.py          →  Vector similarity search (cosine, pgvector)
```

---

## Công nghệ chính

| Thành phần | Công nghệ |
|---|---|
| Cơ sở dữ liệu | PostgreSQL + pgvector |
| Mô hình nhúng | `BAAI/bge-m3` (vector 1024 chiều, hỗ trợ tiếng Việt) |
| Chỉ mục vector | HNSW — Hierarchical Navigable Small World: cấu trúc chỉ mục tìm kiếm vector tương đồng nhanh |
| Tìm kiếm toàn văn | `pg_trgm` (đã có sẵn trong schema) |
| Ngôn ngữ | Python (môi trường Conda `chatbot`) |
| Kết nối DB | `psycopg` / `psycopg2` |

---

## Trạng thái hiện tại

| Giai đoạn | Trạng thái |
|---|:---:|
| Thu thập & Xử lý dữ liệu (617 chunks pháp lý) | ✅ Hoàn thành |
| Chunking & Chuẩn hóa cấu trúc (Điều/Khoản/Điểm) | ✅ Hoàn thành |
| Cơ sở dữ liệu PostgreSQL + pgvector (`init.sql`) | ✅ Hoàn thành |
| Vector hóa BGE-M3 — 617/617 chunks, 1024 chiều | ✅ Hoàn thành |
| Chỉ mục HNSW (`vector_cosine_ops`, m=16, ef_construction=64) | ✅ Hoàn thành |
| Module Vector Retrieval (`retrieval.py`) | ✅ Hoàn thành |
| Đánh giá Retrieval (15 câu hỏi, ground truth phân cấp) | ✅ Hoàn thành |
| Bộ kiểm thử tự động — 239/239 tests passed | ✅ Hoàn thành |
| Nhúng tài liệu nhận biết siêu dữ liệu (Metadata-aware) | ✅ Giai đoạn hiện tại |

---

## Phương pháp truy xuất hiện tại

Tầng Retrieval hiện dùng **metadata-aware document embedding**: mỗi đoạn pháp lý được biểu diễn kèm tiền tố cấu trúc (tên văn bản, chương, điều, khoản, điểm) trước khi đưa vào BGE-M3. Câu hỏi được nhúng nguyên văn, không biến đổi.

```text
[Document] <document_title>
[Chapter]  <chapter>
[Article]  <article>
[Clause]   <clause>
[Content]  <nội dung gốc>
```

Chi tiết đầy đủ: [Lịch sử phát triển Retrieval](docs/retrieval-development-history.md).

---

## Kết quả đánh giá Retrieval

Đo lường trên 15 câu hỏi pháp lý, top-10, `ef_search=80`, metadata-aware embedding:

| Chỉ số | Baseline (text-only) | Hiện tại (metadata-aware) |
|---|---:|---:|
| Hit@3 | 0.400 | **0.467** |
| Hit@5 | 0.400 | **0.533** |
| Hit@10 | 0.667 | **0.733** |
| Recall@3 | 0.333 | **0.400** |
| Recall@5 | 0.333 | **0.467** |
| Recall@10 | 0.600 | **0.667** |
| MRR | 0.389 | **0.430** |

> Bộ đánh giá 15 câu hỏi phục vụ so sánh kỹ thuật trong giai đoạn phát triển. Các chỉ số không nên được hiểu là độ chính xác câu trả lời cuối cùng hay hiệu năng tổng quát trên diện rộng.

---

## Cấu trúc thư mục

```text
init.sql                     ← schema PostgreSQL (nguồn sự thật)
.env.example                 ← mẫu cấu hình biến môi trường
requirements.txt
README.md
docs/                        ← tài liệu chi tiết
scripts/
├── legal_chunker.py
├── import_legal_data.py
├── validate_legal_chunks.py
├── embedding.py             ← định dạng văn bản + bộ bao bọc BGE-M3
├── index_embeddings.py      ← quy trình indexing
├── retrieval.py             ← lớp Retriever
├── test_retrieval.py        ← kịch bản đánh giá (15 câu hỏi)
└── gpu_smoke_test.py
tests/
├── test_embedding.py        ← 30 unit tests
└── test_retrieval.py        ← 209 unit tests
logs/
└── retrieval_*.txt          ← báo cáo đánh giá kèm nhãn thời gian
data/
├── markdown/
├── raw/
└── processed/
    └── legal_chunks.jsonl
```

---

## Tài liệu chi tiết

| Tài liệu | Nội dung |
|---|---|
| [Cài đặt & Cấu hình](docs/setup.md) | Yêu cầu môi trường, Conda, pip, biến `.env` |
| [Chuẩn bị dữ liệu](docs/data-preparation.md) | Nguồn pháp lý, pipeline chunking, JSONL |
| [Cơ sở dữ liệu & Import](docs/database-and-import.md) | Schema `init.sql`, import, xác minh DB |
| [Embedding & Indexing](docs/embedding-and-indexing.md) | BGE-M3, HNSW, cấu hình, kết quả indexing |
| [Retrieval](docs/retrieval.md) | Kiến trúc, module, cách dùng, unit test |
| [Lịch sử phát triển Retrieval](docs/retrieval-development-history.md) | Text-only → cải tiến đánh giá → metadata-aware |
| [Đánh giá Retrieval](docs/retrieval-evaluation.md) | Phương pháp Hit@K/Recall@K/MRR, ground truth, kết quả |
