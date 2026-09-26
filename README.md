# Xây dựng hệ thống hỏi đáp Nghị định 116/2020/NĐ-CP dựa trên RAG

## Giới thiệu

Đây là dự án xây dựng hệ thống hỏi đáp pháp luật tập trung vào **Nghị định 116/2020/NĐ-CP**. Dữ liệu gồm nghị định chính, **Nghị định 60/2025/NĐ-CP** (văn bản sửa đổi), và **Luật Giáo dục 2019** để bổ sung định nghĩa và ngữ cảnh pháp lý. Bộ câu hỏi–trả lời về Nghị định 116 được dùng làm tham khảo, không thay thế nguồn pháp lý chính thức.

Theo định hướng RAG (Retrieval-Augmented Generation), hệ thống nhận câu hỏi, truy xuất các đoạn pháp lý liên quan, rồi cung cấp câu trả lời dựa trên nội dung đã truy xuất.

---

## Kiến trúc tổng quát

```text
Tệp Markdown pháp lý
        ↓
scripts/legal_chunker.py  →  data/processed/legal_chunks.jsonl
        ↓
scripts/validate_legal_chunks.py  →  kiểm tra JSONL
        ↓
scripts/import_legal_data.py  →  PostgreSQL (documents, legal_chunks)
        ↓
scripts/index_embeddings.py   →  embedding + HNSW index
        ↓
scripts/test_retrieval.py     →  đánh giá retrieval HNSW
```

## Quy trình chạy Python chính

Chạy các lệnh từ thư mục gốc repository trong PowerShell. Nếu sử dụng `conda`, cần kích hoạt môi trường `chatbot` trước mọi lệnh Python:

```powershell
conda activate chatbot
```

1. Tạo chunks từ tài liệu trong `data/markdown/` và `data/raw/qa/`; đầu ra mặc định là `data/processed/legal_chunks.jsonl`:

   ```powershell
   python scripts/legal_chunker.py
   ```

2. Kiểm tra JSONL đã tạo. Bước này chỉ đọc dữ liệu và in báo cáo, không sửa database:

   ```powershell
   python scripts/validate_legal_chunks.py
   ```

3. Import `data/processed/legal_chunks.jsonl` vào PostgreSQL. Lệnh này chạy `init.sql` rồi cập nhật các bảng, nên cần PostgreSQL và `DATABASE_URL` đã cấu hình:

   ```powershell
   python scripts/import_legal_data.py
   ```

4. Sinh embeddings và tạo/cập nhật cột cùng chỉ mục HNSW cho model đã chọn (mặc định `BAAI/bge-m3`). Đây là thao tác ghi database; chỉ chạy sau khi chunks đã import:

   ```powershell
   python scripts/index_embeddings.py --model bge-m3
   ```

5. Chạy bộ đánh giá retrieval HNSW sau khi có embeddings. Kết quả được ghi vào `logs/`:

   ```powershell
   $env:EMBEDDING_MODEL = "bge-m3"
   python scripts/test_retrieval.py --top-k 10 --ef-search 80
   ```
   hoặc thay đổi mô hình trong .env và chạy:
   ```powershell
   python scripts/test_retrieval.py --top-k 10 --ef-search 80
   ```     

Các script cấp cao nhất trong `scripts/` là CLI; package con như `scripts/embeddings/`, `scripts/indexing/`, `scripts/retrievers/` và `scripts/evaluation/` chứa implementation được các CLI import. BM25 là retriever implementation riêng, chưa có CLI benchmark riêng trong repository.

---

## Công nghệ chính

| Thành phần | Công nghệ |
|---|---|
| Cơ sở dữ liệu | PostgreSQL + pgvector |
| Mô hình nhúng | `BAAI/bge-m3` (mặc định), `dxtech-asia/deepx-embedding-v1` (DeepX), và 4 mô hình khác |
| Chỉ mục vector | HNSW — Hierarchical Navigable Small World: cấu trúc chỉ mục tìm kiếm vector tương đồng nhanh |
| Tìm kiếm Trigram | `pg_trgm` (đã có sẵn trong schema) |
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
| Module Vector Retrieval (`scripts/retrievers/hnsw.py`) | ✅ Hoàn thành |
| Đánh giá Retrieval (15 câu hỏi, ground truth phân cấp) | ✅ Hoàn thành |
| Bộ kiểm thử tự động — 91/91 tests passed (embedding + retrieval) | ✅ Hoàn thành |
| Nhúng tài liệu nhận biết siêu dữ liệu (Metadata-aware) | ✅ Hoàn thành |
| Đánh giá so sánh 5 mô hình embedding | ✅ Hoàn thành |
| Tích hợp DeepX Embedding v1 (`dxtech-asia/deepx-embedding-v1`) | ✅ Hoàn thành |

---

## Phương pháp truy xuất hiện tại

Tầng Retrieval hiện dùng **metadata-aware document embedding**: mỗi đoạn pháp lý được biểu diễn kèm tiền tố cấu trúc (tên văn bản, chương, điều, khoản, điểm) trước khi đưa vào BGE-M3. Câu hỏi được nhúng nguyên văn, không biến đổi.

```text
[Document] <document_title>
[Chapter]  <chapter>
[Article]  <article>
[Clause]   <clause>
[Point]    <point>
[Content]  <nội dung gốc>
```

Chi tiết đầy đủ: [Lịch sử phát triển Retrieval](docs/retrieval-development-history.md).

---

## Kết quả đánh giá Retrieval

Đánh giá thực hiện trên **15 câu hỏi pháp lý**, top-10, `ef_search=80`, metadata-aware embedding. Năm mô hình đã được tích hợp và đánh giá thành công. DeepX đã được tích hợp đầy đủ nhưng kết quả đánh giá thực tế trên bộ dữ liệu Nghị định 116 chưa có (cần chạy indexing với model thực).

| Mô hình | Emb. time | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `BAAI/bge-m3` | 37.5s | **46.7%** | **53.3%** | **73.3%** | **40.0%** | **46.7%** | **66.7%** | **0.430** |
| `mainguyen9/vietlegal-e5` | 52.6s | 33.3% | 40.0% | 53.3% | 33.3% | 36.7% | 50.0% | 0.278 |
| `mainguyen9/vietlegal-harrier-0.6b` | 45.9s | 20.0% | 40.0% | 66.7% | 20.0% | 40.0% | 66.7% | 0.275 |
| `darklethelong/vnlegal-lal` | 40.3s | 26.7% | 26.7% | 33.3% | 26.7% | 26.7% | 33.3% | 0.274 |
| `jinaai/jina-embeddings-v3-hf` | 38.6s | 20.0% | 26.7% | 33.3% | 20.0% | 26.7% | 33.3% | 0.188 |
| `dxtech-asia/deepx-embedding-v1` (DeepX-1024) | — | — | — | — | — | — | — | — |

**Nhận xét chính:**
- Trong phạm vi tập đánh giá hiện tại, `BAAI/bge-m3` đạt kết quả truy xuất tổng thể cao nhất trong số năm mô hình đã được đánh giá, đặc biệt ở Recall@K và MRR.
- `mainguyen9/vietlegal-harrier-0.6b` đạt Recall@10 tương đương BGE-M3 (66.7%), nhưng MRR thấp hơn, cho thấy các kết quả liên quan có xu hướng xuất hiện ở vị trí thấp hơn trong danh sách.
- Các giá trị similarity không thể so sánh trực tiếp giữa các mô hình khác nhau — ưu tiên dùng Recall@K / Hit@K / MRR để so sánh mô hình.
- **`dxtech-asia/deepx-embedding-v1` (DeepX) đã được tích hợp đầy đủ vào kiến trúc pipeline.** Kết quả đánh giá trên bộ dữ liệu Nghị định 116 sẽ được cập nhật sau khi chạy indexing thực tế. Điểm benchmark chính thức từ DeepX (nDCG@10 = 0.8162 trên Zalo Legal Text Retrieval) không phải là kết quả đánh giá của dự án này.

> Bộ đánh giá 15 câu hỏi phục vụ so sánh kỹ thuật trong giai đoạn phát triển. Các chỉ số không nên được hiểu là độ chính xác câu trả lời cuối cùng hay hiệu năng tổng quát trên diện rộng. Chi tiết: [Đánh giá Retrieval](docs/retrieval-evaluation.md).

---

## Cấu trúc thư mục

```text
init.sql                     ← schema PostgreSQL (nguồn sự thật)
.env.example                 ← mẫu cấu hình biến môi trường
requirements.txt
README.md
docs/                        ← tài liệu chi tiết
scripts/
├── data_pipeline/           ← chunking, validation
├── embeddings/              ← embedding backend, model registry
├── evaluation/              ← dataset, matching, metrics, runner, reporting
├── indexing/                ← embedding_index, import_data
├── retrievers/              ← bm25, hnsw
├── environment.py           ← cấu hình môi trường (.env)
├── database.py              ← kết nối PostgreSQL
├── retrieval_types.py       ← hợp đồng RetrievalResult dùng chung
├── legal_chunker.py         ← CLI wrapper cho chunking
├── import_legal_data.py     ← CLI wrapper cho import_data
├── validate_legal_chunks.py ← CLI wrapper cho validation
├── index_embeddings.py      ← CLI wrapper cho indexing
├── test_retrieval.py        ← CLI đánh giá HNSW (15 câu hỏi)
└── gpu_smoke_test.py
tests/
├── test_embedding.py        ← 30 unit tests
└── test_retrieval.py        ← 209 unit tests
logs/
├── retrieval_*.txt          ← báo cáo đánh giá kèm nhãn thời gian
└── <model>_<timestamp>.txt  ← báo cáo benchmark từng mô hình embedding
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
| [Chuyển đổi mô hình Embedding](docs/embedding-model-switching.md) | Hướng dẫn chuyển đổi giữa các mô hình embedding |
| [DeepX Embedding v1](docs/deepx-embedding.md) | Tích hợp DeepX, cài đặt, cấu hình, cách dùng |
| [Retrieval](docs/retrieval.md) | Kiến trúc, module, cách dùng, unit test |
| [Lịch sử phát triển Retrieval](docs/retrieval-development-history.md) | Text-only → cải tiến đánh giá → metadata-aware |
| [Đánh giá Retrieval](docs/retrieval-evaluation.md) | Phương pháp Hit@K/Recall@K/MRR, ground truth, kết quả |
