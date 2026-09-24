# Retrieval — Truy xuất văn bản pháp lý

Tài liệu này mô tả kiến trúc, cách sử dụng và cấu hình của tầng Retrieval hiện tại — tìm kiếm vector ngữ nghĩa trên PostgreSQL + pgvector.

---

## Tổng quan

Tầng Retrieval nhận câu hỏi bằng ngôn ngữ tự nhiên, chuyển thành vector 1024 chiều bằng `BAAI/bge-m3`, và tìm kiếm cosine similarity qua chỉ mục HNSW trên PostgreSQL + pgvector.

**Phạm vi của triển khai hiện tại:** Vector Retrieval thuần túy với metadata-aware document embedding.

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

---

## Module

```text
scripts/
├── retrieval.py       ← module retrieval chính: Retriever + RetrievalResult
├── test_retrieval.py  ← script đánh giá thực tế với 15 câu hỏi pháp lý
tests/
└── test_retrieval.py  ← bộ unit test (209 test cases, không cần DB thật)
```

---

## Chạy đánh giá retrieval

```powershell
# Chạy toàn bộ 15 câu hỏi đánh giá mặc định (top-5)
conda activate chatbot
python scripts/test_retrieval.py

# Tuỳ chỉnh top-K và ef_search
conda activate chatbot
python scripts/test_retrieval.py --top-k 10 --ef-search 80
```

---

## Sử dụng trong code

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
                           # retrieval_priority, chapter, article, clause,
                           # point, content_type
```

---

## Tham số cấu hình

| Tham số | Mặc định | Mô tả |
|---|:---:|---|
| `top_k` | `5` | Số chunk trả về. Phải là số nguyên dương ≤ 1000. |
| `ef_search` | `40` | HNSW ef_search — tăng để nâng recall, giảm tốc độ. Phạm vi khuyến nghị: 40–200. |

`ef_search` được thiết lập mỗi kết nối bằng `SELECT set_config('hnsw.ef_search', ..., true)` (transaction-scoped, không ảnh hưởng phiên khác).

---

## Chiến lược kết nối (NeonDB-safe)

```text
1. embed_texts([query])           ← không có DB connection
2. with connect(url) as conn:    ← kết nối ngắn hạn
3.     set_config ef_search
4.     SELECT ... ORDER BY embedding <=> vector LIMIT k
5. # connection tự đóng
```

Embedding model (`BGE-M3`) được nạp **một lần** khi khởi tạo `Retriever`, không nạp lại theo từng truy vấn.

---

## Cấu hình đầu ra đánh giá

Biến môi trường `RETRIEVAL_TEST_OUTPUT` kiểm soát đầu ra của script đánh giá `scripts/test_retrieval.py`:

```env
RETRIEVAL_TEST_OUTPUT=development
```

| Giá trị | Hành vi |
|---|---|
| `development` | In ra console **và** ghi file `.txt` vào `logs/` với nhãn thời gian |
| Thiếu / rỗng / giá trị không hợp lệ | Chỉ in ra console |

---

## Xác minh trạng thái database

```python
retriever = Retriever()
state = retriever.verify_database_state()
print(state)
# {'total_chunks': 617, 'embedded_chunks': 617,
#  'missing_embeddings': 0, 'hnsw_index_exists': True}
```

Phương thức này chỉ đọc — **không sửa đổi embedding hay index**.

---

## Unit Test — Retrieval

```powershell
conda activate chatbot
python -m pytest tests/ -v
```

Kết quả kỳ vọng: **239/239 passed** (30 test embedding + 209 test retrieval), không cần DB thật hay tải model.

---

## Xem thêm

- [Embedding và Indexing](embedding-and-indexing.md)
- [Lịch sử phát triển Retrieval](retrieval-development-history.md)
- [Đánh giá Retrieval](retrieval-evaluation.md)
- [Quay lại README](../README.md)
