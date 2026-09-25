# Retrieval — Truy xuất văn bản pháp lý

Tài liệu này mô tả kiến trúc, cách sử dụng và cấu hình của tầng Retrieval — bao gồm HNSW vector retrieval và BM25 lexical retrieval trên corpus Nghị định 116/2020/NĐ-CP.

---

## Tổng quan

Tầng Retrieval nhận câu hỏi bằng ngôn ngữ tự nhiên và trả về các chunk văn bản pháp lý liên quan nhất từ bảng `legal_chunks` trong PostgreSQL. Hiện tại hỗ trợ hai phương pháp độc lập:

| Phương pháp | Module | Cơ chế |
|---|---|---|
| **HNSW Vector** | `scripts/retrievers/hnsw.py` (`Retriever`) | Cosine similarity qua pgvector + HNSW index. Cần embedding model. |
| **BM25** | `scripts/retrievers/bm25.py` (`BM25Retriever`) | Tần suất từ (Okapi BM25). Không cần embedding model. |

Cả hai đều trả về danh sách `RetrievalResult` — định dạng kết quả chung. CLI đánh giá hiện tại (`scripts/test_retrieval.py`) chỉ chạy HNSW; BM25 là implementation có thể gọi từ code, chưa có CLI để benchmark BM25 riêng.

### Kiến trúc HNSW Vector Retrieval

```text
Câu hỏi pháp lý (tiếng Việt)
          ↓
   Embedding Model
  (bge-m3 / vietlegal-e5 / ...)
          ↓
   Vector 1024 chiều
          ↓
  PostgreSQL + pgvector
   (cosine distance <=>)
          ↓
    HNSW index search
          ↓
   Top-K chunks (có điểm cosine similarity)
```

### Kiến trúc BM25 Retrieval

```text
Câu hỏi pháp lý (tiếng Việt)
          ↓
   Phân tách token
 (khoảng trắng + dấu câu)
          ↓
  BM25.get_scores(tokens)
  (tính toán trong bộ nhớ)
          ↓
   Top-K chunks (có BM25 score)
```

---

## Module

```text
scripts/
├── retrievers/hnsw.py       ← implementation HNSW: Retriever
├── retrievers/bm25.py       ← implementation BM25: BM25Retriever
├── retrieval_types.py       ← hợp đồng kết quả dùng chung
├── embeddings/              ← embedding backends và model registry
├── evaluation/              ← dataset, matching, metrics, runner, reporting
└── test_retrieval.py        ← CLI đánh giá HNSW với câu hỏi pháp lý

tests/
├── test_retrieval.py      ← Unit tests HNSW (209 test cases)
├── test_bm25_retriever.py ← Unit tests BM25 (61 test cases)
└── test_embedding.py      ← Unit tests Embedding (30 test cases)
```

---

## Chạy đánh giá retrieval

```powershell
# HNSW — chạy toàn bộ 15 câu hỏi đánh giá mặc định (top-5)
conda activate chatbot
python scripts/test_retrieval.py

# HNSW — tuỳ chỉnh top-K và ef_search
python scripts/test_retrieval.py --top-k 10 --ef-search 80
```

CLI này nhận `--query TEXT`, `--top-k N` và `--ef-search N`. Nó không nhận tùy chọn method/model riêng: HNSW lấy model từ `EMBEDDING_MODEL` hoặc `.env` (mặc định BGE-M3). BM25 và HNSW có thể được gọi riêng từ implementation; hiện không có lệnh CLI để đánh giá BM25 hoặc so sánh hai phương pháp.

Kết quả retrieval evaluation được lưu tự động trong `logs/` với tên model, thời điểm chạy và hậu tố duy nhất. Không có biến `RETRIEVAL_TEST_OUTPUT` hay tùy chọn `--output-dir`.

---

## Sử dụng trong code

### HNSW Vector Retrieval

```python
from scripts.retrievers.hnsw import Retriever

retriever = Retriever()   # tải embedding model một lần
results = retriever.retrieve(
    "Điều kiện để được hưởng chính sách hỗ trợ là gì?",
    top_k=5,
)
for r in results:
    print(r.chunk_id, f"{r.score:.4f}", r.rank, r.metadata["article"])
    print(r.text[:200])
```

### BM25 Retrieval

```python
from scripts.retrievers.bm25 import BM25Retriever

retriever = BM25Retriever()   # tải corpus một lần, không cần model
results = retriever.retrieve(
    "Điều kiện để được hưởng chính sách hỗ trợ là gì?",
    top_k=5,
)
for r in results:
    print(r.chunk_id, f"{r.score:.4f}", r.rank, r.metadata["article"])
    print(r.text[:200])
```

### Cấu trúc kết quả (`RetrievalResult`)

```python
@dataclass
class RetrievalResult:
    chunk_id: str          # Khóa chính trong legal_chunks
    text: str              # Nội dung pháp lý của chunk
    score: float           # Điểm số — thang đo phụ thuộc vào score_type
    retrieval_method: str  # "hnsw" hoặc "bm25" (mặc định "hnsw")
    score_type: str        # "cosine_similarity" hoặc "bm25" (mặc định "cosine_similarity")
    rank: int | None       # Thứ hạng 1-based (hạng 1 = tốt nhất)
    metadata: dict         # document_id, document_title, document_number,
                           # source_type, document_role, authority_level,
                           # retrieval_priority, chapter, article, clause,
                           # point, content_type
```

> **Lưu ý**: `score` từ HNSW (cosine similarity) và BM25 là **hai thang đo khác nhau** và không được so sánh trực tiếp. Dùng `score_type` để nhận biết ý nghĩa của điểm số.

---

## Tham số cấu hình

### HNSW

| Tham số | Mặc định | Mô tả |
|---|:---:|---|
| `top_k` | `5` | Số chunk trả về. Phải là số nguyên dương ≤ 1000. |
| `ef_search` | `40` | HNSW ef_search — tăng để nâng recall, giảm tốc độ. Khuyến nghị: 40–200. |

`ef_search` được thiết lập mỗi kết nối bằng `SELECT set_config('hnsw.ef_search', ..., true)` (transaction-scoped, không ảnh hưởng phiên khác).

### BM25

| Tham số | Mặc định | Mô tả |
|---|:---:|---|
| `top_k` | `5` | Số chunk trả về. Phải là số nguyên dương ≤ 1000. |
| `k1` | `1.5` | Độ bão hòa tần suất từ. |
| `b` | `0.75` | Chuẩn hóa độ dài tài liệu. |

Xem chi tiết tại [tài liệu BM25](retrieval-bm25.md#5-tham-số-bm25).

---

## Chiến lược kết nối (NeonDB-safe)

### HNSW

```text
1. embed_query(query)            ← không có DB connection
2. with connect(url) as conn:    ← kết nối ngắn hạn
3.     set_config ef_search
4.     SELECT ... ORDER BY embedding <=> vector LIMIT k
5. # connection tự đóng
```

Embedding model được nạp **một lần** khi khởi tạo `Retriever`, không nạp lại theo từng truy vấn.

### BM25

```text
1. __init__: with connect(url) as conn   ← kết nối ngắn hạn để tải corpus
2.             fetchall() → 617 rows
3.           # connection tự đóng
4. retrieve(): BM25.get_scores(tokens)   ← không cần DB, tính trong bộ nhớ
```

`retrieve()` của `BM25Retriever` **không mở kết nối database** — toàn bộ tính toán trong bộ nhớ.

---

## Xác minh trạng thái database (HNSW)

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

Suite kiểm tra implementation của embedding, HNSW, BM25 và DeepX; một số integration check có thể cần cấu hình riêng.

---

## Xem thêm

- [BM25 Retriever và Kiến trúc Mô-đun](retrieval-bm25.md)
- [Embedding và Indexing](embedding-and-indexing.md)
- [Lịch sử phát triển Retrieval](retrieval-development-history.md)
- [Đánh giá Retrieval](retrieval-evaluation.md)
- [Quay lại README](../README.md)
