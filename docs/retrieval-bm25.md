# BM25 Retriever và Kiến trúc Retrieval Mô-đun

Tài liệu này mô tả module BM25 mới, định dạng kết quả truy xuất thống nhất, và kiến trúc retrieval sau khi tái cấu trúc.

---

## 1. Tổ chức module Retrieval

Sau khi tái cấu trúc, các module retrieval được tổ chức trong `scripts/` như sau:

```text
scripts/
├── retrievers/hnsw.py       ← HNSW implementation (Retriever)
├── retrievers/bm25.py       ← BM25 implementation (BM25Retriever)
├── retrieval_types.py       ← hợp đồng kết quả dùng chung
├── embeddings/              ← registry và embedding backends
├── indexing/                ← implementation import và indexing
├── evaluation/              ← implementation đánh giá dùng lại
├── index_embeddings.py      ← CLI sinh và lưu embedding vào PostgreSQL
└── test_retrieval.py        ← CLI đánh giá HNSW (15 câu hỏi pháp lý)

tests/
├── test_retrieval.py      ← Unit tests cho HNSW Retriever (209 test cases)
├── test_bm25_retriever.py ← Unit tests cho BM25Retriever (61 test cases, mới)
└── test_embedding.py      ← Unit tests cho EmbeddingModel
```

### Phân chia trách nhiệm

| Module | Trách nhiệm |
|---|---|
| `scripts/retrievers/hnsw.py` | HNSW vector retrieval qua pgvector. Quản lý cột embedding theo từng model. |
| `scripts/retrievers/bm25.py` | BM25 lexical retrieval. Không phụ thuộc vào embedding model. |
| `scripts/embeddings/model_registry.py` | Cấu hình tập trung cho embedding models được hỗ trợ. |
| `scripts/embeddings/embedding.py` | Implementation tải và chạy inference; được import bởi indexing/retrieval. |

---

## 2. BM25 — Giới thiệu

**BM25 (Best Match 25)** là phương pháp **truy xuất từ khóa** (lexical retrieval). BM25 xếp hạng các đoạn văn bản dựa trên tần suất xuất hiện của các từ trong câu hỏi, có tính đến độ dài của từng tài liệu.

### Đặc điểm chính

- **Không cần embedding model**: BM25 chỉ dùng thống kê tần suất từ — không tải model nào.
- **Kết quả tất định**: Cùng corpus, cùng cấu hình, cùng câu hỏi → luôn cho kết quả giống nhau.
- **Điểm số không thể so sánh với cosine similarity**: BM25 score và cosine similarity là hai thang đo hoàn toàn khác nhau — không được cộng, trừ hay so sánh trực tiếp.
- **Tải corpus một lần**: Corpus được nạp từ PostgreSQL khi khởi tạo và lưu trong bộ nhớ, không cần kết nối cơ sở dữ liệu mỗi lần truy vấn.

### BM25 khác HNSW như thế nào?

| Đặc điểm | BM25 | HNSW (Vector) |
|---|---|---|
| Cơ chế | Từ khóa (tần suất từ) | Ngữ nghĩa (cosine similarity) |
| Cần embedding model | Không | Có |
| Thang đo điểm số | BM25 score (≥ 0, không giới hạn trên) | Cosine similarity ([-1, 1]) |
| Có thể so sánh điểm? | **Không** — cần chuẩn hóa riêng | **Không** — khác thang đo |
| Kết quả tất định | Có | Có (với cùng model và cùng index) |
| Xây dựng chỉ mục | Tức thì trong bộ nhớ (<100 ms) | Cần build HNSW index trên PostgreSQL |
| Phù hợp nhất khi | Câu hỏi chứa từ khóa pháp lý rõ ràng | Câu hỏi ngôn ngữ tự nhiên, ngữ nghĩa phức tạp |

---

## 3. Tải corpus và khởi tạo

BM25 corpus được nạp tự động từ bảng `legal_chunks` khi khởi tạo `BM25Retriever`. Không cần bước chuẩn bị nào ngoài việc đảm bảo `DATABASE_URL` được cấu hình trong `.env`.

```python
from scripts.retrievers.bm25 import BM25Retriever

retriever = BM25Retriever(database_url="postgresql://...")
# Corpus (~617 chunks) được nạp và lập chỉ mục trong <100ms
print(f"Số chunk đã lập chỉ mục: {retriever.corpus_size}")
```

Nếu bảng `legal_chunks` thay đổi sau khi khởi tạo, gọi `reload_corpus()` để cập nhật chỉ mục mà không cần tạo lại đối tượng:

```python
retriever.reload_corpus()
```

**Chiến lược kết nối (NeonDB-safe)**:

```text
1. __init__:  with connect(url) as conn   ← kết nối ngắn hạn để tải corpus
2.              fetchall() → 617 rows
3.            # connection tự đóng
4. retrieve():  BM25.get_scores(tokens)   ← không cần DB, tính toán trong bộ nhớ
```

Khác với HNSW, mỗi lần gọi `retrieve()` của BM25 **không mở kết nối cơ sở dữ liệu** — toàn bộ tính toán thực hiện trong bộ nhớ trên corpus đã tải sẵn.

**Cài đặt thư viện**:

```powershell
conda activate chatbot
pip install -r requirements.txt   # đã bao gồm rank-bm25>=0.2.2 và numpy>=1.24.0
```

---

## 4. Thực thi truy vấn BM25

```python
from scripts.retrievers.bm25 import BM25Retriever

retriever = BM25Retriever(database_url="postgresql://...")

results = retriever.retrieve(
    "Điều kiện để được hưởng chính sách hỗ trợ học phí là gì?",
    top_k=5,
)
for r in results:
    print(f"Hạng {r.rank}: [{r.score:.4f}] {r.chunk_id}")
    print(f"  Phương pháp: {r.retrieval_method} | Loại điểm: {r.score_type}")
    print(f"  Điều: {r.metadata.get('article')} — Khoản: {r.metadata.get('clause')}")
    print(f"  Nội dung: {r.text[:200]}")
```

Ví dụ đầu ra:

```text
Hạng 1: [8.4231] nd116-dieu7-khoan1
  Phương pháp: bm25 | Loại điểm: bm25
  Điều: Điều 7 — Khoản: Khoản 1
  Nội dung: Điều 7. Điều kiện để sinh viên sư phạm được hưởng chính sách hỗ trợ...
```

---

## 5. Tham số BM25

| Tham số | Mặc định | Ý nghĩa |
|---|:---:|---|
| `k1` | `1.5` | **Độ bão hòa tần suất từ** — kiểm soát mức độ ảnh hưởng của tần suất lặp từ lên điểm số. Thông thường 1.2–2.0. Giá trị cao hơn nghĩa là lặp từ nhiều lần vẫn tiếp tục tăng điểm. |
| `b` | `0.75` | **Chuẩn hóa độ dài tài liệu** — 0.0 = tắt chuẩn hóa, 1.0 = chuẩn hóa hoàn toàn theo độ dài trung bình corpus. |
| `epsilon` | `0.25` | **Ngưỡng sàn IDF** — giới hạn dưới cho trọng số IDF, ngăn điểm số âm với từ xuất hiện trong phần lớn tài liệu. |
| `tokenizer` | `default_tokenizer` | Hàm chuyển văn bản thành danh sách token. Áp dụng đồng nhất cho cả corpus và câu hỏi. |

> **Lý do chọn giá trị mặc định**: Với corpus ~617 chunks văn bản pháp lý tiếng Việt, các giá trị này tuân theo khuyến nghị chuẩn của Okapi BM25. Nên tinh chỉnh dựa trên kết quả đánh giá thực tế trước khi đưa vào vận hành.

```python
# Ví dụ tùy chỉnh tham số
retriever = BM25Retriever(
    database_url="postgresql://...",
    k1=1.2,    # bão hòa nhanh hơn
    b=0.8,     # chuẩn hóa độ dài mạnh hơn
    epsilon=0.25,
)
```

---

## 6. Phân tách token

Hàm `default_tokenizer` trong `scripts/retrievers/bm25.py` chia văn bản theo khoảng trắng và dấu câu phổ biến, sau đó chuyển toàn bộ về chữ thường. Không áp dụng tách từ hình thái học (stemming) hay loại bỏ từ dừng (stopword) vì:

- Tiếng Việt là ngôn ngữ đơn lập — stemming không áp dụng trực tiếp được.
- Danh sách từ dừng pháp lý chuyên biệt chưa được xây dựng cho dự án này.
- Cách tiếp cận đơn giản đảm bảo tính tất định và dễ kiểm tra.

Có thể thay thế bộ tách từ bằng cách truyền tham số `tokenizer`:

```python
import underthesea   # ví dụ: dùng thư viện xử lý ngôn ngữ tự nhiên tiếng Việt

def tach_tu_viet(text: str) -> list[str]:
    return underthesea.word_tokenize(text.lower())

retriever = BM25Retriever(
    database_url="postgresql://...",
    tokenizer=tach_tu_viet,
)
```

> **Lưu ý quan trọng**: Phải dùng **cùng một bộ tách từ** cho cả corpus lẫn câu hỏi. Nếu đổi `tokenizer` sau khi corpus đã tải, hãy gọi `reload_corpus()` để xây dựng lại chỉ mục với bộ tách từ mới.

---

## 7. Định dạng kết quả truy xuất thống nhất (`RetrievalResult`)

`RetrievalResult` trong `scripts/retrieval_types.py` là **kiểu dữ liệu kết quả chung** cho các thuật toán retrieval. Mỗi retriever trả về kiểu kết quả này.

```python
@dataclass
class RetrievalResult:
    chunk_id: str          # Khóa chính trong legal_chunks (luôn là chuỗi không rỗng)
    text: str              # Nội dung đầy đủ của chunk
    score: float           # Điểm số — ý nghĩa phụ thuộc vào score_type (xem bảng bên dưới)
    retrieval_method: str  # Thuật toán tạo ra kết quả: "hnsw", "bm25", "fts" (tương lai)
    score_type: str        # Thang đo điểm: "cosine_similarity", "bm25", "ts_rank"
    rank: int | None       # Thứ hạng 1-based (hạng 1 = liên quan nhất); None nếu chưa xếp hạng
    metadata: dict         # Siêu dữ liệu pháp lý: document_id, document_title, article, clause...
```

### Quy ước giữa các thuật toán

| Trường | HNSW | BM25 | FTS (tương lai) |
|---|---|---|---|
| `retrieval_method` | `"hnsw"` | `"bm25"` | `"fts"` |
| `score_type` | `"cosine_similarity"` | `"bm25"` | `"ts_rank"` |
| Khoảng điểm số | [-1, 1] | [0, +∞) | [0, +∞) |
| `rank` | 1-based | 1-based | 1-based |
| Phụ thuộc embedding model | Có | **Không** | Không |

### ⚠️ Điểm số không thể so sánh trực tiếp giữa các thuật toán

BM25 score `8.5` và cosine similarity `0.87` là **hai thang đo hoàn toàn khác nhau** — không được cộng, so sánh hay kết hợp trực tiếp mà không có bước chuẩn hóa. Luôn kiểm tra `score_type` trước khi diễn giải điểm số. Chuẩn hóa cho hybrid retrieval sẽ được thực hiện riêng khi triển khai RRF (xem mục 9).

---

## 8. Hỗ trợ nhiều embedding model

Kiến trúc retrieval được thiết kế để chạy song song nhiều embedding model mà không thay đổi tầng đánh giá:

```python
from scripts.retrievers.hnsw import Retriever
from scripts.retrievers.bm25 import BM25Retriever

# HNSW với BGE-M3
retriever_bge = Retriever(model_name="bge-m3", database_url="postgresql://...")
ket_qua_bge = retriever_bge.retrieve("Điều kiện hưởng hỗ trợ?", top_k=10)

# HNSW với VietLegal-E5 (cột embedding khác, HNSW index khác)
retriever_e5 = Retriever(model_name="vietlegal-e5", database_url="postgresql://...")
ket_qua_e5 = retriever_e5.retrieve("Điều kiện hưởng hỗ trợ?", top_k=10)

# BM25 — không phụ thuộc model nào
retriever_bm25 = BM25Retriever(database_url="postgresql://...")
ket_qua_bm25 = retriever_bm25.retrieve("Điều kiện hưởng hỗ trợ?", top_k=10)
```

Mỗi embedding model có **cột embedding riêng** trong `legal_chunks` (ví dụ `embedding`, `embedding_vietlegal_e5`) và **HNSW index riêng**. Vector từ các model khác nhau **không bao giờ bị trộn lẫn**.

Trường `retrieval_method` trong `RetrievalResult` luôn là `"hnsw"` — không nhúng tên model vào kết quả để giữ tính nhất quán khi so sánh với BM25 và các phương pháp khác. Tên model được ghi riêng vào file báo cáo đánh giá (`logs/*.txt`).

Các module đánh giá `scripts/evaluation/` cung cấp logic dùng lại. CLI `scripts/test_retrieval.py` hiện khởi tạo HNSW và không nhận tùy chọn BM25/method; do đó chưa có lệnh CLI benchmark BM25 hoặc so sánh BM25 với HNSW.

---

## 9. Các tính năng chưa được triển khai

| Tính năng | Trạng thái | Ghi chú |
|---|:---:|---|
| **Hybrid retrieval (BM25 + HNSW)** | ❌ Chưa triển khai | Cần chuẩn hóa điểm số từ hai nguồn |
| **Reciprocal Rank Fusion (RRF)** | ❌ Chưa triển khai | Kết hợp theo thứ hạng, không cần chuẩn hóa điểm |
| **Xếp hạng lại (Reranking)** | ❌ Chưa triển khai | Cross-encoder reranker |
| **Chuẩn hóa điểm số** | ❌ Chưa triển khai | Min-max hoặc Z-score cho hybrid |

### Lưu ý thiết kế cho tương lai

- **BM25 là truy xuất từ khóa**, **HNSW là truy xuất ngữ nghĩa** — hai phương pháp bổ sung cho nhau, phù hợp để kết hợp theo kiến trúc hybrid.
- **Chuẩn hóa điểm số** phải là bước xử lý độc lập, không được thay đổi `score` gốc trong `RetrievalResult`.
- **RRF chỉ dùng thứ hạng (`rank`)**, không cần điểm số — đây là lý do `rank` được lưu trong `RetrievalResult`.
- Kết quả hybrid nên dùng `retrieval_method = "rrf"` để phân biệt rõ nguồn gốc.

---

## 10. Chạy unit tests

```powershell
conda activate chatbot

# Chỉ BM25 tests (61 test cases, không cần DB hay model)
python -m pytest tests/test_bm25_retriever.py -v

# Toàn bộ test suite (BM25 + HNSW + Embedding)
python -m pytest tests/ -v
```

Kết quả phụ thuộc vào môi trường và các kiểm tra integration hiện có; không dùng một số kết quả lịch sử làm cam kết cho các bản checkout khác.

---

## 11. Xem thêm

- [Retrieval — HNSW Vector Search](retrieval.md)
- [Embedding và Indexing](embedding-and-indexing.md)
- [Chuyển đổi Embedding Model](embedding-model-switching.md)
- [Đánh giá Retrieval](retrieval-evaluation.md)
- [Quay lại README](../README.md)
