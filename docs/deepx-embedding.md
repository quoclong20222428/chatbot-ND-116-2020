# DeepX Embedding v1 (`dxtech-asia/deepx-embedding-v1`)

Tài liệu này mô tả chi tiết về việc tích hợp, kiến trúc kỹ thuật, cấu hình và hướng dẫn sử dụng mô hình embedding **DeepX Embedding v1** (`dxtech-asia/deepx-embedding-v1`) trong hệ thống RAG Chatbot Nghị định 116/2020/NĐ-CP.

---

## 1. Tổng quan mô hình

| Thuộc tính | Giá trị |
|---|---|
| **Model ID** | `dxtech-asia/deepx-embedding-v1` |
| **Alias trong hệ thống** | `deepx` |
| **Nhà phát triển / Provider** | DXTech Asia (`dxtech-asia`) |
| **Kiến trúc mạng** | Gated DeltaNet-2 (GDN-2) / Linear Attention |
| **Chiều vector gốc (Native)** | 1536 chiều |
| **Các chiều Matryoshka hỗ trợ** | 256, 512, 768, 1024, 1536 |
| **Chiều vector sử dụng trong hệ thống** | 1024 chiều (tương thích pgvector hiện hành) |
| **Chiều dài ngữ cảnh tối đa** | 8192 tokens |
| **Chuẩn hóa (Normalization)** | L2 Normalization (mặc định kích hoạt) |
| **Query / Passage Instruction** | Không yêu cầu (None) — giữ nguyên văn bản gốc |
| **Backend tích hợp** | `deepx_embed` (thư viện chuyên dụng từ GitHub) |

### Điểm khác biệt kiến trúc

Khác với các mô hình Transformer truyền thống sử dụng cơ chế Softmax Self-Attention bậc hai ($O(N^2)$) như `BAAI/bge-m3` hay `mainguyen9/vietlegal-e5`, DeepX Embedding v1 xây dựng trên nền tảng **Gated DeltaNet-2** — một dạng kiến trúc linear-attention với độ phức tạp tính toán tuyến tính $O(N)$ theo độ dài chuỗi. 

Mô hình không tương thích với việc nạp trực tiếp qua `SentenceTransformer("dxtech-asia/deepx-embedding-v1")` chuẩn, mà đòi hỏi gói suy diễn chuyên biệt `deepx_embed`.

---

## 2. Cài đặt môi trường và Thư viện phụ thuộc

Để sử dụng DeepX, môi trường Conda `chatbot` cần cài đặt thư viện chính thức `deepx_embed` trực tiếp từ kho mã nguồn:

```powershell
conda activate chatbot

# Cài đặt gói suy diễn DeepX chính thức
pip install git+https://github.com/dx-tech-ai/deepx-embed.git
```

### Thư viện tùy chọn cho GPU (Khuyến nghị nếu có CUDA)

Để tối ưu tốc độ suy diễn linear attention trên phần cứng GPU NVIDIA:

```powershell
pip install git+https://github.com/fla-org/flash-linear-attention.git
```

> **Lưu ý:** Gói `deepx_embed` yêu cầu các dependency phụ trợ: `torch`, `transformers`, `huggingface_hub`, `einops`. Khi cài qua `pip install git+...`, các thư viện này sẽ được tự động giải quyết nếu chưa có.

---

## 3. Quản lý chiều vector và Tương thích Cơ sở dữ liệu

### Cơ chế Matryoshka Embeddings

DeepX Embedding v1 hỗ trợ **Matryoshka Representation Learning (MRL)**, cho phép cắt giảm số chiều vector về các mốc `256`, `512`, `768`, `1024`, hoặc `1536` mà vẫn giữ được độ bảo toàn thông tin biểu diễn cao.

Trong backend của hệ thống (`scripts/embeddings/embedding.py`), việc giảm chiều được thực hiện thông qua tham số chính thức của model:

```python
# Gọi suy diễn với chiều vector mục tiêu
embeddings = model.encode(texts, truncate_dim=1024)
```

Vector sau khi cắt giảm chiều sẽ tự động được chuẩn hóa L2 để đảm bảo tính toán chính xác với thước đo khoảng cách **Cosine Similarity** (`vector_cosine_ops` trong PostgreSQL).

### Tương thích với PostgreSQL + pgvector

- **Schema hiện hành:** Bảng `legal_chunks` được thiết kế mặc định với các cột vector kích thước **1024 chiều**.
- **Cột dành riêng cho DeepX:** Khi chạy indexing với model DeepX, hệ thống sẽ sử dụng cột `embedding_deepx vector(1024)` (tự động tạo nếu chưa có) kèm chỉ mục HNSW tương ứng.
- **Không thay đổi schema đột ngột:** Không tùy tiện thay đổi kích thước cột cơ sở dữ liệu hiện tại lên 1536 chiều để đảm bảo tính tương thích và bảo tồn dữ liệu của các mô hình khác (`bge-m3`, `vietlegal-e5`, `jina-v3`, v.v.).
- **Tùy chọn 1536 chiều:** Nếu muốn thử nghiệm kích thước đầy đủ 1536 chiều trong tương lai, cần tạo cột mới riêng biệt (ví dụ `embedding_deepx_1536 vector(1536)`) kèm HNSW index riêng thông qua migration tường minh.

---

## 4. Hướng dẫn cấu hình và Chuyển đổi mô hình

Hệ thống cho phép lựa chọn và chuyển đổi sang DeepX hoàn toàn thông qua cấu hình, không cần can thiệp mã nguồn.

### Cách 1: Thiết lập qua file `.env`

Mở file `.env` và cập nhật:

```dotenv
# Sử dụng alias ngắn
EMBEDDING_MODEL=deepx

# Hoặc sử dụng định danh đầy đủ của Hugging Face
# EMBEDDING_MODEL=dxtech-asia/deepx-embedding-v1
```

### Cách 2: Thiết lập qua tham số dòng lệnh (CLI)

Các script của hệ thống đều nhận tham số `--model`:

```powershell
# Chạy indexing cho DeepX
python scripts/index_embeddings.py --model deepx

# Chạy retrieval kiểm tra với DeepX
python scripts/test_retrieval.py --model deepx --top-k 10
```

---

## 5. Chạy Indexing (Vector hóa dữ liệu)

Để sinh vector embedding cho toàn bộ 617 chunks văn bản luật của Nghị định 116 với mô hình DeepX:

```powershell
conda activate chatbot

# Chạy indexing với batch size an toàn cho GPU VRAM 6GB
python scripts/index_embeddings.py --model deepx --batch-size 8
```

Các tham số hữu ích:
- `--batch-size 8`: Kích thước batch phù hợp với card đồ họa NVIDIA GeForce RTX 3050 6GB, giúp tránh hiện tượng tràn bộ nhớ (Out-Of-Memory).
- `--rebuild`: Tùy chọn xóa vector cũ trong cột `embedding_deepx` và tính toán lại toàn bộ từ đầu.
- `--dry-run`: Kiểm tra kết nối và số lượng chunks cần nhúng mà không ghi vào database.

---

## 6. Chạy Đánh giá Retrieval (Evaluation)

Sau khi dữ liệu đã được index vào database, chạy quy trình benchmark đánh giá độ chính xác truy xuất (Recall@k, Hit@k, MRR) trên cùng 15 câu hỏi kiểm thử chuẩn:

```powershell
conda activate chatbot

# Đánh giá retrieval với DeepX
python scripts/test_retrieval.py --model deepx --top-k 10 --ef-search 80
```

Kết quả chi tiết cùng ma trận chẩn đoán sẽ được lưu tự động trong thư mục `logs/` theo định dạng `retrieval_eval_deepx_*.json` và `.txt`.

---

## 7. Cân nhắc phần cứng (GPU và CPU)

- **Tự động nhận diện thiết bị:** Backend `DeepXEmbeddingBackend` tự động kiểm tra `torch.cuda.is_available()`. Nếu có CUDA, mô hình sẽ nạp lên GPU; nếu không, sẽ chạy mượt mà trên CPU mà không gây lỗi crash.
- **Quản lý bộ nhớ VRAM (RTX 3050 6GB):**
  - Trọng số mô hình chiếm xấp xỉ 1.8 GB – 2.5 GB VRAM.
  - Khi xử lý chuỗi văn bản dài (lên tới 8192 tokens), linear attention tiết kiệm bộ nhớ đáng kể so với full self-attention.
  - Khuyến nghị đặt `batch_size` trong khoảng từ `4` đến `16` khi chạy trên GPU 6GB.
- **Xác minh GPU hoạt động:** Có thể kiểm tra bằng lệnh:
  ```powershell
  python -c "from scripts.embeddings.embedding import get_embedding_model; m = get_embedding_model('deepx'); print('Model loaded on:', m.device)"
  ```

---

## 8. Phân biệt Benchmark Công bố và Đánh giá Thực nghiệm Nghị định 116

Khi đánh giá hiệu năng của DeepX Embedding v1, cần phân định rõ ràng giữa hai nguồn thông tin:

1. **Benchmark công bố trên Model Card Hugging Face:**
   - Do đơn vị phát triển DXTech Asia công bố trên các tập dữ liệu tổng quát (MTEB, đa lĩnh vực).
   - Thể hiện tiềm năng kỹ thuật của mô hình trên các bài toán chung và văn bản tiếng Việt quy mô lớn.
2. **Đánh giá thực nghiệm trên Nghị định 116/2020/NĐ-CP của dự án:**
   - Được đo lường trực tiếp trên 617 legal chunks có cấu trúc (Điều, Khoản, Điểm) và 15 câu hỏi nghiệp vụ đặc thù ngành sư phạm.
   - Hiệu quả thực tế của mô hình (so sánh với baseline `BAAI/bge-m3`) chỉ được kết luận dựa trên kết quả chạy từ `scripts/test_retrieval.py`.
   - Tuyệt đối không suy đoán hoặc khẳng định DeepX vượt trội hơn các mô hình khác khi chưa có dữ liệu benchmark đối sánh thực tế trên hệ thống.

---

## 9. Kiểm thử tự động (Test Suite)

Hệ thống đi kèm bộ kiểm thử toàn diện cho DeepX, kiểm tra từ registry, backend factory, batching, dimension validation cho đến xử lý chuỗi rỗng:

```powershell
conda activate chatbot

# Chạy riêng các test liên quan đến DeepX
pytest tests/test_deepx_integration.py -v

# Chạy toàn bộ test suite embedding (91 test cases)
pytest tests/test_embedding.py -v

# Chạy toàn bộ test suite của toàn dự án (361 test cases)
pytest tests/ -v
```

---

## Xem thêm

- [README](../README.md)
- [Hướng dẫn chuyển đổi mô hình Embedding](embedding-model-switching.md)
- [Embedding và Indexing (Vector hóa)](embedding-and-indexing.md)
- [Đánh giá Retrieval](retrieval-evaluation.md)
- [Lịch sử phát triển Retrieval](retrieval-development-history.md)
