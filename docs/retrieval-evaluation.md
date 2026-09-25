# Đánh giá Retrieval

Tài liệu này mô tả phương pháp luận đánh giá, cấu trúc ground truth phân cấp, kết quả đo lường và giới hạn của bộ dữ liệu đánh giá hiện tại.

---

## Phương pháp luận

### Hit@K

**Hit@K** nhận giá trị 1.0 nếu có ít nhất một đoạn trích phù hợp xuất hiện trong K kết quả đầu tiên, và nhận giá trị 0.0 nếu không có kết quả phù hợp nào. Giá trị báo cáo là trung bình cộng trên toàn bộ các câu hỏi được đánh giá.

Ví dụ:
```text
Hit@5 = 0.533  →  Khoảng 53.3% số câu hỏi có ít nhất một đoạn
                   pháp lý đúng nằm trong top-5 kết quả trả về.
```

### Recall@K

**Recall@K** đo lường tỷ lệ các *mục pháp lý liên quan kỳ vọng* được tìm thấy trong K kết quả đầu tiên. Chỉ số này đặc biệt hữu ích khi một câu hỏi đòi hỏi thông tin từ nhiều điều khoản khác nhau (ví dụ: một câu hỏi về điều kiện hưởng chính sách có thể cần các đoạn trích từ cả Điều 1 và Điều 7).

Mỗi mục kỳ vọng chỉ được tính tối đa một lần vào Recall@K, bất kể có bao nhiêu chunk được truy xuất cùng thỏa mãn mục đó.

### MRR (Mean Reciprocal Rank)

**MRR** phản ánh mức độ ưu tiên vị trí của kết quả đúng đầu tiên. Nếu đoạn phù hợp đầu tiên nằm ở vị trí số 1, điểm là 1.0; nếu ở vị trí số 3, điểm là 1/3 ≈ 0.333. MRR là trung bình cộng của các giá trị nghịch đảo thứ hạng này trên toàn bộ tập câu hỏi.

Chỉ số MRR càng cao chứng tỏ hệ thống có xu hướng đẩy đoạn pháp lý chính xác lên gần đỉnh danh sách kết quả hơn.

> **Lưu ý:** Hit@K, Recall@K và MRR đo lường chất lượng truy xuất theo tiêu chuẩn ground truth phân cấp của dự án. Các chỉ số này **không** đồng nghĩa với độ chính xác của câu trả lời cuối cùng từ chatbot (Generation accuracy). Điểm similarity cũng là tín hiệu chẩn đoán, không phải thước đo trực tiếp độ đúng của câu trả lời pháp lý.

---

## Cấu trúc Ground Truth phân cấp

Mỗi kết quả kỳ vọng được định nghĩa theo hệ thống phân cấp pháp lý:

```text
Văn bản (Document)
    → Điều (Article)
        → Khoản (Clause)
            → Điểm (Point)
```

### Quy tắc khớp Ground Truth

Một chunk được truy xuất được coi là khớp với một mục ground truth kỳ vọng khi và chỉ khi thỏa mãn **tất cả** các điều kiện sau:

1. Chunk có `content_type = 'legal_text'` (các chunk dạng Hỏi–Đáp bị loại khỏi đối chiếu cấu trúc).
2. `document_title` của chunk khớp với tên văn bản kỳ vọng (không phân biệt chữ hoa/thường).
3. `article` của chunk khớp với điều luật kỳ vọng.
4. Nếu ground truth có chỉ định danh sách khoản (`clause`), thì `clause` của chunk phải khớp với ít nhất một khoản trong danh sách đó.
5. Nếu ground truth có chỉ định danh sách điểm (`point`), thì `point` của chunk phải khớp với ít nhất một điểm trong danh sách đó. Chunk không có thông tin điểm sẽ không thể thỏa mãn ràng buộc này.

Một đoạn trích có cùng số hiệu điều nhưng thuộc văn bản khác sẽ **không** được công nhận là khớp.

Nếu ground truth chỉ chỉ định `article` (không ràng buộc `clause` hay `point`), bất kỳ chunk nào thuộc điều luật đó đều được tính là khớp.

---

## Cấu hình đánh giá

| Cấu hình | Giá trị |
|---|---|
| Số lượng câu hỏi đánh giá | 15 câu |
| Số chunk đã nhúng | 617 / 617 |
| Tỷ lệ phủ nhúng | 100% |
| Số chiều vector | 1024 |
| Phương pháp truy xuất | Vector similarity search (cosine) |
| Chỉ mục | HNSW (`vector_cosine_ops`) |
| `ef_search` | 80 |
| `top_k` đánh giá | 10 |
| Biểu diễn tài liệu | Metadata-aware |
| Biến đổi câu hỏi | Không (None) |
| Ground truth matching | Hierarchical: Document → Article → Clause → Point |

## Chạy đánh giá hiện tại

Chạy từ thư mục gốc sau khi import chunks, tạo embeddings cho model đã chọn và kích hoạt Conda `chatbot`:

```powershell
conda activate chatbot
$env:EMBEDDING_MODEL = "bge-m3"
python scripts/test_retrieval.py --top-k 10 --ef-search 80
```

`--query TEXT` chạy một câu hỏi riêng; `--top-k N` đặt số kết quả (mặc định 5), còn `--ef-search N` đặt tham số HNSW (mặc định 40). Model được chọn qua `EMBEDDING_MODEL` hoặc `.env`; model-specific embeddings phải tồn tại trong database trước khi chạy. Script hiện đánh giá HNSW; chưa có option chọn BM25 hoặc `--output-dir`. Báo cáo được ghi tự động vào `logs/`.

---

## Benchmark so sánh mô hình embedding

Năm mô hình embedding đã được tích hợp và đánh giá thành công với cùng cấu hình truy xuất và cùng bộ câu hỏi đánh giá.

> **Lưu ý về DeepX:** `dxtech-asia/deepx-embedding-v1` (DeepX) là mô hình thứ sáu được xem xét, nhưng **chưa được tích hợp thành công** vào hệ thống đánh giá. Do đó, DeepX không xuất hiện trong bảng so sánh dưới đây. Xem thêm: [DeepX Embedding Backend](deepx-backend.md).

### Kết quả tổng hợp

| Mô hình | Emb. time | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `BAAI/bge-m3` | 37.49s | **46.7%** | **53.3%** | **73.3%** | **40.0%** | **46.7%** | **66.7%** | **0.430** |
| `mainguyen9/vietlegal-e5` | 52.60s | 33.3% | 40.0% | 53.3% | 33.3% | 36.7% | 50.0% | 0.278 |
| `mainguyen9/vietlegal-harrier-0.6b` | 45.87s | 20.0% | 40.0% | 66.7% | 20.0% | 40.0% | 66.7% | 0.275 |
| `darklethelong/vnlegal-lal` | 40.27s | 26.7% | 26.7% | 33.3% | 26.7% | 26.7% | 33.3% | 0.274 |
| `jinaai/jina-embeddings-v3-hf` | 38.62s | 20.0% | 26.7% | 33.3% | 20.0% | 26.7% | 33.3% | 0.188 |

### Chỉ số similarity chẩn đoán

Bảng dưới đây ghi lại giá trị similarity trung bình từ các lần chạy benchmark. Lưu ý rằng thang đo similarity **khác nhau giữa các mô hình** và không thể so sánh trực tiếp.

| Mô hình | Avg top-1 similarity | Avg best relevant similarity |
|---|---:|---:|
| `BAAI/bge-m3` | 0.6821 | 0.6523 |
| `mainguyen9/vietlegal-e5` | 0.6743 | 0.5672 |
| `mainguyen9/vietlegal-harrier-0.6b` | 0.5495 | 0.5108 |
| `darklethelong/vnlegal-lal` | 0.9283 | 0.9262 |
| `jinaai/jina-embeddings-v3-hf` | 0.7059 | 0.7183 |

> **Avg top-1 similarity**: Trung bình điểm tương đồng của kết quả được xếp hạng cao nhất trên tất cả các truy vấn.
> **Avg best relevant similarity**: Trung bình điểm tương đồng của chunk liên quan nhất được tìm thấy (nếu có) trên tất cả các truy vấn.

---

## So sánh với Baseline văn bản thuần (BGE-M3)

| Chỉ số | Baseline văn bản thuần | Metadata-aware (BGE-M3) | Mức thay đổi |
|---|---:|---:|---:|
| Hit@3 | 0.400 | 0.467 | +0.067 |
| Hit@5 | 0.400 | 0.533 | +0.133 |
| Hit@10 | 0.667 | 0.733 | +0.067 |
| Recall@3 | 0.333 | 0.400 | +0.067 |
| Recall@5 | 0.333 | 0.467 | +0.133 |
| Recall@10 | 0.600 | 0.667 | +0.067 |
| MRR | 0.389 | 0.430 | +0.041 |

---

## Diễn giải kết quả

Trong tập đánh giá gồm 15 truy vấn và với cùng cấu hình truy xuất có sử dụng metadata-aware embedding, `BAAI/bge-m3` đạt kết quả truy xuất tổng thể cao nhất trong số năm mô hình được đánh giá, đặc biệt ở Recall@K và MRR.

**Chi tiết từng mô hình:**

- **`BAAI/bge-m3`**: Đạt Recall@3, Recall@5, và MRR cao nhất trong tập đánh giá. Là mô hình duy nhất vượt mốc 40% Recall@3. Kết quả nhất quán ở cả các ngưỡng top-3, top-5 và top-10.

- **`mainguyen9/vietlegal-harrier-0.6b`**: Recall@10 đạt 66.7%, tương đương BGE-M3. Tuy nhiên, MRR chỉ đạt 0.275 (thấp hơn đáng kể so với 0.430 của BGE-M3), cho thấy một phần đáng kể các chunk liên quan xuất hiện ở vị trí thấp hơn trong bảng xếp hạng, điều này được phản ánh qua khoảng cách giữa Recall@10 và MRR. Đây là mô hình có khoảng cách lớn nhất giữa Recall@10 và các chỉ số Recall@3/Recall@5.

- **`mainguyen9/vietlegal-e5`**: Cung cấp hiệu suất truy xuất ở mức trung gian — Recall@3 = 33.3%, MRR = 0.278. Thời gian embedding dài nhất trong năm mô hình (52.60s).

- **`darklethelong/vnlegal-lal`**: Recall@10 thấp (33.3%). Điểm similarity trung bình rất cao (top-1 sim ≈ 0.928), nhưng điều này không chuyển thành kết quả truy xuất tốt hơn trong tập đánh giá này. Đây là ví dụ điển hình cho thấy thang đo similarity không thể dùng để so sánh chất lượng truy xuất giữa các mô hình.

- **`jinaai/jina-embeddings-v3-hf`**: MRR thấp nhất trong năm mô hình được đánh giá trong tập đánh giá hiện tại. Các chỉ số Recall@K cũng thuộc nhóm thấp nhất trong bảng so sánh này.

**Lưu ý quan trọng về similarity:**

Giá trị similarity của `darklethelong/vnlegal-lal` (avg top-1 sim ≈ 0.928) cao hơn nhiều so với các mô hình khác, nhưng Recall@10 của mô hình này lại thấp nhất. Điều này xảy ra vì các thang đo similarity giữa các mô hình embedding khác nhau là hoàn toàn không tương đương. Do đó, **không nên dùng giá trị similarity thô làm tiêu chí chính để so sánh các mô hình embedding khác nhau**. Thay vào đó, hãy ưu tiên các chỉ số dựa trên ground truth như Recall@K, Hit@K và MRR.

> **Lưu ý phạm vi:** Các kết luận trên được giới hạn trong phạm vi tập đánh giá hiện tại (15 câu hỏi, corpus pháp lý hiện có, cấu hình truy xuất cụ thể). Không nên khẳng định BGE-M3 là mô hình tốt nhất trên mọi tình huống hay corpus khác.

---

## Giới hạn của bộ dữ liệu đánh giá

Bộ đánh giá hiện có **15 câu hỏi**. Chỉ những câu hỏi đã có ground truth xác minh mới được đưa vào tính toán các chỉ số định lượng. Những câu hỏi chưa có ground truth hoàn chỉnh được giữ lại phục vụ kiểm tra trực quan thủ công, không gán điểm số ước đoán.

**Các hạn chế cụ thể cần lưu ý:**

- **Kích thước tập đánh giá nhỏ**: 15 câu hỏi là đủ cho đối sánh kỹ thuật ở giai đoạn phát triển, nhưng chưa đủ để khẳng định tính tổng quát hóa trên diện rộng.
- **Thời gian embedding chưa được đo chuẩn**: Các thời gian trong bảng (37.49s, 52.60s, v.v.) được ghi từ các lần chạy benchmark riêng lẻ, không phải từ nhiều lần đo lặp lại. Không nên dùng các con số này để so sánh tốc độ một cách chính xác.
- **Kết quả đặc thù theo cấu hình**: Kết quả phụ thuộc vào corpus pháp lý hiện có, chiến lược chunking, cách biểu diễn metadata-aware, cấu hình HNSW (`m=16`, `ef_construction=64`, `ef_search=80`), và tập câu hỏi đánh giá. Thay đổi bất kỳ yếu tố nào trong số này có thể ảnh hưởng đến thứ hạng tương đối của các mô hình.
- **Thang đo similarity không tương đương**: Như đã phân tích ở phần trên, giá trị similarity không thể so sánh trực tiếp giữa các mô hình khác nhau.
- **Chưa có đánh giá đủ rộng**: Cần thêm câu hỏi đánh giá đa dạng hơn và nhiều lần chạy lặp lại trước khi đưa ra kết luận rộng hơn về hiệu năng mô hình.

---

## Hướng phát triển tiếp theo

Các bước phát triển tiếp theo có thể cải thiện thêm chất lượng retrieval và độ tin cậy của đánh giá:

- **Hybrid vector + trigram retrieval**: Kết hợp vector search với tìm kiếm trigram (`pg_trgm`) để tăng khả năng thu hồi. (Hệ thống cũng có thể kết hợp với PostgreSQL Full-Text Search qua `tsvector`/`tsquery` nếu cần).
- **Reciprocal Rank Fusion (RRF)**: Hợp nhất kết quả từ nhiều nguồn truy xuất bằng RRF để cải thiện thứ hạng tổng hợp.
- **Reranking**: Áp dụng cross-encoder reranking để tinh chỉnh thứ tự kết quả sau retrieval.
- **Tập đánh giá lớn hơn và đa dạng hơn**: Mở rộng bộ câu hỏi đánh giá để kết quả benchmark có độ tin cậy thống kê cao hơn.
- **Đo thời gian lặp lại**: Chạy nhiều lần đo để có baseline tốc độ embedding đáng tin cậy hơn.
- **Tích hợp DeepX**: Hoàn thiện tích hợp `dxtech-asia/deepx-embedding-v1` để đưa vào so sánh benchmark trong các lần đánh giá tiếp theo.

> Các hướng phát triển trên chưa được triển khai tại thời điểm hiện tại.

---

## Xem thêm

- [Retrieval — Kiến trúc và sử dụng](retrieval.md)
- [Lịch sử phát triển Retrieval](retrieval-development-history.md)
- [Chuyển đổi mô hình embedding](embedding-model-switching.md)
- [DeepX Embedding Backend](deepx-backend.md)
- [Quay lại README](../README.md)

