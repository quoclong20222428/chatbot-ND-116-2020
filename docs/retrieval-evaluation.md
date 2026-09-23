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
| Mô hình nhúng | `BAAI/bge-m3` |
| Số chiều vector | 1024 |
| Phương pháp truy xuất | Tìm kiếm độ tương đồng vector (Vector similarity search) |
| Chỉ mục | HNSW (`vector_cosine_ops`) |
| `ef_search` | 80 |
| `top_k` đánh giá | 10 |
| Số lượng câu hỏi đánh giá | 15 câu |
| Số chunk đã nhúng | 617 / 617 |
| Tỷ lệ phủ nhúng | 100% |
| Biểu diễn tài liệu | Nhận biết siêu dữ liệu (Metadata-aware) |
| Biến đổi câu hỏi | Không (None) |

---

## Kết quả đo lường (Metadata-aware, top-10, ef_search=80)

```text
Hit@3:     0.467
Hit@5:     0.533
Hit@10:    0.733

Recall@3:  0.400
Recall@5:  0.467
Recall@10: 0.667

MRR:       0.430

Độ tương đồng top-1 trung bình:               0.682
Độ tương đồng điểm liên quan nhất trung bình: 0.652
```

---

## So sánh với Baseline văn bản thuần

| Chỉ số | Baseline văn bản thuần | Hiện tại (Metadata-aware) | Mức thay đổi |
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

Sự cải thiện khi bổ sung siêu dữ liệu cấu trúc vào vector nhúng thể hiện rõ rệt nhất ở **top-3 và top-5** — những vị trí quan trọng nhất đối với người dùng hoặc mô hình sinh ngôn ngữ (LLM) vốn chỉ tổng hợp câu trả lời từ các kết quả hàng đầu. Các đoạn trích pháp lý phù hợp xuất hiện ở những vị trí đầu bảng thường xuyên hơn so với baseline văn bản thuần. Chỉ số Recall@10 và MRR cũng đều có sự cải thiện tích cực.

```text
Phương pháp văn bản thuần:
  Câu hỏi → Tìm kiếm ngữ nghĩa → Danh sách kết quả
  → Chunk liên quan có thể rơi vào hạng 6–10 hoặc không nằm trong danh sách

Phương pháp nhận biết siêu dữ liệu:
  Câu hỏi → Tìm kiếm ngữ nghĩa đối chiếu cả nội dung + cấu trúc phân cấp
  → Chunk liên quan xuất hiện nhiều hơn ở top 3–5 kết quả đầu tiên
```

Phương pháp nhúng nhận biết siêu dữ liệu được hiểu là một **bước cải tiến từng phần (incremental improvement)** so với baseline ban đầu, chưa phải là giải pháp hoàn chỉnh cho toàn bộ bài toán truy xuất.

---

## Giới hạn của bộ dữ liệu đánh giá

Bộ đánh giá hiện có **15 câu hỏi**. Chỉ những câu hỏi đã có ground truth xác minh mới được đưa vào tính toán các chỉ số định lượng. Những câu hỏi chưa có ground truth hoàn chỉnh được giữ lại phục vụ kiểm tra trực quan thủ công, không gán điểm số ước đoán.

15 câu hỏi là đủ cho việc đối sánh phát triển kỹ thuật giữa các phương án truy xuất ở giai đoạn hiện tại. Bộ dữ liệu này chưa đủ để khẳng định tính tổng quát hóa trên diện rộng, và vẫn có một số câu hỏi chưa tìm được đúng chunk pháp lý ngay trong top-10.

---

## Xem thêm

- [Retrieval — Kiến trúc và sử dụng](retrieval.md)
- [Lịch sử phát triển Retrieval](retrieval-development-history.md)
- [Quay lại README](../README.md)
