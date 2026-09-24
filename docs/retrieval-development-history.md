# Lịch sử phát triển tầng Retrieval

Tài liệu này ghi lại tiến trình phát triển từng bước của tầng Retrieval — từ baseline văn bản thuần đến triển khai metadata-aware hiện tại.

---

## Tổng quan tiến trình

```text
Giai đoạn 1: Baseline văn bản thuần (Text-only)
        ↓
Giai đoạn 2: Cải tiến phương pháp luận đánh giá
        ↓
Giai đoạn 3: Nhúng tài liệu nhận biết siêu dữ liệu (Metadata-aware) — Hiện tại
```

---

## Giai đoạn 1 — Baseline văn bản thuần (Text-only baseline)

Phương án nhúng tài liệu ban đầu chỉ biểu diễn mỗi chunk pháp lý thông qua nội dung văn bản thuần túy — toàn bộ nội dung của điều khoản hoặc đoạn văn bản pháp luật được đưa trực tiếp vào mô hình BGE-M3 mà không kèm theo ngữ cảnh cấu trúc.

Phương án này tạo nên một baseline truy xuất ngữ nghĩa hoạt động ổn định: các câu hỏi được diễn đạt bằng ngôn ngữ pháp lý tiếng Việt tự nhiên có thể truy xuất các đoạn pháp lý liên quan về mặt chủ đề. Tuy nhiên, vector nhúng không mã hóa tường minh vị trí của đoạn trích trong hệ thống phân cấp văn bản pháp luật — như thuộc văn bản, chương, điều, khoản hay điểm nào.

**Chỉ số baseline văn bản thuần** (đo lường trên tập 15 câu hỏi đánh giá hiện tại, top-10, `ef_search=80`):

| Chỉ số | Giá trị |
|---|---:|
| Hit@3 | 0.400 |
| Hit@5 | 0.400 |
| Hit@10 | 0.667 |
| Recall@3 | 0.333 |
| Recall@5 | 0.333 |
| Recall@10 | 0.600 |
| MRR | 0.389 |

> Các số liệu trên là baseline tham chiếu ở giai đoạn phát triển ban đầu trên một tập đánh giá nhỏ (15 câu hỏi), đóng vai trò mốc so sánh cho các thử nghiệm tiếp theo chứ không phải lời khẳng định về hiệu năng tổng quát.

---

## Giai đoạn 2 — Cải tiến phương pháp luận đánh giá

Trước khi tinh chỉnh mô hình truy xuất, tầng đánh giá được hoàn thiện để phép đo chất lượng có tính nhất quán, chính xác và khả năng tái lập cao hơn.

Giai đoạn này tập trung nâng cao **chất lượng đo lường**, không làm thay đổi bản thân mô hình truy xuất.

**Các điểm cải tiến cốt lõi trong phương pháp đánh giá:**

- **Cơ sở đối chiếu phân cấp pháp lý (Hierarchical legal ground truth)** — mỗi kết quả kỳ vọng được định nghĩa dưới dạng bộ thuộc tính (văn bản / điều / khoản / điểm), không chỉ dừng lại ở việc khớp từ khóa đơn thuần.
- **Bắt buộc đúng định danh văn bản (Document identity enforcement)** — một chunk thuộc *Điều 6 của Nghị định 60/2025* sẽ không được coi là thỏa mãn ground truth kỳ vọng *Điều 6 của Nghị định 116/2020*, ngay cả khi nội dung văn bản có sự tương đồng cao.
- **Lọc theo nguồn pháp lý chính thức (Legal-source filtering)** — chỉ các chunk có `content_type = 'legal_text'` mới được tham gia khớp cấu trúc ground truth. Các chunk dạng Hỏi–Đáp (QA/FAQ) không bao giờ thỏa mãn một mục cấu trúc kỳ vọng, bất kể nội dung văn bản là gì.
- **Khử trùng lặp khi tính Recall (Recall deduplication)** — mỗi mục kỳ vọng (`ExpectedSection`) chỉ được tính tối đa một lần vào `Recall@K`, bất kể có bao nhiêu chunk được truy xuất cùng thỏa mãn mục đó.
- **Tính linh hoạt ở các cấp không ràng buộc (Unconstrained levels)** — nếu một mục ground truth chỉ chỉ định `article` (không ràng buộc `clause` hay `point`), bất kỳ chunk nào thuộc điều luật đó đều được tính là khớp.
- **Gắn cờ kiểm tra thủ công (Manual verification flag)** — các câu hỏi chưa có ground truth đã xác minh vẫn được giữ trong bộ dữ liệu để kiểm tra trực quan thủ công, nhưng được tách khỏi việc tính toán các chỉ số định lượng (không gán điểm số giả lập).

---

## Giai đoạn 3 — Nhúng tài liệu nhận biết siêu dữ liệu (Metadata-aware — Hiện tại)

Triển khai hiện tại cải tiến cách thức biểu diễn các đoạn pháp lý trước khi đưa vào mô hình BGE-M3 để sinh vector.

### Động lực

> Thông tin trong văn bản pháp luật có tính phân cấp thứ bậc chặt chẽ. Một câu hay một đoạn văn không đứng độc lập; nó thuộc về một văn bản, chương, điều, khoản và có thể là điểm cụ thể. Việc đưa ngữ cảnh cấu trúc này vào vector nhúng tài liệu cung cấp thêm thông tin về *nguồn gốc vị trí* của nội dung — giúp mô hình phân biệt rõ ràng, ví dụ giữa *Điều 4 của Nghị định 116* và *Điều 4 của Luật Giáo dục*.

### Định dạng nhúng tài liệu — Nguồn pháp lý (`legal_text`)

Đối với các chunk có `content_type = 'legal_text'`, chuỗi văn bản đưa vào BGE-M3 là một tiền tố cấu trúc có tổ chức kết hợp với nội dung gốc của chunk:

```text
[Document]
<document_title>

[Chapter]
<chapter>

[Article]
<article>

[Clause]
<clause>

[Point]
<point>

[Content]
<nội dung đoạn pháp lý gốc>
```

**Quy tắc triển khai:**

- Các nhãn cấu trúc (`[Document]`, `[Chapter]`, `[Article]`, `[Clause]`, `[Point]`, `[Content]`) được giữ bằng tiếng Anh theo thiết kế hệ thống.
- Giá trị của siêu dữ liệu giữ nguyên văn bản tiếng Việt gốc từ cơ sở dữ liệu.
- `document_title` luôn được đưa vào khi có giá trị khác null.
- `document_number` **không** được đưa vào biểu diễn nhúng.
- `chapter`, `article`, `clause`, và `point` chỉ được thêm vào khi có giá trị khác null và không rỗng.
- Các trường null hoặc chỉ chứa khoảng trắng được lược bỏ hoàn toàn — không bao giờ chèn văn bản giữ chỗ (`N/A`, `NULL`, `None`).
- Mục `[Content]` luôn chứa nguyên văn toàn bộ nội dung của chunk.

### Định dạng nhúng tài liệu — Nguồn Hỏi–Đáp / Phi pháp lý

Đối với các chunk có `content_type != 'legal_text'` (ví dụ các cặp Hỏi–Đáp / FAQ), nội dung gốc được giữ nguyên — không thêm bất kỳ tiền tố cấu trúc nào:

```text
<nội dung văn bản gốc>
```

### Vector nhúng câu hỏi (Query embedding)

Câu hỏi bằng ngôn ngữ tự nhiên của người dùng được nhúng trực tiếp và nguyên vẹn:

```text
<câu hỏi ngôn ngữ tự nhiên gốc>
```

- Không thêm tiền tố siêu dữ liệu vào câu hỏi.
- Không áp dụng viết lại câu hỏi (query rewriting) hay phân loại câu hỏi (query classification).
- Không áp dụng bất kỳ phép biến đổi truy vấn nào ở giai đoạn hiện tại.

---

## Xem thêm

- [Retrieval — Kiến trúc và sử dụng](retrieval.md)
- [Đánh giá Retrieval — Kết quả đo lường](retrieval-evaluation.md)
- [Embedding và Indexing](embedding-and-indexing.md)
- [Quay lại README](../README.md)
