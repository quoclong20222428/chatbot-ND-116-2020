# Xây dựng hệ thống hỏi đáp Nghị định 116/2020/NĐ-CP dựa trên Retrieval-Augmented Generation (RAG)

## 1. Giới thiệu dự án

Đây là dự án xây dựng hệ thống hỏi đáp pháp luật tập trung vào **Nghị định 116/2020/NĐ-CP**. Dữ liệu gồm nghị định chính, các văn bản liên quan, **Nghị định 60/2025/NĐ-CP** trong vai trò văn bản sửa đổi khi có quan hệ áp dụng, và **Luật Giáo dục 2019** để bổ sung định nghĩa hoặc ngữ cảnh pháp lý.

Bộ câu hỏi–trả lời về Nghị định 116 chỉ là nguồn tham khảo, không thay thế và không có mức ưu tiên cao hơn nguồn pháp luật chính thức.

Theo định hướng RAG, hệ thống sẽ nhận câu hỏi, truy xuất các đoạn pháp lý liên quan, ưu tiên nguồn có thẩm quyền, xem xét quan hệ giữa văn bản gốc và văn bản sửa đổi, rồi cung cấp câu trả lời dựa trên nội dung đã truy xuất.

Hiện repository tập trung vào pipeline chuẩn bị, kiểm tra và import dữ liệu vào PostgreSQL. Các thành phần truy vấn, sinh câu trả lời, giao diện và tích hợp mô hình ngôn ngữ chưa được xem là hoàn thiện nếu chưa có mã tương ứng.

## 2. Mục tiêu

- Chuẩn hóa văn bản pháp lý thành các đoạn (\`legal chunks\`) có cấu trúc.
- Lưu siêu dữ liệu về văn bản, thẩm quyền, ưu tiên truy xuất và quan hệ văn bản.
- Lưu tham chiếu pháp lý ở dạng dữ liệu có cấu trúc.
- Cung cấp quy trình khởi tạo và import có thể chạy lại an toàn.

## 3. Kiến trúc xử lý dữ liệu hiện tại

\`\`\`text
Tệp Markdown pháp lý và tệp hỏi–đáp
        ↓
scripts/legal_chunker.py
        ↓
data/processed/legal_chunks.jsonl
        ↓
scripts/import_legal_data.py
        ↓
init.sql + PostgreSQL
        ↓
documents, legal_chunks, legal_chunk_references
\`\`\`

\`init.sql\` là nguồn sự thật cho schema và quan hệ cơ sở dữ liệu. Script Python thực thi file này, sau đó đọc JSONL từng dòng để cập nhật dữ liệu và tái tạo bảng tham chiếu.

## 4. Nguồn dữ liệu pháp lý

- \`data/markdown/116_2020_ND-CP.md\`: Nghị định 116/2020/NĐ-CP, nguồn chính.
- \`data/markdown/60_2025_ND-CP.md\`: Nghị định 60/2025/NĐ-CP, văn bản sửa đổi liên quan.
- \`data/markdown/43_2019_QH14_Luat_Giao_duc_2019.md\`: Luật Giáo dục 2019, nguồn hỗ trợ.
- \`data/raw/qa/hoi-dap-nghi-dinh-116-2020-nd-cp.md\`: dữ liệu hỏi–đáp tham khảo.
- Các tệp DOCX trong \`data/raw/legal/\`: nguồn thô tương ứng.

Tệp được import là \`data/processed/legal_chunks.jsonl\`, không phải các tệp nguồn thô.

## 5. Công nghệ sử dụng

- Python trong môi trường Conda \`chatbot\`.
- PostgreSQL với extension \`pg_trgm\`, được tạo bởi \`init.sql\` nếu chưa tồn tại.
- \`psql\` để thực thi file SQL có lệnh PostgreSQL \`\\copy\`.
- \`psycopg\` hoặc \`psycopg2\` để kết nối từ Python.
- Định dạng JSONL, mỗi đoạn dữ liệu nằm trên một dòng.

## 6. Yêu cầu môi trường

Cần có Conda, PostgreSQL đang chạy và có thể kết nối bằng \`DATABASE_URL\`, PostgreSQL client \`psql\` trong \`PATH\` của môi trường \`chatbot\`, cùng một trong hai thư viện Python \`psycopg\` hoặc \`psycopg2\`.

Repository không yêu cầu Docker và không có \`requirements.txt\` hoặc \`pyproject.toml\`. PostgreSQL có thể chạy cục bộ hoặc trên máy chủ từ xa; máy chạy script phải truy cập được máy chủ và tài khoản phải có quyền tạo bảng, index và extension \`pg_trgm\`.

## 7. Cấu trúc thư mục liên quan

\`\`\`text
init.sql
README.md
scripts/
├── import_legal_data.py
├── legal_chunker.py
└── validate_legal_chunks.py
data/
├── markdown/
├── raw/
└── processed/
    └── legal_chunks.jsonl
\`\`\`

## 8. Hướng dẫn cài đặt

### Bước 1: Sao chép dự án và chuyển vào thư mục

\`\`\`powershell
git clone <URL_KHO_LUU_TRU>
cd RAG-chatbot
\`\`\`

Nếu đã có mã nguồn:

\`\`\`powershell
cd <DUONG_DAN_TOI_THU_MUC_RAG-chatbot>
\`\`\`

### Bước 2: Tạo và kích hoạt môi trường Conda

Chỉ tạo môi trường nếu môi trường chưa tồn tại:

\`\`\`powershell
conda create -n chatbot python
conda activate chatbot
\`\`\`

Cài \`psql\` và thư viện kết nối:

\`\`\`powershell
conda install -c conda-forge postgresql psycopg
\`\`\`

Nếu đã cài \`psycopg2\` thì có thể dùng thư viện đó thay cho \`psycopg\`.

\`\`\`powershell
where.exe psql
python --version
\`\`\`

## 9. Cấu hình biến môi trường

Tạo hoặc chỉnh sửa \`.env\` ở thư mục gốc:

\`\`\`env
DATABASE_URL=postgresql://<ten_nguoi_dung>:<mat_khau>@<may_chu>:<cong>/<ten_co_so_du_lieu>
\`\`\`

Đây chỉ là mẫu; không đưa thông tin đăng nhập thật vào README hoặc mã nguồn. Script ưu tiên biến \`DATABASE_URL\` trong môi trường, sau đó mới đọc \`.env\`.

Hoặc cấu hình cho phiên PowerShell hiện tại:

\`\`\`powershell
$env:DATABASE_URL = "postgresql://<ten_nguoi_dung>:<mat_khau>@<may_chu>:<cong>/<ten_co_so_du_lieu>"
\`\`\`

## 10. Khởi tạo database và import dữ liệu

Đảm bảo PostgreSQL đang chạy (cục bộ hoặc từ xa), database trong \`DATABASE_URL\` đã tồn tại, và các tệp sau còn nguyên:

\`\`\`text
init.sql
data/processed/legal_chunks.jsonl
scripts/import_legal_data.py
\`\`\`

Chạy từ thư mục gốc:

\`\`\`powershell
conda activate chatbot
python scripts/import_legal_data.py
\`\`\`

Script thực hiện:

\`\`\`text
Kiểm tra cấu hình môi trường và tệp đầu vào
    ↓
Đọc và thực thi init.sql
    ↓
Khởi tạo/cập nhật schema PostgreSQL
    ↓
Đọc data/processed/legal_chunks.jsonl từng dòng
    ↓
Upsert documents và legal_chunks
    ↓
Tái tạo legal_chunk_references
    ↓
Kiểm tra thống kê documents, chunks và references
\`\`\`

Các khóa chính và \`ON CONFLICT\` giúp chạy lại an toàn, không tạo bản ghi trùng. Phần import Python dùng transaction; khi lỗi, transaction được rollback và script kết thúc với thông báo lỗi.

## 11. Kiểm tra dữ liệu sau khi import

Các lệnh sau tương thích với schema hiện tại và chạy trong PowerShell:

\`\`\`powershell
psql --dbname "$env:DATABASE_URL" --command "SELECT count(*) AS documents FROM documents;"
psql --dbname "$env:DATABASE_URL" --command "SELECT count(*) AS legal_chunks FROM legal_chunks;"
psql --dbname "$env:DATABASE_URL" --command "SELECT count(*) AS reference_count FROM legal_chunk_references;"
\`\`\`

Kiểm tra phân bố theo vai trò và mức thẩm quyền:

\`\`\`powershell
psql --dbname "$env:DATABASE_URL" --command "SELECT document_role, count(*) AS chunks FROM legal_chunks GROUP BY document_role ORDER BY document_role;"
psql --dbname "$env:DATABASE_URL" --command "SELECT authority_level, count(*) AS chunks FROM legal_chunks GROUP BY authority_level ORDER BY authority_level;"
\`\`\`

Kiểm tra danh sách văn bản:

\`\`\`powershell
psql --dbname "$env:DATABASE_URL" --command "SELECT document_id, document_number, document_title FROM documents ORDER BY document_id;"
\`\`\`

## 12. Trạng thái phát triển

Đã có pipeline tạo/kiểm tra JSONL và import vào ba bảng \`documents\`, \`legal_chunks\`, \`legal_chunk_references\`, cùng cơ chế upsert để chạy lại. Truy xuất theo câu hỏi, xếp hạng kết quả, sinh câu trả lời bằng mô hình ngôn ngữ, giao diện và đánh giá chất lượng là các giai đoạn tiếp theo; README này không coi chúng là tính năng đã hoàn thành.
