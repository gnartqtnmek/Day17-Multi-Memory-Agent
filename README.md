# 🚀 Lab 17: Hệ thống Agent Đa Trí Nhớ (Multi-Memory Agent)

## 👨‍💻 Thông tin học viên

* **Họ và tên:** Nguyễn Thị Quỳnh Trang
* **Mã học viên / Lớp:** 2A202600406
* **Mục tiêu ứng dụng:** Module lõi cho dự án LMS Chatbot có trí nhớ (AI20K-015).

---

## 🎯 Tổng quan

> "Nếu Agent không có trí nhớ, nó chỉ là một cỗ máy vô hồn phản hồi theo khuôn mẫu."

Dự án này xây dựng một **Hệ thống bộ nhớ đa tầng (Multi-Memory Stack)** chuyên nghiệp sử dụng **LangGraph**. Hệ thống chứng minh khả năng lưu trữ, phân tích và truy xuất ngữ cảnh đa luồng tương tự tư duy con người, phục vụ trực tiếp cho các nền tảng đòi hỏi tính cá nhân hóa cao.

---

## 🛠️ Kiến trúc Bộ nhớ (The Memory Stack)

### 1. Short-term & Long-term Memory (Cá nhân hóa)

* **Short-term (Window Buffer):** Giữ ngữ cảnh hội thoại hiện tại, cắt tỉa thông minh (auto-trim) bằng Tiktoken để tối ưu Context Window.
* **Long-term (Redis):** Tích hợp LLM-based Extraction (Pydantic Structured Output) để tự động nhận diện, lưu trữ và **giải quyết xung đột (Conflict Handling)** các sở thích, thông tin cá nhân vĩnh viễn.

### 2. Episodic & Semantic Memory (Tri thức & Kinh nghiệm)

* **Episodic (JSON Log):** Ghi nhận chuỗi hành động và kết quả (trajectory & outcome) để Agent tự rút kinh nghiệm.
* **Semantic (ChromaDB):** Ứng dụng OpenAI Embeddings để lưu trữ kiến thức chuyên ngành. Hệ thống được nạp sẵn `dataset.json` để thực hiện RAG (Retrieval-Augmented Generation) chuẩn xác.

---

## 🔧 Hướng dẫn cài đặt & Chạy hệ thống

```bash
# 1. Cài đặt các thư viện phụ thuộc (Dependencies)
pip install -r requirements.txt

# 2. Cấu hình môi trường (Bắt buộc)
# Tạo file .env tại thư mục gốc và cấu hình:
# OPENAI_API_KEY="sk-proj-..."
# REDIS_URL="redis://localhost:6379/0"

# 3. Chạy Benchmark & Tạo Report tự động
python main.py benchmark

# 4. Chạy chế độ Tương tác trực tiếp (Interactive Chat)
python main.py chat
```

---

## 🛡️ 5. Reflection: Quyền riêng tư & Giới hạn hệ thống (Privacy & Limitations)

### ✅ Rủi ro về Quyền riêng tư (Privacy & PII Risk)

**Rủi ro cốt lõi:** Long-term Memory (Redis) lưu trữ thông tin cá nhân (PII) dưới dạng Entity vĩnh viễn xuyên suốt các phiên làm việc. Nếu triển khai trên hệ thống phân quyền lỏng lẻo, nguy cơ rò rỉ dữ liệu nhạy cảm của người dùng là cực kỳ lớn.

**Giải pháp Privacy-by-Design:**

* Hệ thống hiện tại đã xử lý thành công *"Right to be Forgotten"* (tự động trích xuất lệnh DELETE khi user yêu cầu xóa thông tin).
* Khi đưa lên Production, cần thiết lập thêm TTL (Time-To-Live) cho Redis keys để dữ liệu tự động hủy sau 30–90 ngày, kèm theo module xin phép (Explicit Consent) trước khi lưu fact.

---

### ✅ Giới hạn Kỹ thuật (Technical Limitations)

**Chi phí & Độ trễ (Cost & Latency):**
Thuật toán LLM-based extraction yêu cầu gọi mô hình AI ít nhất 2 lần/lượt chat (1 lần sinh phản hồi, 1 lần chạy ngầm để Upsert/Delete DB). Việc này làm tăng gấp đôi chi phí token và độ trễ, có thể trở thành bottleneck khi scale lên hàng chục ngàn user.

**Xung đột ngầm định (Implicit Conflict):**
Hệ thống xử lý tốt các xung đột trực tiếp (ví dụ: “À nhầm, tôi thích B chứ không phải A”). Tuy nhiên, với sự thay đổi hành vi ngầm định qua thời gian dài, prompt phân tích có thể gặp hiện tượng *hallucination* khi quyết định ghi đè hay giữ nguyên dữ liệu.

---

## ⚠️ Lưu ý quan trọng

Bắt buộc chạy lệnh:

```bash
python main.py benchmark
```

để hệ thống sinh ra file báo cáo `benchmark/Benchmark_Report.md`. Đây là file quan trọng nhất để đánh giá các chỉ số KPI như:

* Response relevance
* Memory hit rate
* Token efficiency

giữa Agent có trí nhớ và không có trí nhớ.
