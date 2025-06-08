# ✅ Checklist học Continual / Lifelong Learning

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Cài đặt các thư viện cần thiết:
  - [ ] `transformers`
  - [ ] `datasets`
  - [ ] `torch`
  - [ ] `evaluate`
  - [ ] Thư viện hỗ trợ kỹ thuật continual learning (ví dụ: `avalanche`, `continuum`)

## 📚 2. Chuẩn bị dữ liệu
- [ ] Chuẩn bị nhiều tập dữ liệu lần lượt theo từng "giai đoạn" hoặc "task"
- [ ] Tiền xử lý dữ liệu (tokenization, padding, truncation)
- [ ] Thiết kế dữ liệu để mô phỏng môi trường học liên tục (streaming hoặc batches)

## 🧠 3. Chuẩn bị mô hình
- [ ] Load mô hình pretrained ban đầu
- [ ] Xác định kiến trúc mô hình và điểm cần áp dụng kỹ thuật chống quên lãng

## 🔧 4. Triển khai kỹ thuật continual learning
- [ ] Elastic Weight Consolidation (EWC):
  - [ ] Tính toán Fisher information matrix
  - [ ] Thêm penalty loss để giữ trọng số quan trọng
- [ ] Rehearsal:
  - [ ] Lưu lại một phần dữ liệu cũ (buffer)
  - [ ] Kết hợp dữ liệu cũ với dữ liệu mới khi huấn luyện
- [ ] Regularization khác (vd: L2 regularization, SI)
- [ ] Chọn cấu hình optimizer, learning rate, batch size phù hợp

## 🚀 5. Huấn luyện mô hình liên tục
- [ ] Huấn luyện mô hình theo từng giai đoạn dữ liệu mới
- [ ] Theo dõi hiệu năng trên dữ liệu cũ và mới để phát hiện quên lãng
- [ ] Điều chỉnh hyperparameters kỹ thuật chống quên lãng khi cần

## 💾 6. Lưu và đánh giá
- [ ] Lưu mô hình sau mỗi giai đoạn học
- [ ] Đánh giá mô hình trên toàn bộ dữ liệu cũ và mới
- [ ] So sánh hiệu quả giữa các kỹ thuật chống quên lãng

## ⚙️ 7. Triển khai và ứng dụng
- [ ] Triển khai mô hình có khả năng học liên tục trong môi trường thực tế
- [ ] Chuẩn bị pipeline cho cập nhật mô hình định kỳ hoặc streaming data

## 📁 8. Quản lý experiment
- [ ] Ghi lại chi tiết các giai đoạn học, kỹ thuật áp dụng, kết quả hiệu năng
- [ ] Sử dụng công cụ theo dõi experiment như `wandb`, `tensorboard`
