# ✅ Checklist học BitFit (Bias Fine-tuning)

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Cài đặt các thư viện cần thiết:
  - [ ] `transformers`
  - [ ] `datasets`
  - [ ] `torch`
  - [ ] `evaluate`

## 📚 2. Chuẩn bị dữ liệu
- [ ] Chuẩn bị tập dữ liệu phù hợp với task
- [ ] Tiền xử lý dữ liệu (tokenization, padding, truncation)
- [ ] Phân chia dữ liệu train/validation/test rõ ràng

## 🧠 3. Chuẩn bị mô hình
- [ ] Load mô hình pretrained phù hợp (BERT, RoBERTa, GPT…)
- [ ] Đóng băng toàn bộ tham số mô hình ngoại trừ các bias term (chỉ cho phép các bias trainable)

## 🔧 4. Thiết lập huấn luyện
- [ ] Cấu hình optimizer chỉ cập nhật các bias parameters
- [ ] Chọn learning rate, batch size phù hợp
- [ ] Cài đặt loss function phù hợp với task

## 🚀 5. Huấn luyện mô hình
- [ ] Huấn luyện mô hình trên tập train chỉ với bias term được cập nhật
- [ ] Theo dõi metric và loss trên tập validation
- [ ] Điều chỉnh hyperparameters nếu cần

## 💾 6. Lưu và kiểm thử
- [ ] Lưu mô hình với các bias parameters đã fine-tune
- [ ] Đánh giá hiệu năng trên tập test
- [ ] So sánh hiệu quả với các phương pháp fine-tuning khác như full fine-tuning, adapter, LoRA

## ⚙️ 7. Triển khai và ứng dụng
- [ ] Chuẩn bị pipeline inference với mô hình đã tinh chỉnh bias
- [ ] Thử áp dụng cho các task tương tự để đánh giá tính khả dụng

## 📁 8. Quản lý experiment
- [ ] Ghi lại các thông số huấn luyện, kết quả và hyperparameters
- [ ] Sử dụng công cụ theo dõi experiment như `wandb` hoặc `tensorboard`
