# ✅ Checklist học Adapter Tuning

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Cài đặt các thư viện cần thiết:
  - [ ] `transformers`
  - [ ] `adapter-transformers` (Hugging Face Adapter Hub)
  - [ ] `datasets`
  - [ ] `torch`
  - [ ] `evaluate`

## 📚 2. Chuẩn bị dữ liệu
- [ ] Chuẩn bị tập dữ liệu phù hợp với task
- [ ] Tiền xử lý dữ liệu (tokenization, padding, truncation)
- [ ] Phân chia train/validation/test

## 🧠 3. Chuẩn bị mô hình và adapter
- [ ] Load mô hình pretrained phù hợp (BERT, RoBERTa, GPT…)
- [ ] Thêm các adapter module (2-layer MLP) vào vị trí thích hợp trong mô hình (sau mỗi layer transformer)
- [ ] Đóng băng toàn bộ trọng số gốc, chỉ train adapter modules

## 🔧 4. Thiết lập huấn luyện adapter
- [ ] Cấu hình optimizer chỉ update tham số adapter
- [ ] Chọn learning rate, batch size phù hợp
- [ ] Cấu hình loss function theo task

## 🚀 5. Huấn luyện adapter
- [ ] Huấn luyện adapter trên dữ liệu train
- [ ] Theo dõi metric trên tập validation
- [ ] Điều chỉnh hyperparameters khi cần thiết

## 💾 6. Lưu và kiểm thử
- [ ] Lưu mô hình cùng adapter modules đã fine-tune
- [ ] Đánh giá hiệu năng trên tập test
- [ ] So sánh với full fine-tuning hoặc các phương pháp PEFT khác

## ⚙️ 7. Triển khai và mở rộng
- [ ] Tích hợp adapter vào pipeline inference
- [ ] Thử áp dụng cho nhiều task khác nhau bằng cách load adapter tương ứng
- [ ] Quản lý nhiều adapter song song trên cùng mô hình

## 📁 8. Quản lý experiment
- [ ] Ghi lại các thông số huấn luyện, kết quả và cấu hình adapter
- [ ] Sử dụng công cụ theo dõi experiment như `wandb` hoặc `tensorboard`
