# ✅ Checklist học Prompt Tuning / Prefix Tuning

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Cài đặt các thư viện cần thiết:
  - [ ] `transformers`
  - [ ] `datasets`
  - [ ] `torch`
  - [ ] `peft` (Hugging Face PEFT thư viện hỗ trợ prompt tuning)
  - [ ] `accelerate`
  - [ ] `evaluate`

## 📚 2. Chuẩn bị dữ liệu
- [ ] Chuẩn bị tập dữ liệu phù hợp cho task cần fine-tune
- [ ] Tiền xử lý dữ liệu (tokenization, padding, truncation)
- [ ] Phân chia dữ liệu train/validation/test

## 🧠 3. Chuẩn bị mô hình
- [ ] Load mô hình pretrained (GPT, T5, hoặc LLM phù hợp)
- [ ] Đóng băng toàn bộ tham số mô hình (không trainable)
- [ ] Khởi tạo chuỗi vector prompt/prefix (prompt embeddings)

## 🔧 4. Thiết lập prompt/prefix tuning
- [ ] Thiết lập để chỉ học các vector prompt/prefix, không update trọng số mô hình chính
- [ ] Cấu hình optimizer cho các tham số prompt
- [ ] Chọn learning rate, batch size phù hợp

## 🚀 5. Huấn luyện prompt/prefix
- [ ] Huấn luyện prompt embeddings trên tập train
- [ ] Theo dõi loss và metric trên tập validation
- [ ] Điều chỉnh hyperparameters nếu cần

## 💾 6. Lưu và kiểm thử
- [ ] Lưu các vector prompt/prefix đã học
- [ ] Đánh giá mô hình với prompt tuning trên tập test
- [ ] So sánh hiệu năng với full fine-tuning và các kỹ thuật PEFT khác

## ⚙️ 7. Triển khai và ứng dụng
- [ ] Tích hợp prompt embeddings vào pipeline inference
- [ ] Thử nghiệm với nhiều task và prompt khác nhau
- [ ] (Tuỳ chọn) Kết hợp với kỹ thuật LoRA, adapter để tăng hiệu quả

## 📁 8. Quản lý experiment
- [ ] Ghi lại chi tiết hyperparameters, kết quả huấn luyện
- [ ] Sử dụng công cụ theo dõi experiment như `wandb` hoặc `tensorboard`
