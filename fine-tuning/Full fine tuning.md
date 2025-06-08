# ✅ Checklist học Full Fine-tuning

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Cài đặt các thư viện cần thiết:
  - [ ] `transformers`
  - [ ] `datasets`
  - [ ] `torch`
  - [ ] `accelerate` (nếu dùng đa GPU)
  - [ ] `evaluate` (đánh giá hiệu năng)

## 📚 2. Chuẩn bị dữ liệu
- [ ] Chuẩn bị tập dữ liệu đủ lớn và chất lượng cao phù hợp task
- [ ] Tiền xử lý dữ liệu:
  - [ ] Tokenization
  - [ ] Xử lý đặc biệt (padding, truncation)
- [ ] Phân chia dữ liệu train/validation/test rõ ràng

## 🧠 3. Chuẩn bị mô hình
- [ ] Chọn mô hình pretrained phù hợp với task (vd: BERT, GPT, T5)
- [ ] Load mô hình và tokenizer

## 🔧 4. Thiết lập fine-tuning
- [ ] Đặt toàn bộ tham số model ở chế độ trainable
- [ ] Cấu hình optimizer (AdamW phổ biến)
- [ ] Chọn learning rate, scheduler phù hợp (vd: linear warmup)
- [ ] Thiết lập batch size và số epoch dựa trên tài nguyên
- [ ] (Tuỳ chọn) Dùng gradient clipping để ổn định training

## 🚀 5. Huấn luyện mô hình
- [ ] Chạy fine-tuning trên tập train
- [ ] Theo dõi loss và metric trên tập validation
- [ ] Kiểm soát overfitting (early stopping, regularization nếu cần)

## 💾 6. Lưu và kiểm thử mô hình
- [ ] Lưu mô hình fine-tuned
- [ ] Đánh giá trên tập test để kiểm tra hiệu năng cuối cùng
- [ ] So sánh kết quả với baseline hoặc mô hình chưa fine-tune

## ⚙️ 7. Tối ưu và triển khai
- [ ] (Tuỳ chọn) Pruning, quantization để giảm model size
- [ ] Chuẩn bị pipeline inference hiệu quả
- [ ] Triển khai model lên môi trường sản xuất (API, embedded devices)

## 📁 8. Quản lý experiment
- [ ] Ghi lại chi tiết hyperparameters, kết quả
- [ ] Sử dụng công cụ theo dõi experiment (wandb, tensorboard)
- [ ] Đánh giá hiệu năng, tài nguyên tiêu thụ
