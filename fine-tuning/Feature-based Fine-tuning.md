# ✅ Checklist học Feature-based Fine-tuning (Transfer Learning)

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Cài đặt các thư viện cần thiết:
  - [ ] `transformers`
  - [ ] `datasets`
  - [ ] `torch`
  - [ ] `evaluate`

## 📚 2. Chuẩn bị dữ liệu
- [ ] Chuẩn bị tập dữ liệu phù hợp cho task mới
- [ ] Tiền xử lý dữ liệu (tokenization, padding, truncation)
- [ ] Phân chia dữ liệu train/validation/test

## 🧠 3. Chuẩn bị mô hình
- [ ] Load mô hình pretrained làm feature extractor
- [ ] Đóng băng (freeze) toàn bộ trọng số của mô hình gốc (không trainable)
- [ ] Thêm tầng classifier hoặc head mới phía trên phù hợp với task (vd: fully connected layer)

## 🔧 4. Thiết lập huấn luyện
- [ ] Chỉ fine-tune tầng classifier mới, giữ nguyên mô hình gốc
- [ ] Cấu hình optimizer chỉ cho các tham số của classifier
- [ ] Chọn learning rate, batch size phù hợp
- [ ] Thiết lập loss function phù hợp với task

## 🚀 5. Huấn luyện mô hình
- [ ] Huấn luyện classifier trên dữ liệu train
- [ ] Theo dõi loss và metric trên tập validation
- [ ] Đảm bảo mô hình gốc không bị update

## 💾 6. Lưu và kiểm thử
- [ ] Lưu mô hình với classifier fine-tuned
- [ ] Đánh giá trên tập test để đo hiệu năng
- [ ] So sánh với baseline hoặc mô hình full fine-tuning

## ⚙️ 7. Triển khai và tối ưu
- [ ] Chuẩn bị pipeline inference cho mô hình feature-based
- [ ] (Tuỳ chọn) Tinh chỉnh hyperparameters để cải thiện hiệu năng

## 📁 8. Quản lý experiment
- [ ] Ghi lại hyperparameters, kết quả huấn luyện
- [ ] Dùng công cụ theo dõi experiment như `wandb` hoặc `tensorboard`
