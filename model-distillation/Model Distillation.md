# ✅ Checklist học Distillation (Model Distillation)

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Cài các thư viện cần thiết:
  - [ ] `transformers`
  - [ ] `datasets`
  - [ ] `torch`
  - [ ] `accelerate`
  - [ ] `evaluate`, `scipy`, `numpy`, `tqdm`
- [ ] (Tuỳ chọn) cài thêm `wandb` để theo dõi experiment

## 🧠 2. Chọn mô hình Teacher và Student
- [ ] Chọn mô hình lớn làm teacher (vd: BERT-large, GPT-2-xl, LLaMA-7B)
- [ ] Chọn mô hình nhỏ làm student (vd: BERT-base, DistilBERT, GPT-2-small)
- [ ] Load cả 2 mô hình và tokenizer tương ứng

## 📚 3. Chuẩn bị dữ liệu distillation
- [ ] Chọn tập dữ liệu huấn luyện phù hợp (có thể nhỏ)
- [ ] Tokenize, chuẩn bị input cho cả teacher và student
- [ ] Nếu cần, chuẩn bị thêm tập validation/test

## 🔄 4. Định nghĩa quy trình distillation
- [ ] Chọn dạng distillation:
  - [ ] Logit distillation (học output logits của teacher)
  - [ ] Feature-based distillation (học layer intermediate)
  - [ ] Task-specific distillation (học trên task cụ thể)
- [ ] Xây loss function kết hợp giữa loss student và loss với teacher
- [ ] Cấu hình optimizer, scheduler phù hợp

## 🚀 5. Huấn luyện student model
- [ ] Huấn luyện student bằng cách học từ teacher
- [ ] Theo dõi loss, accuracy trong quá trình train
- [ ] Kiểm tra tốc độ training và sử dụng tài nguyên

## 💾 6. Lưu và kiểm tra mô hình distill
- [ ] Lưu student model đã học
- [ ] Kiểm tra hiệu năng trên tập validation/test
- [ ] So sánh tốc độ inference và độ chính xác so với teacher

## ⚙️ 7. Tùy chọn nâng cao
- [ ] Thử nghiệm với các tỉ lệ loss khác nhau giữa teacher và student
- [ ] Áp dụng knowledge distillation kết hợp với quantization / pruning
- [ ] Thử distill trên nhiều task khác nhau
- [ ] Tinh chỉnh hyperparameters để tối ưu hiệu quả

## 📁 8. Quản lý experiment
- [ ] Ghi lại kết quả, loss curves, accuracy, inference time
- [ ] Sử dụng công cụ quản lý experiment (wandb, tensorboard, mlflow)
- [ ] So sánh nhiều mô hình distill khác nhau
