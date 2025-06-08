# ✅ Checklist học Multitask Fine-tuning

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Cài đặt các thư viện cần thiết:
  - [ ] `transformers`
  - [ ] `datasets`
  - [ ] `torch`
  - [ ] `evaluate`
  - [ ] `accelerate` (nếu dùng đa GPU)

## 📚 2. Chuẩn bị dữ liệu
- [ ] Thu thập dữ liệu cho nhiều task khác nhau (vd: classification, QA, summarization)
- [ ] Tiền xử lý từng tập dữ liệu (tokenization, padding, truncation)
- [ ] Chuẩn hóa định dạng dữ liệu (vd: dùng prompt/instruction để phân biệt task)
- [ ] Kết hợp dữ liệu thành một dataset đa nhiệm
- [ ] Phân chia train/validation/test hợp lý cho từng task và tổng thể

## 🧠 3. Chuẩn bị mô hình
- [ ] Load mô hình pretrained làm backbone chung
- [ ] Thiết kế hoặc lựa chọn head/output layers phù hợp từng task (có thể dùng shared head với token hướng dẫn)
- [ ] Đảm bảo mô hình có khả năng xử lý input đa dạng (multi-input hoặc multi-prompt)

## 🔧 4. Thiết lập huấn luyện đa nhiệm
- [ ] Cấu hình optimizer, learning rate, batch size phù hợp
- [ ] Cân bằng trọng số loss giữa các task (task weighting)
- [ ] Cài đặt kỹ thuật điều phối dữ liệu (vd: sampling, batch mix)
- [ ] (Tuỳ chọn) Dùng kỹ thuật prompt hoặc adapter để hỗ trợ đa nhiệm

## 🚀 5. Huấn luyện mô hình
- [ ] Huấn luyện trên tập đa nhiệm, theo dõi loss và metric riêng từng task
- [ ] Điều chỉnh cân bằng loss và sampling nếu task nào quá yếu hoặc quá mạnh
- [ ] Kiểm soát overfitting và underfitting cho từng task

## 💾 6. Lưu và đánh giá
- [ ] Lưu mô hình fine-tuned đa nhiệm
- [ ] Đánh giá hiệu năng trên từng task riêng biệt và tổng thể
- [ ] So sánh với các mô hình đơn nhiệm hoặc baseline

## ⚙️ 7. Triển khai và mở rộng
- [ ] Chuẩn bị pipeline inference đa nhiệm, nhận biết task hoặc prompt tương ứng
- [ ] Thử nghiệm khả năng mở rộng sang task mới hoặc thêm dữ liệu mới
- [ ] Quản lý model và phiên bản cho từng nhiệm vụ

## 📁 8. Quản lý experiment
- [ ] Ghi lại chi tiết hyperparameters, loss weighting, kết quả cho từng task
- [ ] Dùng công cụ theo dõi experiment như `wandb`, `tensorboard`
