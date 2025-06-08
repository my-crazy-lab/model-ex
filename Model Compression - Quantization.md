# ✅ Checklist học Model Compression / Quantization

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Cài các thư viện cần thiết:
  - [ ] `transformers`
  - [ ] `bitsandbytes` (hỗ trợ quantization 8-bit, 4-bit)
  - [ ] `torch`
  - [ ] `accelerate`
  - [ ] `numpy`
  - [ ] `datasets` (tuỳ chọn)

## 🧠 2. Chọn mô hình cần nén
- [ ] Chọn mô hình pre-trained phù hợp (GPT, BERT, LLaMA, ...)
- [ ] Tải mô hình full precision (FP32) làm baseline

## 🔧 3. Thực hiện quantization
- [ ] Chọn dạng quantization phù hợp:
  - [ ] 8-bit (INT8)
  - [ ] 4-bit (INT4 hoặc 4-bit quantrimization)
- [ ] Sử dụng `bitsandbytes` hoặc các công cụ hỗ trợ khác để quantize mô hình
- [ ] Kiểm tra mô hình đã quantize có tải lên và inference đúng

## 🚀 4. Đánh giá hiệu năng và tốc độ
- [ ] So sánh kích thước mô hình trước và sau khi quantize
- [ ] Đánh giá tốc độ inference trên GPU/CPU
- [ ] Đo lường độ chính xác, loss, hoặc metric phù hợp trên tập test
- [ ] Cân bằng giữa chất lượng (accuracy) và tốc độ/inference time

## 🔄 5. Tinh chỉnh và tối ưu
- [ ] Thử các kỹ thuật bổ trợ để cải thiện chất lượng khi quantize (vd: quantrimization aware training - QAT)
- [ ] Thử kết hợp quantization với pruning hoặc knowledge distillation
- [ ] Thử nhiều thiết lập batch size, sequence length để tối ưu tốc độ

## 💾 6. Lưu và triển khai mô hình compressed
- [ ] Lưu mô hình quantized
- [ ] Tạo pipeline hoặc script inference tối ưu cho mô hình đã nén
- [ ] Kiểm tra deploy trên các môi trường có tài nguyên hạn chế

## 📁 7. Quản lý experiment
- [ ] Ghi lại các kết quả đo lường (accuracy, tốc độ, kích thước)
- [ ] Sử dụng công cụ theo dõi như `wandb`, `tensorboard` nếu cần
- [ ] So sánh nhiều phương pháp compression khác nhau
