# ✅ Checklist học Multi-modal Fusion & Cross-modal Training

## 📦 1. Chuẩn bị môi trường
- [ ] Cài Python >= 3.8
- [ ] Cài các thư viện chính:
  - [ ] `transformers` (có hỗ trợ các model multi-modal như CLIP, BLIP, Flamingo)
  - [ ] `datasets`
  - [ ] `torch`
  - [ ] `timm` (nếu dùng vision backbone)
  - [ ] `accelerate`
  - [ ] `evaluate`
  - [ ] Các thư viện xử lý ảnh/audio như `PIL`, `opencv-python`, `torchaudio`

## 📚 2. Chuẩn bị dữ liệu multi-modal
- [ ] Chọn tập dữ liệu có chứa nhiều modal (vd: COCO captions, VQA, AudioCaps)
- [ ] Tiền xử lý dữ liệu:
  - [ ] Text tokenization
  - [ ] Image preprocessing (resize, normalize)
  - [ ] Audio feature extraction (spectrogram, mel-frequency cepstral coefficients)
- [ ] Tạo dataset với input multi-modal (ví dụ: cặp text + image, text + audio)

## 🧠 3. Chọn mô hình multi-modal
- [ ] Lựa chọn các kiến trúc tiêu biểu:
  - [ ] CLIP (text-image)
  - [ ] BLIP (Bootstrapped Language-Image Pre-training)
  - [ ] Flamingo (multi-modal few-shot learning)
- [ ] Load pre-trained model và tokenizer tương ứng

## 🧩 4. Fine-tune hoặc huấn luyện cross-modal
- [ ] Cấu hình fine-tuning cho multi-modal input
- [ ] Thiết lập pipeline xử lý input đa modal (vd: text + image)
- [ ] Huấn luyện model trên task multi-modal (captioning, VQA, retrieval)
- [ ] Theo dõi metric riêng cho từng modal và tổng thể

## 🔗 5. Xây dựng pipeline inference đa modal
- [ ] Tạo pipeline nhận input đa modal và output kết quả
- [ ] Test pipeline với dữ liệu thực tế
- [ ] Tối ưu inference (batching, quantization nếu cần)

## 🧪 6. Đánh giá & thử nghiệm
- [ ] Đánh giá model trên các benchmark multi-modal (ví dụ: COCO, VQA)
- [ ] Thử nghiệm khả năng generalization với dữ liệu cross-modal khác
- [ ] So sánh hiệu suất với baseline (mô hình đơn modal)

## ⚙️ 7. Tùy chọn nâng cao
- [ ] Kết hợp PEFT / LoRA để fine-tune multi-modal model tiết kiệm tài nguyên
- [ ] Thử kiến trúc multi-modal mới hoặc tự thiết kế fusion layer
- [ ] Áp dụng kỹ thuật self-supervised learning cho multi-modal data
- [ ] Tích hợp thêm modal khác (ví dụ audio, video)

## 📁 8. Quản lý experiment và mô hình
- [ ] Lưu trữ mô hình, tokenizer, config
- [ ] Ghi log quá trình huấn luyện, metric, lỗi
- [ ] Dùng công cụ như `wandb`, `tensorboard` để theo dõi
