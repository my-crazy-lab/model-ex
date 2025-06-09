# ✅ Checklist học LoRA / PEFT (Parameter-Efficient Fine-Tuning)

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Tạo môi trường ảo (virtualenv hoặc conda)
- [ ] Cài đặt PyTorch (phiên bản hỗ trợ GPU)
- [ ] Cài các thư viện chính:
  - [ ] `transformers` (Hugging Face)
  - [ ] `datasets` (Hugging Face)
  - [ ] `peft` (Hugging Face PEFT library)
  - [ ] `accelerate` (tối ưu tốc độ train)
  - [ ] `bitsandbytes` (hỗ trợ 8-bit, 4-bit)
  - [ ] `scipy`, `numpy`, `evaluate`, `tqdm`, `wandb` (tuỳ chọn)

## 📚 2. Chuẩn bị dữ liệu
- [ ] Lựa chọn tập dữ liệu phù hợp (ví dụ: từ Hugging Face Hub hoặc dữ liệu riêng)
- [ ] Tiền xử lý dữ liệu:
  - [ ] Tokenization
  - [ ] Padding/Truncation
- [ ] Tạo `Dataset` train/validation

## 🧠 3. Chọn mô hình nền (base model)
- [ ] Chọn mô hình pre-trained (ví dụ: `bert-base-uncased`, `llama`, `mistral`, `falcon`, v.v.)
- [ ] Tải model và tokenizer từ `transformers`

## 🧩 4. Cấu hình PEFT / LoRA
- [ ] Chọn phương pháp PEFT: `LoRA`, `Prefix-Tuning`, `Prompt-Tuning`, `IA3`
- [ ] Định nghĩa cấu hình PEFT (`LoraConfig`, `PrefixTuningConfig`...)
- [ ] Áp dụng cấu hình vào mô hình bằng `get_peft_model`

## 🚀 5. Huấn luyện mô hình
- [ ] Cấu hình huấn luyện (`TrainingArguments`, batch size, lr, epochs, v.v.)
- [ ] Dùng `Trainer` hoặc `Accelerate` để huấn luyện mô hình
- [ ] Theo dõi loss, eval metrics (tuỳ chọn: log bằng `wandb`)

## 💾 6. Lưu và kiểm tra mô hình
- [ ] Lưu mô hình PEFT đã fine-tune (`save_pretrained`)
- [ ] Load lại và kiểm tra mô hình trên tập test hoặc câu hỏi thực tế

## 🧪 7. Đánh giá & triển khai
- [ ] Đánh giá hiệu suất mô hình (accuracy, BLEU, F1, v.v.)
- [ ] So sánh với mô hình gốc hoặc fine-tune full
- [ ] Triển khai inference bằng `pipeline` hoặc custom script

## 💡 8. Tùy chọn nâng cao
- [ ] Sử dụng quantization (4-bit, 8-bit) với `bitsandbytes`
- [ ] Kết hợp PEFT với QLoRA để tiết kiệm hơn nữa
- [ ] Thử nghiệm với nhiều mô hình và phương pháp PEFT khác nhau
