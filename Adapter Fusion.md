# ✅ Checklist: Adapter Fusion / Multi-task Learning với Adapter (PEFT)

## 📦 1. Chuẩn bị môi trường
- [ ] Đảm bảo môi trường đã cài:
  - [ ] Python >= 3.8
  - [ ] `transformers`
  - [ ] `peft` (>= 0.4.0 nếu có hỗ trợ fusion)
  - [ ] `datasets`, `accelerate`, `scipy`, `numpy`
  - [ ] `bitsandbytes` (nếu muốn dùng low-bit model)

## 🧠 2. Chọn mô hình nền và adapter method
- [ ] Chọn mô hình pre-trained (BERT, T5, LLaMA, v.v.)
- [ ] Chọn adapter-based PEFT method: `LoRA`, `Prefix`, `Adapter`, v.v.
- [ ] Tạo adapter riêng cho từng task (vd: QA, sentiment, NLI...)

## 🎯 3. Chuẩn bị dữ liệu cho multi-task
- [ ] Chọn 2–3 task khác nhau (vd: SST-2, MNLI, SQuAD)
- [ ] Tải và xử lý từng tập dữ liệu theo đúng format
- [ ] Tạo bộ `DatasetDict` cho từng task
- [ ] Cân nhắc mix mẫu từ các task hoặc học song song (multi-task sampling)

## 🧩 4. Cấu hình và huấn luyện adapter riêng biệt
- [ ] Tạo `PeftConfig` và `get_peft_model` cho từng task adapter
- [ ] Huấn luyện từng adapter độc lập trên task riêng
- [ ] Lưu mỗi adapter vào folder riêng (vd: `adapter_sst2/`, `adapter_mnli/`)

## 🔗 5. Thực hiện Adapter Fusion
- [ ] Load lại mô hình gốc
- [ ] Load nhiều adapter đã huấn luyện
- [ ] Kết hợp bằng phương pháp **adapter fusion**
  - [ ] Dùng `peft` hoặc AdapterHub nếu hỗ trợ
  - [ ] Cấu hình `AdapterFusionConfig` (nếu dùng library khác)

## 🧪 6. Đánh giá mô hình fused
- [ ] Thử nghiệm adapter fused trên task kết hợp
- [ ] So sánh với từng adapter riêng
- [ ] Đánh giá khả năng tổng hợp kiến thức đa nhiệm

## ⚙️ 7. Tùy chọn nâng cao
- [ ] Thử fine-tune tiếp adapter fusion trên tập hợp multi-task
- [ ] Gán trọng số cho từng adapter trong quá trình inference
- [ ] So sánh với baseline full fine-tuning / multi-head

## 📁 8. Quản lý mô hình và kết quả
- [ ] Lưu từng adapter, config, tokenizer
- [ ] Ghi log và theo dõi kết quả huấn luyện (vd: dùng `wandb`, `mlflow`, `tensorboard`)
- [ ] Xuất báo cáo: loss, accuracy theo từng task và tổng thể
