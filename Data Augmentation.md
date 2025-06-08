# ✅ Checklist học Data Augmentation / Synthetic Data Generation

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Cài các thư viện cần thiết:
  - [ ] `transformers`
  - [ ] `datasets`
  - [ ] `nlpaug` hoặc các thư viện augment text khác (ví dụ: `textattack`, `augmenty`)
  - [ ] `torch`, `numpy`
  - [ ] Các thư viện hỗ trợ tạo dữ liệu synthetic (vd: LLM API, TTS, tạo ảnh)

## 📚 2. Chuẩn bị dữ liệu gốc
- [ ] Lựa chọn bộ dữ liệu ban đầu để augment
- [ ] Phân tích đặc điểm dữ liệu (đa dạng, cân bằng, thiếu dữ liệu ở nhãn nào…)

## 🔄 3. Triển khai Data Augmentation
- [ ] Áp dụng các kỹ thuật augment text cơ bản:
  - [ ] Synonym replacement (thay từ đồng nghĩa)
  - [ ] Random insertion, deletion, swap
  - [ ] Back translation
- [ ] Sử dụng mô hình ngôn ngữ lớn (LLM) để tạo data synthetic:
  - [ ] Sinh câu hỏi mới, câu trả lời mới
  - [ ] Sinh dữ liệu theo template
- [ ] Tạo dữ liệu synthetic cho các dạng khác (nếu cần):
  - [ ] TTS (chuyển văn bản thành giọng nói)
  - [ ] Tạo ảnh/ảnh kèm text (sử dụng diffusion model, GAN…)

## 🧹 4. Tiền xử lý và lọc dữ liệu sinh
- [ ] Loại bỏ dữ liệu không hợp lệ, lặp hoặc nhiễu
- [ ] Kiểm tra chất lượng dữ liệu bằng metric (fluency, diversity)
- [ ] Cân bằng nhãn và phân bố dữ liệu mới

## 🧠 5. Kết hợp và huấn luyện mô hình
- [ ] Kết hợp dữ liệu gốc và dữ liệu augment/synthetic
- [ ] Huấn luyện/fine-tune mô hình với tập dữ liệu mở rộng
- [ ] Theo dõi hiệu quả mô hình (độ chính xác, robustness…)

## 📊 6. Đánh giá tác động augmentation
- [ ] So sánh hiệu suất mô hình trước và sau augmentation
- [ ] Thử nghiệm trên các tập dữ liệu ngoài để kiểm tra generalization
- [ ] Đánh giá tính đa dạng và chất lượng dữ liệu tạo ra

## ⚙️ 7. Tùy chọn nâng cao
- [ ] Tạo data synthetic với prompt engineering cho LLM
- [ ] Sử dụng kỹ thuật active learning để chọn dữ liệu augment hữu ích
- [ ] Thử nghiệm augmentation cho các task phức tạp (ví dụ: summarization, QA)
- [ ] Kết hợp augmentation với PEFT / fine-tuning để tăng hiệu quả

## 📁 8. Quản lý dữ liệu và experiment
- [ ] Lưu trữ rõ ràng dữ liệu augment và dữ liệu gốc
- [ ] Ghi chép chi tiết quá trình và kết quả huấn luyện
- [ ] Sử dụng công cụ quản lý experiment (wandb, tensorboard, mlflow)
