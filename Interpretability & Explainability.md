# ✅ Checklist học Interpretability & Explainability

## 📦 1. Chuẩn bị môi trường
- [ ] Cài đặt Python >= 3.8
- [ ] Cài các thư viện cần thiết:
  - [ ] `transformers`
  - [ ] `torch`
  - [ ] `lime`
  - [ ] `shap`
  - [ ] `captum` (dành cho PyTorch)
  - [ ] Các thư viện hỗ trợ visualization (matplotlib, seaborn)
  - [ ] (Tuỳ chọn) `bertviz` hoặc các công cụ chuyên biệt cho attention visualization

## 🧠 2. Chọn mô hình và task để phân tích
- [ ] Chọn mô hình NLP hoặc vision phù hợp
- [ ] Chuẩn bị dữ liệu đầu vào mẫu để phân tích (vd: câu, đoạn văn, ảnh)

## 🔍 3. Sử dụng công cụ giải thích đầu ra mô hình
- [ ] Áp dụng **LIME** để giải thích dự đoán đầu ra:
  - [ ] Tạo explainer phù hợp với kiểu dữ liệu
  - [ ] Trực quan hóa trọng số và tầm ảnh hưởng từng feature/token
- [ ] Áp dụng **SHAP** để đo tầm quan trọng từng input:
  - [ ] Chọn explainer SHAP phù hợp (DeepExplainer, KernelExplainer)
  - [ ] Visualize giá trị SHAP (force plot, summary plot)
- [ ] Sử dụng **Captum** (PyTorch) để phân tích attribution:
  - [ ] Gradient-based attribution (Integrated Gradients, Saliency)
  - [ ] Layer-wise relevance propagation

## 👁️ 4. Visualize attention và cơ chế bên trong model
- [ ] Dùng **bertviz** hoặc tương tự để xem attention weights
- [ ] Phân tích attention layers, heads có ảnh hưởng lớn đến output
- [ ] Thử thay đổi input và quan sát thay đổi attention

## 🧩 5. Hiểu cách model ra quyết định
- [ ] So sánh các giải thích từ các phương pháp khác nhau
- [ ] Tìm hiểu các pattern, xu hướng model ưu tiên
- [ ] Phân tích lỗi hoặc dự đoán sai để hiểu nguyên nhân

## ⚙️ 6. Tùy chọn nâng cao
- [ ] Kết hợp interpretability với fine-tuning để cải thiện mô hình
- [ ] Triển khai giải thích cho multi-modal hoặc các mô hình lớn hơn
- [ ] Tạo dashboard visualization cho việc giải thích model

## 📁 7. Quản lý và báo cáo
- [ ] Lưu lại kết quả giải thích và các visualizations
- [ ] Ghi chú các phát hiện, insight từ phân tích
- [ ] So sánh interpretability giữa các mô hình hoặc phiên bản
