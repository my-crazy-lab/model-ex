# Tổng hợp các kỹ thuật Fine-tuning hiện nay

Fine-tuning là quá trình điều chỉnh lại một mô hình đã được huấn luyện trước (pretrained model) để phù hợp hơn với một tác vụ cụ thể. Dưới đây là các kỹ thuật fine-tuning phổ biến và hiện đại, kèm theo giải thích chi tiết.

---

## 1. Full Fine-tuning

* **Mô tả:** Tinh chỉnh toàn bộ trọng số của mô hình pretrained trên tập dữ liệu mới.
* **Chi tiết:** Bạn cần một tập dữ liệu đủ lớn và tài nguyên phần cứng mạnh (như GPU/TPU). Phù hợp khi dữ liệu mới khác biệt rõ rệt so với dữ liệu gốc.
* **Ưu điểm:** Hiệu quả tối ưu cho bài toán cụ thể.
* **Nhược điểm:** Rất tốn bộ nhớ và thời gian; dễ bị overfitting nếu dữ liệu huấn luyện nhỏ.

---

## 2. Feature-based Fine-tuning (Transfer Learning)

* **Mô tả:** Sử dụng mô hình pretrained như một bộ trích xuất đặc trưng (feature extractor), sau đó huấn luyện một tầng mới phía trên.
* **Chi tiết:** Các lớp gốc giữ nguyên, chỉ huấn luyện classifier phía trên.
* **Ưu điểm:** Nhanh, tiết kiệm tài nguyên, vẫn tận dụng tốt mô hình lớn.
* **Nhược điểm:** Kém linh hoạt hơn, đôi khi không tối ưu nếu tác vụ mới khác biệt nhiều.

---

## 3. Prompt Tuning / Prefix Tuning

* **Mô tả:** Thay vì thay đổi trọng số mô hình, ta huấn luyện một chuỗi vector đặc biệt (prompt hoặc prefix) để thêm vào đầu input.
* **Chi tiết:** Mô hình chính không thay đổi, chỉ học các vector ngữ cảnh đầu vào.
* **Ưu điểm:** Rất nhẹ, hiệu quả cao với LLMs như GPT, T5. Dễ mở rộng sang nhiều tác vụ.
* **Nhược điểm:** Cần kỹ thuật prompt tốt; đôi khi khó đạt hiệu năng như fine-tuning đầy đủ.

---

## 4. Adapter Tuning

* **Mô tả:** Thêm các module nhỏ (adapter) vào giữa các lớp của mô hình, chỉ tinh chỉnh các module này.
* **Chi tiết:** Các adapter thường là mạng nhỏ (2-layer MLP) chèn vào sau mỗi lớp transformer.
* **Ưu điểm:** Hiệu quả bộ nhớ cao, tránh phải tinh chỉnh toàn bộ mô hình. Dễ triển khai trên nhiều tác vụ.
* **Nhược điểm:** Cần tích hợp vào kiến trúc mô hình; đôi khi cần điều chỉnh lại codebase.

---

## 5. LoRA (Low-Rank Adaptation)

* **Mô tả:** Thay vì cập nhật trọng số trực tiếp, LoRA thêm các ma trận có thứ hạng thấp để điều chỉnh các phép biến đổi tuyến tính.
* **Chi tiết:** Giảm số lượng tham số cần học bằng cách giả định các thay đổi có thể được biểu diễn trong không gian thấp hơn.
* **Ưu điểm:** Rất tiết kiệm tài nguyên; hiệu năng gần bằng fine-tuning đầy đủ.
* **Nhược điểm:** Phức tạp hơn để tích hợp; yêu cầu hỗ trợ từ framework như PEFT (HuggingFace).

---

## 6. BitFit

* **Mô tả:** Chỉ tinh chỉnh các bias term trong mô hình.
* **Chi tiết:** Bias (độ lệch) là phần nhỏ nhưng có thể giúp mô hình thích nghi với nhiệm vụ mới.
* **Ưu điểm:** Rất nhẹ, nhanh, tiết kiệm bộ nhớ.
* **Nhược điểm:** Không hiệu quả với tác vụ quá khác biệt; hiệu năng thấp hơn các kỹ thuật khác trong nhiều trường hợp.

---

## 7. Knowledge Distillation

* **Mô tả:** Huấn luyện một mô hình nhỏ hơn (student) học theo đầu ra của mô hình lớn hơn (teacher).
* **Chi tiết:** Mô hình student cố gắng bắt chước phân phối xác suất hoặc embedding từ teacher.
* **Ưu điểm:** Giảm kích thước mô hình, tăng tốc độ suy luận, dễ triển khai thực tế.
* **Nhược điểm:** Cần mô hình teacher tốt; quá trình distillation có thể phức tạp.

---

## 8. Multitask Fine-tuning

* **Mô tả:** Tinh chỉnh mô hình trên nhiều tác vụ khác nhau đồng thời.
* **Chi tiết:** Mô hình chia sẻ phần lớn tham số, chỉ khác biệt ở đầu ra hoặc các token hướng dẫn (instruction).
* **Ưu điểm:** Tăng khả năng tổng quát hóa, cải thiện khả năng đa nhiệm.
* **Nhược điểm:** Khó thiết kế dữ liệu và cân bằng các tác vụ hợp lý.

---

## 9. Continual / Lifelong Learning

* **Mô tả:** Cho phép mô hình học liên tục với dữ liệu mới mà không quên kiến thức cũ.
* **Chi tiết:** Dùng kỹ thuật như Elastic Weight Consolidation (EWC), rehearsal (phát lại dữ liệu cũ), hoặc regularization.
* **Ưu điểm:** Phù hợp trong môi trường dữ liệu thay đổi liên tục.
* **Nhược điểm:** Khó kiểm soát hiện tượng "quên lãng thảm họa" (catastrophic forgetting).

---

## So sánh nhanh các kỹ thuật

| Kỹ thuật               | Tinh chỉnh trọng số | Tiết kiệm bộ nhớ | Phù hợp cho            | Ưu điểm chính                        |
| ---------------------- | ------------------- | ---------------- | ---------------------- | ------------------------------------ |
| Full Fine-tuning       | Toàn bộ             | ❌                | Tập dữ liệu lớn        | Hiệu quả cao                         |
| Feature-based          | Một phần            | ✅                | Dữ liệu nhỏ, nhanh     | Nhanh, dễ triển khai                 |
| Prompt / Prefix Tuning | Không               | ✅✅               | LLMs như GPT, T5       | Rất nhẹ, linh hoạt                   |
| Adapter                | Adapter modules     | ✅✅               | Nhiều tác vụ           | Modular, dễ tái sử dụng              |
| LoRA                   | Ma trận low-rank    | ✅✅✅              | Mô hình lớn (LLMs)     | Hiệu quả + tiết kiệm                 |
| BitFit                 | Bias terms          | ✅✅✅              | Nhanh, nhẹ             | Rất tiết kiệm                        |
| Knowledge Distillation | -                   | ✅✅               | Triển khai mô hình nhỏ | Sinh mô hình nhẹ hơn, nhanh hơn      |
| Multitask Fine-tuning  | Toàn bộ / một phần  | ✅                | Nhiệm vụ đa tác vụ     | Tổng quát hóa tốt                    |
| Continual Learning     | Có (chọn lọc)       | ✅                | Dữ liệu liên tục       | Học liên tục không quên kiến thức cũ |

