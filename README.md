# Xây Dựng SQuAD 2.0 Cho Bài Toán Hỏi Đáp Tự Động Trong Lĩnh Vực Giáo Dục Tiếng Việt

**Tác giả:** Ngô Hữu Lễ & Lê Dương Minh Khoa  
**Trường:** Trường Đại học Tôn Đức Thắng  
**Người hướng dẫn:** TS. Trần Thanh Phước  
**Năm:** 2024  

## 📌 Mục Tiêu Dự Án
- Xây dựng bộ dữ liệu SQuAD 2.0 chất lượng cao dành cho tiếng Việt trong lĩnh vực giáo dục phổ thông.
- Huấn luyện mô hình hỏi đáp (QA) sử dụng kỹ thuật học chuyển giao với mô hình BERT.
- Phát triển ứng dụng web hỏi đáp tự động thân thiện với người dùng.

## 🧠 Phương Pháp
- Áp dụng **transfer learning** với mô hình **BERT**.
- Kết hợp **Intent Classification (IC)** bằng Logistic Regression và **Machine Reading Comprehension (MRC)**.
- Sử dụng **Flask** để xây dựng giao diện web tương tác.

## 📊 Dữ Liệu
- Tổng cộng **5000 cặp câu hỏi – trả lời** từ **858 đoạn văn bản** và **265 chủ đề**.
- Dữ liệu lấy từ sách giáo khoa, trắc nghiệm “Cánh diều”, “Chân trời sáng tạo” và Wikipedia.

## ✅ Kết Quả Thực Nghiệm
| Mô Hình              | F1 Score | Exact Match | BLEU Score |
|----------------------|----------|--------------|-------------|
| Logistic Regression (IC) | 0.55     | -            | -           |
| BERT (MRC)           | 0.053    | 0.029        | 0.027       |
| IC + MRC             | 0.17     | -            | -           |

- Kết hợp IC + MRC cải thiện đáng kể so với mô hình đơn lẻ.
- Tuy nhiên, hiệu quả còn hạn chế do dữ liệu chưa phong phú và mô hình chưa được tối ưu.

## 🌐 Ứng Dụng Web
- Được phát triển bằng Flask.
- Cho phép người dùng đặt câu hỏi trực tiếp qua trình duyệt.
- Lưu lịch sử tin nhắn và cho điểm F1 để đánh giá chất lượng câu trả lời.

## 🔍 Hướng Phát Triển
- Mở rộng và làm phong phú tập dữ liệu SQuAD tiếng Việt.
- Cải tiến kỹ thuật NLP và sử dụng các mô hình lớn hơn như GPT.
- Tối ưu hóa quá trình huấn luyện và đánh giá trên môi trường thực tế.

---

> Dự án này là bước đầu nghiên cứu và triển khai hệ thống hỏi đáp tiếng Việt ứng dụng trong giáo dục, kết hợp giữa công nghệ học sâu và dữ liệu ngôn ngữ tự nhiên.

---
