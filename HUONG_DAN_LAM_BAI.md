# HƯỚNG DẪN BÀI TẬP LỚN – NHẬP MÔN HỌC MÁY (2026–2027)
## Đề tài 3: Phân loại tin nhắn rác (Spam Detection)

GV: ThS. Lê Thị Thùy Trang · Nhóm 2–4 thành viên · Nhóm NLP

> Dùng **đúng bộ dữ liệu GV chỉ định** (link "đây" trong đề). Muốn đổi dữ liệu / cách đặt bài toán → phải được GV đồng ý trước.
> Các gợi ý bên dưới giả định đây là bài toán nhị phân `ham` / `spam` (kiểu SMS Spam Collection). Nếu dữ liệu của bạn khác (nhiều cột, email…), điều chỉnh cho phù hợp.

---

## 1. Yêu cầu bắt buộc

- [ ] Chạy **đủ pipeline** từ dữ liệu thô → đánh giá; không chỉ sao chép notebook mẫu.
- [ ] Có **baseline** làm mốc so sánh.
- [ ] **≥ 2 phương án thử nghiệm có ý nghĩa** (ưu tiên 2 thuật toán đã học).
- [ ] Tách train/test đúng; vectorizer **chỉ `fit` trên train** → tránh **data leakage** (dùng `Pipeline`).
- [ ] Ghi rõ tham số, cách chia dữ liệu, `random_state`.
- [ ] Phân tích kết quả, hạn chế, trường hợp sai; **không chỉ dùng Accuracy**.
- [ ] Mọi thành viên giải thích được pipeline, code, kết quả.

---

## 2. Pipeline cho bài spam

### Bước 1 – EDA
- Kích thước, tên cột, kiểu dữ liệu, giá trị thiếu, **dòng trùng lặp** (spam dataset thường có nhiều tin trùng → ảnh hưởng split, xử lý và giải thích).
- Phân phối nhãn: spam thường chỉ ~13% → **mất cân bằng lớp** (biểu đồ cột/pie).
- Độ dài tin nhắn (số ký tự, số từ) theo từng lớp – spam thường dài hơn.
- Đặc điểm gợi ý để khảo sát: số lượng chữ số, có URL, ký hiệu tiền (£, $), từ viết hoa toàn bộ, dấu `!`.
- Top từ phổ biến theo từng lớp (bar chart / wordcloud) → nêu giả thuyết ban đầu (vd: "free", "win", "call", "txt", "prize").

### Bước 2 – Tiền xử lý
- Lowercase, bỏ dấu câu/ký tự đặc biệt, khoảng trắng thừa.
- Với spam, **URL, số điện thoại, số, tiền tệ là tín hiệu quan trọng** → không xóa máy móc. Có thể thay bằng token (`<URL>`, `<NUM>`, `<MONEY>`) thay vì xóa.
- Stopword: thử **có/không bỏ** (stopword như "you", "to", "call" đôi khi mang tín hiệu).
- Thí nghiệm ablation: so sánh kết quả khi bật/tắt từng bước → là bằng chứng cho tiêu chí rubric 15%.
- Mã hóa nhãn: `ham=0`, `spam=1` (spam là lớp dương để Precision/Recall có ý nghĩa).

### Bước 3 – Vector hóa (cốt lõi)
- [ ] `CountVectorizer` (BoW)
- [ ] `TfidfVectorizer`
- [ ] So sánh: unigram vs (1,2)-gram; `min_df` (1, 2, 5); `max_df` (0.9, 1.0); `max_features`
- (Mở rộng, tùy chọn) Sentence embeddings, char n-gram, đặc trưng thủ công (độ dài, số chữ số) – không thay thế BoW/TF-IDF.

### Bước 4 – Mô hình
- **Multinomial Naive Bayes** – mô hình chính (hợp với đếm/TF-IDF). Thử `alpha` (0.1, 0.5, 1.0…).
- **Thuật toán thứ 2** (chọn 1 để so sánh): K-NN (thử `k`, cosine) hoặc Decision Tree (thử `max_depth`, `min_samples_leaf`).
- Có thể dùng `GridSearchCV` (giải thích phạm vi + tiêu chí chọn; với dữ liệu mất cân bằng nên chọn theo **F1 của lớp spam**, không phải accuracy).
- Chia dữ liệu: `train_test_split(test_size=0.2, stratify=y, random_state=42)`; có thể thêm cross-validation phân tầng (StratifiedKFold).

Bảng thí nghiệm gợi ý:

| ID | Vector hóa | Mô hình | Vai trò |
|---|---|---|---|
| E0 | BoW mặc định | MultinomialNB mặc định | **Baseline** |
| E1 | TF-IDF unigram | MultinomialNB | |
| E2 | TF-IDF (1,2)-gram | MultinomialNB (tune alpha) | |
| E3 | TF-IDF | KNN / Decision Tree | Thuật toán thứ 2 |
| E4 | Biến thể tiền xử lý (giữ/bỏ stopword, token hóa URL/số) | mô hình tốt nhất | Ablation |

### Bước 5 – Đánh giá
- **Accuracy, Precision, Recall, F1 (lớp spam)**, Confusion Matrix; có thể thêm ROC-AUC / PR-curve.
- Hiểu ý nghĩa nghiệp vụ:
  - **FP** (ham bị chặn nhầm thành spam) → mất tin quan trọng → thường nghiêm trọng hơn → quan tâm **Precision**.
  - **FN** (spam lọt qua) → phiền người dùng → quan tâm **Recall**.
- Bảng + biểu đồ so sánh mọi phương án; thảo luận ảnh hưởng của vector hóa & tham số.
- Chỉ ra vì sao Accuracy lừa dối: mô hình luôn đoán "ham" đã đạt ~87%.

### Bước 6 – Phân tích lỗi
- In ra **một số tin FP và FN cụ thể** (nên 5–10 mỗi loại) và giải thích:
  - FP: tin ham có từ như "free", "call", số điện thoại…
  - FN: spam viết khéo, ngắn, không có từ khóa điển hình, dùng ký hiệu/viết tắt.
- Từ nào có trọng số/log-prob cao nhất cho spam (từ `feature_log_prob_` của NB) → minh họa mô hình học gì.
- Liên hệ giả định "các từ độc lập" của Naive Bayes, mất cân bằng lớp, dữ liệu trùng lặp.

### Bước 7 – Kết luận & khuyến nghị
- Phương án tốt nhất + bằng chứng; hạn chế (dữ liệu nhỏ, ngôn ngữ, spam thay đổi theo thời gian); đề xuất cải thiện dựa trên lỗi quan sát được (class weight, ngưỡng quyết định, đặc trưng thêm, embeddings); ứng dụng thực tế (bộ lọc tin nhắn, ngưỡng ưu tiên Precision).

---

## 3. Cấu trúc repo đề xuất

```
machine_learning-btl/
├── README.md              # đề tài, cách chạy, cấu trúc, phân công, kết quả chính
├── requirements.txt
├── data/                  # hoặc hướng dẫn tải dữ liệu
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_vectorize.ipynb
│   ├── 03_models_eval.ipynb
│   └── 04_error_analysis.ipynb
├── src/                   # hàm tiền xử lý, train, evaluate
├── reports/               # báo cáo LaTeX + hình
├── slides/
└── AI_USAGE.md            # khai báo dùng AI (prompt, mục đích, phần nào)
```

- Mỗi thành viên commit phần của mình, message rõ ràng.
- README đủ để người khác tái lập (version thư viện, `random_state`, lệnh chạy).
- Giữ **bảng log thí nghiệm** (tham số → kết quả).

---

## 4. Báo cáo LaTeX – dàn ý

1. Giới thiệu bài toán, dữ liệu (nguồn, kích thước, nhãn), ý nghĩa ứng dụng
2. EDA (mất cân bằng, độ dài, từ phổ biến, trùng lặp)
3. Tiền xử lý & biểu diễn đặc trưng (giải thích quyết định + ablation)
4. Mô hình & thiết lập thí nghiệm (baseline, tham số, split, seed)
5. Kết quả & so sánh
6. Phân tích lỗi (FP/FN cụ thể)
7. Kết luận, hạn chế, hướng phát triển
8. Phân công, tài liệu tham khảo, **khai báo AI**

Bảo vệ – câu hỏi hay gặp cần chuẩn bị:
- Vì sao Naive Bayes hợp với văn bản? Vì sao TF-IDF thay vì BoW (hay ngược lại)?
- Vì sao không dùng Accuracy? Vì sao fit vectorizer chỉ trên train?
- FP và FN cái nào tệ hơn? Bạn sẽ chỉnh ngưỡng thế nào?
- Vì sao giữ/bỏ stopword, URL, số?

---

## 5. Rubric A3 – nắm điểm ở đâu

| Tiêu chí | Trọng số | Để đạt 9–10 |
|---|---|---|
| Xác định bài toán, dữ liệu & nguồn | 10% | Bối cảnh rõ, nguồn minh bạch, nêu giả định & giới hạn |
| Tiền xử lý & feature/representation | 15% | Xử lý tối ưu, **chứng minh tác động** tới mô hình |
| Xây dựng pipeline | 20% | Hoàn chỉnh, code tổ chức tốt, dễ mở rộng |
| Đánh giá & so sánh | 20% | Toàn diện, so sánh hợp lý, **đề xuất cải thiện dựa trên bằng chứng** |
| Diễn giải & khuyến nghị ứng dụng | 15% | Thuyết phục, có căn cứ, khả thi |
| Tái lập, repo, log, quản lý nhóm | 10% | Evidence đầy đủ, repo chuyên nghiệp, tái lập cao |
| Báo cáo, slide, demo, vấn đáp | 10% | Thuyết phục, demo ổn định, làm chủ sản phẩm |
| Liêm chính & sử dụng AI | điều kiện/trừ | Minh bạch nguồn, code, prompt; giải trình nhất quán |

- Điểm cá nhân có thể điều chỉnh theo đóng góp & khả năng giải trình.
- Dùng AI/sao chép mà không khai báo hoặc không giải trình được → bị trừ điểm.

---

## 6. Phân công gợi ý

| Vai trò | Việc |
|---|---|
| A | EDA, trực quan hóa, README/repo |
| B | Tiền xử lý + vector hóa + ablation |
| C | Mô hình, tuning, log thí nghiệm |
| D | Đánh giá, error analysis, báo cáo, slide |

Review chéo; ai cũng phải hiểu toàn bộ.

---

## 7. Checklist cuối

- [ ] Đúng dataset được giao, ghi rõ nguồn
- [ ] Xử lý trùng lặp/thiếu và nói rõ trong báo cáo
- [ ] Baseline + ≥2 phương án
- [ ] BoW và TF-IDF đều thử; so sánh n-gram, `min_df`, `max_df`
- [ ] Split có `stratify`, `random_state`; vectorizer chỉ fit trên train
- [ ] Accuracy, Precision, Recall, F1 (lớp spam), Confusion Matrix
- [ ] Thảo luận FP vs FN
- [ ] Error analysis có ví dụ cụ thể
- [ ] Notebook chạy lại từ đầu không lỗi
- [ ] README, requirements, log, commit từng thành viên
- [ ] Báo cáo LaTeX, slide, demo (ví dụ nhập 1 tin nhắn → dự đoán spam/ham)
- [ ] Khai báo AI/nguồn tham khảo
