# Phân loại tin nhắn rác (Spam) – Bài tập lớn Nhập môn Học máy

Pipeline: **EDA → Tiền xử lý → BoW/TF-IDF → Naive Bayes / K-NN / Decision Tree → Đánh giá → Phân tích lỗi**.
Hướng dẫn chi tiết & rubric: [HUONG_DAN_LAM_BAI.md](HUONG_DAN_LAM_BAI.md).

## Cài đặt & chạy

Yêu cầu: Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

**Chạy giao diện web (demo)** – cần có kết quả trong `outputs/` (thư mục này không commit lên git, nên sau khi
clone phải chạy `python -m spam_filter run` một lần để train; thêm `viz` và `check` để giao diện đủ biểu đồ):

```bash
python -m spam_filter run     # chỉ cần lần đầu (~vài phút)
streamlit run app.py          # mở http://localhost:8501 (lần đầu hỏi email: nhấn Enter bỏ trống)
```

Giao diện đi theo đúng luồng của đề: Demo dự đoán → Data Source → EDA → Preprocessing → Feature → Model →
Evaluation → Error Analysis → Conclusion. Trang Demo cho nhập tin nhắn, kéo ngưỡng quyết định, xem từ nào đẩy
về spam/ham, và có ô **test hàng loạt** (mỗi dòng một tin, tùy chọn tiền tố `spam|` / `ham|` để chấm đúng/sai).

**Chạy pipeline bằng dòng lệnh** (tạo lại `outputs/`):

```bash
python -m spam_filter eda      # thống kê + biểu đồ EDA
python -m spam_filter run      # chạy tất cả thí nghiệm E0..E6 (lưu mô hình, bảng, biểu đồ, lỗi)
python -m spam_filter run --ids E0 E2     # chỉ chạy một vài thí nghiệm
python -m spam_filter viz      # bộ biểu đồ cho báo cáo (viz_*.png); thêm --stability để lặp nhiều random_state
python -m spam_filter check    # kiểm tra rò rỉ dữ liệu / học thuộc / baseline đơn giản (--model E5)
python -m spam_filter predict "Free entry! Call 0906... to win £500"
pytest                         # kiểm thử
```

Thứ tự khuyến nghị khi chạy từ đầu: `eda` → `run` → `viz` → `check` → `streamlit run app.py`.
Notebook theo từng bước nằm ở `notebooks/` (xem `notebooks/README.md`).

## Dataset

Mặc định dùng **SMS Spam Collection (UCI)** đặt tại `data/sms+spam+collection/SMSSpamCollection`
(tab-separated, không header, dạng `label<TAB>text`). 5574 tin, sau khi bỏ trùng còn 5160.

Dùng dataset khác (vd: `spam.csv` của Kaggle có header, cột `v1/v2`, latin-1):

```bash
python -m spam_filter run --data data/raw/spam.csv --header --encoding latin-1
python -m spam_filter run --data data/raw/abc.csv --header --text-col content --label-col type
```

Nhãn chấp nhận: `spam/1` và `ham/0` (sửa trong `DataConfig` nếu khác). Tin trùng bị bỏ **trước khi chia** để tránh rò rỉ train/test.

## Cấu trúc

```
spam_filter/
├── config.py          # đường dẫn, cột dữ liệu, random_state, tỉ lệ chia  <- sửa khi đổi dataset
├── data.py            # đọc, chuẩn hóa nhãn, chia train/test (stratify)
├── preprocessing.py   # TextCleaner: các bước bật/tắt được (ablation)
├── features.py        # BoW / TF-IDF (n-gram, min_df, max_df)
├── models.py          # MultinomialNB, KNN, DecisionTree
├── pipeline.py        # ExperimentSpec + build_pipeline (Pipeline sklearn, chống leakage)
├── experiments.py     # DANH SÁCH THÍ NGHIỆM E0..E6  <- thêm thí nghiệm ở đây
├── evaluation.py      # Accuracy/Precision/Recall/F1/ROC-AUC, CV, confusion matrix, biểu đồ so sánh
├── error_analysis.py  # tin FP/FN, từ đặc trưng spam của Naive Bayes
├── eda.py             # thống kê và biểu đồ khám phá
├── runner.py          # chạy danh sách thí nghiệm -> bảng kết quả (CLI và notebook dùng chung)
├── viz.py             # bộ biểu đồ cho báo cáo (viz_1..8)
├── sanity.py          # kiểm tra rò rỉ / học thuộc / baseline + learning curve (viz_9)
└── cli.py             # lệnh eda | run | viz | check | predict
app.py                 # giao diện web Streamlit (đọc kết quả trong outputs/)
tests/                 # pytest
notebooks/             # 8 notebook 00..07, mỗi khối của quy trình một file (xem notebooks/README.md)
data/raw/              # dataset (không commit)
outputs/{figures,results,models}/   # kết quả sinh ra
```

## Thiết kế

- **Thêm thí nghiệm** = thêm một `ExperimentSpec` vào `experiments.py`; không cần sửa chỗ khác.
- Mọi bước học từ dữ liệu nằm trong `Pipeline`, nên vectorizer chỉ fit trên train (kể cả khi cross-validate).
- Mô hình tốt nhất được chọn theo **F1 lớp spam qua CV trên tập train**, không dựa vào tập test.
- `random_state=42` đặt ở `config.py`; chia dữ liệu `test_size=0.2`, `stratify` theo nhãn.

## Kết quả

Chia train/test 80/20 (stratify, `random_state=42`); mô hình tốt nhất (⭐) chọn theo **F1 lớp spam qua CV trên tập train**.
Số liệu trên tập test (1032 tin, 128 spam), từ `outputs/results/metrics.csv`:

| ID | Mô tả | Accuracy | Precision | Recall | F1 (spam) | ROC-AUC | CV F1 (train) |
|----|-------|----------|-----------|--------|-----------|---------|---------------|
| E0 | Baseline: BoW + MultinomialNB mặc định | 0.9826 | 0.9508 | 0.9062 | 0.9280 | 0.9790 | 0.9385 |
| E1 | TF-IDF unigram + MultinomialNB | 0.9719 | 0.9901 | 0.7812 | 0.8734 | 0.9832 | 0.8512 |
| E2 | TF-IDF (1,2)-gram + MultinomialNB | 0.9845 | 0.9746 | 0.8984 | 0.9350 | 0.9882 | 0.9365 |
| E3 | TF-IDF + K-NN (cosine) | 0.9758 | 0.9640 | 0.8359 | 0.8954 | 0.9682 | 0.8935 |
| E4 | TF-IDF + Decision Tree | 0.9700 | 0.9450 | 0.8047 | 0.8692 | 0.8954 | 0.8086 |
| E5 ⭐ | Ablation: E2 + token hóa URL/tiền/số, bỏ dấu câu | 0.9864 | 0.9750 | 0.9141 | 0.9435 | 0.9905 | 0.9414 |
| E6 | Ablation: E2 + bỏ stopword | 0.9845 | 0.9746 | 0.8984 | 0.9350 | 0.9920 | 0.9283 |

**Kiểm tra độ tin cậy** (`python -m spam_filter check`, `outputs/results/sanity_checks.csv`): không có tin trùng chính xác
giữa train và test; xáo nhãn train thì F1 test rơi về mức đoán mò (≈ 0.1) ⇒ không học thuộc; F1 train − test ≈ 0.04.
E1 và E4 kém hơn baseline đơn giản (độ dài + số chữ số, F1 0.886).

## Hạn chế

- Chỉ **tiếng Anh**: tin tiếng Việt gần như không có từ nào nằm trong từ vựng nên kết quả không đáng tin.
- Dữ liệu **cũ (~2012), nhỏ, và ham/spam đến từ hai nguồn khác nhau** (ham chủ yếu từ sinh viên NUS Singapore, spam từ
  diễn đàn Grumbletext của Anh) ⇒ điểm cao đúng trong phạm vi bộ dữ liệu này, có thể lạc quan với spam hiện đại.
  Thử nhanh bằng vài tin tự viết cho thấy mô hình bỏ sót nhiều spam kiểu mới (giả mạo ngân hàng, link rút gọn).
- Tập test chỉ có 128 tin spam nên F1 dao động theo cách chia dữ liệu (xem `viz --stability`).
- Cải thiện: thêm dữ liệu spam hiện đại/đa ngôn ngữ, đặc trưng link rút gọn, class weight, chỉnh ngưỡng ưu tiên Precision.

## Phân công

| Thành viên | Nhiệm vụ |
|---|---|
| | |

## Khai báo sử dụng AI

Xem `AI_USAGE.md` (cần bổ sung: công cụ, mục đích, phần nào được hỗ trợ).
