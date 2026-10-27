# Phân loại tin nhắn rác (Spam) – Bài tập lớn Nhập môn Học máy

Pipeline: **EDA → Tiền xử lý → BoW/TF-IDF → Naive Bayes / K-NN / Decision Tree → Đánh giá → Phân tích lỗi**.
Hướng dẫn chi tiết & rubric: [HUONG_DAN_LAM_BAI.md](HUONG_DAN_LAM_BAI.md).

## Cài đặt & chạy

```bash
pip install -r requirements.txt

python -m spam_filter eda      # thống kê + biểu đồ EDA   -> outputs/
python -m spam_filter run      # chạy tất cả thí nghiệm E0..E6
python -m spam_filter run --ids E0 E2     # chỉ chạy một vài thí nghiệm
python -m spam_filter predict "Free entry! Call 0906... to win £500"
pytest                         # kiểm thử
```

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
└── cli.py             # lệnh eda | run | predict
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

## Kết quả (điền sau khi chạy)

| ID | Mô tả | Precision | Recall | F1 |
|----|-------|-----------|--------|----|
| E0 | Baseline | | | |

## Phân công

| Thành viên | Nhiệm vụ |
|---|---|
| | |

## Khai báo sử dụng AI

Xem `AI_USAGE.md` (cần bổ sung: công cụ, mục đích, phần nào được hỗ trợ).
