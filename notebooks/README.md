# Notebook theo từng bước (đúng Hình 1 của đề)

Chạy theo thứ tự, mỗi file một khối. Chỉ `import` từ `spam_filter/`, không copy logic.

| File | Khối | Nội dung chính |
|---|---|---|
| `00_data_source.ipynb` | Data Source | Nguồn, định dạng, bỏ trùng, chia train/test, giả định |
| `01_eda.ipynb` | EDA | Thiếu/trùng, phân phối nhãn, độ dài, dấu hiệu spam, từ phổ biến |
| `02_preprocessing.ipynb` | Preprocessing | Trước/sau xử lý + ablation từng bước |
| `03_features.ipynb` | Feature / Representation | BoW vs TF-IDF, n-gram, min_df, max_df, max_features (đổi 1 biến/lần) |
| `04_model.ipynb` | Model | Tinh chỉnh Naive Bayes, K-NN, Decision Tree cùng tiêu chí CV |
| `05_evaluation.ipynb` | Evaluation | Bảng E0–E6, confusion matrix, ROC/PR, tham số tái lập |
| `06_error_analysis.ipynb` | Error Analysis | FP/FN cụ thể, từ đặc trưng, giả thuyết nguyên nhân |
| `07_conclusion.ipynb` | Conclusion | Kết luận, hạn chế, khuyến nghị (ngưỡng quyết định), hướng cải thiện |

Mở từ thư mục `notebooks/` (notebook dùng `sys.path.insert(0, "..")`). Các ô "Nhận xét (nhóm điền)" cần nhóm tự viết.
