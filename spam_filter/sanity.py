"""Kiểm tra độ tin cậy của kết quả: rò rỉ dữ liệu (leakage) và học thuộc (memorization).

Chạy:  python -m spam_filter check   ->  bảng kiểm tra + outputs/figures/viz_9_learning_curve.png
Mỗi phép kiểm trả về một dòng: kiểm tra | giá trị | tiêu chí | trạng thái (OK / LƯU Ý / LỖI).
"""
import logging
import re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, learning_curve

from .config import RANDOM_STATE, DataConfig
from .data import load_dataset, split_data
from .experiments import get_experiments
from .pipeline import build_pipeline
from .viz import GRID, INK, MUTED, PURPLE, SURFACE, _save  # dùng chung style biểu đồ

log = logging.getLogger(__name__)

NEAR_DUP_THRESHOLD = 0.8   # cosine TF-IDF: coi là "bản gần giống"


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _row(name, value, criterion, status):
    return {"kiểm tra": name, "giá trị": value, "tiêu chí": criterion, "trạng thái": status}


def run_checks(model_id: str = "E5", cfg: Optional[DataConfig] = None) -> pd.DataFrame:
    cfg = cfg or DataConfig()
    df = load_dataset(cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    spec = get_experiments([model_id])[0]
    pipe = build_pipeline(spec).fit(X_train, y_train)
    pred_test = pipe.predict(X_test)
    f1_test = f1_score(y_test, pred_test)
    rows = []

    # --- A. Rò rỉ dữ liệu -------------------------------------------------------------------
    exact = len(set(X_train) & set(X_test))
    rows.append(_row("A1. Tin trùng chính xác giữa train và test", exact, "= 0",
                     "OK" if exact == 0 else "LỖI"))

    norm = len(set(X_train.map(_normalize)) & set(X_test.map(_normalize)))
    rows.append(_row("A2. Trùng sau chuẩn hóa (bỏ hoa/thường, dấu câu)", norm, "= 0",
                     "OK" if norm == 0 else "LƯU Ý"))

    vec = TfidfVectorizer().fit(X_train)
    max_sim = cosine_similarity(vec.transform(X_test), vec.transform(X_train)).max(axis=1)
    near = max_sim >= NEAR_DUP_THRESHOLD
    share = near.mean()
    rows.append(_row(f"A3. Tin test có bản gần giống trong train (cosine ≥ {NEAR_DUP_THRESHOLD})",
                     f"{near.sum()} tin ({share:.1%})", "< 10%", "OK" if share < 0.10 else "LƯU Ý"))

    f1_novel = f1_score(y_test[~near], pred_test[~near])
    drop = f1_test - f1_novel
    rows.append(_row("A4. F1 trên tin 'mới' (không có bản gần giống)",
                     f"{f1_novel:.3f} (toàn test {f1_test:.3f}, giảm {drop:.3f})", "giảm < 0.05",
                     "OK" if drop < 0.05 else "LƯU Ý"))

    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = []
    for _ in range(5):
        y_shuf = pd.Series(rng.permutation(y_train.values), index=y_train.index)
        shuffled.append(f1_score(y_test, build_pipeline(spec).fit(X_train, y_shuf).predict(X_test)))
    rows.append(_row("A5. Xáo nhãn train rồi đo trên test (5 lần)",
                     f"F1 trung bình {np.mean(shuffled):.3f}", "< 0.30 (≈ đoán mò)",
                     "OK" if np.mean(shuffled) < 0.30 else "LỖI"))

    # --- B. Học thuộc -----------------------------------------------------------------------
    f1_train = f1_score(y_train, pipe.predict(X_train))
    gap = f1_train - f1_test
    rows.append(_row("B1. Chênh lệch F1 train − test",
                     f"{f1_train:.3f} − {f1_test:.3f} = {gap:.3f}", "< 0.10",
                     "OK" if gap < 0.10 else "LƯU Ý"))

    from .evaluation import cross_val_f1
    cv_f1 = cross_val_f1(build_pipeline(spec), X_train, y_train, cfg.cv_folds)
    rows.append(_row("B2. F1 cross-validation (train) so với test",
                     f"CV {cv_f1:.3f} vs test {f1_test:.3f}", "lệch < 0.03",
                     "OK" if abs(cv_f1 - f1_test) < 0.03 else "LƯU Ý"))

    # --- C. Bài toán có thật sự dễ không? ---------------------------------------------------
    feats = lambda X: np.c_[X.str.len(), X.str.count(r"\d")]
    simple = LogisticRegression(max_iter=1000).fit(feats(X_train), y_train)
    f1_simple = f1_score(y_test, simple.predict(feats(X_test)))
    rows.append(_row("C1. Baseline chỉ dùng độ dài + số chữ số",
                     f"F1 {f1_simple:.3f} (mô hình {f1_test:.3f}, hơn {f1_test - f1_simple:.3f})",
                     "mô hình phải hơn", "OK" if f1_test > f1_simple else "LỖI"))
    f1_rand = f1_score(y_test, DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
                       .fit(X_train, y_train).predict(X_test))
    rows.append(_row("C2. Đoán ngẫu nhiên theo tỉ lệ lớp", f"F1 {f1_rand:.3f}", "mô hình phải hơn nhiều",
                     "OK" if f1_test > f1_rand + 0.5 else "LỖI"))
    return pd.DataFrame(rows)


def plot_learning_curve(model_id: str, out: Path, cfg: Optional[DataConfig] = None) -> pd.DataFrame:
    """Đường cong học: nếu điểm train và điểm validation tiến gần nhau khi thêm dữ liệu thì không học thuộc."""
    cfg = cfg or DataConfig()
    df = load_dataset(cfg)
    X_train, _, y_train, _ = split_data(df, cfg)
    sizes, train_s, val_s = learning_curve(
        build_pipeline(get_experiments([model_id])[0]), X_train, y_train, scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for scores, color, label in ((train_s, MUTED, "Điểm trên dữ liệu đã học (train)"),
                                 (val_s, PURPLE, "Điểm trên dữ liệu chưa thấy (validation)")):
        m, s = scores.mean(axis=1), scores.std(axis=1)
        ax.fill_between(sizes, m - s, m + s, color=color, alpha=0.15)
        ax.plot(sizes, m, color=color, linewidth=2, marker="o", markersize=5)
        ax.text(sizes[-1], m[-1], f"  {label}", color=INK, va="center", fontsize=9)
    ax.set(title=f"Đường cong học ({model_id}): validation ổn định ~0,94, chỉ kém train ~0,04",
           xlabel="Số tin nhắn dùng để huấn luyện", ylabel="F1 lớp spam",
           xlim=(sizes[0] * 0.9, sizes[-1] * 1.7))
    _save(fig, out / "viz_9_learning_curve.png")
    return pd.DataFrame({"n_train": sizes, "train_f1": train_s.mean(axis=1), "val_f1": val_s.mean(axis=1)})
