"""Chỉ số đánh giá và biểu đồ kết quả (spam = lớp dương)."""
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import LABEL_NAMES, RANDOM_STATE


def _scores(pipe, X) -> np.ndarray:
    """Điểm liên tục cho lớp spam (để tính ROC-AUC); None nếu mô hình không hỗ trợ."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(X)
    return None


def evaluate(pipe, X, y) -> Dict[str, float]:
    pred = pipe.predict(X)
    scores = _scores(pipe, X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, scores) if scores is not None else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def cross_val_f1(pipe, X, y, folds: int) -> float:
    """F1 (lớp spam) trung bình qua CV phân tầng trên tập train – dùng để chọn mô hình."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return float(cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean())


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=names, yticklabels=names,
           xlabel="Dự đoán", ylabel="Thực tế", title=title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(results, path: Path) -> None:
    """Biểu đồ cột Precision / Recall / F1 cho tất cả thí nghiệm."""
    metrics = ["precision", "recall", "f1"]
    ids = results["id"].tolist()
    x = np.arange(len(ids))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(6, len(ids) * 1.2), 4))
    for k, m in enumerate(metrics):
        ax.bar(x + (k - 1) * width, results[m], width, label=m)
    ax.set(xticks=x, xticklabels=ids, ylim=(0, 1.05), ylabel="Điểm (lớp spam)",
           title="So sánh các thí nghiệm trên tập test")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
