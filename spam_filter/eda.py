"""Khám phá dữ liệu: thống kê và biểu đồ."""
import logging
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import LABEL_NAMES

log = logging.getLogger(__name__)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng thống kê theo lớp: số mẫu, tỉ lệ, độ dài trung bình."""
    length = df["text"].str.len()
    words = df["text"].str.split().str.len()
    g = df.assign(chars=length, words=words).groupby("label")
    out = g.agg(n=("text", "size"), mean_chars=("chars", "mean"), mean_words=("words", "mean"))
    out["ratio"] = out["n"] / out["n"].sum()
    out.index = [LABEL_NAMES[i] for i in out.index]
    return out.round(3)


def _top_words(texts: pd.Series, n: int = 15):
    counter = Counter(w for t in texts for w in t.lower().split()
                      if w.isalpha() and w not in ENGLISH_STOP_WORDS)
    return counter.most_common(n)


def plot_eda(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [LABEL_NAMES[i] for i in (0, 1)]

    # 1. Phân phối nhãn
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.bar(names, counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set(title="Phân phối nhãn", ylabel="Số tin nhắn")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_label_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Phân phối độ dài
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for label, color in ((0, "#4C72B0"), (1, "#C44E52")):
        ax.hist(df.loc[df.label == label, "text"].str.len(), bins=40, alpha=0.6,
                color=color, label=LABEL_NAMES[label], density=True)
    ax.set(title="Độ dài tin nhắn (ký tự)", xlabel="Số ký tự", ylabel="Mật độ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "eda_length_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Từ phổ biến theo lớp
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, label in zip(axes, (0, 1)):
        top = _top_words(df.loc[df.label == label, "text"])
        ax.barh([w for w, _ in top][::-1], [c for _, c in top][::-1],
                color="#4C72B0" if label == 0 else "#C44E52")
        ax.set_title(f"Top từ – {LABEL_NAMES[label]}")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_top_words.png", dpi=150)
    plt.close(fig)
    log.info("Đã lưu biểu đồ EDA vào %s", out_dir)
