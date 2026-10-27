"""Biểu đồ trực quan cho báo cáo: python -m spam_filter viz  ->  outputs/figures/viz_*.png

Màu: ham = xanh, spam = cam (cố định cho mọi biểu đồ); tím/đỏ cho cặp thước đo khác;
xám cho phần phụ. Mọi cột/điểm quan trọng đều có nhãn số trực tiếp.
"""
import logging
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .config import DataConfig
from .data import load_dataset, split_data
from .experiments import EXPERIMENTS
from .runner import run_specs

log = logging.getLogger(__name__)

HAM, SPAM = "#2a78d6", "#eb6834"      # xanh / cam
PURPLE, RED = "#6250d6", "#e34948"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log.info("Đã lưu %s", path.name)


def _top_words(texts, n=12):
    c = Counter(w for t in texts for w in t.lower().split()
                if w.isalpha() and len(w) > 2 and w not in ENGLISH_STOP_WORDS)
    return c.most_common(n)


def chart_balance_length(df, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
    counts = df["label"].value_counts().sort_index()
    bars = a.bar(["ham", "spam"], counts.values, color=[HAM, SPAM], width=0.55)
    for bar, v in zip(bars, counts.values):
        a.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}\n({v / counts.sum():.1%})",
               ha="center", va="bottom", color=INK)
    a.set(title="Spam chỉ chiếm ~12% dữ liệu", ylabel="Số tin nhắn", ylim=(0, counts.max() * 1.18))
    a.grid(axis="x", visible=False)
    for label, color, name in ((0, HAM, "ham"), (1, SPAM, "spam")):
        b.hist(df.loc[df.label == label, "text"].str.len(), bins=45, range=(0, 400),
               density=True, alpha=0.75, color=color, label=name, edgecolor=SURFACE, linewidth=0.5)
    med = df.groupby("label")["text"].apply(lambda s: s.str.len().median())
    b.set(title="Spam dài hơn hẳn tin thường", xlabel="Số ký tự (cắt ở 400)", ylabel="Mật độ")
    b.legend(frameon=False)
    b.text(0.98, 0.62, f"trung vị: ham {med[0]:.0f} · spam {med[1]:.0f} ký tự",
           transform=b.transAxes, ha="right", color=MUTED)
    _save(fig, out / "viz_1_class_balance.png")


def chart_top_words(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, label, color, name in zip(axes, (0, 1), (HAM, SPAM), ("ham", "spam")):
        top = _top_words(df.loc[df.label == label, "text"])[::-1]
        bars = ax.barh([w for w, _ in top], [c for _, c in top], color=color, height=0.65)
        for bar, (_, c) in zip(bars, top):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {c}", va="center", color=MUTED)
        ax.set(title=f"Từ phổ biến nhất – {name}", xlabel="Số lần xuất hiện")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, top[-1][1] * 1.12)
    _save(fig, out / "viz_2_top_words.png")


def _dumbbell(ax, labels, a, b, a_name, b_name, a_color, b_color, xlim):
    y = np.arange(len(labels))
    ax.hlines(y, np.minimum(a, b), np.maximum(a, b), color=GRID, linewidth=3, zorder=1)
    ax.scatter(a, y, s=70, color=a_color, zorder=3, edgecolor=SURFACE, linewidth=1.5, label=a_name)
    ax.scatter(b, y, s=70, color=b_color, zorder=3, edgecolor=SURFACE, linewidth=1.5, label=b_name)
    for yi, ai, bi in zip(y, a, b):
        ax.text(max(ai, bi) + 0.004, yi, f"{ai:.3f} / {bi:.3f}", va="center", color=MUTED, fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper left")


def chart_models(results, out):
    r = results.sort_values("f1", ascending=False)  # tốt nhất ở trên cùng
    labels = [f"{i}  {d.split(': ', 1)[-1][:38]}" for i, d in zip(r["id"], r["description"])]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    _dumbbell(ax, labels, r["cv_f1_train"].values, r["f1"].values,
              "F1 qua CV (train)", "F1 trên test", MUTED, PURPLE, (0.8, 1.0))
    ax.set(title="So sánh các thí nghiệm – F1 lớp spam", xlabel="F1 (nhãn: CV / test)")
    _save(fig, out / "viz_3_model_comparison.png")


def chart_precision_recall(results, out):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.scatter(results["recall"], results["precision"], s=110, color=PURPLE,
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
    # Các thí nghiệm trùng điểm (cùng precision/recall) gộp chung một nhãn, vd "E2 = E6".
    groups = results.round(4).groupby(["recall", "precision"])["id"].apply(" = ".join)
    for (rec, prec), name in groups.items():
        left = "=" in name  # nhãn gộp đặt bên trái-dưới để không đè điểm lân cận
        ax.annotate(name, (rec, prec), xytext=(-9, -15) if left else (7, 5),
                    textcoords="offset points", ha="right" if left else "left",
                    color=INK, fontweight="bold")
    ax.set(title="Precision và Recall – góc trên-phải là tốt nhất",
           xlabel="Recall (bắt được bao nhiêu spam)",
           ylabel="Precision (chặn đúng bao nhiêu)")
    ax.set_xlim(results["recall"].min() - 0.02, 1.0)
    ax.set_ylim(results["precision"].min() - 0.02, 1.0)
    _save(fig, out / "viz_4_precision_recall.png")


def chart_confusion(y_test, pred, best_id, out):
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    names = ["ham", "spam"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=names, yticklabels=names,
           xlabel="Mô hình dự đoán", ylabel="Thực tế", title=f"Confusion matrix – {best_id}")
    tags = [["đúng", "chặn nhầm (FP)"], ["bỏ lọt (FN)", "đúng"]]
    for i in range(2):
        for j in range(2):
            color = "white" if norm[i, j] > 0.5 else INK
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.1%}\n{tags[i][j]}", ha="center", va="center", color=color)
    _save(fig, out / "viz_5_confusion_matrix.png")


def chart_overfit(rows, out):
    rows = sorted(rows, key=lambda r: -r["test"])  # tốt nhất ở trên cùng
    labels = [r["id"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _dumbbell(ax, labels, np.array([r["train"] for r in rows]), np.array([r["test"] for r in rows]),
              "F1 trên train", "F1 trên test", MUTED, PURPLE, (0.8, 1.02))
    ax.set(title="Không học thuộc: điểm train chỉ cao hơn test 0,04–0,07",
           xlabel="F1 lớp spam (nhãn: train / test)")
    _save(fig, out / "viz_6_overfitting_check.png")


def chart_threshold(pipe, X_test, y_test, best_id, out):
    proba = pipe.predict_proba(X_test)[:, 1]
    ts = np.linspace(0.05, 0.99, 60)
    prec = [precision_score(y_test, proba >= t, zero_division=1) for t in ts]
    rec = [recall_score(y_test, proba >= t) for t in ts]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ts, prec, color=PURPLE, linewidth=2)
    ax.plot(ts, rec, color=RED, linewidth=2)
    ax.text(ts[-1], prec[-1], " Precision", color=INK, va="center")
    ax.text(ts[-1], rec[-1], " Recall", color=INK, va="center")
    ax.axvline(0.5, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(0.51, 0.43, "ngưỡng mặc định 0,5", color=MUTED, fontsize=9)
    ax.set(title=f"Đổi ngưỡng quyết định ({best_id}): tăng Precision thì giảm Recall",
           xlabel="Ngưỡng xác suất để gọi là spam", ylabel="Điểm", ylim=(0.4, 1.02), xlim=(0.05, 1.08))
    _save(fig, out / "viz_7_threshold_tradeoff.png")


def make_all(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = DataConfig()
    df = load_dataset(cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    results, pipes = run_specs(EXPERIMENTS, X_train, y_train, X_test, y_test, cfg.cv_folds)
    best_id = results.sort_values("cv_f1_train", ascending=False).iloc[0]["id"]
    overfit = [{"id": i, "train": f1_score(y_train, p.predict(X_train)),
                "test": f1_score(y_test, p.predict(X_test))} for i, p in pipes.items()]

    chart_balance_length(df, out)
    chart_top_words(df, out)
    chart_models(results, out)
    chart_precision_recall(results, out)
    chart_confusion(y_test, pipes[best_id].predict(X_test), best_id, out)
    chart_overfit(overfit, out)
    chart_threshold(pipes[best_id], X_test, y_test, best_id, out)


def stability(out: Path, n_seeds: int = 10) -> "pd.DataFrame":
    """Lặp lại cả quy trình (chia + fit + test) với nhiều random_state để đo độ dao động của F1."""
    import pandas as pd
    from dataclasses import replace
    from .pipeline import build_pipeline

    out.mkdir(parents=True, exist_ok=True)
    base = DataConfig()
    df = load_dataset(base)
    rows = []
    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = split_data(df, replace(base, random_state=seed))
        for spec in EXPERIMENTS:
            pipe = build_pipeline(spec).fit(X_train, y_train)
            rows.append({"seed": seed, "id": spec.id, "f1": f1_score(y_test, pipe.predict(X_test))})
        log.info("seed %d/%d xong", seed + 1, n_seeds)
    res = pd.DataFrame(rows)
    res.to_csv(out.parent / "results" / "stability_f1.csv", index=False)

    stats = res.groupby("id")["f1"].agg(["mean", "std", "min", "max"]).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for y, (exp_id, row) in enumerate(stats.iterrows()):
        vals = res.loc[res.id == exp_id, "f1"].values
        ax.hlines(y, row["min"], row["max"], color=GRID, linewidth=6, zorder=1)
        ax.scatter(vals, np.full_like(vals, y), s=26, color=PURPLE, alpha=0.55, zorder=2, edgecolor="none")
        ax.scatter(row["mean"], y, s=110, color=PURPLE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(row["max"] + 0.004, y, f"{row['mean']:.3f} ± {row['std']:.3f}", va="center", color=MUTED, fontsize=9)
    ax.set_yticks(range(len(stats)), stats.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(title=f"F1 qua {n_seeds} lần chia ngẫu nhiên: các mô hình tốt nhất chồng lên nhau",
           xlabel="F1 lớp spam trên test (chấm nhỏ = mỗi lần chia, chấm lớn = trung bình ± độ lệch chuẩn)",
           xlim=(0.78, 1.0))
    _save(fig, out / "viz_8_f1_stability.png")
    return stats
