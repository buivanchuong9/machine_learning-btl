"""Dòng lệnh:  python -m spam_filter <eda|run|viz|predict> [tùy chọn]"""
import argparse
import logging
from dataclasses import replace
from pathlib import Path

import joblib

from . import eda
from .config import OUT_FIGURES, OUT_MODELS, OUT_RESULTS, DataConfig
from .data import load_dataset, split_data
from .error_analysis import error_examples, top_spam_words
from .evaluation import plot_comparison, plot_confusion_matrix
from .experiments import get_experiments
from .runner import run_specs

log = logging.getLogger("spam_filter")
BEST_FILE = OUT_RESULTS / "best_model.txt"


def _data_config(args) -> DataConfig:
    cfg = DataConfig()
    overrides = {
        "path": Path(args.data) if args.data else None,
        "text_col": args.text_col, "label_col": args.label_col,
        "encoding": args.encoding, "sep": args.sep,
    }
    if args.header:
        overrides["has_header"] = True
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


def cmd_eda(args) -> None:
    df = load_dataset(_data_config(args))
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    summary = eda.summarize(df)
    summary.to_csv(OUT_RESULTS / "eda_summary.csv")
    print(summary.to_string())
    eda.plot_eda(df, OUT_FIGURES)


def cmd_run(args) -> None:
    cfg = _data_config(args)
    df = load_dataset(cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    log.info("Train=%d  Test=%d  (spam train=%.1f%%)", len(X_train), len(X_test),
             100 * y_train.mean())
    for d in (OUT_FIGURES, OUT_MODELS, OUT_RESULTS):
        d.mkdir(parents=True, exist_ok=True)

    results, pipes = run_specs(get_experiments(args.ids), X_train, y_train, X_test, y_test,
                               cfg.cv_folds, log=log.info)
    for exp_id, pipe in pipes.items():
        joblib.dump(pipe, OUT_MODELS / f"{exp_id}.joblib")
        plot_confusion_matrix(y_test, pipe.predict(X_test), exp_id, OUT_FIGURES / f"cm_{exp_id}.png")
    results.round(4).to_csv(OUT_RESULTS / "metrics.csv", index=False)
    plot_comparison(results, OUT_FIGURES / "comparison.png")
    print(results.drop(columns=["description", "params"]).round(4).to_string(index=False))

    # Chọn mô hình tốt nhất theo CV trên train (không nhìn test -> tránh chọn thiên lệch).
    best_id = results.sort_values("cv_f1_train", ascending=False).iloc[0]["id"]
    BEST_FILE.write_text(best_id)
    best = pipes[best_id]
    errors = error_examples(X_test, y_test, best.predict(X_test))
    errors.to_csv(OUT_RESULTS / f"errors_{best_id}.csv", index=False)
    words = top_spam_words(best)
    if words is not None:
        words.to_csv(OUT_RESULTS / f"top_words_{best_id}.csv", index=False)
    log.info("Mô hình tốt nhất (theo CV): %s | FP=%d FN=%d -> outputs/results/",
             best_id, (errors.error == "FP").sum(), (errors.error == "FN").sum())


def cmd_viz(args) -> None:
    from .viz import make_all, stability  # import muộn: chỉ cần khi vẽ
    make_all(OUT_FIGURES)
    if args.stability:
        print(stability(OUT_FIGURES, args.seeds).round(4).to_string())


def cmd_predict(args) -> None:
    model_id = args.model or (BEST_FILE.read_text().strip() if BEST_FILE.exists() else None)
    if not model_id or not (OUT_MODELS / f"{model_id}.joblib").exists():
        raise SystemExit("Chưa có mô hình. Chạy `python -m spam_filter run` trước.")
    pipe = joblib.load(OUT_MODELS / f"{model_id}.joblib")
    for text in args.texts:
        label = "SPAM" if pipe.predict([text])[0] == 1 else "ham"
        print(f"[{model_id}] {label:4s} | {text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spam_filter", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data_args(p):
        p.add_argument("--data", help="Đường dẫn file dữ liệu (mặc định: data/sms+spam+collection/SMSSpamCollection)")
        p.add_argument("--header", action="store_true", help="File có dòng tiêu đề (vd: spam.csv của Kaggle)")
        p.add_argument("--text-col"), p.add_argument("--label-col")
        p.add_argument("--encoding", help="vd: latin-1 cho SMS Spam Collection")
        p.add_argument("--sep")

    p_eda = sub.add_parser("eda", help="Thống kê + biểu đồ khám phá dữ liệu")
    add_data_args(p_eda)
    p_eda.set_defaults(func=cmd_eda)

    p_run = sub.add_parser("run", help="Chạy thí nghiệm, đánh giá, phân tích lỗi")
    add_data_args(p_run)
    p_run.add_argument("--ids", nargs="+", help="Chỉ chạy các thí nghiệm này, vd: E0 E2")
    p_run.set_defaults(func=cmd_run)

    p_viz = sub.add_parser("viz", help="Vẽ bộ biểu đồ trực quan cho báo cáo (outputs/figures/viz_*.png)")
    p_viz.add_argument("--stability", action="store_true", help="Thêm thí nghiệm lặp nhiều random_state (chậm hơn)")
    p_viz.add_argument("--seeds", type=int, default=10, help="Số lần chia ngẫu nhiên cho --stability")
    p_viz.set_defaults(func=cmd_viz)

    p_pred = sub.add_parser("predict", help="Dự đoán tin nhắn mới")
    p_pred.add_argument("texts", nargs="+")
    p_pred.add_argument("--model", help="ID thí nghiệm (mặc định: mô hình tốt nhất)")
    p_pred.set_defaults(func=cmd_predict)
    return parser


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):  # in được tiếng Việt trên console Windows
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = build_parser().parse_args()
    args.func(args)
