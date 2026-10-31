from types import SimpleNamespace

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from spam_filter import cli, sanity, viz
from spam_filter.config import RANDOM_STATE, DataConfig
from spam_filter.data import load_dataset, split_data
from spam_filter.experiments import EXPERIMENTS
from spam_filter.pipeline import build_pipeline


def test_cli_metrics_csv_matches_model_and_rounds_to_four_decimals(tmp_path, monkeypatch):
    figures = tmp_path / "figures"
    models = tmp_path / "models"
    results = tmp_path / "results"
    monkeypatch.setattr(cli, "OUT_FIGURES", figures)
    monkeypatch.setattr(cli, "OUT_MODELS", models)
    monkeypatch.setattr(cli, "OUT_RESULTS", results)
    monkeypatch.setattr(cli, "BEST_FILE", results / "best_model.txt")
    args = SimpleNamespace(
        data=None, text_col=None, label_col=None, encoding=None, sep=None,
        header=False, ids=["E0"],
    )

    cli.cmd_run(args)

    report = pd.read_csv(results / "metrics.csv").set_index("id")
    row = report.loc["E0"]
    cfg = DataConfig()
    df = load_dataset(cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    pipe = joblib.load(models / "E0.joblib")
    pred = pipe.predict(X_test)
    true = y_test.to_numpy()
    tn = int(np.count_nonzero((true == 0) & (pred == 0)))
    fp = int(np.count_nonzero((true == 0) & (pred == 1)))
    fn = int(np.count_nonzero((true == 1) & (pred == 0)))
    tp = int(np.count_nonzero((true == 1) & (pred == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    expected = {
        "accuracy": (tn + tp) / len(true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]),
    }

    for name, value in expected.items():
        assert row[name] == pytest.approx(round(value, 4))
    assert (int(row.TN), int(row.FP), int(row.FN), int(row.TP)) == (tn, fp, fn, tp)
    assert tn + fp + fn + tp == len(y_test)

    spec = next(item for item in EXPERIMENTS if item.id == "E0")
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    independent_cv_f1 = cross_val_score(
        build_pipeline(spec), X_train, y_train, cv=cv, scoring="f1"
    ).mean()
    assert row.cv_f1_train == pytest.approx(round(independent_cv_f1, 4))


def test_balance_chart_title_uses_the_input_class_distribution(tmp_path, monkeypatch):
    saved_figures = []
    monkeypatch.setattr(viz, "_save", lambda fig, path: saved_figures.append(fig))

    for spam_count in (1, 3):
        labels = [0] * (4 - spam_count) + [1] * spam_count
        df = pd.DataFrame({"label": labels, "text": [f"message {i}" for i in range(4)]})
        viz.chart_balance_length(df, tmp_path)

    titles = [fig.axes[0].get_title(loc="left") for fig in saved_figures]
    try:
        assert "25.0%" in titles[0]
        assert "75.0%" in titles[1]
        assert titles[0] != titles[1]
    finally:
        for fig in saved_figures:
            plt.close(fig)


def test_overfit_chart_reports_computed_gap_and_fits_low_scores(tmp_path, monkeypatch):
    saved_figures = []
    monkeypatch.setattr(viz, "_save", lambda fig, path: saved_figures.append(fig))
    rows = [
        {"id": "E0", "train": 0.42, "test": 0.30},
        {"id": "E1", "train": 0.35, "test": 0.32},
    ]

    viz.chart_overfit(rows, tmp_path)

    fig = saved_figures[0]
    try:
        title = fig.axes[0].get_title(loc="left")
        assert "+0.030" in title and "+0.120" in title
        lower, upper = fig.axes[0].get_xlim()
        assert lower < 0.30
        assert upper > 0.42
    finally:
        plt.close(fig)


def test_model_comparison_chart_keeps_low_performance_values_visible(tmp_path, monkeypatch):
    saved_figures = []
    monkeypatch.setattr(viz, "_save", lambda fig, path: saved_figures.append(fig))
    results = pd.DataFrame({
        "id": ["E0", "E1"],
        "description": ["Baseline: bag of words", "TF-IDF: unigram"],
        "cv_f1_train": [0.52, 0.61],
        "f1": [0.45, 0.58],
    })

    viz.chart_models(results, tmp_path)

    fig = saved_figures[0]
    try:
        lower, upper = fig.axes[0].get_xlim()
        assert lower <= 0.45
        assert upper >= 0.61
        assert lower < 0.8
    finally:
        plt.close(fig)


def test_learning_curve_title_uses_computed_cv_and_train_scores(tmp_path, monkeypatch):
    captured = {}
    train_scores = np.array([[0.60, 0.70], [0.80, 0.90]])
    validation_scores = np.array([[0.50, 0.60], [0.70, 0.80]])
    monkeypatch.setattr(sanity, "load_dataset", lambda cfg: object())
    monkeypatch.setattr(sanity, "split_data", lambda df, cfg: ([], [], [], []))
    monkeypatch.setattr(sanity, "get_experiments", lambda ids: [object()])
    monkeypatch.setattr(sanity, "build_pipeline", lambda spec: object())
    monkeypatch.setattr(
        sanity, "learning_curve",
        lambda *args, **kwargs: (np.array([2, 4]), train_scores, validation_scores),
    )

    def capture_learning_curve(fig, path):
        captured["title"] = fig.axes[0].get_title(loc="left")
        plt.close(fig)

    monkeypatch.setattr(sanity, "_save", capture_learning_curve)

    result = sanity.plot_learning_curve("E3", tmp_path)

    assert "E3" in captured["title"]
    assert "0.750" in captured["title"]
    assert "+0.100" in captured["title"]
    assert result["train_f1"].tolist() == pytest.approx([0.65, 0.85])
    assert result["val_f1"].tolist() == pytest.approx([0.55, 0.75])


class _FixedPipeline:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([0, 1])[:len(X)]


def test_stability_chart_title_reports_requested_seed_count(tmp_path, monkeypatch):
    captured = {}
    X_train = pd.Series(["ham train", "spam train"])
    y_train = pd.Series([0, 1])
    X_test = pd.Series(["ham test", "spam test"])
    y_test = pd.Series([0, 1])
    out = tmp_path / "figures"
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(viz, "load_dataset", lambda cfg: object())
    monkeypatch.setattr(viz, "split_data", lambda df, cfg: (X_train, X_test, y_train, y_test))
    monkeypatch.setattr(viz, "EXPERIMENTS", [EXPERIMENTS[0]])
    monkeypatch.setattr("spam_filter.pipeline.build_pipeline", lambda spec: _FixedPipeline())

    def capture_stability(fig, path):
        captured["title"] = fig.axes[0].get_title(loc="left")
        plt.close(fig)

    monkeypatch.setattr(viz, "_save", capture_stability)

    stats = viz.stability(out, n_seeds=2)

    assert "2 lần chia ngẫu nhiên" in captured["title"]
    assert "chồng lên nhau" not in captured["title"]
    assert stats.loc["E0", "mean"] == 1


class _FixedProbabilityModel:
    def predict_proba(self, X):
        probabilities = np.linspace(0.1, 0.9, len(X))
        return np.column_stack((1 - probabilities, probabilities))


def test_threshold_chart_uses_selected_model_and_neutral_title(tmp_path, monkeypatch):
    captured = {}

    def capture_threshold(fig, path):
        captured["title"] = fig.axes[0].get_title(loc="left")
        plt.close(fig)

    monkeypatch.setattr(viz, "_save", capture_threshold)

    viz.chart_threshold(
        _FixedProbabilityModel(), ["ham", "spam", "ham", "spam"],
        np.array([0, 1, 0, 1]), "E3", tmp_path,
    )

    assert captured["title"] == "Precision và Recall theo ngưỡng (E3)"
