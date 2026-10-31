import numpy as np
import pytest

from spam_filter.evaluation import evaluate


class FixedPredictor:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def predict(self, X):
        assert len(X) == len(self.predictions)
        return self.predictions


class FixedProbabilityPredictor(FixedPredictor):
    def __init__(self, predictions, spam_probabilities):
        super().__init__(predictions)
        spam_probabilities = np.asarray(spam_probabilities)
        self.probabilities = np.column_stack((1 - spam_probabilities, spam_probabilities))

    def predict_proba(self, X):
        assert len(X) == len(self.probabilities)
        return self.probabilities


def test_evaluate_calculates_spam_metrics_and_confusion_orientation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])

    result = evaluate(FixedPredictor(y_pred), np.arange(6), y_true)

    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert (result["TN"], result["FP"], result["FN"], result["TP"]) == (2, 1, 1, 2)
    assert np.isnan(result["roc_auc"])


def test_evaluate_handles_no_predicted_spam():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 0, 0])

    result = evaluate(FixedPredictor(y_pred), np.arange(3), y_true)

    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["precision"] == 0
    assert result["recall"] == 0
    assert result["f1"] == 0
    assert (result["TN"], result["FP"], result["FN"], result["TP"]) == (1, 0, 2, 0)


def test_evaluate_calculates_roc_auc_from_spam_probabilities():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    # Of the four positive-negative pairs, three are ranked correctly.
    model = FixedProbabilityPredictor(y_pred, [0.1, 0.7, 0.4, 0.8])

    result = evaluate(model, np.arange(4), y_true)

    assert result["roc_auc"] == pytest.approx(3 / 4)
    assert (result["TN"], result["FP"], result["FN"], result["TP"]) == (1, 1, 1, 1)


def test_evaluate_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = y_true.copy()
    model = FixedProbabilityPredictor(y_pred, [0.05, 0.2, 0.8, 0.95])

    result = evaluate(model, np.arange(4), y_true)

    assert result["accuracy"] == 1
    assert result["precision"] == 1
    assert result["recall"] == 1
    assert result["f1"] == 1
    assert result["roc_auc"] == 1
    assert (result["TN"], result["FP"], result["FN"], result["TP"]) == (2, 0, 0, 2)
