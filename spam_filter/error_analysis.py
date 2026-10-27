"""Phân tích lỗi: tin bị dự đoán sai và từ mà mô hình coi là 'spam'."""
from typing import Optional

import numpy as np
import pandas as pd


def error_examples(X_test: pd.Series, y_test: pd.Series, y_pred) -> pd.DataFrame:
    """FP = ham bị đoán spam; FN = spam bị đoán ham."""
    df = pd.DataFrame({"text": X_test.values, "actual": y_test.values, "predicted": y_pred})
    df["error"] = np.select(
        [(df.actual == 0) & (df.predicted == 1), (df.actual == 1) & (df.predicted == 0)],
        ["FP", "FN"], default="",
    )
    df["length"] = df["text"].str.len()
    return df[df.error != ""].sort_values("error").reset_index(drop=True)


def top_spam_words(pipe, n: int = 20) -> Optional[pd.DataFrame]:
    """Với Naive Bayes: từ có log P(w|spam) - log P(w|ham) lớn nhất. Model khác -> None."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_log_prob_"):
        return None
    words = np.array(pipe.named_steps["vec"].get_feature_names_out())
    diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    order = np.argsort(diff)
    return pd.DataFrame({
        "spam_indicative": words[order[::-1][:n]], "spam_score": diff[order[::-1][:n]],
        "ham_indicative": words[order[:n]], "ham_score": diff[order[:n]],
    })
