"""Chạy một danh sách ExperimentSpec và gom kết quả thành bảng (dùng chung cho CLI và notebook)."""
from typing import Dict, Iterable, Tuple

import pandas as pd

from .evaluation import cross_val_f1, evaluate
from .pipeline import ExperimentSpec, build_pipeline


def describe_params(spec: ExperimentSpec) -> str:
    """Chuỗi ghi lại đầy đủ tham số để tái lập."""
    return (f"clean={spec.cleaner or 'lowercase'} | vec={spec.vectorizer or 'bow'} | "
            f"model={spec.model}{spec.model_params or ''}")


def run_specs(specs: Iterable[ExperimentSpec], X_train, y_train, X_test, y_test,
              folds: int = 5, log=None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Với mỗi spec: CV trên train -> fit trên train -> đánh giá trên test.

    Trả về (bảng kết quả, {id: pipeline đã fit}).
    """
    rows, pipes = [], {}
    for spec in specs:
        if log:
            log("[%s] %s", spec.id, spec.description)
        pipe = build_pipeline(spec)
        cv_f1 = cross_val_f1(pipe, X_train, y_train, folds)
        pipe.fit(X_train, y_train)
        rows.append({"id": spec.id, "description": spec.description,
                     "cv_f1_train": cv_f1, **evaluate(pipe, X_test, y_test),
                     "params": describe_params(spec)})
        pipes[spec.id] = pipe
    return pd.DataFrame(rows), pipes
