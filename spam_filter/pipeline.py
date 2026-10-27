"""Ghép tiền xử lý -> vector hóa -> mô hình thành một sklearn Pipeline.

Dùng Pipeline để vectorizer luôn được fit CHỈ trên tập train (cả khi cross-validate).
"""
from dataclasses import dataclass, field
from typing import Optional

from sklearn.pipeline import Pipeline

from .features import build_vectorizer
from .models import build_model
from .preprocessing import TextCleaner


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    description: str
    model: str
    model_params: dict = field(default_factory=dict)
    cleaner: dict = field(default_factory=dict)      # tham số của TextCleaner
    vectorizer: dict = field(default_factory=dict)   # tham số của build_vectorizer
    note: Optional[str] = None


def build_pipeline(spec: ExperimentSpec) -> Pipeline:
    return Pipeline([
        ("clean", TextCleaner(**spec.cleaner)),
        ("vec", build_vectorizer(**spec.vectorizer)),
        ("clf", build_model(spec.model, **spec.model_params)),
    ])
