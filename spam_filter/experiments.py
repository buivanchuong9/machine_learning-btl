"""Danh sách thí nghiệm. Thêm một thí nghiệm = thêm một dòng ExperimentSpec."""
from typing import List, Optional

from .pipeline import ExperimentSpec

EXPERIMENTS: List[ExperimentSpec] = [
    ExperimentSpec(
        id="E0", description="Baseline: BoW + MultinomialNB mặc định",
        model="nb",
        vectorizer={"kind": "bow"},
    ),
    ExperimentSpec(
        id="E1", description="TF-IDF unigram + MultinomialNB",
        model="nb", model_params={"alpha": 0.5},
        vectorizer={"kind": "tfidf"},
    ),
    ExperimentSpec(
        id="E2", description="TF-IDF (1,2)-gram + MultinomialNB",
        model="nb", model_params={"alpha": 0.1},
        vectorizer={"kind": "tfidf", "ngram_range": (1, 2), "min_df": 2},
    ),
    ExperimentSpec(
        id="E3", description="TF-IDF + K-NN (cosine)",
        model="knn", model_params={"n_neighbors": 5},
        vectorizer={"kind": "tfidf", "min_df": 2},
    ),
    ExperimentSpec(
        id="E4", description="TF-IDF + Decision Tree",
        model="tree", model_params={"max_depth": 20, "min_samples_leaf": 2},
        vectorizer={"kind": "tfidf", "min_df": 2},
    ),
    ExperimentSpec(
        id="E5", description="Ablation: E2 + token hóa URL/tiền/số, bỏ dấu câu",
        model="nb", model_params={"alpha": 0.1},
        cleaner={"replace_url": True, "replace_money": True,
                 "replace_number": True, "remove_punct": True},
        vectorizer={"kind": "tfidf", "ngram_range": (1, 2), "min_df": 2},
    ),
    ExperimentSpec(
        id="E6", description="Ablation: E2 + bỏ stopword",
        model="nb", model_params={"alpha": 0.1},
        cleaner={"remove_punct": True, "remove_stopwords": True},
        vectorizer={"kind": "tfidf", "ngram_range": (1, 2), "min_df": 2},
    ),
]


def get_experiments(ids: Optional[List[str]] = None) -> List[ExperimentSpec]:
    if not ids:
        return EXPERIMENTS
    by_id = {e.id: e for e in EXPERIMENTS}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"Không có thí nghiệm {missing}. Có: {list(by_id)}")
    return [by_id[i] for i in ids]
