"""Các thuật toán đã học: Naive Bayes, K-NN, Decision Tree."""
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE

MODELS = {
    "nb": lambda **p: MultinomialNB(**p),
    "knn": lambda **p: KNeighborsClassifier(**{"metric": "cosine", "algorithm": "brute", **p}),
    "tree": lambda **p: DecisionTreeClassifier(**{"random_state": RANDOM_STATE, **p}),
}


def build_model(name: str, **params):
    if name not in MODELS:
        raise ValueError(f"Model phải thuộc {list(MODELS)}, nhận {name!r}")
    return MODELS[name](**params)
