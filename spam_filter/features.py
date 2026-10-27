"""Biểu diễn văn bản: Bag-of-Words và TF-IDF."""
from typing import Optional, Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

VECTORIZERS = {"bow": CountVectorizer, "tfidf": TfidfVectorizer}


def build_vectorizer(kind: str = "bow", ngram_range: Tuple[int, int] = (1, 1),
                     min_df: int = 1, max_df: float = 1.0,
                     max_features: Optional[int] = None):
    if kind not in VECTORIZERS:
        raise ValueError(f"kind phải thuộc {list(VECTORIZERS)}, nhận {kind!r}")
    return VECTORIZERS[kind](ngram_range=tuple(ngram_range), min_df=min_df,
                             max_df=max_df, max_features=max_features)
