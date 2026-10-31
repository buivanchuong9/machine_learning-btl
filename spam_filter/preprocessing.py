"""Tiền xử lý văn bản dưới dạng transformer của scikit-learn (dùng được trong Pipeline)."""
import re
from typing import Iterable, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:co\.uk|com|net|org|uk|ly|xyz)(?![\w-])\S*)")
MONEY_RE = re.compile(r"[£$€]\s?\d[\d,.]*|\d[\d,.]*\s?(?:p|pounds|usd)\b")
NUMBER_RE = re.compile(r"\d+")
PUNCT_RE = re.compile(r"[^\w\s]")
SPACE_RE = re.compile(r"\s+")

# Token thay thế: toàn chữ cái để không bị bước bỏ dấu câu / số xóa mất.
URL_TOKEN, MONEY_TOKEN, NUM_TOKEN = " urltoken ", " moneytoken ", " numtoken "
URL_TRAILING_PUNCTUATION = ".,!?:;)]}\"'>"


def _replace_url(match: re.Match) -> str:
    """Replace a URL while leaving sentence punctuation outside the token."""
    url = match.group(0)
    trimmed_url = url.rstrip(URL_TRAILING_PUNCTUATION)
    if trimmed_url == url:
        return URL_TOKEN
    trailing_punctuation = url[len(trimmed_url):]
    return f" {URL_TOKEN.strip()}{trailing_punctuation} "


class TextCleaner(BaseEstimator, TransformerMixin):
    """Làm sạch văn bản. Mỗi bước bật/tắt được để làm thí nghiệm ablation.

    - replace_*: thay URL / tiền / số bằng token thay vì xóa (giữ tín hiệu spam).
    - remove_stopwords: bỏ stopword tiếng Anh.
    """

    def __init__(self, lowercase: bool = True, replace_url: bool = False,
                 replace_money: bool = False, replace_number: bool = False,
                 remove_punct: bool = False, remove_stopwords: bool = False):
        self.lowercase = lowercase
        self.replace_url = replace_url
        self.replace_money = replace_money
        self.replace_number = replace_number
        self.remove_punct = remove_punct
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        return self  # không học tham số từ dữ liệu -> không có nguy cơ leakage

    def transform(self, X: Iterable[str]) -> List[str]:
        return [self._clean(text) for text in X]

    def _clean(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        # Thứ tự quan trọng: URL -> tiền -> số (tiền chứa số).
        if self.replace_url:
            text = URL_RE.sub(_replace_url, text)
        if self.replace_money:
            text = MONEY_RE.sub(MONEY_TOKEN, text)
        if self.replace_number:
            text = NUMBER_RE.sub(NUM_TOKEN, text)
        if self.remove_punct:
            text = PUNCT_RE.sub(" ", text)
        text = SPACE_RE.sub(" ", text).strip()
        if self.remove_stopwords:
            text = " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)
        return text
