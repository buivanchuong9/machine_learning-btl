"""Cấu hình tập trung: đường dẫn, tên cột, seed, cách chia dữ liệu.

Chỉ cần sửa file này (hoặc truyền cờ CLI) khi đổi dataset.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_MODELS = ROOT / "outputs" / "models"
OUT_RESULTS = ROOT / "outputs" / "results"

RANDOM_STATE = 42


@dataclass(frozen=True)
class DataConfig:
    """Mô tả file dataset. Mặc định: SMS Spam Collection (UCI) – tab, không header, `label<TAB>text`."""

    path: Path = DATA_DIR / "sms+spam+collection" / "SMSSpamCollection"
    text_col: str = "text"
    label_col: str = "label"
    sep: str = "	"
    has_header: bool = False  # False: đặt tên cột theo thứ tự (label_col, text_col)
    encoding: str = "utf-8"
    spam_values: tuple = ("spam", "1")
    ham_values: tuple = ("ham", "0")
    drop_duplicates: bool = True  # bỏ tin trùng trước khi chia -> tránh rò rỉ train/test
    test_size: float = 0.2
    random_state: int = RANDOM_STATE
    cv_folds: int = 5


# Tên cột hay gặp, dùng để tự nhận diện khi chưa chỉnh config.
COMMON_COLUMN_PAIRS = [("v2", "v1"), ("text", "label"), ("message", "label"), ("sms", "label"),
                       ("Message", "Category"), ("text", "spam")]

# Nhãn mô hình dùng nội bộ: spam là lớp dương (1).
LABEL_NAMES = {0: "ham", 1: "spam"}
