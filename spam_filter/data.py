"""Đọc, kiểm tra và chia dữ liệu."""
import csv
import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import COMMON_COLUMN_PAIRS, DataConfig

log = logging.getLogger(__name__)


def _resolve_columns(raw: pd.DataFrame, cfg: DataConfig) -> Tuple[str, str]:
    """Dùng cột trong config; nếu không có thì thử các cặp tên phổ biến."""
    if cfg.text_col in raw.columns and cfg.label_col in raw.columns:
        return cfg.text_col, cfg.label_col
    for text_col, label_col in COMMON_COLUMN_PAIRS:
        if text_col in raw.columns and label_col in raw.columns:
            log.info("Tự nhận diện cột: text=%s, label=%s", text_col, label_col)
            return text_col, label_col
    raise ValueError(
        f"Không tìm thấy cột text/label. Cột hiện có: {list(raw.columns)}. "
        "Dùng --text-col/--label-col hoặc sửa DataConfig."
    )


def _encode_label(value: object, cfg: DataConfig) -> int:
    key = str(value).strip().lower()
    if key in cfg.spam_values:
        return 1
    if key in cfg.ham_values:
        return 0
    raise ValueError(f"Nhãn không nhận diện được: {value!r}")


def _read_table(cfg: DataConfig) -> pd.DataFrame:
    kwargs = {"sep": cfg.sep}
    if cfg.sep == "	":
        kwargs["quoting"] = csv.QUOTE_NONE  # tin nhắn chứa dấu " không phải dấu bao trường
    if not cfg.has_header:
        kwargs.update(header=None, names=[cfg.label_col, cfg.text_col])
    try:
        return pd.read_csv(cfg.path, encoding=cfg.encoding, **kwargs)
    except UnicodeDecodeError:
        log.warning("Không đọc được bằng %s, thử latin-1", cfg.encoding)
        return pd.read_csv(cfg.path, encoding="latin-1", **kwargs)


def load_dataset(cfg: DataConfig) -> pd.DataFrame:
    """Trả về DataFrame chuẩn hóa gồm 2 cột: `text` (str) và `label` (0=ham, 1=spam)."""
    if not cfg.path.exists():
        raise FileNotFoundError(
            f"Chưa có dataset tại {cfg.path}. Xem README mục 'Thêm dataset'."
        )
    raw = _read_table(cfg)
    text_col, label_col = _resolve_columns(raw, cfg)

    df = raw[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    n_raw = len(df)
    df = df.dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].map(lambda v: _encode_label(v, cfg))

    n_dup = int(df.duplicated(subset="text").sum())
    if cfg.drop_duplicates:
        df = df.drop_duplicates(subset="text")
    df = df.reset_index(drop=True)
    log.info("Đọc %d dòng -> %d dòng sau làm sạch (trùng text: %d)", n_raw, len(df), n_dup)
    return df


def split_data(df: pd.DataFrame, cfg: DataConfig):
    """Chia train/test có phân tầng theo nhãn."""
    return train_test_split(
        df["text"], df["label"],
        test_size=cfg.test_size, stratify=df["label"], random_state=cfg.random_state,
    )
