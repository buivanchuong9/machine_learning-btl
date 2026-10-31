from dataclasses import replace

import pandas as pd
import pytest

from spam_filter.config import DataConfig
from spam_filter.data import load_dataset, split_data


def _write_dataset(path, contents):
    path.write_text(contents, encoding="utf-8")
    return DataConfig(path=path)


@pytest.fixture(scope="module")
def provided_corpus():
    cfg = DataConfig()
    return cfg, load_dataset(cfg)


def test_load_dataset_returns_canonical_columns_and_maps_labels(tmp_path):
    cfg = _write_dataset(
        tmp_path / "labels.tsv",
        "ham\tA ham message\nspam\tA spam message\n0\tNumeric ham\n1\tNumeric spam\n",
    )

    df = load_dataset(cfg)

    assert df.columns.tolist() == ["text", "label"]
    assert df.to_dict("records") == [
        {"text": "A ham message", "label": 0},
        {"text": "A spam message", "label": 1},
        {"text": "Numeric ham", "label": 0},
        {"text": "Numeric spam", "label": 1},
    ]


def test_load_dataset_drops_rows_with_missing_fields(tmp_path):
    cfg = _write_dataset(
        tmp_path / "missing.tsv",
        "ham\tkept message\n\tmissing label\nspam\t\n",
    )

    df = load_dataset(cfg)

    assert df.to_dict("records") == [{"text": "kept message", "label": 0}]


def test_load_dataset_rejects_unknown_nonempty_labels(tmp_path):
    cfg = _write_dataset(tmp_path / "invalid.tsv", "maybe\tA message\n")

    with pytest.raises(ValueError, match="Nhãn không nhận diện được"):
        load_dataset(cfg)


def test_load_dataset_strips_text_but_retains_whitespace_only_messages(tmp_path):
    cfg = _write_dataset(
        tmp_path / "empty.tsv",
        "ham\t   \nspam\tusable message\nspam\t\n",
    )

    df = load_dataset(cfg)

    # A missing TSV field is dropped; a whitespace-only field becomes an empty string.
    assert df.to_dict("records") == [
        {"text": "", "label": 0},
        {"text": "usable message", "label": 1},
    ]


def test_load_dataset_removes_exact_duplicates_by_default(tmp_path):
    path = tmp_path / "duplicates.tsv"
    contents = "ham\tRepeated text\nham\tRepeated text\nspam\tOther text\n"
    cfg = _write_dataset(path, contents)

    df = load_dataset(cfg)
    with_duplicates = load_dataset(replace(cfg, drop_duplicates=False))

    assert df.to_dict("records") == [
        {"text": "Repeated text", "label": 0},
        {"text": "Other text", "label": 1},
    ]
    assert len(with_duplicates) == 3


def test_load_dataset_keeps_first_label_for_conflicting_duplicate_text(tmp_path):
    cfg = _write_dataset(
        tmp_path / "conflicting.tsv",
        "ham\tConflicting text\nspam\tConflicting text\n",
    )

    df = load_dataset(cfg)

    # Current deduplication is text-only and keeps the first occurrence.
    assert df.to_dict("records") == [{"text": "Conflicting text", "label": 0}]


def test_load_dataset_is_deterministic(tmp_path):
    cfg = _write_dataset(
        tmp_path / "stable.tsv",
        "spam\tSecond\nham\tFirst\nspam\tThird\n",
    )

    first = load_dataset(cfg)
    second = load_dataset(cfg)

    pd.testing.assert_frame_equal(first, second)


def test_provided_corpus_has_expected_size_and_class_distribution(provided_corpus):
    _, df = provided_corpus

    assert df.columns.tolist() == ["text", "label"]
    assert len(df) == 5160
    assert df["label"].value_counts().sort_index().to_dict() == {0: 4518, 1: 642}
    assert not df["text"].duplicated().any()


def test_split_has_expected_sizes_and_stratified_distribution(provided_corpus):
    cfg, df = provided_corpus
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    assert len(X_train) == len(y_train) == 4128
    assert len(X_test) == len(y_test) == 1032
    prevalence = df["label"].mean()
    assert abs(y_train.mean() - prevalence) < 0.01
    assert abs(y_test.mean() - prevalence) < 0.01


def test_split_has_no_exact_text_overlap(provided_corpus):
    cfg, df = provided_corpus
    X_train, X_test, _, _ = split_data(df, cfg)

    assert set(X_train).isdisjoint(set(X_test))


def test_split_is_deterministic_for_the_same_seed(provided_corpus):
    cfg, df = provided_corpus
    first = split_data(df, cfg)
    second = split_data(df, replace(cfg, random_state=42))

    for first_part, second_part in zip(first, second):
        pd.testing.assert_series_equal(first_part, second_part)


def test_different_split_seed_changes_membership_not_sizes(provided_corpus):
    cfg, df = provided_corpus
    seed_42 = split_data(df, cfg)
    seed_7 = split_data(df, replace(cfg, random_state=7))

    assert len(seed_42[0]) == len(seed_7[0]) == 4128
    assert len(seed_42[1]) == len(seed_7[1]) == 1032
    assert set(seed_42[1].index) != set(seed_7[1].index)
