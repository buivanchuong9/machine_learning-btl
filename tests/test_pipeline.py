import pandas as pd
import pytest

from spam_filter.experiments import EXPERIMENTS
from spam_filter.pipeline import build_pipeline
from spam_filter.preprocessing import TextCleaner

TEXTS = ["WIN a FREE prize now! Call 09061701461 or visit www.win.com to claim £500",
         "Are we still meeting for lunch tomorrow?"] * 20
LABELS = [1, 0] * 20


def test_cleaner_replaces_signals():
    out = TextCleaner(replace_url=True, replace_money=True, replace_number=True,
                      remove_punct=True).transform([TEXTS[0]])[0]
    assert "urltoken" in out and "moneytoken" in out and "numtoken" in out
    assert "www" not in out and "£" not in out


def test_cleaner_replaces_https_urls():
    out = TextCleaner(replace_url=True).transform(
        ["Visit https://example.com/login?next=1"]
    )[0]
    assert out == "visit urltoken"


def test_cleaner_replaces_www_urls():
    out = TextCleaner(replace_url=True).transform(["Visit www.example.com/deal"])[0]
    assert out == "visit urltoken"


def test_cleaner_replaces_bare_bitly_short_links():
    out = TextCleaner(replace_url=True).transform(["Click bit.ly/3xYz now"])[0]
    assert out == "click urltoken now"


def test_cleaner_replaces_bare_xyz_domains():
    out = TextCleaner(replace_url=True).transform(
        ["Visit secure-bank-login.xyz to verify"]
    )[0]
    assert out == "visit urltoken to verify"


def test_cleaner_preserves_punctuation_around_urls():
    out = TextCleaner(replace_url=True).transform(
        ["Open (https://example.com/path), then continue!"]
    )[0]
    assert out == "open ( urltoken), then continue!"


def test_cleaner_replaces_urls_in_unicode_text():
    out = TextCleaner(replace_url=True).transform(
        ["Nhận quà tại https://example.com/đăng-nhập nhé"]
    )[0]
    assert out == "nhận quà tại urltoken nhé"


def test_cleaner_keeps_non_url_dotted_text():
    text = ("The version is 1.2.3, e.g. this is not a URL; "
            "process.networking and type..lyk are not URLs either")
    assert TextCleaner(replace_url=True).transform([text]) == [text.lower()]


def test_cleaner_default_only_lowercases():
    assert TextCleaner().transform(["Hello  World"]) == ["hello world"]


def test_cleaner_stopwords():
    assert TextCleaner(remove_stopwords=True).transform(["this is a free prize"]) == ["free prize"]


@pytest.mark.parametrize("spec", EXPERIMENTS, ids=lambda s: s.id)
def test_every_experiment_fits_and_predicts(spec):
    pipe = build_pipeline(spec)
    pipe.fit(pd.Series(TEXTS), pd.Series(LABELS))
    assert list(pipe.predict(["Call now to win a free prize"])) in ([0], [1])
