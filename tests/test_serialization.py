import numpy as np
from joblib import dump, load
from sklearn.pipeline import Pipeline

from spam_filter.experiments import EXPERIMENTS
from spam_filter.pipeline import build_pipeline
from spam_filter.preprocessing import TextCleaner


def test_joblib_round_trip_preserves_pipeline_predictions_and_cleaner(tmp_path):
    spec = next(item for item in EXPERIMENTS if item.id == "E0")
    pipe = build_pipeline(spec)
    X_train = [
        "WIN a free prize now",
        "claim your free reward",
        "cheap offer call today",
        "meeting for lunch tomorrow",
        "see you after class",
        "let us meet for coffee",
    ]
    y_train = [1, 1, 1, 0, 0, 0]
    probes = ["FREE PRIZE NOW", "Let's meet for lunch", "a cheap reward"]

    pipe.fit(X_train, y_train)
    predictions_before = pipe.predict(probes)
    artifact = tmp_path / "e0.joblib"
    dump(pipe, artifact)
    loaded = load(artifact)

    assert isinstance(loaded, Pipeline)
    assert list(loaded.named_steps) == ["clean", "vec", "clf"]
    assert isinstance(loaded.named_steps["clean"], TextCleaner)
    assert loaded.named_steps["clean"].transform(["WIN FREE"]) == ["win free"]
    np.testing.assert_array_equal(loaded.predict(probes), predictions_before)
