"""Software-validation tests for the P2 LightGBM model."""

from __future__ import annotations

import numpy as np

from ml.p2_boosting.config import LightGBMConfig, P2Config
from ml.p2_boosting.lightgbm_model import (
    build_lightgbm_pipeline,
    predict_lightgbm,
    train_lightgbm,
)
from ml.p2_boosting.serialization import load_p2_model, save_p2_model
from ml.p2_boosting.tests.fixtures import make_split


def _data(seed: int = 42):
    split = make_split("train", n_rows=120, seed=seed)
    X = split.X.copy()
    y = split.y.copy()
    y.iloc[::7] = 0
    y.iloc[0] = 1
    return X, y


def test_build_lightgbm_pipeline():
    config = P2Config(lightgbm=LightGBMConfig(n_estimators=10, verbosity=-1))
    X, y = _data()
    pipeline = build_lightgbm_pipeline(config, y)
    assert "classifier" in pipeline.named_steps


def test_train_lightgbm_imbalanced():
    config = P2Config(lightgbm=LightGBMConfig(n_estimators=10, verbosity=-1))
    X, y = _data()
    assert y.mean() < 0.5
    result = train_lightgbm(X, y, config, list(X.columns))
    assert result.model is not None
    assert result.class_weight == "balanced"


def test_lightgbm_prediction_shape_and_range():
    config = P2Config(lightgbm=LightGBMConfig(n_estimators=10, verbosity=-1))
    X, y = _data()
    result = train_lightgbm(X, y, config, list(X.columns))
    predictions = predict_lightgbm(result, X, threshold=0.5)
    assert len(predictions) == len(X)
    assert predictions["predicted_probability"].between(0, 1).all()
    assert predictions["predicted_class"].isin([0, 1]).all()


def test_lightgbm_reproducible():
    config = P2Config(lightgbm=LightGBMConfig(n_estimators=10, verbosity=-1))
    X, y = _data()
    first = train_lightgbm(X, y, config, list(X.columns))
    second = train_lightgbm(X, y, config, list(X.columns))
    np.testing.assert_allclose(
        predict_lightgbm(first, X)["predicted_probability"].to_numpy(),
        predict_lightgbm(second, X)["predicted_probability"].to_numpy(),
    )


def test_lightgbm_serialization_roundtrip(tmp_path):
    config = P2Config(lightgbm=LightGBMConfig(n_estimators=10, verbosity=-1))
    X, y = _data()
    result = train_lightgbm(X, y, config, list(X.columns))
    path = save_p2_model(result, tmp_path)
    loaded = load_p2_model(path)
    np.testing.assert_allclose(
        predict_lightgbm(result, X)["predicted_probability"].to_numpy(),
        predict_lightgbm(loaded, X)["predicted_probability"].to_numpy(),
    )


def test_lightgbm_invalid_input_message():
    config = P2Config(lightgbm=LightGBMConfig(n_estimators=10, verbosity=-1))
    X, y = _data()
    result = train_lightgbm(X, y, config, list(X.columns))
    bad = X.copy()
    bad["f1"] = "not-a-number"
    try:
        predict_lightgbm(result, bad)
    except ValueError as exc:
        assert "must be numeric" in str(exc)
    else:
        raise AssertionError("Expected non-numeric input to raise.")
