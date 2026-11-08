"""Software-validation tests for the P2 XGBoost model."""

from __future__ import annotations

import numpy as np

from ml.p2_boosting.config import P2Config, XGBoostConfig
from ml.p2_boosting.serialization import load_p2_model, save_p2_model
from ml.p2_boosting.tests.fixtures import make_split
from ml.p2_boosting.xgboost_model import (
    build_xgboost_pipeline,
    predict_xgboost,
    train_xgboost,
)


def _data(seed: int = 42):
    split = make_split("train", n_rows=120, seed=seed)
    X = split.X.copy()
    y = split.y.copy()
    y.iloc[::8] = 0
    y.iloc[0] = 1
    return X, y


def test_build_xgboost_pipeline():
    config = P2Config(xgboost=XGBoostConfig(n_estimators=10))
    X, y = _data()
    pipeline = build_xgboost_pipeline(config, y)
    classifier = pipeline.named_steps["classifier"]
    assert classifier.scale_pos_weight > 0


def test_train_xgboost_imbalanced():
    config = P2Config(xgboost=XGBoostConfig(n_estimators=10))
    X, y = _data()
    result = train_xgboost(X, y, config, list(X.columns))
    assert result.model is not None
    assert result.hyperparameters["n_estimators"] == 10


def test_xgboost_prediction_shape_and_range():
    config = P2Config(xgboost=XGBoostConfig(n_estimators=10))
    X, y = _data()
    result = train_xgboost(X, y, config, list(X.columns))
    predictions = predict_xgboost(result, X, threshold=0.6)
    assert len(predictions) == len(X)
    assert predictions["predicted_probability"].between(0, 1).all()
    assert predictions["predicted_class"].isin([0, 1]).all()


def test_xgboost_reproducible():
    config = P2Config(xgboost=XGBoostConfig(n_estimators=10))
    X, y = _data()
    first = train_xgboost(X, y, config, list(X.columns))
    second = train_xgboost(X, y, config, list(X.columns))
    np.testing.assert_allclose(
        predict_xgboost(first, X)["predicted_probability"].to_numpy(),
        predict_xgboost(second, X)["predicted_probability"].to_numpy(),
    )


def test_xgboost_serialization_roundtrip(tmp_path):
    config = P2Config(xgboost=XGBoostConfig(n_estimators=10))
    X, y = _data()
    result = train_xgboost(X, y, config, list(X.columns))
    path = save_p2_model(result, tmp_path)
    loaded = load_p2_model(path)
    np.testing.assert_allclose(
        predict_xgboost(result, X)["predicted_probability"].to_numpy(),
        predict_xgboost(loaded, X)["predicted_probability"].to_numpy(),
    )


def test_xgboost_single_class_training_rejected():
    config = P2Config(xgboost=XGBoostConfig(n_estimators=10))
    X, _y = _data()
    y = np.zeros(len(X), dtype=int)
    try:
        train_xgboost(X, y, config, list(X.columns))
    except ValueError as exc:
        assert "both classes" in str(exc)
    else:
        raise AssertionError("Expected single-class target to raise.")
