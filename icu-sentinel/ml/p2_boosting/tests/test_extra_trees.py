"""Software-validation tests for the P2 Extra Trees model."""

from __future__ import annotations

import numpy as np

from ml.p2_boosting.config import ExtraTreesConfig, P2Config
from ml.p2_boosting.extra_trees import (
    build_extra_trees_pipeline,
    predict_extra_trees,
    train_extra_trees,
)
from ml.p2_boosting.serialization import load_p2_model, save_p2_model
from ml.p2_boosting.tests.fixtures import make_split


def _data(seed: int = 42):
    split = make_split("train", n_rows=100, seed=seed)
    return split.X, split.y


def test_build_extra_trees_pipeline():
    config = P2Config(extra_trees=ExtraTreesConfig(n_estimators=10))
    X, y = _data()
    pipeline = build_extra_trees_pipeline(config, y)
    assert "imputer" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_train_extra_trees_handles_missing_values():
    config = P2Config(extra_trees=ExtraTreesConfig(n_estimators=10))
    X, y = _data()
    assert X.isna().any().any()
    result = train_extra_trees(X, y, config, list(X.columns))
    assert result.model is not None
    assert result.selected_features == list(X.columns)
    assert result.training_timestamp


def test_extra_trees_prediction_shape_and_range():
    config = P2Config(extra_trees=ExtraTreesConfig(n_estimators=10))
    X, y = _data()
    result = train_extra_trees(X, y, config, list(X.columns))
    predictions = predict_extra_trees(result, X, threshold=0.4)
    assert len(predictions) == len(X)
    assert list(predictions.columns) == [
        "predicted_probability",
        "predicted_class",
    ]
    assert predictions["predicted_probability"].between(0, 1).all()
    assert predictions["predicted_class"].isin([0, 1]).all()
    assert (predictions["predicted_class"] == (predictions["predicted_probability"] >= 0.4)).all()


def test_extra_trees_reproducible():
    config = P2Config(extra_trees=ExtraTreesConfig(n_estimators=10))
    X, y = _data()
    first = train_extra_trees(X, y, config, list(X.columns))
    second = train_extra_trees(X, y, config, list(X.columns))
    p1 = predict_extra_trees(first, X)["predicted_probability"].to_numpy()
    p2 = predict_extra_trees(second, X)["predicted_probability"].to_numpy()
    np.testing.assert_allclose(p1, p2)


def test_extra_trees_schema_mismatch():
    config = P2Config(extra_trees=ExtraTreesConfig(n_estimators=10))
    X, y = _data()
    result = train_extra_trees(X, y, config, list(X.columns))
    bad = X.drop(columns=["f0"])
    try:
        predict_extra_trees(result, bad)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected feature schema mismatch to raise.")


def test_extra_trees_serialization_roundtrip(tmp_path):
    config = P2Config(extra_trees=ExtraTreesConfig(n_estimators=10))
    X, y = _data()
    result = train_extra_trees(X, y, config, list(X.columns), source="synthetic")
    path = save_p2_model(result, tmp_path)
    assert path.exists()

    loaded = load_p2_model(path)
    before = predict_extra_trees(result, X)["predicted_probability"].to_numpy()
    after = predict_extra_trees(loaded, X)["predicted_probability"].to_numpy()
    np.testing.assert_allclose(before, after)
    assert loaded.model_name == "extra_trees"
    assert loaded.selected_features == list(X.columns)


def test_extra_trees_save_refuses_overwrite(tmp_path):
    config = P2Config(extra_trees=ExtraTreesConfig(n_estimators=10))
    X, y = _data()
    result = train_extra_trees(X, y, config, list(X.columns))
    save_p2_model(result, tmp_path)
    try:
        save_p2_model(result, tmp_path)
    except FileExistsError:
        return
    raise AssertionError("Expected existing artifact to be protected.")
