"""Tests for random_forest.py.

Tests training, prediction, feature importance, and serialization.
Uses synthetic data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.classical_ml.model_config import ModelConfig
from ml.p1_clinical.classical_ml.model_serialization import (
    load_feature_list,
    load_model_artifact,
)
from ml.p1_clinical.classical_ml.random_forest import (
    build_rf_pipeline,
    predict_rf,
    save_rf_artifact,
    train_rf,
)


def _make_train_data(n_rows=100, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(0, 1, size=(n_rows, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    linear = X["f0"] * 1.5 + X["f1"] * -1.0 + 0.5
    prob = 1 / (1 + np.exp(-linear))
    y = (rng.random(n_rows) < prob).astype(int)
    return X, y


# ------------------------------------------------------------------
# Build pipeline
# ------------------------------------------------------------------


def test_build_rf_pipeline():
    config = ModelConfig()
    features = ["f0", "f1", "f2"]
    pipeline = build_rf_pipeline(config, features)
    assert pipeline is not None
    assert "imputer" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    # No scaler for RF.
    assert "scaler" not in pipeline.named_steps


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------


def test_train_rf():
    config = ModelConfig(rf_n_estimators=10, rf_max_depth=5)
    X, y = _make_train_data(n_rows=200, n_features=5, seed=42)
    result = train_rf(X, y, config, list(X.columns))
    assert result.model is not None
    assert result.preprocessing_pipeline is not None
    assert result.selected_features == list(X.columns)
    assert result.n_trees == 10
    assert result.max_depth == 5


def test_train_rf_feature_importance():
    config = ModelConfig(rf_n_estimators=10, rf_max_depth=5)
    X, y = _make_train_data(n_rows=200, n_features=5, seed=42)
    result = train_rf(X, y, config, list(X.columns))
    assert result.feature_importance is not None
    assert "feature" in result.feature_importance.columns
    assert "importance" in result.feature_importance.columns
    assert len(result.feature_importance) == 5


def test_train_rf_training_metrics():
    config = ModelConfig(rf_n_estimators=10, rf_max_depth=5)
    X, y = _make_train_data(n_rows=200, n_features=5, seed=42)
    result = train_rf(X, y, config, list(X.columns))
    assert "auroc" in result.training_metrics
    assert "auprc" in result.training_metrics
    assert "f1" in result.training_metrics


# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------


def test_predict_rf():
    config = ModelConfig(rf_n_estimators=10, rf_max_depth=5)
    X_train, y_train = _make_train_data(n_rows=200, n_features=5, seed=42)
    result = train_rf(X_train, y_train, config, list(X_train.columns))

    X_test, _y_test = _make_train_data(n_rows=50, n_features=5, seed=7)
    pred_df = predict_rf(result, X_test, threshold=0.5)
    assert "predicted_probability" in pred_df.columns
    assert "predicted_class" in pred_df.columns
    assert len(pred_df) == 50
    assert pred_df["predicted_class"].isin([0, 1]).all()
    assert (pred_df["predicted_probability"] >= 0).all()
    assert (pred_df["predicted_probability"] <= 1).all()


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------


def test_save_rf_artifact(tmp_path):
    config = ModelConfig(rf_n_estimators=10, rf_max_depth=5)
    X, y = _make_train_data(n_rows=100, n_features=4, seed=42)
    result = train_rf(X, y, config, list(X.columns))

    artifact_path = save_rf_artifact(result, tmp_path)
    assert artifact_path.exists()

    artifact = load_model_artifact(artifact_path)
    assert "model" in artifact
    assert "preprocessing_pipeline" in artifact
    assert "metadata" in artifact

    feature_file = tmp_path / "random_forest_1.0.0_features.json"
    features = load_feature_list(feature_file)
    assert len(features) == 4


# ------------------------------------------------------------------
# Label isolation
# ------------------------------------------------------------------


def test_rf_target_not_feature():
    config = ModelConfig(rf_n_estimators=10)
    X, y = _make_train_data(n_rows=200, n_features=5, seed=42)
    assert "SepsisLabel" not in list(X.columns)
    result = train_rf(X, y, config, list(X.columns))
    assert "SepsisLabel" not in result.selected_features
