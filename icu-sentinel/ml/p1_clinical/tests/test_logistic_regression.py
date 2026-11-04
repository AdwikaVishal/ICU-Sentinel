"""Tests for logistic_regression.py.

Tests training, prediction, coefficient extraction, and serialization.
Uses synthetic data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.classical_ml.logistic_regression import (
    build_lr_pipeline,
    predict_lr,
    save_lr_artifact,
    train_lr,
)
from ml.p1_clinical.classical_ml.model_config import ModelConfig
from ml.p1_clinical.classical_ml.model_serialization import (
    load_feature_list,
    load_model_artifact,
)


def _make_train_data(n_rows=100, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(0, 1, size=(n_rows, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # Create a linear signal + noise.
    linear = X["f0"] * 1.5 + X["f1"] * -1.0 + 0.5
    prob = 1 / (1 + np.exp(-linear))
    y = (rng.random(n_rows) < prob).astype(int)
    return X, y


# ------------------------------------------------------------------
# Build pipeline
# ------------------------------------------------------------------


def test_build_lr_pipeline():
    config = ModelConfig()
    features = ["f0", "f1", "f2"]
    pipeline = build_lr_pipeline(config, features)
    assert pipeline is not None
    assert "imputer" in pipeline.named_steps
    assert "scaler" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------


def test_train_lr():
    config = ModelConfig(lr_max_iter=1000)
    X, y = _make_train_data(n_rows=200, n_features=5, seed=42)
    result = train_lr(X, y, config, list(X.columns))
    assert result.model is not None
    assert result.preprocessing_pipeline is not None
    assert result.selected_features == list(X.columns)
    assert result.converged is not None
    assert result.n_iter is not None


def test_train_lr_coefficients():
    config = ModelConfig(lr_max_iter=1000)
    X, y = _make_train_data(n_rows=200, n_features=5, seed=42)
    result = train_lr(X, y, config, list(X.columns))
    assert result.coefficients is not None
    assert "feature" in result.coefficients.columns
    assert "coefficient" in result.coefficients.columns
    assert len(result.coefficients) == 5


def test_train_lr_training_metrics():
    config = ModelConfig(lr_max_iter=1000)
    X, y = _make_train_data(n_rows=200, n_features=5, seed=42)
    result = train_lr(X, y, config, list(X.columns))
    assert "auroc" in result.training_metrics
    assert "auprc" in result.training_metrics
    assert "f1" in result.training_metrics
    assert "precision" in result.training_metrics
    assert "recall" in result.training_metrics


# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------


def test_predict_lr():
    config = ModelConfig(lr_max_iter=1000)
    X_train, y_train = _make_train_data(n_rows=200, n_features=5, seed=42)
    result = train_lr(X_train, y_train, config, list(X_train.columns))

    X_test, _y_test = _make_train_data(n_rows=50, n_features=5, seed=7)
    pred_df = predict_lr(result, X_test, threshold=0.5)
    assert "predicted_probability" in pred_df.columns
    assert "predicted_class" in pred_df.columns
    assert len(pred_df) == 50
    assert pred_df["predicted_class"].isin([0, 1]).all()
    assert (pred_df["predicted_probability"] >= 0).all()
    assert (pred_df["predicted_probability"] <= 1).all()


def test_predict_lr_threshold():
    config = ModelConfig(lr_max_iter=1000)
    X_train, y_train = _make_train_data(n_rows=200, n_features=5, seed=42)
    result = train_lr(X_train, y_train, config, list(X_train.columns))

    X_test, _ = _make_train_data(n_rows=50, n_features=5, seed=7)
    pred_high = predict_lr(result, X_test, threshold=0.9)
    pred_low = predict_lr(result, X_test, threshold=0.1)
    assert pred_high["predicted_class"].sum() <= pred_low["predicted_class"].sum()


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------


def test_save_lr_artifact(tmp_path):
    config = ModelConfig(lr_max_iter=500)
    X, y = _make_train_data(n_rows=100, n_features=4, seed=42)
    result = train_lr(X, y, config, list(X.columns))

    artifact_path = save_lr_artifact(result, tmp_path)
    assert artifact_path.exists()

    # Load and verify.
    artifact = load_model_artifact(artifact_path)
    assert "model" in artifact
    assert "preprocessing_pipeline" in artifact
    assert "metadata" in artifact

    # Load feature list.
    feature_file = tmp_path / "logistic_regression_1.0.0_features.json"
    features = load_feature_list(feature_file)
    assert len(features) == 4


# ------------------------------------------------------------------
# Label isolation
# ------------------------------------------------------------------


def test_lr_target_not_feature():
    config = ModelConfig(lr_max_iter=1000)
    X, y = _make_train_data(n_rows=200, n_features=5, seed=42)
    # SepsisLabel should not be in selected features.
    assert "SepsisLabel" not in list(X.columns)
    result = train_lr(X, y, config, list(X.columns))
    assert "SepsisLabel" not in result.selected_features
