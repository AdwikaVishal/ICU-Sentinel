"""Tests for svm.py.

Tests training, prediction, serialization, and computational manageability.
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
from ml.p1_clinical.classical_ml.svm import (
    build_svm_pipeline,
    predict_svm,
    save_svm_artifact,
    train_svm,
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


def test_build_svm_pipeline():
    config = ModelConfig()
    features = ["f0", "f1", "f2"]
    pipeline = build_svm_pipeline(config, features)
    assert pipeline is not None
    assert "imputer" in pipeline.named_steps
    assert "scaler" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------


def test_train_svm():
    config = ModelConfig(svm_max_iter=1000)
    X, y = _make_train_data(n_rows=100, n_features=5, seed=42)
    result = train_svm(X, y, config, list(X.columns))
    assert result.model is not None
    assert result.preprocessing_pipeline is not None
    assert result.selected_features == list(X.columns)


def test_train_svm_training_metrics():
    config = ModelConfig(svm_max_iter=1000)
    X, y = _make_train_data(n_rows=100, n_features=5, seed=42)
    result = train_svm(X, y, config, list(X.columns))
    assert "auroc" in result.training_metrics
    assert "auprc" in result.training_metrics
    assert "f1" in result.training_metrics


# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------


def test_predict_svm():
    config = ModelConfig(svm_max_iter=1000)
    X_train, y_train = _make_train_data(n_rows=100, n_features=5, seed=42)
    result = train_svm(X_train, y_train, config, list(X_train.columns))

    X_test, _ = _make_train_data(n_rows=30, n_features=5, seed=7)
    pred_df = predict_svm(result, X_test, threshold=0.5)
    assert "predicted_probability" in pred_df.columns
    assert "predicted_class" in pred_df.columns
    assert len(pred_df) == 30
    assert pred_df["predicted_class"].isin([0, 1]).all()


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------


def test_save_svm_artifact(tmp_path):
    config = ModelConfig(svm_max_iter=500)
    X, y = _make_train_data(n_rows=100, n_features=4, seed=42)
    result = train_svm(X, y, config, list(X.columns))

    artifact_path = save_svm_artifact(result, tmp_path)
    assert artifact_path.exists()

    artifact = load_model_artifact(artifact_path)
    assert "model" in artifact
    assert "preprocessing_pipeline" in artifact
    assert "metadata" in artifact

    feature_file = tmp_path / "svm_1.0.0_features.json"
    features = load_feature_list(feature_file)
    assert len(features) == 4


# ------------------------------------------------------------------
# Label isolation
# ------------------------------------------------------------------


def test_svm_target_not_feature():
    config = ModelConfig(svm_max_iter=500)
    X, y = _make_train_data(n_rows=100, n_features=5, seed=42)
    assert "SepsisLabel" not in list(X.columns)
    result = train_svm(X, y, config, list(X.columns))
    assert "SepsisLabel" not in result.selected_features
