"""Tests for feature_ablation.py.

Tests feature ablation group correctness and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.evaluation.evaluation_schema import FeatureAblationSummary
from ml.p1_clinical.evaluation.feature_ablation import (
    compute_feature_ablation,
    validate_feature_ablation_result,
)


def _make_model_config():
    """Create a minimal model config for ablation testing."""
    class MockConfig:
        def __init__(self):
            self.random_seed = 42
            self.impute_strategy = "median"
            self.impute_add_indicator = False
            self.scale_with_mean = True
            self.scale_with_std = True
    return MockConfig()


def _make_train_data(n=100, n_features=6, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.random((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = (X.iloc[:, 0] + X.iloc[:, 1] > 1.0).astype(int)
    return X, y


def test_compute_feature_ablation_basic():
    """Test basic feature ablation computation."""
    # Create a simple feature schema mock
    class MockFeatureSchema:
        def by_category(self, category):
            return []

    X_train, y_train = _make_train_data(n=200, n_features=4, seed=42)
    X_test, y_test = _make_train_data(n=50, n_features=4, seed=7)

    # Simple model that just predicts based on first feature
    class MockModel:
        def __init__(self):
            self.feature_names = None
            self.model_name = "MockModel"
        def fit(self, X, y):
            self.feature_names = list(X.columns)
            self.coef_ = np.ones(len(X.columns))
            return self
        def predict_proba(self, X):
            X_sel = X[self.feature_names]
            scores = X_sel.values @ self.coef_
            probs = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - probs, probs])
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def train_func(X, y, config):
        model = MockModel()
        model.fit(X, y)
        return model

    def predict_func(model, X, threshold):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        return pd.DataFrame({
            "predicted_probability": probs,
            "predicted_class": preds,
        }, index=X.index)

    config = _make_model_config()

    # Create a minimal feature schema
    class MockSchema:
        def by_category(self, category):
            return []

    result = compute_feature_ablation(
        train_func, predict_func,
        X_train, y_train, X_test, y_test,
        MockSchema(), config,
        threshold=0.5, source="synthetic",
    )

    assert result.model_name == "MockModel"
    assert result.model_version == "1.0.0"
    assert result.split == "test"
    assert result.source == "synthetic"
    assert result.n_features_full == 4


def test_validate_feature_ablation_result_valid():
    """Test validation of valid feature ablation result."""
    from ml.p1_clinical.evaluation.feature_ablation import FeatureAblationResult

    result = FeatureAblationSummary(
        model_name="test_model",
        model_version="1.0.0",
        split="test",
        n_features_full=10,
        ablations=[
            FeatureAblationResult(
                feature_group="current",
                n_features_removed=2,
                metrics_full={"auroc": 0.8},
                metrics_ablated={"auroc": 0.7},
                metric_changes={"auroc": -0.1},
                metric_changes_relative={"auroc": -0.125},
                source="synthetic",
            )
        ],
        source="synthetic",
    )

    validation = validate_feature_ablation_result(result)
    assert validation["valid"] is True
    assert len(validation["issues"]) == 0


def test_validate_feature_ablation_result_invalid():
    """Test validation of invalid feature ablation result."""
    from ml.p1_clinical.evaluation.feature_ablation import FeatureAblationResult

    result = FeatureAblationSummary(
        model_name="test_model",
        model_version="1.0.0",
        split="test",
        n_features_full=5,
        ablations=[
            FeatureAblationResult(
                feature_group="current",
                n_features_removed=10,  # More than n_features_full
                metrics_full={"auroc": 0.8},
                metrics_ablated={"auroc": 0.7},
                metric_changes={"auroc": -0.1},
                metric_changes_relative={"auroc": -0.125},
                source="synthetic",
            )
        ],
        source="synthetic",
    )

    validation = validate_feature_ablation_result(result)
    assert validation["valid"] is False
    assert len(validation["issues"]) > 0