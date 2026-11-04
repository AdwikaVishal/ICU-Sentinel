"""Tests for model_serialization.py.

Tests save/load of model artifacts, preprocessing pipelines, and feature
lists. Uses synthetic data only.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ml.p1_clinical.classical_ml.logistic_regression import train_lr
from ml.p1_clinical.classical_ml.model_config import ModelConfig
from ml.p1_clinical.classical_ml.model_registry import ModelMetadata
from ml.p1_clinical.classical_ml.model_serialization import (
    load_feature_list,
    load_model_artifact,
    load_preprocessing_pipeline,
    save_feature_list,
    save_model_artifact,
    save_preprocessing_pipeline,
)


def _make_train_data(n_rows=50, n_features=4, seed=42):
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
# Save/load model artifact
# ------------------------------------------------------------------


def test_save_load_model_artifact(tmp_path):
    config = ModelConfig(lr_max_iter=500)
    X, y = _make_train_data(n_rows=100, n_features=4, seed=42)
    result = train_lr(X, y, config, list(X.columns))

    tmp_path / "model.joblib"
    from ml.p1_clinical.classical_ml.model_registry import ModelMetadata

    metadata = ModelMetadata(
        model_name="logistic_regression",
        model_version="1.0.0",
        training_timestamp=datetime.now(UTC).isoformat(),
        random_seed=42,
        feature_names=list(X.columns),
        selected_feature_names=list(X.columns),
        n_features_input=4,
        n_features_output=4,
        n_patients_train=10,
        n_rows_train=100,
        n_positive_train=30,
        n_negative_train=70,
        class_weight="balanced",
        hyperparameters={"C": 1.0},
        preprocessing_config={"impute_strategy": "median"},
        validation_metrics={"auroc": 0.85},
        split_identity="patient_level",
        source="synthetic",
    )

    save_model_artifact(
        model=result.model,
        preprocessing_pipeline=result.preprocessing_pipeline,
        metadata=metadata,
        artifact_dir=tmp_path,
    )

    # Verify artifact file exists.
    expected = tmp_path / "logistic_regression_1.0.0.joblib"
    assert expected.exists()

    # Load artifact.
    artifact = load_model_artifact(expected)
    assert "model" in artifact
    assert "preprocessing_pipeline" in artifact
    assert "metadata" in artifact
    assert artifact["metadata"]["model_name"] == "logistic_regression"


# ------------------------------------------------------------------
# Save/load preprocessing pipeline
# ------------------------------------------------------------------


def test_save_load_preprocessing_pipeline(tmp_path):
    config = ModelConfig(lr_max_iter=500)
    X, y = _make_train_data(n_rows=100, n_features=4, seed=42)
    result = train_lr(X, y, config, list(X.columns))

    prep_path = save_preprocessing_pipeline(
        result.preprocessing_pipeline,
        tmp_path,
        "logistic_regression",
        "1.0.0",
    )
    assert prep_path.exists()

    loaded = load_preprocessing_pipeline(prep_path)
    assert loaded is not None


# ------------------------------------------------------------------
# Save/load feature list
# ------------------------------------------------------------------


def test_save_load_feature_list(tmp_path):
    features = ["f0", "f1", "f2", "f3"]
    path = save_feature_list(
        features, tmp_path, "logistic_regression", "1.0.0"
    )
    assert path.exists()

    loaded = load_feature_list(path)
    assert loaded == features


# ------------------------------------------------------------------
# Metadata serialization
# ------------------------------------------------------------------


def test_metadata_save_load(tmp_path):
    metadata = ModelMetadata(
        model_name="random_forest",
        model_version="1.0.0",
        training_timestamp=datetime.now(UTC).isoformat(),
        random_seed=42,
        feature_names=["f0", "f1"],
        selected_feature_names=["f0"],
        n_features_input=2,
        n_features_output=1,
        n_patients_train=10,
        n_rows_train=100,
        n_positive_train=30,
        n_negative_train=70,
        class_weight="balanced",
        hyperparameters={"n_estimators": 100},
        preprocessing_config={"impute_strategy": "median"},
        validation_metrics={"auroc": 0.8},
        split_identity="patient_level",
        source="synthetic",
    )
    path = tmp_path / "metadata.json"
    metadata.save(path)
    loaded = ModelMetadata.load(path)
    assert loaded.model_name == "random_forest"
    assert loaded.selected_feature_names == ["f0"]
    assert loaded.validation_metrics["auroc"] == 0.8
