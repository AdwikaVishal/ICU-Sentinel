"""Support Vector Machine model for P1 classical ML.

Implements a reproducible SVM pipeline with:
- Standardized numeric features (fitted on training data only).
- Missing-value imputation (fitted on training data only).
- Configurable kernel and regularization.
- Probability estimates via calibrated probabilities.
- Model serialization.

If SVM becomes computationally expensive, the configuration uses
reasonable defaults for local execution. Probability calibration is
performed internally by scikit-learn using training-only cross-validation
(when probability=True), never on the test set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .model_config import ModelConfig
from .model_serialization import save_feature_list, save_model_artifact

logger = logging.getLogger("icu_sentinel.p1_clinical.classical_ml.svm")


@dataclass
class SVMResult:
    """Result of training/evaluating SVM."""

    model_name: str = "svm"
    model_version: str = "1.0.0"
    model: Pipeline | None = None
    preprocessing_pipeline: Pipeline | None = None
    selected_features: list[str] = field(default_factory=list)
    training_metrics: dict = field(default_factory=dict)
    validation_metrics: dict = field(default_factory=dict)
    support_vectors: int | None = None


def build_svm_pipeline(
    config: ModelConfig,
    selected_features: list[str],
) -> Pipeline:
    """Build an SVM pipeline.

    Pipeline steps:
      1. SimpleImputer (median, fitted on train only)
      2. StandardScaler (fitted on train only)
      3. SVC with probability estimates

    Parameters
    ----------
    config : ModelConfig
        Model configuration.
    selected_features : list[str]
        Selected feature names.

    Returns
    -------
    Pipeline
        Configured scikit-learn Pipeline.
    """
    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy=config.impute_strategy,
                    add_indicator=config.impute_add_indicator,
                ),
            ),
            (
                "scaler",
                StandardScaler(
                    with_mean=config.scale_with_mean,
                    with_std=config.scale_with_std,
                ),
            ),
            (
                "classifier",
                SVC(
                    C=config.svm_c,
                    kernel=config.svm_kernel,
                    gamma=config.svm_gamma,
                    class_weight=config.svm_class_weight,
                    probability=config.svm_probability,
                    max_iter=config.svm_max_iter,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )
    return pipeline


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelConfig,
    selected_features: list[str],
) -> SVMResult:
    """Train SVM on training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (SepsisLabel).
    config : ModelConfig
        Model configuration.
    selected_features : list[str]
        Selected feature names.

    Returns
    -------
    SVMResult
        Training result with fitted model and metrics.
    """
    result = SVMResult(
        model_name="svm",
        model_version="1.0.0",
        selected_features=list(selected_features),
    )

    X = X_train[selected_features].copy()
    y = y_train.copy()

    pipeline = build_svm_pipeline(config, selected_features)
    pipeline.fit(X, y)

    result.model = pipeline
    result.preprocessing_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy=config.impute_strategy,
                    add_indicator=config.impute_add_indicator,
                ),
            ),
            (
                "scaler",
                StandardScaler(
                    with_mean=config.scale_with_mean,
                    with_std=config.scale_with_std,
                ),
            ),
        ]
    )
    result.preprocessing_pipeline.fit(X, y)

    # Extract support vector count.
    clf = pipeline.named_steps["classifier"]
    try:
        result.support_vectors = int(clf.support_.shape[0])
    except (AttributeError, ValueError):
        result.support_vectors = None

    # Training metrics.
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]
    result.training_metrics = _compute_metrics(y, y_pred, y_prob)

    logger.info(
        "SVM trained: n_features=%d, kernel=%s, C=%s, support_vectors=%s",
        len(selected_features),
        config.svm_kernel,
        config.svm_c,
        result.support_vectors,
    )
    return result


def predict_svm(
    result: SVMResult,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate predictions with a trained SVM model.

    Parameters
    ----------
    result : SVMResult
        Trained result.
    X : pd.DataFrame
        Features (validation or test).
    threshold : float
        Decision threshold for predicted class.

    Returns
    -------
    pd.DataFrame
        DataFrame with predicted_probability and predicted_class.
    """
    if result.model is None:
        raise ValueError("Model not trained. Call train_svm() first.")

    X_sel = X[result.selected_features].copy()
    y_prob = result.model.predict_proba(X_sel)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    out = pd.DataFrame(
        {
            "predicted_probability": y_prob,
            "predicted_class": y_pred,
        }
    )
    return out


def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute standard binary classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = float("nan")

    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auprc"] = float("nan")

    try:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    except ValueError:
        metrics["brier"] = float("nan")

    metrics["tp"] = int(tp)
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)

    return metrics


def save_svm_artifact(
    result: SVMResult,
    artifact_dir: str | Path,
) -> Path:
    """Save SVM artifact and metadata."""
    from datetime import datetime

    from .model_registry import ModelMetadata

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    metadata = ModelMetadata(
        model_name="svm",
        model_version="1.0.0",
        training_timestamp=datetime.now(UTC).isoformat(),
        random_seed=0,
        feature_names=[],
        selected_feature_names=list(result.selected_features),
        n_features_input=0,
        n_features_output=len(result.selected_features),
        n_patients_train=0,
        n_rows_train=0,
        n_positive_train=0,
        n_negative_train=0,
        class_weight=None,
        hyperparameters={},
        preprocessing_config={},
        validation_metrics=dict(result.validation_metrics),
        split_identity="patient_level",
        source="synthetic",
    )

    save_model_artifact(
        model=result.model,
        preprocessing_pipeline=result.preprocessing_pipeline,
        metadata=metadata,
        artifact_dir=artifact_dir,
    )
    save_feature_list(
        result.selected_features,
        artifact_dir,
        "svm",
        "1.0.0",
    )
    return artifact_path / "svm_1.0.0.joblib"
