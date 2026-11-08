"""Random Forest model for P1 classical ML.

Implements a reproducible Random Forest pipeline with:
- Missing-value imputation fitted on training data only.
- Configurable trees, depth, samples per split/leaf.
- Configurable class weighting.
- Probability predictions.
- Feature importance extraction.
- Model serialization.

Avoids unnecessarily expensive configurations for local execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .model_config import ModelConfig
from .model_serialization import save_feature_list, save_model_artifact

logger = logging.getLogger("icu_sentinel.p1_clinical.classical_ml.random_forest")


@dataclass
class RandomForestResult:
    """Result of training/evaluating Random Forest."""

    model_name: str = "random_forest"
    model_version: str = "1.0.0"
    model: Pipeline | None = None
    preprocessing_pipeline: Pipeline | None = None
    selected_features: list[str] = field(default_factory=list)
    feature_importance: pd.DataFrame | None = None
    training_metrics: dict = field(default_factory=dict)
    validation_metrics: dict = field(default_factory=dict)
    n_trees: int = 0
    max_depth: int | None = None


def build_rf_pipeline(
    config: ModelConfig,
    selected_features: list[str],
) -> Pipeline:
    """Build a Random Forest pipeline.

    Pipeline steps:
      1. SimpleImputer (median, fitted on train only)
      2. RandomForestClassifier

    Note: StandardScaler is NOT included for Random Forest as tree-based
    models are insensitive to feature scaling.
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
                "classifier",
                RandomForestClassifier(
                    n_estimators=config.rf_n_estimators,
                    max_depth=config.rf_max_depth,
                    min_samples_split=config.rf_min_samples_split,
                    min_samples_leaf=config.rf_min_samples_leaf,
                    class_weight=config.rf_class_weight,
                    n_jobs=config.rf_n_jobs,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )
    return pipeline


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelConfig,
    selected_features: list[str],
) -> RandomForestResult:
    """Train Random Forest on training data.

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
    RandomForestResult
        Training result with fitted model and metrics.
    """
    result = RandomForestResult(
        model_name="random_forest",
        model_version="1.0.0",
        selected_features=list(selected_features),
        n_trees=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
    )

    if list(X_train.columns) == list(selected_features):
        X = X_train
    else:
        X = X_train[selected_features]
    y = y_train.copy()

    pipeline = build_rf_pipeline(config, selected_features)
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
        ]
    )
    result.preprocessing_pipeline.fit(X, y)

    # Extract feature importance.
    clf = pipeline.named_steps["classifier"]
    importance = clf.feature_importances_
    imp_df = pd.DataFrame(
        {
            "feature": selected_features,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)
    result.feature_importance = imp_df

    # Training metrics.
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]
    result.training_metrics = _compute_metrics(y, y_pred, y_prob)

    logger.info(
        "Random Forest trained: n_features=%d, n_trees=%d, max_depth=%s",
        len(selected_features),
        config.rf_n_estimators,
        config.rf_max_depth,
    )
    return result


def predict_rf(
    result: RandomForestResult,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate predictions with a trained Random Forest model.

    Parameters
    ----------
    result : RandomForestResult
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
        raise ValueError("Model not trained. Call train_rf() first.")

    X_sel = X if list(X.columns) == list(result.selected_features) else X[result.selected_features]
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


def save_rf_artifact(
    result: RandomForestResult,
    artifact_dir: str | Path,
) -> Path:
    """Save Random Forest artifact and metadata."""
    from datetime import datetime

    from .model_registry import ModelMetadata

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    metadata = ModelMetadata(
        model_name="random_forest",
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
        "random_forest",
        "1.0.0",
    )
    return artifact_path / "random_forest_1.0.0.joblib"
