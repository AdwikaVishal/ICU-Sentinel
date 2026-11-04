"""Inference interface for P1 classical ML models.

Provides a unified interface for loading trained models and generating
predictions on validation or test data. Ensures test-set isolation by
never refitting preprocessing on evaluation data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model_serialization import load_feature_list, load_model_artifact
from .prediction_schema import PredictionBatch, PredictionRecord

logger = logging.getLogger("icu_sentinel.p1_clinical.classical_ml.inference")


@dataclass
class InferenceResult:
    """Result of running inference on a split."""

    model_name: str
    model_version: str
    split: str
    n_predictions: int
    metrics: dict
    predictions: PredictionBatch
    threshold: float
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "split": self.split,
            "n_predictions": self.n_predictions,
            "metrics": dict(self.metrics),
            "threshold": self.threshold,
            "source": self.source,
        }


class ModelInference:
    """Unified inference interface for classical ML models."""

    def __init__(
        self,
        artifact_dir: str | Path,
        source: str = "synthetic",
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.source = source
        self.logger = logging.getLogger(
            "icu_sentinel.p1_clinical.classical_ml.inference.ModelInference"
        )
        self._loaded_models: dict[str, dict] = {}

    def load_model(
        self,
        model_name: str,
        model_version: str = "1.0.0",
    ) -> dict:
        """Load a trained model artifact.

        Parameters
        ----------
        model_name : str
            Model name (logistic_regression, random_forest, svm).
        model_version : str
            Model version string.

        Returns
        -------
        dict
            Loaded artifact with model, preprocessing_pipeline, metadata.
        """
        key = f"{model_name}_{model_version}"
        if key in self._loaded_models:
            return self._loaded_models[key]

        artifact_file = self.artifact_dir / model_name / f"{model_name}_{model_version}.joblib"
        if not artifact_file.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {artifact_file}"
            )

        artifact = load_model_artifact(artifact_file)
        self._loaded_models[key] = artifact
        self.logger.info("Loaded model %s version %s", model_name, model_version)
        return artifact

    def load_features(
        self,
        model_name: str,
        model_version: str = "1.0.0",
    ) -> list[str]:
        """Load the selected feature list for a model."""
        feature_file = (
            self.artifact_dir
            / model_name
            / f"{model_name}_{model_version}_features.json"
        )

        return load_feature_list(feature_file)

    def predict(
        self,
        model_name: str,
        X: pd.DataFrame,
        model_version: str = "1.0.0",
        threshold: float = 0.5,
        split: str = "test",
    ) -> InferenceResult:
        """Generate predictions for a model on evaluation data.

        Parameters
        ----------
        model_name : str
            Model name.
        X : pd.DataFrame
            Evaluation features (validation or test). Preprocessing
            is applied using the training-fitted pipeline - no
            refitting occurs.
        model_version : str
            Model version.
        threshold : float
            Decision threshold.
        split : str
            Split identity (validation or test).

        Returns
        -------
        InferenceResult
            Inference result with metrics and predictions.
        """
        artifact = self.load_model(model_name, model_version)
        model = artifact["model"]
        artifact["metadata"]

        selected_features = self.load_features(model_name, model_version)
        X_sel = X[selected_features].copy()

        # Generate predictions using the training-fitted model.
        # No refitting - preprocessing parameters come from training only.
        y_prob = model.predict_proba(X_sel)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        # Compute metrics.
        from .logistic_regression import _compute_metrics

        y_true = X.get("SepsisLabel")
        metrics = {}
        if y_true is not None:
            metrics = _compute_metrics(
                y_true, y_pred, y_prob
            )

        # Build prediction records.
        records = []
        n = len(y_prob)
        patient_ids = X.get("patient_id", pd.Series([f"unknown_{i}" for i in range(n)]))
        hours = X.get("hour", pd.Series(range(n)))

        for i in range(n):
            records.append(
                PredictionRecord(
                    patient_id=str(patient_ids.iloc[i]),
                    hour=int(hours.iloc[i]),
                    true_label=int(y_true.iloc[i]) if y_true is not None else -1,
                    predicted_probability=float(y_prob[i]),
                    predicted_class=int(y_pred[i]),
                    model_name=model_name,
                    model_version=model_version,
                    split=split,
                    threshold=threshold,
                )
            )

        batch = PredictionBatch(
            model_name=model_name,
            model_version=model_version,
            split=split,
            n_predictions=n,
            n_positive_true=int(sum(r.true_label for r in records if r.true_label >= 0)),
            n_positive_pred=int(sum(r.predicted_class for r in records)),
            predictions=records,
        )

        result = InferenceResult(
            model_name=model_name,
            model_version=model_version,
            split=split,
            n_predictions=n,
            metrics=metrics,
            predictions=batch,
            threshold=threshold,
            source=self.source,
        )

        self.logger.info(
            "Inference complete: %s on %s split, %d predictions, AUROC=%.4f",
            model_name,
            split,
            n,
            metrics.get("auroc", float("nan")),
        )
        return result

    def predict_all(
        self,
        model_names: list[str],
        X: pd.DataFrame,
        model_version: str = "1.0.0",
        threshold: float = 0.5,
        split: str = "test",
    ) -> dict[str, InferenceResult]:
        """Run inference for multiple models on the same data."""
        results = {}
        for name in model_names:
            results[name] = self.predict(
                model_name=name,
                X=X,
                model_version=model_version,
                threshold=threshold,
                split=split,
            )
        return results

    def select_threshold(
        self,
        model_name: str,
        X_val: pd.DataFrame,
        metric: str = "f1",
        grid_step: float = 0.01,
        model_version: str = "1.0.0",
    ) -> float:
        """Select optimal threshold on validation data only.

        Parameters
        ----------
        model_name : str
            Model name.
        X_val : pd.DataFrame
            Validation features with SepsisLabel column.
        metric : str
            Metric to optimize (f1, precision, recall, balanced_accuracy).
        grid_step : float
            Grid search step size.
        model_version : str
            Model version.

        Returns
        -------
        float
            Selected threshold.

        Note
        ----
        Threshold selection uses validation data only. Test data is
        never used for threshold selection.
        """
        if "SepsisLabel" not in X_val.columns:
            raise ValueError(
                "X_val must contain SepsisLabel column for threshold selection."
            )

        artifact = self.load_model(model_name, model_version)
        model = artifact["model"]
        selected_features = self.load_features(model_name, model_version)
        X_sel = X_val[selected_features].copy()
        y_true = X_val["SepsisLabel"].copy()

        y_prob = model.predict_proba(X_sel)[:, 1]

        best_threshold = 0.5
        best_score = -1.0

        from sklearn.metrics import (
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        thresholds = np.arange(0.0, 1.0 + grid_step, grid_step)
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == "balanced_accuracy":
                score = balanced_accuracy_score(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = float(t)

        self.logger.info(
            "Threshold selection: metric=%s, best_threshold=%.3f, best_score=%.4f",
            metric,
            best_threshold,
            best_score,
        )
        return best_threshold
