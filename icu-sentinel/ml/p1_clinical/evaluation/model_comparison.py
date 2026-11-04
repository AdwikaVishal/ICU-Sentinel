"""Model comparison for P1 research evaluation.

Provides standardized comparison of classical ML models, clinical
baselines, and future P2/P3 models on the same evaluation population.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .classification_metrics import compute_binary_metrics
from .evaluation_schema import ModelComparisonEntry, ModelComparisonResult

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.model_comparison")


def compute_model_comparison(
    predictions_by_model: dict[str, pd.DataFrame],
    split: str,
    threshold: float = 0.5,
    source: str = "synthetic",
) -> ModelComparisonResult:
    """Compute model comparison results for all models on the same split.

    Parameters
    ----------
    predictions_by_model : dict[str, pd.DataFrame]
        Dictionary mapping model name to predictions DataFrame.
        Each DataFrame must have columns: patient_id, hour,
        true_label, predicted_probability, predicted_class.
    split : str
        Split identity (train/validation/test).
    threshold : float
        Decision threshold for positive prediction.
    source : str
        Data source label.

    Returns
    -------
    ModelComparisonResult
        Model comparison result.
    """
    entries: list[ModelComparisonEntry] = []

    for model_name, df in predictions_by_model.items():
        required_cols = ["patient_id", "hour", "true_label",
                         "predicted_probability", "predicted_class"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(
                "Model %s missing columns %s; skipping", model_name, missing
            )
            continue

        y_true = df["true_label"].astype(int).values
        y_pred = df["predicted_class"].astype(int).values
        y_prob = df["predicted_probability"].astype(float).values

        metrics = compute_binary_metrics(y_true, y_pred, y_prob)

        # Determine model type from name
        model_type = _classify_model_type(model_name)

        n_patients = int(df["patient_id"].nunique())
        n_rows = len(df)

        entries.append(
            ModelComparisonEntry(
                model_name=model_name,
                model_version="1.0.0",
                model_type=model_type,
                split=split,
                metrics={k: float(v) for k, v in metrics.items()
                         if k not in ("tp", "tn", "fp", "fn",
                                      "n_positive_pred", "n_negative_pred",
                                      "n_positive_true", "n_negative_true")},
                n_patients=n_patients,
                n_rows=n_rows,
                source=source,
            )
        )

    # Sort by AUROC descending for consistent ordering
    entries.sort(
        key=lambda e: e.metrics.get("auroc", float("-inf")),
        reverse=True,
    )

    return ModelComparisonResult(
        split=split,
        models=entries,
        source=source,
    )


def _classify_model_type(model_name: str) -> str:
    """Classify model type based on model name.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    str
        Model type: "classical_ml", "clinical_baseline", "boosting",
        or "temporal".
    """
    name_lower = model_name.lower()

    if name_lower in ("logistic_regression", "random_forest", "svm",
                      "extra_trees", "xgboost", "lightgbm", "catboost"):
        return "classical_ml"
    elif name_lower in ("sirs", "qsofa", "news2_style"):
        return "clinical_baseline"
    elif name_lower in ("gru", "lstm", "transformer"):
        return "temporal"
    else:
        return "unknown"


def compute_pairwise_comparison(
    result_a: ModelComparisonEntry,
    result_b: ModelComparisonEntry,
    metric: str = "auroc",
) -> dict:
    """Compute pairwise comparison between two models.

    Parameters
    ----------
    result_a : ModelComparisonEntry
        First model's comparison entry.
    result_b : ModelComparisonEntry
        Second model's comparison entry.
    metric : str
        Metric to compare.

    Returns
    -------
    dict
        Pairwise comparison result with difference and relative change.
    """
    val_a = result_a.metrics.get(metric, float("nan"))
    val_b = result_b.metrics.get(metric, float("nan"))

    diff = val_b - val_a if np.isfinite(val_a) and np.isfinite(val_b) else float("nan")
    rel_change = (
        diff / abs(val_a) if np.isfinite(val_a) and abs(val_a) > 1e-10
        else float("nan")
    )

    return {
        "model_a": result_a.model_name,
        "model_b": result_b.model_name,
        "metric": metric,
        "value_a": float(val_a),
        "value_b": float(val_b),
        "difference": float(diff),
        "relative_change": float(rel_change),
        "model_b_better": bool(diff > 0),
    }


def validate_model_comparison_result(
    result: ModelComparisonResult,
) -> dict:
    """Validate a ModelComparisonResult.

    Returns a dict with validation results.
    """
    issues: list[str] = []

    if not result.models:
        issues.append("No models in comparison result")

    for entry in result.models:
        if entry.n_patients <= 0:
            issues.append(
                f"Model {entry.model_name}: n_patients is non-positive"
            )
        if entry.n_rows <= 0:
            issues.append(
                f"Model {entry.model_name}: n_rows is non-positive"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_models": len(result.models),
    }