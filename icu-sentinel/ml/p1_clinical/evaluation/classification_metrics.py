"""Classification metrics for P1 research evaluation.

Implements reusable binary classification metrics:
- AUROC, AUPRC
- Sensitivity, specificity, precision, F1-score, accuracy
- Brier score
- Confusion matrix (TP, TN, FP, FN)
- Positive and negative prediction counts
- Positive-label prevalence

All metrics handle single-class subsets safely (returning NaN instead of
raising errors) and never silently replace undefined metrics with zero.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from .evaluation_schema import ClassificationMetricEntry, ClassificationMetricsResult

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.classification_metrics")


def compute_binary_metrics(
    y_true: np.ndarray | list[int] | pd.Series,
    y_pred: np.ndarray | list[int] | pd.Series,
    y_prob: np.ndarray | list[float] | pd.Series | None,
    n_patients: int = 0,
    n_rows: int | None = None,
) -> dict:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_pred : array-like
        Predicted binary labels (0 or 1).
    y_prob : array-like, optional
        Predicted probabilities for the positive class. If None,
        threshold-dependent metrics that require probabilities (AUROC,
        AUPRC, Brier) will be NaN.
    n_patients : int
        Number of unique patients.
    n_rows : int, optional
        Number of hourly rows.

    Returns
    -------
    dict
        Dictionary with AUROC, AUPRC, sensitivity, specificity,
        precision, F1, Brier score, accuracy, and confusion-matrix
        counts. Undefined metrics return NaN (never silently replaced
        with zero).
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(float)
    else:
        y_prob = None

    if n_rows is None:
        n_rows = len(y_true)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = sensitivity  # alias
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else float("nan")
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float("nan")

    # Prevalence
    prevalence = n_pos / len(y_true) if len(y_true) > 0 else float("nan")

    # AUROC, AUPRC, Brier require probabilities
    try:
        auroc = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(y_true, y_prob) if y_prob is not None else float("nan")
    except ValueError:
        auprc = float("nan")

    try:
        brier = brier_score_loss(y_true, y_prob) if y_prob is not None else float("nan")
    except ValueError:
        brier = float("nan")

    metrics = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "brier": float(brier),
        "prevalence": float(prevalence),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n_positive_pred": int((y_pred == 1).sum()),
        "n_negative_pred": int((y_pred == 0).sum()),
        "n_positive_true": n_pos,
        "n_negative_true": n_neg,
    }

    return metrics


def metrics_result_from_dict(
    model_name: str,
    model_version: str,
    split: str,
    metrics: dict,
    source: str = "synthetic",
) -> ClassificationMetricsResult:
    """Build a ClassificationMetricsResult from a metrics dict.

    Parameters
    ----------
    model_name : str
        Name of the model.
    model_version : str
        Version of the model.
    split : str
        Split identity (train/validation/test).
    metrics : dict
        Dictionary of metric values (as returned by compute_binary_metrics).
    source : str
        Data source label.

    Returns
    -------
    ClassificationMetricsResult
        Structured metrics result.
    """
    metric_entries = []
    for name, value in metrics.items():
        if name in ("tp", "tn", "fp", "fn", "n_positive_pred", "n_negative_pred",
                    "n_positive_true", "n_negative_true"):
            continue  # Skip confusion matrix counts, stored separately
        metric_entries.append(
            ClassificationMetricEntry(
                metric_name=name,
                value=float(value) if value is not None else float("nan"),
            )
        )

    confusion = {
        "tp": int(metrics.get("tp", 0)),
        "tn": int(metrics.get("tn", 0)),
        "fp": int(metrics.get("fp", 0)),
        "fn": int(metrics.get("fn", 0)),
    }

    return ClassificationMetricsResult(
        model_name=model_name,
        model_version=model_version,
        split=split,
        metrics=metric_entries,
        confusion_matrix=confusion,
        source=source,
    )


def compute_all_metrics(
    y_true: np.ndarray | list[int] | pd.Series,
    y_pred: np.ndarray | list[int] | pd.Series,
    y_prob: np.ndarray | list[float] | pd.Series | None,
    model_name: str,
    model_version: str,
    split: str,
    source: str = "synthetic",
) -> ClassificationMetricsResult:
    """Compute all classification metrics and return a structured result.

    Convenience function that combines compute_binary_metrics with
    metrics_result_from_dict.
    """
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    return metrics_result_from_dict(
        model_name=model_name,
        model_version=model_version,
        split=split,
        metrics=metrics,
        source=source,
    )


def validate_metrics_result(
    result: ClassificationMetricsResult,
) -> dict:
    """Validate a ClassificationMetricsResult.

    Checks that all metric values are finite numbers (or NaN for
    undefined metrics) and that confusion matrix counts are non-negative.

    Returns a dict with validation results.
    """
    issues: list[str] = []

    for entry in result.metrics:
        if entry.value is not None and not np.isfinite(entry.value) and not np.isnan(entry.value):
            issues.append(
                f"Metric {entry.metric_name}: value {entry.value} is not finite"
            )

    cm = result.confusion_matrix
    for key in ("tp", "tn", "fp", "fn"):
        if cm.get(key, 0) < 0:
            issues.append(f"Confusion matrix {key}: {cm[key]} is negative")
        total = cm.get("tp", 0) + cm.get("tn", 0) + cm.get("fp", 0) + cm.get("fn", 0)
        if total == 0:
            issues.append("Confusion matrix has zero total count")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_metrics": len(result.metrics),
    }