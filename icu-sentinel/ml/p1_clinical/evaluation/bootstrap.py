"""Bootstrap confidence intervals for P1 research evaluation.

Implements patient-level bootstrap confidence intervals for classification metrics.
Resamples patients (not individual hourly rows) to keep all observations
from a sampled patient together.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .evaluation_schema import BootstrapResult, BootstrapSummary

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.bootstrap")

SUPPORTED_METRICS = (
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "accuracy",
    "brier",
)


def compute_bootstrap_confidence_intervals(
    predictions_df: pd.DataFrame,
    model_name: str,
    model_version: str,
    split: str,
    metric_names: list[str],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    source: str = "synthetic",
) -> BootstrapSummary:
    """Compute bootstrap confidence intervals for classification metrics.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with patient_id, true_label, predicted_probability and
        predicted_class columns.
    model_name : str
        Name of the model.
    model_version : str
        Model version.
    split : str
        Split identity (train/validation/test).
    metric_names : list[str]
        List of metric names to compute bootstrap CIs for.
        Supported: "auroc", "auprc", "sensitivity", "specificity",
        "precision", "f1", "accuracy", "brier".
    n_iterations : int
        Number of bootstrap iterations.
    confidence_level : float
        Confidence level for intervals (e.g., 0.95 for 95% CI).
    random_seed : int
        Random seed for reproducibility.
    source : str
        Data source label.

    Returns
    -------
    BootstrapSummary
        Bootstrap confidence interval summary.
    """
    required_cols = [
        "patient_id",
        "true_label",
        "predicted_probability",
        "predicted_class",
    ]
    missing = [c for c in required_cols if c not in predictions_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if n_iterations <= 0:
        raise ValueError(f"n_iterations must be positive, got {n_iterations}.")
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}."
        )
    if not metric_names:
        raise ValueError("metric_names must contain at least one metric.")

    unknown_metrics = [m for m in metric_names if m not in SUPPORTED_METRICS]
    if unknown_metrics:
        raise ValueError(
            f"Unsupported bootstrap metrics: {unknown_metrics}. "
            f"Supported metrics: {list(SUPPORTED_METRICS)}."
        )

    rng = np.random.default_rng(random_seed)

    # Get unique patient IDs
    patient_ids = predictions_df["patient_id"].unique()
    n_patients = len(patient_ids)

    if n_patients == 0:
        return BootstrapSummary(
            model_name=model_name,
            model_version=model_version,
            split=split,
            metrics=[],
            source=source,
        )

    # Map metric names to functions with a uniform (y_true, y_pred, y_prob)
    # signature. Each function ignores the inputs it does not need.
    metric_funcs = {
        "auroc": _safe_roc_auc,
        "auprc": _safe_auprc,
        "sensitivity": _safe_sensitivity,
        "specificity": _safe_specificity,
        "precision": _safe_precision,
        "f1": _safe_f1,
        "accuracy": _safe_accuracy,
        "brier": _safe_brier,
    }

    # Precompute patient row positions once so each bootstrap iteration only
    # gathers integer indices. Values and row order match repeated boolean
    # filtering of the original frame, including first-appearance patient
    # order within each resample.
    row_positions_by_patient = predictions_df.groupby(
        "patient_id", sort=False
    ).indices
    true_labels = predictions_df["true_label"].astype(int).to_numpy()
    predicted_classes = predictions_df["predicted_class"].astype(int).to_numpy()
    probabilities = predictions_df["predicted_probability"].astype(float).to_numpy()

    # Collect bootstrap samples for each metric
    bootstrap_samples: dict[str, list[float]] = {m: [] for m in metric_names}

    for iteration in range(n_iterations):
        # Resample patients with replacement. This call is intentionally kept
        # identical to the original implementation to preserve seeded draws.
        sampled_patient_ids = rng.choice(patient_ids, size=n_patients, replace=True)

        sampled_positions = []
        total_rows = 0
        for pid in sampled_patient_ids:
            positions = row_positions_by_patient.get(pid)
            if positions is not None and len(positions) > 0:
                sampled_positions.append(positions)
                total_rows += len(positions)

        row_indices = np.empty(total_rows, dtype=np.intp)
        cursor = 0
        for positions in sampled_positions:
            row_indices[cursor:cursor + len(positions)] = positions
            cursor += len(positions)

        # Compute metrics for each metric
        y_true = true_labels[row_indices]
        y_pred = predicted_classes[row_indices]
        y_prob = probabilities[row_indices]

        for metric_name in metric_names:
            metric_func = metric_funcs[metric_name]
            # Undefined metrics (e.g. AUROC on a single-class resample) return
            # NaN by design. Unexpected programming errors are not caught here
            # and propagate so they cannot be silently hidden.
            value = metric_func(y_true, y_pred, y_prob)
            if np.isfinite(value):
                bootstrap_samples[metric_name].append(value)

    # Compute confidence intervals
    bootstrap_results: list[BootstrapResult] = []
    alpha = 1.0 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    for metric_name in metric_names:
        if not bootstrap_samples[metric_name]:
            logger.warning(
                "Bootstrap metric '%s' was undefined in all %d resamples "
                "(e.g. single-class resamples) and is omitted from the summary.",
                metric_name,
                n_iterations,
            )
            continue

        samples = np.array(bootstrap_samples[metric_name])
        point_estimate = float(np.mean(samples))
        ci_lower = float(np.percentile(samples, lower_percentile))
        ci_upper = float(np.percentile(samples, upper_percentile))

        bootstrap_results.append(
            BootstrapResult(
                metric_name=metric_name,
                point_estimate=point_estimate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                confidence_level=confidence_level,
                n_iterations=n_iterations,
                n_patients=n_patients,
                source=source,
            )
        )

    return BootstrapSummary(
        model_name=model_name,
        model_version=model_version,
        split=split,
        metrics=bootstrap_results,
        source=source,
    )


def _safe_roc_auc(y_true, y_pred, y_prob):
    """Safe AUROC computation returning NaN on failure."""
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _safe_auprc(y_true, y_pred, y_prob):
    """Safe AUPRC computation returning NaN when undefined.

    Average precision is undefined without any positive labels, so this is
    treated as explicitly undefined rather than silently returning 0.0.
    """
    from sklearn.metrics import average_precision_score

    if not np.any(y_true == 1):
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _safe_sensitivity(y_true, y_pred, y_prob):
    """Safe sensitivity computation returning NaN on failure."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")


def _safe_specificity(y_true, y_pred, y_prob):
    """Safe specificity computation returning NaN on failure."""
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def _safe_precision(y_true, y_pred, y_prob):
    """Safe precision computation returning NaN on failure."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")


def _safe_f1(y_true, y_pred, y_prob):
    """Safe F1 computation returning NaN on failure."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * precision * sensitivity / (precision + sensitivity)
                 if (precision + sensitivity) > 0 else 0.0)


def _safe_accuracy(y_true, y_pred, y_prob):
    """Safe accuracy computation returning NaN on failure."""
    return float(np.mean(y_true == y_pred))


def _safe_brier(y_true, y_pred, y_prob):
    """Safe Brier score computation returning NaN on failure."""
    from sklearn.metrics import brier_score_loss
    try:
        return float(brier_score_loss(y_true, y_prob))
    except ValueError:
        return float("nan")


def validate_bootstrap_result(
    result: BootstrapSummary,
) -> dict:
    """Validate a BootstrapSummary.

    Returns a dict with validation results.
    """
    issues: list[str] = []

    for result_entry in result.metrics:
        if result_entry.n_iterations <= 0:
            issues.append(
                f"Metric {result_entry.metric_name}: "
                f"n_iterations is non-positive"
            )
        if not (0 < result_entry.confidence_level < 1):
            issues.append(
                f"Metric {result_entry.metric_name}: "
                f"confidence_level is out of (0,1) range"
            )
        if result_entry.ci_lower is not None and result_entry.ci_upper is not None \
                and result_entry.ci_lower > result_entry.ci_upper:
            issues.append(
                f"Metric {result_entry.metric_name}: "
                f"ci_lower > ci_upper"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_metrics": len(result.metrics),
    }