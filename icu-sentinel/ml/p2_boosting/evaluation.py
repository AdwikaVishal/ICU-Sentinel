"""Unified evaluation for P2 models.

Reuses the shared metric implementation in ``ml.common.metrics`` and P1's
validation-only threshold selector. Training, validation, and test results
are always returned under distinct keys by the training orchestrator.
"""

from __future__ import annotations

import numpy as np

from ml.common.metrics import compute_binary_metrics
from ml.p1_clinical.classical_ml.training import (
    select_threshold_on_validation as _p1_select_threshold,
)

from .feature_contract import validate_prediction_target


def evaluate_predictions(
    y_true,
    y_prob,
    *,
    threshold: float = 0.5,
    model_name: str | None = None,
    split: str | None = None,
) -> dict:
    """Evaluate probabilities with the shared binary metric functions."""
    y = validate_prediction_target(
        y_true, "evaluation target", require_both=False
    )
    probabilities = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(probabilities) != len(y):
        raise ValueError("y_true and y_prob must have the same length.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    predicted = (probabilities >= threshold).astype(int)
    metrics = compute_binary_metrics(y, predicted, probabilities)

    return {
        "model_name": model_name,
        "split": split,
        "threshold": float(threshold),
        "n_rows": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "metrics": metrics,
        "confusion_matrix": {
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        },
    }


def select_validation_threshold(
    y_true,
    y_prob,
    *,
    metric: str = "f1",
    grid_step: float = 0.01,
    partition: str = "validation",
) -> float:
    """Select a threshold on validation only.

    ``partition`` is explicit so fitting/selection on the test split is
    rejected by construction.
    """
    if partition == "test":
        raise ValueError("Threshold selection on the test split is forbidden.")
    if partition not in {"validation", "calibration"}:
        raise ValueError(
            "partition must be 'validation' or 'calibration' for threshold "
            f"selection; got '{partition}'."
        )
    y = validate_prediction_target(y_true, "validation target")
    probabilities = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(probabilities) != len(y):
        raise ValueError("y_true and y_prob must have the same length.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")
    return _p1_select_threshold(
        y,
        probabilities,
        metric=metric,
        grid_step=grid_step,
    )
