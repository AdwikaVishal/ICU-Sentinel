"""Shared result contract for P2 tabular models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .feature_contract import (
    sanitize_infinities,
    validate_feature_frame,
)


@dataclass
class P2ModelResult:
    """A fitted P2 model and its provenance.

    ``training_metrics``, ``validation_metrics``, and ``test_metrics`` are
    intentionally separate. Only validation metrics may influence model or
    threshold selection.
    """

    model_name: str
    model_version: str = "1.0.0"
    model: Any = None
    preprocessing_pipeline: Any = None
    selected_features: list[str] = field(default_factory=list)
    hyperparameters: dict = field(default_factory=dict)
    training_metrics: dict = field(default_factory=dict)
    validation_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    threshold: float | None = None
    random_seed: int = 42
    training_timestamp: str = ""
    n_patients_train: int = 0
    n_rows_train: int = 0
    n_positive_train: int = 0
    n_negative_train: int = 0
    class_weight: str | None = None
    calibration_method: str | None = None
    calibrator: Any = None
    source: str = "synthetic"


def predict_probabilities(
    result: P2ModelResult,
    X: pd.DataFrame,
    threshold: float = 0.5,
    *,
    already_sanitized: bool = False,
) -> pd.DataFrame:
    """Return positive-class probabilities and thresholded classes.

    When ``already_sanitized=True`` the caller guarantees ``X`` has already
    been validated and infinities replaced with NaN exactly once; the
    sanitisation pass is skipped to avoid a repeated full-frame copy. Default
    ``False`` preserves the previous behaviour (safe copy).
    """
    if result.model is None:
        raise ValueError(
            f"Model '{result.model_name}' is not fitted; train it first."
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    # copy=False: we do not mutate the caller's frame.
    selected = validate_feature_frame(X, result.selected_features, copy=False)
    if not already_sanitized:
        # Cannot use inplace=True here: sklearn pipeline validation rejects
        # an in-place-modified frame. One copy only.
        selected = sanitize_infinities(selected, inplace=False)
    
    probabilities = np.asarray(
        result.model.predict_proba(selected), dtype=float
    )
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "predict_proba must return two columns for binary classification."
        )
    positive = probabilities[:, 1]
    if not np.all(np.isfinite(positive)):
        raise ValueError("Model returned non-finite probabilities.")

    return pd.DataFrame(
        {
            "predicted_probability": positive,
            "predicted_class": (positive >= threshold).astype(int),
        },
        index=selected.index,
    )
