"""Alert burden evaluation for P1 research evaluation.

Computes alert burden metrics including false alerts per patient-day,
alerts per patient, and alert episode statistics.

Delegates to alert_episodes.compute_alert_burden for the core computation.
"""

from __future__ import annotations

import logging

import pandas as pd

from .alert_episodes import (
    AlertBurdenResult,
)
from .alert_episodes import (
    compute_alert_burden as _compute_alert_burden,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.alert_burden")


def compute_alert_burden(
    predictions_df: pd.DataFrame,
    model_name: str,
    model_version: str,
    split: str,
    threshold: float = 0.5,
    cooldown_hours: int = 6,
    min_duration_hours: int = 1,
    min_consecutive: int = 1,
    source: str = "synthetic",
) -> AlertBurdenResult:
    """Compute alert burden metrics for all patients.

    This is a wrapper around alert_episodes.compute_alert_burden that
    provides the alert burden computation interface for the evaluation
    pipeline.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with columns: patient_id, source_dataset (optional),
        hour, true_label, predicted_probability.
    model_name : str
        Name of the model.
    model_version : str
        Version of the model.
    split : str
        Split identity.
    threshold : float
        Decision threshold for positive prediction.
    cooldown_hours : int
        Minimum hours between alert episodes.
    min_duration_hours : int
        Minimum duration (in hours) for an alert episode.
    min_consecutive : int
        Minimum number of consecutive positive predictions required.
    source : str
        Data source label.

    Returns
    -------
    AlertBurdenResult
        Alert burden result.
    """
    return _compute_alert_burden(
        predictions_df=predictions_df,
        model_name=model_name,
        model_version=model_version,
        split=split,
        threshold=threshold,
        cooldown_hours=cooldown_hours,
        min_duration_hours=min_duration_hours,
        min_consecutive=min_consecutive,
        source=source,
    )


def validate_alert_burden_result(
    result: AlertBurdenResult,
) -> dict:
    """Validate an AlertBurdenResult.

    Returns a dict with validation results.
    """
    from .alert_episodes import validate_alert_burden_result as _validate

    return _validate(result)