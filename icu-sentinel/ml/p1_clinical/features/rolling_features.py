"""Rolling temporal features.

Computes causal trailing rolling statistics over 6-, 12-, and 24-hour
windows for relevant clinical variables:

- mean, min, max, std, latest available value

Windows are causal and include the current hour. Insufficient
observations yield NaN (never fabricated).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..preprocessing.schemas import ProcessedPatient
from .feature_config import (
    FEATURE_VARIABLES,
    ROLLING_AGGREGATIONS,
    ROLLING_MIN_OBSERVATIONS,
    ROLLING_WINDOW_HOURS,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.features.rolling_features")


def _rolling_stat(
    values: np.ndarray,
    hours: np.ndarray,
    idx: int,
    window_hours: int,
    agg: str,
    min_obs: int,
) -> float:
    """Compute a causal rolling statistic for row idx.

    Includes all rows j <= idx where hours[j] > hours[idx] - window_hours
    (i.e., the trailing window ending at and including the current hour).
    """
    t = hours[idx]
    window_start = t - window_hours

    valid_vals = []
    for j in range(idx, -1, -1):
        hj = hours[j]
        if hj <= window_start:
            break
        v = values[j]
        if pd.notna(v):
            valid_vals.append(float(v))

    if len(valid_vals) < min_obs:
        return float("nan")

    arr = np.array(valid_vals, dtype=float)
    if agg == "mean":
        return float(np.mean(arr))
    if agg == "min":
        return float(np.min(arr))
    if agg == "max":
        return float(np.max(arr))
    if agg == "std":
        if len(arr) < 2:
            return float("nan")
        return float(np.std(arr, ddof=1))
    if agg == "latest":
        return float(arr[0])  # most recent (j starts at idx and goes backward)
    return float("nan")


def add_rolling_features(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    window_hours: list[int] | None = None,
    aggregations: list[str] | None = None,
    min_observations: int = ROLLING_MIN_OBSERVATIONS,
    hour_column: str = "hour",
) -> pd.DataFrame:
    """Add causal rolling feature columns to a patient-hour DataFrame.

    For each variable, window, and aggregation, creates a column:
        {var}_rolling_{agg}_{window}h
    """
    vars_ = variables or FEATURE_VARIABLES
    windows = window_hours or ROLLING_WINDOW_HOURS
    aggs = aggregations or ROLLING_AGGREGATIONS

    out = df.copy()

    if hour_column not in out.columns:
        raise ValueError(f"Hour column '{hour_column}' not found in DataFrame")

    hours = out[hour_column].to_numpy()
    n = len(out)

    for var in vars_:
        if var not in out.columns:
            continue
        values = out[var].to_numpy()
        for window in windows:
            for agg in aggs:
                col_name = f"{var}_rolling_{agg}_{window}h"
                result = np.full(n, np.nan, dtype=float)
                for i in range(n):
                    result[i] = _rolling_stat(
                        values, hours, i, window, agg, min_observations
                    )
                out[col_name] = result

    return out


def compute_rolling_features(
    patient: ProcessedPatient,
    variables: list[str] | None = None,
    window_hours: list[int] | None = None,
    aggregations: list[str] | None = None,
    min_observations: int = ROLLING_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Compute all rolling features for one patient.

    Returns a DataFrame with identifier columns, SepsisLabel, and all
    rolling feature columns.
    """
    from .current_features import extract_current_features

    vars_ = variables or FEATURE_VARIABLES
    df = extract_current_features(patient, feature_variables=vars_)
    return add_rolling_features(
        df,
        variables=vars_,
        window_hours=window_hours,
        aggregations=aggregations,
        min_observations=min_observations,
    )


def validate_rolling_features(df: pd.DataFrame) -> list[str]:
    """Validate rolling features output.

    Returns a list of issue messages (empty if valid).
    """
    issues: list[str] = []
    if len(df) == 0:
        issues.append("Empty rolling features DataFrame")
        return issues

    required = ["patient_id", "source_dataset", "hour", "SepsisLabel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # Rolling columns should be numeric and have no inf values.
    roll_cols = [c for c in df.columns if "_rolling_" in c]
    for col in roll_cols:
        vals = df[col]
        if np.isinf(vals).any():
            issues.append(f"{col} contains infinite values")
    return issues
