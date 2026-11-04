"""Trend and slope features.

Calculates temporal changes for relevant clinical measurements using only
historical/current observations:

- Absolute change: value[t] - last_valid_value_at_or_before(t - window)
- Relative change: (value[t] - value[t-window]) / |value[t-window]|
- Slope: change per hour over a trailing window

All computations are causal: the reference observation is the last valid
one at or before t - window (never the current observation itself). Zero
denominators and insufficient observations are handled explicitly (NaN).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..preprocessing.schemas import ProcessedPatient
from .feature_config import (
    FEATURE_VARIABLES,
    TREND_MIN_OBSERVATIONS,
    TREND_WINDOW_HOURS,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.features.trend_features")


def _last_valid_strictly_before(
    values: np.ndarray,
    hours: np.ndarray,
    idx: int,
    window_hours: int,
) -> tuple[float, float] | None:
    """Return (value, hour) of the last valid observation strictly before
    row idx, within (hours[idx] - window_hours, hours[idx]).

    The current observation itself is excluded. Returns None if no
    qualifying observation exists.
    """
    t = hours[idx]
    window_start = t - window_hours
    best: tuple[float, float] | None = None
    for j in range(len(values)):
        if j == idx:
            continue
        hj = hours[j]
        # Before current, within the trailing window (inclusive of lower bound).
        if not (window_start <= hj < t):
            continue
        v = values[j]
        if pd.notna(v) and (best is None or hj > best[1]):
            best = (float(v), float(hj))
    return best


def add_trend_features(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    window_hours: list[int] | None = None,
    min_observations: int = TREND_MIN_OBSERVATIONS,
    hour_column: str = "hour",
) -> pd.DataFrame:
    """Add causal trend/slope feature columns.

    For each variable and window, creates:
        {var}_abs_change_{window}h   : value[t] - ref_value (ref strictly before t, within window)
        {var}_rel_change_{window}h   : (value[t] - ref_value) / |ref_value|
        {var}_slope_{window}h        : (value[t] - ref_value) / (t - ref_hour)

    All computations are causal: only observations strictly before the
    current hour, within the trailing window, are used as reference.
    """
    vars_ = variables or FEATURE_VARIABLES
    windows = window_hours or TREND_WINDOW_HOURS

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
            abs_col = f"{var}_abs_change_{window}h"
            rel_col = f"{var}_rel_change_{window}h"
            slope_col = f"{var}_slope_{window}h"

            abs_vals = np.full(n, np.nan, dtype=float)
            rel_vals = np.full(n, np.nan, dtype=float)
            slope_vals = np.full(n, np.nan, dtype=float)

            for i in range(n):
                cur = values[i]
                if not pd.notna(cur):
                    continue

                # Reference observation: last valid strictly before t, within window.
                ref = _last_valid_strictly_before(values, hours, i, window)
                if ref is None:
                    continue

                ref_val, ref_hour = ref
                abs_vals[i] = float(cur) - ref_val

                # Relative change (NaN if zero denominator).
                if abs(ref_val) > 0:
                    rel_vals[i] = (float(cur) - ref_val) / abs(ref_val)

                # Slope: change per hour over elapsed time.
                elapsed = hours[i] - ref_hour
                if elapsed > 0:
                    slope_vals[i] = (float(cur) - ref_val) / elapsed

            out[abs_col] = abs_vals
            out[rel_col] = rel_vals
            out[slope_col] = slope_vals

    return out


def compute_trend_features(
    patient: ProcessedPatient,
    variables: list[str] | None = None,
    window_hours: list[int] | None = None,
    min_observations: int = TREND_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Compute all trend features for one patient.

    Returns a DataFrame with identifier columns, SepsisLabel, and all
    trend feature columns.
    """
    from .current_features import extract_current_features

    vars_ = variables or FEATURE_VARIABLES
    df = extract_current_features(patient, feature_variables=vars_)
    return add_trend_features(
        df,
        variables=vars_,
        window_hours=window_hours,
        min_observations=min_observations,
    )


def validate_trend_features(df: pd.DataFrame) -> list[str]:
    """Validate trend features output.

    Returns a list of issue messages (empty if valid).
    """
    issues: list[str] = []
    if len(df) == 0:
        issues.append("Empty trend features DataFrame")
        return issues

    required = ["patient_id", "source_dataset", "hour", "SepsisLabel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    trend_cols = [c for c in df.columns
                  if "_abs_change_" in c or "_rel_change_" in c or "_slope_" in c]
    for col in trend_cols:
        if np.isinf(df[col]).any():
            issues.append(f"{col} contains infinite values")

    return issues
