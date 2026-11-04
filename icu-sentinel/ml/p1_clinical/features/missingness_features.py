"""Missingness features.

Generates, for each clinical measurement:

1. Missingness indicator (1 if missing at current hour, else 0).
2. Time since last valid measurement (causal; hours since last valid obs).
3. Measurement availability over recent windows (fraction of valid hours).

Missingness indicators are computed from the raw input BEFORE any
imputation or forward filling. Time-since-last-measurement uses only
observations available at the current hour or earlier.

Never uses future measurements to fill earlier values. Distinguishes
never-observed measurements (inf) from previously-observed ones.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..preprocessing.schemas import ProcessedPatient
from .feature_config import FEATURE_VARIABLES

logger = logging.getLogger("icu_sentinel.p1_clinical.features.missingness_features")


def add_missingness_indicators(
    df: pd.DataFrame,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Add missingness indicator columns (1 if missing, 0 if present).

    Computed from raw input before imputation/forward fill.
    """
    vars_ = variables or FEATURE_VARIABLES
    out = df.copy()
    for var in vars_:
        if var in out.columns:
            out[f"{var}_is_missing"] = out[var].isna().astype(int)
        else:
            out[f"{var}_is_missing"] = 1
    return out


def add_time_since_last_measurement(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    hour_column: str = "hour",
) -> pd.DataFrame:
    """Add time-since-last-valid-measurement columns (causal).

    For each row at hour t, computes hours since the last hour <= t where
    the variable had a valid (non-missing) measurement.

    Returns inf (np.inf) if the variable has never been observed up to
    and including the current hour. Returns 0 if the current hour itself
    has a valid measurement.

    Never uses future observations.
    """
    vars_ = variables or FEATURE_VARIABLES
    out = df.copy()

    if hour_column not in out.columns:
        raise ValueError(f"Hour column '{hour_column}' not found in DataFrame")

    hours = out[hour_column].to_numpy()
    n = len(out)

    for var in vars_:
        if var not in out.columns:
            out[f"{var}_hours_since_last"] = np.inf
            continue

        values = out[var].to_numpy()
        result = np.full(n, np.inf, dtype=float)

        last_valid_hour: float | None = None
        for i in range(n):
            h = hours[i]
            v = values[i]
            if pd.notna(v):
                # Current hour has a valid measurement.
                last_valid_hour = h
                result[i] = 0.0
            else:
                if last_valid_hour is not None:
                    result[i] = h - last_valid_hour
                else:
                    # Never observed up to this hour.
                    result[i] = np.inf

        out[f"{var}_hours_since_last"] = result

    return out


def add_measurement_availability(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    window_hours: int = 6,
    hour_column: str = "hour",
) -> pd.DataFrame:
    """Add measurement availability over a trailing causal window.

    For each row at hour t, computes the fraction of hours in
    (t - window_hours, t] where the variable had a valid measurement.

    The window is causal: includes the current hour, excludes future hours.
    Hours with no data at all (gaps in the hour sequence) are treated as
    unavailable for that variable.

    Returns 0 if no valid observations in the window.
    """
    vars_ = variables or FEATURE_VARIABLES
    out = df.copy()

    if hour_column not in out.columns:
        raise ValueError(f"Hour column '{hour_column}' not found in DataFrame")

    hours = out[hour_column].to_numpy()
    n = len(out)

    for var in vars_:
        if var not in out.columns:
            out[f"{var}_availability_{window_hours}h"] = 0.0
            continue

        values = out[var].to_numpy()
        result = np.zeros(n, dtype=float)

        for i in range(n):
            t = hours[i]
            window_start = t - window_hours
            count_valid = 0
            count_total = 0
            for j in range(i, -1, -1):
                hj = hours[j]
                if hj <= window_start:
                    break
                count_total += 1
                if pd.notna(values[j]):
                    count_valid += 1
            if count_total > 0:
                result[i] = count_valid / count_total
            else:
                result[i] = 0.0

        out[f"{var}_availability_{window_hours}h"] = result

    return out


def compute_missingness_features(
    patient: ProcessedPatient,
    variables: list[str] | None = None,
    availability_window_hours: int = 6,
) -> pd.DataFrame:
    """Compute all missingness features for one patient.

    Returns a DataFrame with columns:
        patient_id, source_dataset, hour,
        <var>_is_missing, <var>_hours_since_last,
        <var>_availability_{window}h
    plus SepsisLabel.
    """
    from .current_features import extract_current_features

    vars_ = variables or FEATURE_VARIABLES
    df = extract_current_features(patient, feature_variables=vars_)

    df = add_missingness_indicators(df, variables=vars_)
    df = add_time_since_last_measurement(df, variables=vars_)
    df = add_measurement_availability(
        df, variables=vars_, window_hours=availability_window_hours
    )

    return df


def validate_missingness_features(df: pd.DataFrame) -> list[str]:
    """Validate missingness features output.

    Returns a list of issue messages (empty if valid).
    """
    issues: list[str] = []
    if len(df) == 0:
        issues.append("Empty missingness features DataFrame")
        return issues

    required = ["patient_id", "source_dataset", "hour", "SepsisLabel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # Check missingness indicators are 0/1.
    miss_cols = [c for c in df.columns if c.endswith("_is_missing")]
    for col in miss_cols:
        vals = df[col].dropna()
        if len(vals) > 0 and not vals.isin([0, 1]).all():
            issues.append(f"{col} contains values other than 0/1")

    # Check hours_since_last >= 0 or inf.
    tsl_cols = [c for c in df.columns if c.endswith("_hours_since_last")]
    for col in tsl_cols:
        vals = df[col].dropna()
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0 and (finite < 0).any():
            issues.append(f"{col} contains negative values")

    return issues
