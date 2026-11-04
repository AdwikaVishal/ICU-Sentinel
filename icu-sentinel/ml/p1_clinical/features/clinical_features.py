"""Derived clinical features.

Implements clinically interpretable derived variables where required
inputs exist:

A. Shock index = HR / SBP (guards against zero/missing denominators)
B. SIRS component count (documents which components are included;
   missing components are not treated as normal)

No unsupported clinical thresholds are invented. All thresholds are
documented from the SIRS definition.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..preprocessing.schemas import ProcessedPatient
from .feature_config import (
    SIRS_HR_HIGH,
    SIRS_RESP_HIGH,
    SIRS_TEMP_HIGH,
    SIRS_TEMP_LOW,
    SIRS_WBC_HIGH,
    SIRS_WBC_LOW,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.features.clinical_features")


def compute_shock_index(
    df: pd.DataFrame,
    hr_column: str = "HR",
    sbp_column: str = "SBP",
    output_column: str = "shock_index",
) -> pd.DataFrame:
    """Compute shock index = HR / SBP.

    Returns NaN where SBP is zero or missing, or HR is missing.
    Does not modify the input DataFrame (returns a copy).
    """
    out = df.copy()
    if hr_column not in out.columns or sbp_column not in out.columns:
        out[output_column] = np.nan
        return out

    hr = pd.to_numeric(out[hr_column], errors="coerce")
    sbp = pd.to_numeric(out[sbp_column], errors="coerce")

    valid_denominator = sbp.notna() & (sbp > 0)
    valid_numerator = hr.notna()

    result = np.full(len(out), np.nan, dtype=float)
    result[valid_denominator & valid_numerator] = (
        hr[valid_denominator & valid_numerator] / sbp[valid_denominator & valid_numerator]
    )
    out[output_column] = result
    return out


def compute_sirs_count(
    df: pd.DataFrame,
    temp_column: str = "Temp",
    hr_column: str = "HR",
    resp_column: str = "Resp",
    wbc_column: str = "WBC",
    count_column: str = "sirs_count",
    available_column: str = "sirs_components_available",
) -> pd.DataFrame:
    """Compute SIRS component count from available measurements.

    SIRS criteria (documented thresholds):
    - Temperature > 38.3 C or < 36.0 C
    - Heart rate > 90 bpm
    - Respiration > 20 breaths/min
    - WBC > 12 (10^3/uL) or < 4 (10^3/uL)

    Missing components are NOT counted as normal. The count is computed
    only over components with valid measurements. A separate column
    reports how many components were available.

    Does not claim a complete SIRS score when components are unavailable.
    """
    out = df.copy()

    def _check_temp(v):
        if pd.isna(v):
            return None  # not available
        return bool(v > SIRS_TEMP_HIGH or v < SIRS_TEMP_LOW)

    def _check_hr(v):
        if pd.isna(v):
            return None
        return bool(v > SIRS_HR_HIGH)

    def _check_resp(v):
        if pd.isna(v):
            return None
        return bool(v > SIRS_RESP_HIGH)

    def _check_wbc(v):
        if pd.isna(v):
            return None
        return bool(v > SIRS_WBC_HIGH or v < SIRS_WBC_LOW)

    checks = []
    if temp_column in out.columns:
        checks.append(("temp", out[temp_column].apply(_check_temp)))
    if hr_column in out.columns:
        checks.append(("hr", out[hr_column].apply(_check_hr)))
    if resp_column in out.columns:
        checks.append(("resp", out[resp_column].apply(_check_resp)))
    if wbc_column in out.columns:
        checks.append(("wbc", out[wbc_column].apply(_check_wbc)))

    n_rows = len(out)
    counts = np.zeros(n_rows, dtype=int)
    available = np.zeros(n_rows, dtype=int)

    for i in range(n_rows):
        n_avail = 0
        n_met = 0
        for _, series in checks:
            val = series.iloc[i]
            if val is not None:
                n_avail += 1
                if val:
                    n_met += 1
        available[i] = n_avail
        counts[i] = n_met

    out[count_column] = counts
    out[available_column] = available
    return out


def compute_clinical_features(
    patient: ProcessedPatient,
) -> pd.DataFrame:
    """Compute all clinical features for one patient.

    Returns a DataFrame with identifier columns, SepsisLabel, shock_index,
    sirs_count, and sirs_components_available.
    """
    from .current_features import extract_current_features

    df = extract_current_features(patient)
    df = compute_shock_index(df)
    df = compute_sirs_count(df)
    return df


def validate_clinical_features(df: pd.DataFrame) -> list[str]:
    """Validate clinical features output.

    Returns a list of issue messages (empty if valid).
    """
    issues: list[str] = []
    if len(df) == 0:
        issues.append("Empty clinical features DataFrame")
        return issues

    required = ["patient_id", "source_dataset", "hour", "SepsisLabel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # Shock index should be non-negative where valid.
    if "shock_index" in df.columns:
        si = df["shock_index"].dropna()
        neg = si[si < 0]
        if len(neg) > 0:
            issues.append(f"{len(neg)} row(s) with negative shock_index")

    # SIRS count should be >= 0 and <= components available.
    if "sirs_count" in df.columns and "sirs_components_available" in df.columns:
        bad = df[(df["sirs_count"] < 0) | (df["sirs_count"] > df["sirs_components_available"])]
        if len(bad) > 0:
            issues.append(f"{len(bad)} row(s) with invalid sirs_count")

    return issues
