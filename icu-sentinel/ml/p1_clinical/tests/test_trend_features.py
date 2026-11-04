"""Tests for trend_features.py.

Uses synthetic ProcessedPatient fixtures. No real patient data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.features.feature_config import FEATURE_VARIABLES
from ml.p1_clinical.features.trend_features import (
    add_trend_features,
    compute_trend_features,
    validate_trend_features,
)
from ml.p1_clinical.preprocessing.schemas import ProcessedPatient


def _expand_values(val, n, default):
    if val is None:
        return [default] * n
    if len(val) == 1 and n > 1:
        return list(val) * n
    if len(val) < n:
        return (list(val) * ((n // len(val)) + 1))[:n]
    return list(val)[:n]


def _make_patient(
    patient_id: str = "p00001",
    hours: list[int] | None = None,
    hr_values: list[float] | None = None,
    sepsis_label: int = 0,
    seed: int = 42,
) -> ProcessedPatient:
    rng = np.random.default_rng(seed)
    if hours is None:
        hours = list(range(1, 9))
    n = len(hours)
    clinical: dict = {}
    for var in FEATURE_VARIABLES:
        if var in ("Age", "Gender", "Unit1", "Unit2"):
            clinical[var] = [float(rng.integers(0, 2))] * n
        else:
            clinical[var] = rng.normal(0, 1, size=n).tolist()
    # ICULOS and HospAdmTime are not in FEATURE_VARIABLES; add separately.
    clinical["ICULOS"] = [float(h) for h in hours]
    clinical["HospAdmTime"] = [float(-50)] * n
    if hr_values is not None:
        clinical["HR"] = _expand_values(hr_values, n, 80.0)
    return ProcessedPatient(
        patient_id=patient_id,
        hospital_id="set_a",
        hour=list(hours),
        clinical=clinical,
        sepsis_label=[sepsis_label] * n,
        source_file=f"{patient_id}.psv",
        source_dataset="set_a",
        n_rows=n,
    )


# ------------------------------------------------------------------
# absolute change tests
# ------------------------------------------------------------------


def test_abs_change_causal():
    df = pd.DataFrame({"HR": [80.0, 85.0, 90.0], "hour": [1, 2, 3]})
    out = add_trend_features(df, variables=["HR"], window_hours=[1])
    # hour 1: no prior observation within window => NaN
    # hour 2: ref=hour 1 (80), diff = 85-80 = 5
    # hour 3: ref=hour 2 (85), diff = 90-85 = 5
    assert np.isnan(out["HR_abs_change_1h"].iloc[0])
    assert out["HR_abs_change_1h"].iloc[1] == 5.0
    assert out["HR_abs_change_1h"].iloc[2] == 5.0


def test_abs_change_no_future_leakage():
    df = pd.DataFrame({"HR": [80.0, 80.0, 100.0], "hour": [1, 2, 3]})
    out = add_trend_features(df, variables=["HR"], window_hours=[3])
    # hour 1: no prior observation => NaN (hour 3's 100 must not leak)
    assert np.isnan(out["HR_abs_change_3h"].iloc[0])


def test_abs_change_insufficient_obs():
    df = pd.DataFrame({"HR": [np.nan, np.nan], "hour": [1, 2]})
    out = add_trend_features(df, variables=["HR"], window_hours=[6])
    assert np.isnan(out["HR_abs_change_6h"].iloc[0])


# ------------------------------------------------------------------
# relative change tests
# ------------------------------------------------------------------


def test_rel_change_normal():
    df = pd.DataFrame({"HR": [80.0, 100.0], "hour": [1, 2]})
    out = add_trend_features(df, variables=["HR"], window_hours=[2])
    # hour 2: (100-80)/|80| = 0.25
    assert abs(out["HR_rel_change_2h"].iloc[1] - 0.25) < 1e-9


def test_rel_change_zero_denominator():
    df = pd.DataFrame({"HR": [0.0, 80.0], "hour": [1, 2]})
    out = add_trend_features(df, variables=["HR"], window_hours=[2])
    assert np.isnan(out["HR_rel_change_2h"].iloc[1])


# ------------------------------------------------------------------
# slope tests
# ------------------------------------------------------------------


def test_slope_positive():
    df = pd.DataFrame({"HR": [80.0, 90.0], "hour": [1, 3]})
    out = add_trend_features(df, variables=["HR"], window_hours=[4])
    # hour 2: (90-80)/(3-1) = 5.0 per hour
    assert abs(out["HR_slope_4h"].iloc[1] - 5.0) < 1e-9


# ------------------------------------------------------------------
# compute_trend_features tests
# ------------------------------------------------------------------


def test_compute_trend_output_columns():
    patient = _make_patient(hr_values=[80.0, 85.0, 90.0])
    df = compute_trend_features(patient, variables=["HR"])
    assert "HR_abs_change_6h" in df.columns
    assert "HR_rel_change_6h" in df.columns
    assert "HR_slope_6h" in df.columns
    assert "patient_id" in df.columns


def test_compute_trend_preserves_sepsis_label():
    patient = _make_patient(sepsis_label=1)
    df = compute_trend_features(patient, variables=["HR"])
    assert (df["SepsisLabel"] == 1).all()


# ------------------------------------------------------------------
# validate tests
# ------------------------------------------------------------------


def test_validate_trend_valid():
    patient = _make_patient()
    df = compute_trend_features(patient, variables=["HR"])
    assert validate_trend_features(df) == []
