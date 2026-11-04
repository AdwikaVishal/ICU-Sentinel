"""Tests for missingness_features.py.

Uses synthetic ProcessedPatient fixtures. No real patient data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.features.feature_config import FEATURE_VARIABLES
from ml.p1_clinical.features.missingness_features import (
    add_measurement_availability,
    add_missingness_indicators,
    add_time_since_last_measurement,
    compute_missingness_features,
    validate_missingness_features,
)
from ml.p1_clinical.preprocessing.schemas import ProcessedPatient


def _expand_values(val, n, default):
    if val is None:
        return [default] * n
    if len(val) == 1 and n > 1:
        return list(val) * n
    if len(val) < n:
        # Repeat cyclically to reach length n.
        return (list(val) * ((n // len(val)) + 1))[:n]
    return list(val)[:n]


def _make_patient(
    patient_id: str = "p00001",
    hours: list[int] | None = None,
    hr_values: list[float] | None = None,
    temp_values: list[float] | None = None,
    sepsis_label: int = 0,
    source_dataset: str = "set_a",
    seed: int = 42,
) -> ProcessedPatient:
    rng = np.random.default_rng(seed)
    if hours is None:
        hours = list(range(1, 8))
    n = len(hours)
    clinical: dict = {}
    for var in FEATURE_VARIABLES:
        if var == "ICULOS":
            clinical[var] = [float(h) for h in hours]
        elif var == "HospAdmTime":
            clinical[var] = [float(-50)] * n
        elif var in ("Age", "Gender", "Unit1", "Unit2"):
            clinical[var] = [float(rng.integers(0, 2))] * n
        else:
            clinical[var] = rng.normal(0, 1, size=n).tolist()
    if hr_values is not None:
        clinical["HR"] = _expand_values(hr_values, n, 80.0)
    if temp_values is not None:
        clinical["Temp"] = _expand_values(temp_values, n, 37.0)
    return ProcessedPatient(
        patient_id=patient_id,
        hospital_id=source_dataset,
        hour=list(hours),
        clinical=clinical,
        sepsis_label=[sepsis_label] * n,
        source_file=f"{patient_id}.psv",
        source_dataset=source_dataset,
        n_rows=n,
    )


# ------------------------------------------------------------------
# missingness indicator tests
# ------------------------------------------------------------------


def test_missingness_indicator_detects_nan():
    df = pd.DataFrame({"HR": [80.0, np.nan, 85.0], "hour": [1, 2, 3]})
    out = add_missingness_indicators(df, variables=["HR"])
    assert out["HR_is_missing"].tolist() == [0, 1, 0]


def test_missingness_indicator_no_nan():
    df = pd.DataFrame({"HR": [80.0, 85.0, 90.0], "hour": [1, 2, 3]})
    out = add_missingness_indicators(df, variables=["HR"])
    assert (out["HR_is_missing"] == 0).all()


def test_missingness_indicator_all_nan():
    df = pd.DataFrame({"HR": [np.nan, np.nan], "hour": [1, 2]})
    out = add_missingness_indicators(df, variables=["HR"])
    assert (out["HR_is_missing"] == 1).all()


# ------------------------------------------------------------------
# time since last measurement tests
# ------------------------------------------------------------------


def test_time_since_last_current_valid():
    df = pd.DataFrame({"HR": [80.0, 85.0, np.nan], "hour": [1, 2, 3]})
    out = add_time_since_last_measurement(df, variables=["HR"])
    assert out["HR_hours_since_last"].tolist() == [0.0, 0.0, 1.0]


def test_time_since_last_long_gap():
    df = pd.DataFrame({"HR": [80.0, np.nan, np.nan, np.nan], "hour": [1, 2, 3, 4]})
    out = add_time_since_last_measurement(df, variables=["HR"])
    assert out["HR_hours_since_last"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_time_since_last_never_observed():
    df = pd.DataFrame({"HR": [np.nan, np.nan], "hour": [1, 2]})
    out = add_time_since_last_measurement(df, variables=["HR"])
    assert np.isinf(out["HR_hours_since_last"].iloc[0])
    assert np.isinf(out["HR_hours_since_last"].iloc[1])


def test_time_since_last_no_future_leakage():
    # A valid measurement at hour 3 must not fill hour 1.
    df = pd.DataFrame({"HR": [np.nan, np.nan, 85.0], "hour": [1, 2, 3]})
    out = add_time_since_last_measurement(df, variables=["HR"])
    assert np.isinf(out["HR_hours_since_last"].iloc[0])
    assert np.isinf(out["HR_hours_since_last"].iloc[1])
    assert out["HR_hours_since_last"].iloc[2] == 0.0


# ------------------------------------------------------------------
# measurement availability tests
# ------------------------------------------------------------------


def test_availability_fraction():
    df = pd.DataFrame({"HR": [80.0, np.nan, 85.0], "hour": [1, 2, 3]})
    out = add_measurement_availability(df, variables=["HR"], window_hours=2)
    # hour 1: window ( -1, 1] => [1] => 1/1 = 1.0
    # hour 2: window ( 0, 2] => [1,2] => 1/2 = 0.5
    # hour 3: window ( 1, 3] => [2,3] => 1/2 = 0.5
    assert abs(out["HR_availability_2h"].iloc[0] - 1.0) < 1e-9
    assert abs(out["HR_availability_2h"].iloc[1] - 0.5) < 1e-9
    assert abs(out["HR_availability_2h"].iloc[2] - 0.5) < 1e-9


def test_availability_no_valid():
    df = pd.DataFrame({"HR": [np.nan, np.nan], "hour": [1, 2]})
    out = add_measurement_availability(df, variables=["HR"], window_hours=6)
    assert (out["HR_availability_6h"] == 0.0).all()


# ------------------------------------------------------------------
# compute_missingness_features tests
# ------------------------------------------------------------------


def test_compute_missingness_output_columns():
    patient = _make_patient(hr_values=[80.0, np.nan, 85.0])
    df = compute_missingness_features(patient, variables=["HR"])
    assert "HR_is_missing" in df.columns
    assert "HR_hours_since_last" in df.columns
    assert "HR_availability_6h" in df.columns
    assert "patient_id" in df.columns
    assert "source_dataset" in df.columns
    assert "hour" in df.columns
    assert "SepsisLabel" in df.columns


def test_compute_missingness_preserves_sepsis_label():
    patient = _make_patient(sepsis_label=1)
    df = compute_missingness_features(patient, variables=["HR"])
    assert (df["SepsisLabel"] == 1).all()


# ------------------------------------------------------------------
# validate tests
# ------------------------------------------------------------------


def test_validate_missingness_valid():
    patient = _make_patient()
    df = compute_missingness_features(patient, variables=["HR"])
    assert validate_missingness_features(df) == []


def test_validate_missingness_bad_indicator():
    patient = _make_patient()
    df = compute_missingness_features(patient, variables=["HR"])
    df.loc[0, "HR_is_missing"] = 2
    issues = validate_missingness_features(df)
    assert any("HR_is_missing" in i for i in issues)
