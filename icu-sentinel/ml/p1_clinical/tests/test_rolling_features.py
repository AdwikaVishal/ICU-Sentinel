"""Tests for rolling_features.py.

Uses synthetic ProcessedPatient fixtures. No real patient data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.p1_clinical.features.feature_config import (
    FEATURE_VARIABLES,
)
from ml.p1_clinical.features.rolling_features import (
    add_rolling_features,
    compute_rolling_features,
    validate_rolling_features,
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
        hours = list(range(1, 11))
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
# rolling stat tests
# ------------------------------------------------------------------


def test_rolling_mean_causal_includes_current():
    df = pd.DataFrame({"HR": [10.0, 20.0, 30.0], "hour": [1, 2, 3]})
    out = add_rolling_features(df, variables=["HR"], window_hours=[2], aggregations=["mean"])
    # hour 1: window (-1,1] => [1] => 10
    # hour 2: window (0,2] => [1,2] => 15
    # hour 3: window (1,3] => [2,3] => 25
    assert out["HR_rolling_mean_2h"].iloc[0] == 10.0
    assert out["HR_rolling_mean_2h"].iloc[1] == 15.0
    assert out["HR_rolling_mean_2h"].iloc[2] == 25.0


def test_rolling_no_future_leakage():
    # A high value at hour 3 must not affect hour 1.
    df = pd.DataFrame({"HR": [10.0, 10.0, 100.0], "hour": [1, 2, 3]})
    out = add_rolling_features(df, variables=["HR"], window_hours=[3], aggregations=["mean"])
    # hour 1: window (-2,1] => [1] => 10 (not affected by hour 3's 100)
    assert out["HR_rolling_mean_3h"].iloc[0] == 10.0


def test_rolling_min_max_std():
    df = pd.DataFrame({"HR": [10.0, 20.0, 30.0], "hour": [1, 2, 3]})
    out = add_rolling_features(
        df, variables=["HR"], window_hours=[3], aggregations=["min", "max", "std"]
    )
    assert out["HR_rolling_min_3h"].iloc[2] == 10.0
    assert out["HR_rolling_max_3h"].iloc[2] == 30.0
    assert out["HR_rolling_std_3h"].iloc[2] == pytest.approx(10.0)


def test_rolling_latest():
    df = pd.DataFrame({"HR": [10.0, np.nan, 30.0], "hour": [1, 2, 3]})
    out = add_rolling_features(df, variables=["HR"], window_hours=[3], aggregations=["latest"])
    # hour 1: latest valid = 10
    # hour 2: latest valid in window = 10 (hour 2 is NaN)
    # hour 3: latest valid = 30
    assert out["HR_rolling_latest_3h"].iloc[0] == 10.0
    assert out["HR_rolling_latest_3h"].iloc[1] == 10.0
    assert out["HR_rolling_latest_3h"].iloc[2] == 30.0


def test_rolling_insufficient_obs_returns_nan():
    df = pd.DataFrame({"HR": [np.nan, np.nan], "hour": [1, 2]})
    out = add_rolling_features(df, variables=["HR"], window_hours=[6], aggregations=["mean"])
    assert np.isnan(out["HR_rolling_mean_6h"].iloc[0])
    assert np.isnan(out["HR_rolling_mean_6h"].iloc[1])


# ------------------------------------------------------------------
# compute_rolling_features tests
# ------------------------------------------------------------------


def test_compute_rolling_output_columns():
    patient = _make_patient(hr_values=[80.0, 85.0, 90.0, 95.0, 100.0])
    df = compute_rolling_features(patient, variables=["HR"])
    assert "HR_rolling_mean_6h" in df.columns
    assert "patient_id" in df.columns
    assert "hour" in df.columns
    assert "SepsisLabel" in df.columns


def test_compute_rolling_preserves_sepsis_label():
    patient = _make_patient(sepsis_label=1)
    df = compute_rolling_features(patient, variables=["HR"])
    assert (df["SepsisLabel"] == 1).all()


# ------------------------------------------------------------------
# validate tests
# ------------------------------------------------------------------


def test_validate_rolling_valid():
    patient = _make_patient()
    df = compute_rolling_features(patient, variables=["HR"])
    assert validate_rolling_features(df) == []
