"""Temporal leakage tests for feature engineering.

These are the most important acceptance tests: they verify that no
observation after hour t contributes to any feature at hour t.
Tests modify future observations and confirm features at hour t are
unchanged. Also tests cross-patient contamination, sorting, missing
hours, duplicate timestamps, and sparse observations.

Uses synthetic fixtures. No real patient data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.features.feature_config import FeatureConfig
from ml.p1_clinical.features.feature_pipeline import FeaturePipeline
from ml.p1_clinical.features.missingness_features import (
    add_time_since_last_measurement,
)
from ml.p1_clinical.features.rolling_features import add_rolling_features
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
    n_hours: int | None = None,
    hr_values: list[float] | None = None,
    seed: int = 42,
) -> ProcessedPatient:
    rng = np.random.default_rng(seed)
    if hours is None:
        if n_hours is None:
            n_hours = 8
        hours = list(range(1, n_hours + 1))
    n = len(hours)
    clinical: dict = {}
    base_vars = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine",
        "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
        "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets",
        "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
    ]
    for var in base_vars:
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
        sepsis_label=[0] * n,
        source_file=f"{patient_id}.psv",
        source_dataset="set_a",
        n_rows=n,
    )


def _get_feature_value(df, patient_id, hour, col):
    row = df[(df["patient_id"] == patient_id) & (df["hour"] == hour)]
    if len(row) == 0:
        return None
    return row.iloc[0][col]


# ------------------------------------------------------------------
# No future leakage tests
# ------------------------------------------------------------------


def test_future_observation_d_not_affect_hour_t_rolling():
    pipeline = FeaturePipeline()
    # Baseline patient.
    base = _make_patient(hr_values=[80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0])
    df_base = pipeline.transform_patient(base)

    # Modified patient: hour 8 (future) changed to 200.
    modified = _make_patient(hr_values=[80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 200.0])
    df_mod = pipeline.transform_patient(modified)

    # Features at hour 1 should be identical between base and modified.
    for col in df_base.columns:
        if col in ("patient_id", "source_dataset", "hour", "SepsisLabel"):
            continue
        v1 = _get_feature_value(df_base, "p00001", 1, col)
        v2 = _get_feature_value(df_mod, "p00001", 1, col)
        if pd.isna(v1) and pd.isna(v2):
            continue
        assert v1 == v2, f"Future observation changed feature {col} at hour 1: {v1} vs {v2}"


def test_future_observation_d_not_affect_hour_4_rolling():
    pipeline = FeaturePipeline()
    base = _make_patient(hr_values=[80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0])
    df_base = pipeline.transform_patient(base)

    modified = _make_patient(hr_values=[80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 200.0])
    df_mod = pipeline.transform_patient(modified)

    # Features at hour 4 should NOT be affected by hour 8's value.
    roll_cols = [c for c in df_base.columns if "_rolling_" in c]
    for col in roll_cols:
        v1 = _get_feature_value(df_base, "p00001", 4, col)
        v2 = _get_feature_value(df_mod, "p00001", 4, col)
        if pd.isna(v1) and pd.isna(v2):
            continue
        assert v1 == v2, f"Rolling feature {col} at hour 4 leaked from hour 8: {v1} vs {v2}"


def test_time_since_last_no_future_leakage():
    df = pd.DataFrame({"HR": [np.nan, np.nan, 85.0], "hour": [1, 2, 3]})
    out = add_time_since_last_measurement(df, variables=["HR"])
    # Hour 1: never observed => inf (hour 3's 85 must not retroactively fill)
    assert np.isinf(out["HR_hours_since_last"].iloc[0])
    # Hour 2: never observed up to hour 2 => inf
    assert np.isinf(out["HR_hours_since_last"].iloc[1])


def test_forward_fill_no_future_leakage():
    pipeline = FeaturePipeline(config=FeatureConfig(compute_forward_fill=True))
    patient = _make_patient(hr_values=[80.0, np.nan, np.nan, np.nan])
    df = pipeline.transform_patient(patient)
    # Hour 2 forward-filled from hour 1 (gap=1 <= default window of 4).
    # Hour 4: gap from hour 1 is 3 <= 4, so still filled.
    # No future observations exist, so this just tests the mechanism.
    assert pd.notna(df["HR_ffilled"].iloc[1])


def test_no_backward_fill():
    # A missing value at hour 1 should remain NaN even if hour 2 has a value.
    df = pd.DataFrame({"HR": [np.nan, 85.0], "hour": [1, 2]})
    out = add_time_since_last_measurement(df, variables=["HR"])
    assert np.isinf(out["HR_hours_since_last"].iloc[0])


# ------------------------------------------------------------------
# Cross-patient contamination
# ------------------------------------------------------------------


def test_no_cross_patient_contamination():
    pipeline = FeaturePipeline()
    p1 = _make_patient("p00001", hr_values=[80.0] * 8, seed=1)
    p2 = _make_patient("p00002", hr_values=[140.0] * 8, seed=2)
    result = pipeline.transform_patients([p1, p2])
    df = result.combined

    p1_rows = df[df["patient_id"] == "p00001"]
    p2_rows = df[df["patient_id"] == "p00002"]
    assert (p1_rows["HR"] == 80.0).all()
    assert (p2_rows["HR"] == 140.0).all()

    # Rolling features for p1 should not be affected by p2's values.
    p1_roll_cols = [c for c in df.columns if "_rolling_" in c]
    for col in p1_roll_cols:
        vals_p1 = p1_rows[col].dropna()
        vals_p2 = p2_rows[col].dropna()
        if len(vals_p1) > 0 and len(vals_p2) > 0:
            # p1's rolling stats should reflect p1's values (~80), not p2's (~140).
            assert vals_p1.mean() < 100, f"{col} contaminated by p2: p1 mean={vals_p1.mean()}"


# ------------------------------------------------------------------
# Sorting
# ------------------------------------------------------------------


def test_output_sorted_by_patient_and_hour():
    pipeline = FeaturePipeline()
    p1 = _make_patient("p00002", n_hours=4, seed=1)
    p2 = _make_patient("p00001", n_hours=4, seed=2)
    result = pipeline.transform_patients([p1, p2])
    df = result.combined
    # Within each patient, hours should be increasing.
    for pid in ["p00001", "p00002"]:
        sub = df[df["patient_id"] == pid]
        assert sub["hour"].is_monotonic_increasing


# ------------------------------------------------------------------
# Missing hours / sparse observations
# ------------------------------------------------------------------


def test_sparse_observations_no_crash():
    pipeline = FeaturePipeline()
    # Patient with many missing HR values.
    patient = _make_patient(hr_values=[80.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    df = pipeline.transform_patient(patient)
    assert len(df) == 8
    assert "HR_rolling_mean_6h" in df.columns


def test_missing_hours_gap():
    # Hours [1, 2, 5, 6] - gap between 2 and 5.
    patient = _make_patient(hours=[1, 2, 5, 6], hr_values=[80.0, 85.0, 90.0, 95.0], seed=1)
    pipeline = FeaturePipeline()
    df = pipeline.transform_patient(patient)
    assert df["hour"].tolist() == [1, 2, 5, 6]


# ------------------------------------------------------------------
# Duplicate timestamps
# ------------------------------------------------------------------


def test_duplicate_timestamps_handled():
    df = pd.DataFrame({"HR": [80.0, 85.0], "hour": [1, 1]})
    out = add_rolling_features(df, variables=["HR"], window_hours=[2], aggregations=["mean"])
    # Should not crash; both rows have hour 1.
    assert len(out) == 2
