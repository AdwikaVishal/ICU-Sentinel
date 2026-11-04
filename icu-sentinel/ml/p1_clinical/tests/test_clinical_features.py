"""Tests for clinical_features.py.

Uses synthetic ProcessedPatient fixtures. No real patient data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.features.clinical_features import (
    compute_clinical_features,
    compute_shock_index,
    compute_sirs_count,
    validate_clinical_features,
)
from ml.p1_clinical.features.feature_config import FEATURE_VARIABLES
from ml.p1_clinical.preprocessing.schemas import ProcessedPatient


def _make_df(
    hr_values: list[float] | None = None,
    sbp_values: list[float] | None = None,
    temp_values: list[float] | None = None,
    resp_values: list[float] | None = None,
    wbc_values: list[float] | None = None,
    hours: list[int] | None = None,
) -> pd.DataFrame:
    if hours is None:
        hours = [1, 2, 3]
    n = len(hours)

    def _expand(val, default):
        if val is None:
            return [default] * n
        if len(val) == 1 and n > 1:
            return list(val) * n
        return list(val)

    df = pd.DataFrame({"hour": hours})
    df["HR"] = _expand(hr_values, 80.0)
    df["SBP"] = _expand(sbp_values, 120.0)
    df["Temp"] = _expand(temp_values, 37.0)
    df["Resp"] = _expand(resp_values, 18.0)
    df["WBC"] = _expand(wbc_values, 7.0)
    for var in FEATURE_VARIABLES:
        if var not in df.columns:
            df[var] = [np.nan] * n
    df["patient_id"] = "p00001"
    df["source_dataset"] = "set_a"
    df["SepsisLabel"] = [0] * n
    return df


# ------------------------------------------------------------------
# shock index tests
# ------------------------------------------------------------------


def test_shock_index_normal():
    df = _make_df(hr_values=[80.0], sbp_values=[120.0])
    out = compute_shock_index(df)
    assert abs(out["shock_index"].iloc[0] - (80.0 / 120.0)) < 1e-9


def test_shock_index_zero_sbp():
    df = _make_df(hr_values=[80.0], sbp_values=[0.0])
    out = compute_shock_index(df)
    assert np.isnan(out["shock_index"].iloc[0])


def test_shock_index_missing_sbp():
    df = _make_df(hr_values=[80.0], sbp_values=[np.nan])
    out = compute_shock_index(df)
    assert np.isnan(out["shock_index"].iloc[0])


def test_shock_index_missing_hr():
    df = _make_df(hr_values=[np.nan], sbp_values=[120.0])
    out = compute_shock_index(df)
    assert np.isnan(out["shock_index"].iloc[0])


def test_shock_index_preserves_original():
    df = _make_df(hr_values=[80.0], sbp_values=[120.0])
    out = compute_shock_index(df)
    assert out["HR"].iloc[0] == 80.0
    assert out["SBP"].iloc[0] == 120.0


# ------------------------------------------------------------------
# SIRS count tests
# ------------------------------------------------------------------


def test_sirs_count_all_normal():
    df = _make_df(temp_values=[37.0], hr_values=[80.0], resp_values=[18.0], wbc_values=[7.0])
    out = compute_sirs_count(df)
    assert out["sirs_count"].iloc[0] == 0
    assert out["sirs_components_available"].iloc[0] == 4


def test_sirs_count_temp_high():
    df = _make_df(temp_values=[39.0], hr_values=[80.0], resp_values=[18.0], wbc_values=[7.0])
    out = compute_sirs_count(df)
    assert out["sirs_count"].iloc[0] == 1


def test_sirs_count_temp_low():
    df = _make_df(temp_values=[35.0], hr_values=[80.0], resp_values=[18.0], wbc_values=[7.0])
    out = compute_sirs_count(df)
    assert out["sirs_count"].iloc[0] == 1


def test_sirs_count_hr_high():
    df = _make_df(temp_values=[37.0], hr_values=[95.0], resp_values=[18.0], wbc_values=[7.0])
    out = compute_sirs_count(df)
    assert out["sirs_count"].iloc[0] == 1


def test_sirs_count_resp_high():
    df = _make_df(temp_values=[37.0], hr_values=[80.0], resp_values=[22.0], wbc_values=[7.0])
    out = compute_sirs_count(df)
    assert out["sirs_count"].iloc[0] == 1


def test_sirs_count_wbc_high():
    df = _make_df(temp_values=[37.0], hr_values=[80.0], resp_values=[18.0], wbc_values=[15.0])
    out = compute_sirs_count(df)
    assert out["sirs_count"].iloc[0] == 1


def test_sirs_count_wbc_low():
    df = _make_df(temp_values=[37.0], hr_values=[80.0], resp_values=[18.0], wbc_values=[3.0])
    out = compute_sirs_count(df)
    assert out["sirs_count"].iloc[0] == 1


def test_sirs_count_missing_not_normal():
    df = _make_df(temp_values=[np.nan], hr_values=[80.0], resp_values=[18.0], wbc_values=[7.0])
    out = compute_sirs_count(df)
    assert out["sirs_components_available"].iloc[0] == 3
    assert out["sirs_count"].iloc[0] == 0


def test_sirs_count_all_missing():
    df = _make_df(temp_values=[np.nan], hr_values=[np.nan], resp_values=[np.nan], wbc_values=[np.nan])
    out = compute_sirs_count(df)
    assert out["sirs_components_available"].iloc[0] == 0
    assert out["sirs_count"].iloc[0] == 0


# ------------------------------------------------------------------
# compute_clinical_features tests
# ------------------------------------------------------------------


def test_compute_clinical_output_columns():
    patient = _make_patient_from_values()
    df = compute_clinical_features(patient)
    assert "shock_index" in df.columns
    assert "sirs_count" in df.columns
    assert "sirs_components_available" in df.columns


def _make_patient_from_values():
    rng = np.random.default_rng(42)
    hours = list(range(1, 5))
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
            clinical[var] = [np.nan] * n
    clinical["HR"] = [80.0, 85.0, 90.0, 95.0]
    clinical["SBP"] = [120.0, 115.0, 110.0, 105.0]
    clinical["Temp"] = [37.0, 37.0, 37.0, 37.0]
    clinical["Resp"] = [18.0, 18.0, 18.0, 18.0]
    clinical["WBC"] = [7.0, 7.0, 7.0, 7.0]
    return ProcessedPatient(
        patient_id="p00001",
        hospital_id="set_a",
        hour=hours,
        clinical=clinical,
        sepsis_label=[0] * n,
        source_file="p00001.psv",
        source_dataset="set_a",
        n_rows=n,
    )


def test_compute_clinical_preserves_sepsis_label():
    patient = _make_patient_from_values()
    patient.sepsis_label = [1] * 4
    df = compute_clinical_features(patient)
    assert (df["SepsisLabel"] == 1).all()


# ------------------------------------------------------------------
# validate tests
# ------------------------------------------------------------------


def test_validate_clinical_valid():
    patient = _make_patient_from_values()
    df = compute_clinical_features(patient)
    assert validate_clinical_features(df) == []


def test_validate_clinical_negative_shock_index():
    patient = _make_patient_from_values()
    df = compute_clinical_features(patient)
    df.loc[0, "shock_index"] = -1.0
    issues = validate_clinical_features(df)
    assert any("shock_index" in i for i in issues)
