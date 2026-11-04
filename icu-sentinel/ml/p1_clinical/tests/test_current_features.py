"""Tests for current_features.py.

Uses synthetic ProcessedPatient fixtures. No real patient data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.features.current_features import (
    extract_current_features,
    validate_current_features,
)
from ml.p1_clinical.features.feature_config import (
    FEATURE_VARIABLES,
    STATIC_VARIABLES,
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
    source_dataset: str = "set_a",
    seed: int = 42,
) -> ProcessedPatient:
    rng = np.random.default_rng(seed)
    if hours is None:
        hours = list(range(1, 6))
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
        hospital_id=source_dataset,
        hour=list(hours),
        clinical=clinical,
        sepsis_label=[sepsis_label] * n,
        source_file=f"{patient_id}.psv",
        source_dataset=source_dataset,
        n_rows=n,
    )


# ------------------------------------------------------------------
# extract_current_features tests
# ------------------------------------------------------------------


def test_extract_preserves_patient_id_and_dataset():
    patient = _make_patient(patient_id="p00042", source_dataset="set_b")
    df = extract_current_features(patient)
    assert (df["patient_id"] == "p00042").all()
    assert (df["source_dataset"] == "set_b").all()


def test_extract_preserves_hour():
    hours = [1, 2, 3, 4, 5]
    patient = _make_patient(hours=hours)
    df = extract_current_features(patient)
    assert df["hour"].tolist() == hours


def test_extract_preserves_sepsis_label_unchanged():
    patient = _make_patient(sepsis_label=1)
    df = extract_current_features(patient)
    assert (df["SepsisLabel"] == 1).all()


def test_extract_preserves_original_values():
    hr_vals = [80.0, 85.0, np.nan, 90.0, 88.0]
    patient = _make_patient(hr_values=hr_vals)
    df = extract_current_features(patient)
    assert df["HR"].tolist()[:2] == hr_vals[:2]
    assert pd.isna(df["HR"].iloc[2])
    assert df["HR"].tolist()[3:] == hr_vals[3:]


def test_extract_includes_all_feature_variables():
    patient = _make_patient()
    df = extract_current_features(patient)
    for var in FEATURE_VARIABLES:
        assert var in df.columns, f"Missing variable: {var}"


def test_extract_includes_static_variables():
    patient = _make_patient()
    df = extract_current_features(patient)
    for var in STATIC_VARIABLES:
        assert var in df.columns, f"Missing static variable: {var}"


def test_extract_output_column_order():
    patient = _make_patient()
    df = extract_current_features(patient)
    expected_start = ["patient_id", "source_dataset", "hour"]
    expected_end = ["SepsisLabel"]
    assert list(df.columns[:3]) == expected_start
    assert df.columns[-1] == expected_end[0]


def test_extract_no_imputation():
    patient = _make_patient(hr_values=[80.0, np.nan, 85.0])
    df = extract_current_features(patient)
    assert pd.isna(df["HR"].iloc[1])


def test_extract_multiple_patients():
    p1 = _make_patient("p00001", hours=[1, 2], seed=1)
    p2 = _make_patient("p00002", hours=[1, 2, 3], seed=2)
    df1 = extract_current_features(p1)
    df2 = extract_current_features(p2)
    assert len(df1) == 2
    assert len(df2) == 3
    assert (df1["patient_id"] == "p00001").all()
    assert (df2["patient_id"] == "p00002").all()


# ------------------------------------------------------------------
# validate_current_features tests
# ------------------------------------------------------------------


def test_validate_valid():
    patient = _make_patient()
    df = extract_current_features(patient)
    assert validate_current_features(df) == []


def test_validate_missing_required_column():
    patient = _make_patient()
    df = extract_current_features(patient)
    df = df.drop(columns=["SepsisLabel"])
    issues = validate_current_features(df)
    assert "SepsisLabel" in str(issues)


def test_validate_invalid_sepsis_label():
    patient = _make_patient(sepsis_label=0)
    df = extract_current_features(patient)
    df.loc[0, "SepsisLabel"] = 5
    issues = validate_current_features(df)
    assert any("invalid SepsisLabel" in i for i in issues)


def test_validate_empty():
    patient = _make_patient(hours=[])
    df = extract_current_features(patient)
    issues = validate_current_features(df)
    assert any("Empty" in i for i in issues)
