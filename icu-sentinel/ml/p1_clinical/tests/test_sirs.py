"""Tests for SIRS clinical baseline.

Tests threshold boundaries, missing components, partial/complete scores,
and temporal integrity. Uses synthetic fixtures only.
"""

from __future__ import annotations

import numpy as np

from ml.p1_clinical.baselines.score_completeness import (
    CompletenessLevel,
    is_score_complete,
    is_score_none,
    is_score_partial,
)
from ml.p1_clinical.baselines.sirs import (
    compute_sirs_for_patient,
    validate_sirs_results,
)
from ml.p1_clinical.preprocessing.schemas import ProcessedPatient


def _make_patient(
    patient_id: str = "p00001",
    hours: list[int] | None = None,
    temp_values: list[float] | None = None,
    hr_values: list[float] | None = None,
    resp_values: list[float] | None = None,
    wbc_values: list[float] | None = None,
    sepsis_label: int = 0,
    seed: int = 42,
) -> ProcessedPatient:
    rng = np.random.default_rng(seed)
    if hours is None:
        hours = list(range(1, 6))
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
    if temp_values is not None:
        clinical["Temp"] = list(temp_values)
    if hr_values is not None:
        clinical["HR"] = list(hr_values)
    if resp_values is not None:
        clinical["Resp"] = list(resp_values)
    if wbc_values is not None:
        clinical["WBC"] = list(wbc_values)
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
# SIRS threshold boundary tests
# ------------------------------------------------------------------


def test_sirs_temp_high_boundary():
    """Temp > 38.3 is positive; 38.3 exactly is NOT positive (> not >=)."""
    patient = _make_patient(
        hours=[1], temp_values=[38.3], hr_values=[80.0],
        resp_values=[18.0], wbc_values=[7.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    temp_comp = next(c for c in r.components if c.component_name == "temperature")
    assert temp_comp.met is False  # 38.3 is not > 38.3


def test_sirs_temp_just_above_high():
    """Temp 38.4 > 38.3 is positive."""
    patient = _make_patient(
        hours=[1], temp_values=[38.4], hr_values=[80.0],
        resp_values=[18.0], wbc_values=[7.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    temp_comp = next(c for c in r.components if c.component_name == "temperature")
    assert temp_comp.met is True


def test_sirs_temp_low_boundary():
    """Temp < 36.0 is positive; 36.0 exactly is NOT positive."""
    patient = _make_patient(
        hours=[1], temp_values=[36.0], hr_values=[80.0],
        resp_values=[18.0], wbc_values=[7.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    temp_comp = next(c for c in r.components if c.component_name == "temperature")
    assert temp_comp.met is False


def test_sirs_temp_just_below_low():
    """Temp 35.9 < 36.0 is positive."""
    patient = _make_patient(
        hours=[1], temp_values=[35.9], hr_values=[80.0],
        resp_values=[18.0], wbc_values=[7.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    temp_comp = next(c for c in r.components if c.component_name == "temperature")
    assert temp_comp.met is True


def test_sirs_hr_boundary():
    """HR > 90 is positive; 90 exactly is NOT positive."""
    patient = _make_patient(
        hours=[1], temp_values=[37.0], hr_values=[90.0],
        resp_values=[18.0], wbc_values=[7.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    hr_comp = next(c for c in r.components if c.component_name == "heart_rate")
    assert hr_comp.met is False


def test_sirs_hr_just_above():
    """HR 90.1 > 90 is positive."""
    patient = _make_patient(
        hours=[1], temp_values=[37.0], hr_values=[90.1],
        resp_values=[18.0], wbc_values=[7.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    hr_comp = next(c for c in r.components if c.component_name == "heart_rate")
    assert hr_comp.met is True


def test_sirs_resp_boundary():
    """Resp > 20 is positive; 20 exactly is NOT positive."""
    patient = _make_patient(
        hours=[1], temp_values=[37.0], hr_values=[80.0],
        resp_values=[20.0], wbc_values=[7.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    resp_comp = next(c for c in r.components if c.component_name == "respiration")
    assert resp_comp.met is False


def test_sirs_resp_just_above():
    """Resp 20.1 > 20 is positive."""
    patient = _make_patient(
        hours=[1], temp_values=[37.0], hr_values=[80.0],
        resp_values=[20.1], wbc_values=[7.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    resp_comp = next(c for c in r.components if c.component_name == "respiration")
    assert resp_comp.met is True


def test_sirs_wbc_high_boundary():
    """WBC > 12 is positive; 12 exactly is NOT positive."""
    patient = _make_patient(
        hours=[1], temp_values=[37.0], hr_values=[80.0],
        resp_values=[18.0], wbc_values=[12.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    wbc_comp = next(c for c in r.components if c.component_name == "wbc")
    assert wbc_comp.met is False


def test_sirs_wbc_low_boundary():
    """WBC < 4 is positive; 4 exactly is NOT positive."""
    patient = _make_patient(
        hours=[1], temp_values=[37.0], hr_values=[80.0],
        resp_values=[18.0], wbc_values=[4.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    wbc_comp = next(c for c in r.components if c.component_name == "wbc")
    assert wbc_comp.met is False


# ------------------------------------------------------------------
# Complete vs partial score tests
# ------------------------------------------------------------------


def test_sirs_complete_all_normal():
    """All 4 components available, all normal => score 0, not positive."""
    patient = _make_patient(
        hours=[1], temp_values=[37.0], hr_values=[80.0],
        resp_values=[18.0], wbc_values=[7.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    assert r.score.complete is True
    assert r.score.score_value == 0
    assert r.score.is_positive is False


def test_sirs_complete_positive():
    """2 criteria met, all 4 available => score 2, positive."""
    patient = _make_patient(
        hours=[1], temp_values=[39.0], hr_values=[95.0],
        resp_values=[18.0], wbc_values=[7.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    assert r.score.complete is True
    assert r.score.score_value == 2
    assert r.score.is_positive is True


def test_sirs_partial_missing_component():
    """One component missing => partial score, is_positive None."""
    patient = _make_patient(
        hours=[1], temp_values=[39.0], hr_values=[95.0],
        resp_values=[float("nan")], wbc_values=[7.0],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    assert r.score.complete is False
    assert r.score.components_available == 3
    assert r.score.is_positive is None  # partial, not labeled positive


def test_sirs_none_all_missing():
    """All components missing => score None, completeness NONE."""
    patient = _make_patient(
        hours=[1], temp_values=[float("nan")], hr_values=[float("nan")],
        resp_values=[float("nan")], wbc_values=[float("nan")],
    )
    results = compute_sirs_for_patient(patient)
    r = results[0]
    assert r.score.complete is False
    assert r.score.components_available == 0
    assert r.score.score_value is None
    assert r.score.is_positive is None


# ------------------------------------------------------------------
# Completeness helper tests
# ------------------------------------------------------------------


def test_completeness_levels():
    from ml.p1_clinical.baselines.score_completeness import (
        ComponentAvailability,
        assess_completeness,
    )
    comps = [
        ComponentAvailability("temp", "Temp", True, 37.0),
        ComponentAvailability("hr", "HR", True, 80.0),
        ComponentAvailability("resp", "Resp", True, 18.0),
        ComponentAvailability("wbc", "WBC", True, 7.0),
    ]
    c = assess_completeness("sirs", comps, 4)
    assert c.level == CompletenessLevel.COMPLETE
    assert is_score_complete(c)

    comps_partial = [
        ComponentAvailability("temp", "Temp", True, 37.0),
        ComponentAvailability("hr", "HR", False, None),
        ComponentAvailability("resp", "Resp", True, 18.0),
        ComponentAvailability("wbc", "WBC", True, 7.0),
    ]
    c2 = assess_completeness("sirs", comps_partial, 4)
    assert c2.level == CompletenessLevel.PARTIAL
    assert is_score_partial(c2)

    comps_none = [
        ComponentAvailability("temp", "Temp", False, None),
        ComponentAvailability("hr", "HR", False, None),
        ComponentAvailability("resp", "Resp", False, None),
        ComponentAvailability("wbc", "WBC", False, None),
    ]
    c3 = assess_completeness("sirs", comps_none, 4)
    assert c3.level == CompletenessLevel.NONE
    assert is_score_none(c3)


# ------------------------------------------------------------------
# SepsisLabel isolation
# ------------------------------------------------------------------


def test_sirs_preserves_sepsis_label():
    patient = _make_patient(sepsis_label=1)
    results = compute_sirs_for_patient(patient)
    assert all(r.sepsis_label == 1 for r in results)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def test_validate_sirs_valid():
    patient = _make_patient()
    results = compute_sirs_for_patient(patient)
    assert validate_sirs_results(results) == []
