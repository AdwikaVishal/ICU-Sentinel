"""Tests for NEWS-style clinical baseline.

Tests NEWS2 parameter scoring, missing components (supplemental O2 and
consciousness NOT in dataset), partial/complete scores, and temporal
integrity. Uses synthetic fixtures only.
"""

from __future__ import annotations

import numpy as np

from ml.p1_clinical.baselines.news_style import (
    compute_news2_style,
    validate_news2_results,
)
from ml.p1_clinical.baselines.score_completeness import (
    CompletenessLevel,
    is_score_partial,
)
from ml.p1_clinical.preprocessing.schemas import ProcessedPatient


def _make_patient(
    patient_id: str = "p00001",
    hours: list[int] | None = None,
    resp_values: list[float] | None = None,
    spo2_values: list[float] | None = None,
    temp_values: list[float] | None = None,
    sbp_values: list[float] | None = None,
    hr_values: list[float] | None = None,
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
    if resp_values is not None:
        clinical["Resp"] = list(resp_values)
    if spo2_values is not None:
        clinical["O2Sat"] = list(spo2_values)
    if temp_values is not None:
        clinical["Temp"] = list(temp_values)
    if sbp_values is not None:
        clinical["SBP"] = list(sbp_values)
    if hr_values is not None:
        clinical["HR"] = list(hr_values)
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
# NEWS2 parameter scoring tests
# ------------------------------------------------------------------


def test_news2_rr_normal_range_score_zero():
    """RR 15 (12-20 range) => score 0."""
    patient = _make_patient(
        hours=[1], resp_values=[15.0], spo2_values=[97.0],
        temp_values=[37.0], sbp_values=[120.0], hr_values=[80.0],
    )
    results = compute_news2_style(patient)
    r = results[0]
    resp_comp = next(c for c in r.components if c.component_name == "respiration")
    assert resp_comp.met is False  # score 0 means not met


def test_news2_rr_high_score():
    """RR 26 (>=25) => score 3."""
    patient = _make_patient(
        hours=[1], resp_values=[26.0], spo2_values=[97.0],
        temp_values=[37.0], sbp_values=[120.0], hr_values=[80.0],
    )
    results = compute_news2_style(patient)
    r = results[0]
    resp_comp = next(c for c in r.components if c.component_name == "respiration")
    assert resp_comp.met is True  # score 3 > 0


def test_news2_spo2_low_score():
    """SpO2 90 (<=91) => score 3."""
    patient = _make_patient(
        hours=[1], resp_values=[16.0], spo2_values=[90.0],
        temp_values=[37.0], sbp_values=[120.0], hr_values=[80.0],
    )
    results = compute_news2_style(patient)
    r = results[0]
    spo2_comp = next(c for c in r.components if c.component_name == "spo2")
    assert spo2_comp.met is True


def test_news2_temp_high_score():
    """Temp 39.5 (>=39.1) => score 2."""
    patient = _make_patient(
        hours=[1], resp_values=[16.0], spo2_values=[97.0],
        temp_values=[39.5], sbp_values=[120.0], hr_values=[80.0],
    )
    results = compute_news2_style(patient)
    r = results[0]
    temp_comp = next(c for c in r.components if c.component_name == "temperature")
    assert temp_comp.met is True  # score 2 > 0


def test_news2_sbp_low_score():
    """SBP 85 (<=90) => score 3."""
    patient = _make_patient(
        hours=[1], resp_values=[16.0], spo2_values=[97.0],
        temp_values=[37.0], sbp_values=[85.0], hr_values=[80.0],
    )
    results = compute_news2_style(patient)
    r = results[0]
    sbp_comp = next(c for c in r.components if c.component_name == "systolic_bp")
    assert sbp_comp.met is True


def test_news2_hr_high_score():
    """HR 145 (>=141) => score 3."""
    patient = _make_patient(
        hours=[1], resp_values=[16.0], spo2_values=[97.0],
        temp_values=[37.0], sbp_values=[120.0], hr_values=[145.0],
    )
    results = compute_news2_style(patient)
    r = results[0]
    hr_comp = next(c for c in r.components if c.component_name == "heart_rate")
    assert hr_comp.met is True


# ------------------------------------------------------------------
# Missing components NOT available tests
# ------------------------------------------------------------------


def test_news2_supplemental_oxygen_not_available():
    """Supplemental oxygen component is always unavailable."""
    patient = _make_patient(
        hours=[1], resp_values=[16.0], spo2_values=[97.0],
        temp_values=[37.0], sbp_values=[120.0], hr_values=[80.0],
    )
    results = compute_news2_style(patient)
    r = results[0]
    o2_comp = next(c for c in r.components if c.component_name == "supplemental_oxygen")
    assert o2_comp.available is False
    assert o2_comp.met is None
    assert o2_comp.variable is None


def test_news2_consciousness_not_available():
    """Consciousness component is always unavailable."""
    patient = _make_patient(
        hours=[1], resp_values=[16.0], spo2_values=[97.0],
        temp_values=[37.0], sbp_values=[120.0], hr_values=[80.0],
    )
    results = compute_news2_style(patient)
    r = results[0]
    conscious_comp = next(c for c in r.components if c.component_name == "consciousness")
    assert conscious_comp.available is False
    assert conscious_comp.met is None
    assert conscious_comp.variable is None


# ------------------------------------------------------------------
# Complete vs partial score tests
# ------------------------------------------------------------------


def test_news2_partial_due_to_missing_components():
    """Missing supplemental O2 and consciousness => partial, is_positive None."""
    patient = _make_patient(
        hours=[1], resp_values=[26.0], spo2_values=[90.0],
        temp_values=[39.5], sbp_values=[85.0], hr_values=[145.0],
    )
    results = compute_news2_style(patient)
    r = results[0]
    assert r.score.complete is False
    assert r.score.components_available == 5
    assert r.score.components_required == 7
    assert r.score.is_positive is None  # partial due to missing O2 + consciousness


def test_news2_none_all_missing():
    """All components missing => score None."""
    patient = _make_patient(
        hours=[1], resp_values=[float("nan")], spo2_values=[float("nan")],
        temp_values=[float("nan")], sbp_values=[float("nan")],
        hr_values=[float("nan")],
    )
    results = compute_news2_style(patient)
    r = results[0]
    assert r.score.complete is False
    assert r.score.components_available == 0
    assert r.score.score_value is None
    assert r.score.is_positive is None


# ------------------------------------------------------------------
# SepsisLabel isolation
# ------------------------------------------------------------------


def test_news2_preserves_sepsis_label():
    patient = _make_patient(sepsis_label=1)
    results = compute_news2_style(patient)
    assert all(r.sepsis_label == 1 for r in results)


# ------------------------------------------------------------------
# Completeness helper tests
# ------------------------------------------------------------------


def test_news2_completeness_partial():
    from ml.p1_clinical.baselines.score_completeness import (
        ComponentAvailability,
        assess_completeness,
    )
    comps = [
        ComponentAvailability("resp", "Resp", True, 26.0),
        ComponentAvailability("spo2", "O2Sat", True, 90.0),
        ComponentAvailability("o2", None, False, None),
        ComponentAvailability("temp", "Temp", True, 39.5),
        ComponentAvailability("sbp", "SBP", True, 85.0),
        ComponentAvailability("hr", "HR", True, 145.0),
        ComponentAvailability("conscious", None, False, None),
    ]
    c = assess_completeness("news2_style", comps, 7)
    assert c.level == CompletenessLevel.PARTIAL
    assert is_score_partial(c)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def test_validate_news2_valid():
    patient = _make_patient()
    results = compute_news2_style(patient)
    assert validate_news2_results(results) == []
