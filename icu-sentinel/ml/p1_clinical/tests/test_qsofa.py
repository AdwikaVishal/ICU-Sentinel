"""Tests for qSOFA clinical baseline.

Tests threshold boundaries, missing components (especially altered mental
status which is NOT in the dataset), partial/complete scores, and
temporal integrity. Uses synthetic fixtures only.
"""

from __future__ import annotations

import numpy as np

from ml.p1_clinical.baselines.qsofa import (
    compute_qsofa,
    validate_qsofa_results,
)
from ml.p1_clinical.baselines.score_completeness import (
    CompletenessLevel,
    is_score_complete,
    is_score_partial,
)
from ml.p1_clinical.preprocessing.schemas import ProcessedPatient


def _make_patient(
    patient_id: str = "p00001",
    hours: list[int] | None = None,
    resp_values: list[float] | None = None,
    sbp_values: list[float] | None = None,
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
    if sbp_values is not None:
        clinical["SBP"] = list(sbp_values)
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
# qSOFA threshold boundary tests
# ------------------------------------------------------------------


def test_qsofa_resp_boundary():
    """RR >= 22 is positive; 21.9 is NOT positive."""
    patient = _make_patient(
        hours=[1], resp_values=[22.0], sbp_values=[120.0],
    )
    results = compute_qsofa(patient)
    r = results[0]
    resp_comp = next(c for c in r.components if c.component_name == "respiration")
    assert resp_comp.met is True


def test_qsofa_resp_just_below():
    """RR 21.9 < 22 is NOT positive."""
    patient = _make_patient(
        hours=[1], resp_values=[21.9], sbp_values=[120.0],
    )
    results = compute_qsofa(patient)
    r = results[0]
    resp_comp = next(c for c in r.components if c.component_name == "respiration")
    assert resp_comp.met is False


def test_qsofa_sbp_boundary():
    """SBP <= 100 is positive; 100 exactly IS positive."""
    patient = _make_patient(
        hours=[1], resp_values=[18.0], sbp_values=[100.0],
    )
    results = compute_qsofa(patient)
    r = results[0]
    sbp_comp = next(c for c in r.components if c.component_name == "systolic_bp")
    assert sbp_comp.met is True


def test_qsofa_sbp_just_above():
    """SBP 100.1 > 100 is NOT positive."""
    patient = _make_patient(
        hours=[1], resp_values=[18.0], sbp_values=[100.1],
    )
    results = compute_qsofa(patient)
    r = results[0]
    sbp_comp = next(c for c in r.components if c.component_name == "systolic_bp")
    assert sbp_comp.met is False


# ------------------------------------------------------------------
# Altered mental status NOT available test
# ------------------------------------------------------------------


def test_qsofa_ams_not_available():
    """Altered mental status component is always unavailable."""
    patient = _make_patient(
        hours=[1], resp_values=[22.0], sbp_values=[90.0],
    )
    results = compute_qsofa(patient)
    r = results[0]
    ams_comp = next(c for c in r.components if c.component_name == "altered_mental_status")
    assert ams_comp.available is False
    assert ams_comp.met is None
    assert ams_comp.variable is None  # no dataset variable for AMS


# ------------------------------------------------------------------
# Complete vs partial score tests
# ------------------------------------------------------------------


def test_qsofa_partial_due_to_ams():
    """RR and SBP available but AMS missing => partial, is_positive None."""
    patient = _make_patient(
        hours=[1], resp_values=[24.0], sbp_values=[90.0],
    )
    results = compute_qsofa(patient)
    r = results[0]
    assert r.score.complete is False
    assert r.score.components_available == 2
    assert r.score.components_required == 3
    assert r.score.is_positive is None  # partial due to missing AMS


def test_qsofa_none_all_missing():
    """All components missing => score None."""
    patient = _make_patient(
        hours=[1], resp_values=[float("nan")], sbp_values=[float("nan")],
    )
    results = compute_qsofa(patient)
    r = results[0]
    assert r.score.complete is False
    assert r.score.components_available == 0
    assert r.score.score_value is None
    assert r.score.is_positive is None


def test_qsofa_complete_positive():
    """2 criteria met (RR + SBP), but AMS missing => still partial."""
    # Even with RR and SBP positive, AMS is missing so complete=False.
    patient = _make_patient(
        hours=[1], resp_values=[24.0], sbp_values=[90.0],
    )
    results = compute_qsofa(patient)
    r = results[0]
    assert r.score.complete is False
    assert r.score.score_value == 2  # RR + SBP met
    assert r.score.is_positive is None  # partial because AMS missing


# ------------------------------------------------------------------
# Completeness helper tests
# ------------------------------------------------------------------


def test_qsofa_completeness_partial():
    from ml.p1_clinical.baselines.score_completeness import (
        ComponentAvailability,
        assess_completeness,
    )
    comps = [
        ComponentAvailability("resp", "Resp", True, 24.0),
        ComponentAvailability("sbp", "SBP", True, 90.0),
        ComponentAvailability("ams", None, False, None),
    ]
    c = assess_completeness("qsofa", comps, 3)
    assert c.level == CompletenessLevel.PARTIAL
    assert is_score_partial(c)
    assert not is_score_complete(c)


# ------------------------------------------------------------------
# SepsisLabel isolation
# ------------------------------------------------------------------


def test_qsofa_preserves_sepsis_label():
    patient = _make_patient(sepsis_label=1)
    results = compute_qsofa(patient)
    assert all(r.sepsis_label == 1 for r in results)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def test_validate_qsofa_valid():
    patient = _make_patient()
    results = compute_qsofa(patient)
    assert validate_qsofa_results(results) == []
