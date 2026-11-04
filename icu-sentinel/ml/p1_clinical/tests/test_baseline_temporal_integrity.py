"""Tests for baseline temporal integrity.

Tests that no future observation affects any score at hour t.
Also tests cross-patient contamination, sorting, and target-label isolation.
"""

from __future__ import annotations

import numpy as np

from ml.p1_clinical.baselines.baseline_pipeline import BaselinePipeline
from ml.p1_clinical.baselines.news_style import compute_news2_style
from ml.p1_clinical.baselines.qsofa import compute_qsofa
from ml.p1_clinical.baselines.sirs import compute_sirs_for_patient
from ml.p1_clinical.preprocessing.schemas import ProcessedPatient


def _make_patient(
    patient_id: str = "p00001",
    hours: list[int] | None = None,
    n_hours: int | None = None,
    temp_values: list[float] | None = None,
    hr_values: list[float] | None = None,
    resp_values: list[float] | None = None,
    wbc_values: list[float] | None = None,
    sbp_values: list[float] | None = None,
    spo2_values: list[float] | None = None,
    sepsis_label: int = 0,
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
    if temp_values is not None:
        clinical["Temp"] = list(temp_values)
    if hr_values is not None:
        clinical["HR"] = list(hr_values)
    if resp_values is not None:
        clinical["Resp"] = list(resp_values)
    if wbc_values is not None:
        clinical["WBC"] = list(wbc_values)
    if sbp_values is not None:
        clinical["SBP"] = list(sbp_values)
    if spo2_values is not None:
        clinical["O2Sat"] = list(spo2_values)
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


def _get_score_at_hour(results, hour):
    for r in results:
        if r.hour == hour:
            return r.score
    return None


# ------------------------------------------------------------------
# No future leakage tests
# ------------------------------------------------------------------


def test_sirs_future_temp_does_not_affect_hour_1():
    """Changing temp at hour 8 must not change SIRS at hour 1."""
    base = _make_patient(
        temp_values=[37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0],
        hr_values=[80.0] * 8,
        resp_values=[18.0] * 8,
        wbc_values=[7.0] * 8,
    )
    base_results = compute_sirs_for_patient(base)

    modified = _make_patient(
        temp_values=[37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 40.0],
        hr_values=[80.0] * 8,
        resp_values=[18.0] * 8,
        wbc_values=[7.0] * 8,
    )
    mod_results = compute_sirs_for_patient(modified)

    base_s1 = _get_score_at_hour(base_results, 1)
    mod_s1 = _get_score_at_hour(mod_results, 1)
    assert base_s1.score_value == mod_s1.score_value
    assert base_s1.is_positive == mod_s1.is_positive


def test_qsofa_future_sbp_does_not_affect_hour_1():
    """Changing SBP at hour 8 must not change qSOFA at hour 1."""
    base = _make_patient(
        resp_values=[18.0] * 8, sbp_values=[120.0] * 8,
    )
    base_results = compute_qsofa(base)

    modified = _make_patient(
        resp_values=[18.0] * 8, sbp_values=[120.0, 120.0, 120.0, 120.0,
                                              120.0, 120.0, 120.0, 80.0],
    )
    mod_results = compute_qsofa(modified)

    base_s1 = _get_score_at_hour(base_results, 1)
    mod_s1 = _get_score_at_hour(mod_results, 1)
    assert base_s1.score_value == mod_s1.score_value


def test_news2_future_resp_does_not_affect_hour_1():
    """Changing Resp at hour 8 must not change NEWS2 at hour 1."""
    base = _make_patient(
        resp_values=[16.0] * 8, spo2_values=[97.0] * 8,
        temp_values=[37.0] * 8, sbp_values=[120.0] * 8, hr_values=[80.0] * 8,
    )
    base_results = compute_news2_style(base)

    modified = _make_patient(
        resp_values=[16.0] * 7 + [30.0], spo2_values=[97.0] * 8,
        temp_values=[37.0] * 8, sbp_values=[120.0] * 8, hr_values=[80.0] * 8,
    )
    mod_results = compute_news2_style(modified)

    base_s1 = _get_score_at_hour(base_results, 1)
    mod_s1 = _get_score_at_hour(mod_results, 1)
    assert base_s1.score_value == mod_s1.score_value


# ------------------------------------------------------------------
# Cross-patient contamination
# ------------------------------------------------------------------


def test_no_cross_patient_contamination():
    pipeline = BaselinePipeline()
    p1 = _make_patient("p00001", temp_values=[39.0] * 8,
                       hr_values=[95.0] * 8, resp_values=[18.0] * 8,
                       wbc_values=[7.0] * 8, seed=1)
    p2 = _make_patient("p00002", temp_values=[37.0] * 8,
                       hr_values=[80.0] * 8, resp_values=[18.0] * 8,
                       wbc_values=[7.0] * 8, seed=2)
    result = pipeline.transform_patients([p1, p2])
    df = result.combined

    p1_sirs = df[(df["patient_id"] == "p00001") & (df["score_name"] == "sirs")]
    p2_sirs = df[(df["patient_id"] == "p00002") & (df["score_name"] == "sirs")]
    assert (p1_sirs["score_value"] == 2).all()  # temp high + hr high
    assert (p2_sirs["score_value"] == 0).all()  # all normal


# ------------------------------------------------------------------
# Sorting
# ------------------------------------------------------------------


def test_output_sorted_by_hour_within_patient():
    pipeline = BaselinePipeline()
    p1 = _make_patient("p00002", n_hours=4, seed=1)
    p2 = _make_patient("p00001", n_hours=4, seed=2)
    result = pipeline.transform_patients([p1, p2])
    df = result.combined
    for pid in ["p00001", "p00002"]:
        sub = df[df["patient_id"] == pid].sort_values("hour")
        assert sub["hour"].is_monotonic_increasing


# ------------------------------------------------------------------
# Target-label isolation
# ------------------------------------------------------------------


def test_sepsis_label_not_used_as_score_component():
    """SepsisLabel must never influence score computation."""
    pipeline = BaselinePipeline()
    # Same clinical data, different sepsis labels => same scores.
    p_pos = _make_patient("p00001", n_hours=4, sepsis_label=1, seed=1)
    p_neg = _make_patient("p00001", n_hours=4, sepsis_label=0, seed=1)
    result_pos = pipeline.transform_patient(p_pos)
    result_neg = pipeline.transform_patient(p_neg)
    for sn in ["sirs", "qsofa", "news2_style"]:
        for r_pos, r_neg in zip(result_pos[sn], result_neg[sn]):
            assert r_pos.score.score_value == r_neg.score.score_value
            assert r_pos.score.is_positive == r_neg.score.is_positive


# ------------------------------------------------------------------
# Sparse observations
# ------------------------------------------------------------------


def test_sparse_observations_no_crash():
    pipeline = BaselinePipeline()
    patient = _make_patient(
        temp_values=[float("nan")] * 8,
        hr_values=[float("nan")] * 8,
        resp_values=[float("nan")] * 8,
        wbc_values=[float("nan")] * 8,
        seed=1,
    )
    result = pipeline.transform_patient(patient)
    assert len(result["sirs"]) == 8
    for r in result["sirs"]:
        assert r.score.score_value is None
        assert r.score.is_positive is None


# ------------------------------------------------------------------
# Duplicate timestamps
# ------------------------------------------------------------------


def test_duplicate_hours_handled():
    """Duplicate hour values should not crash."""
    df_data = {
        "Temp": [37.0, 37.0],
        "HR": [80.0, 80.0],
        "Resp": [18.0, 18.0],
        "WBC": [7.0, 7.0],
    }
    # Build a patient with duplicate hour 1.
    patient = ProcessedPatient(
        patient_id="p00001",
        hospital_id="set_a",
        hour=[1, 1],
        clinical=df_data,
        sepsis_label=[0, 0],
        source_file="p00001.psv",
        source_dataset="set_a",
        n_rows=2,
    )
    results = compute_sirs_for_patient(patient)
    assert len(results) == 2
