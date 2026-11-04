"""Tests for baseline_pipeline module.

Tests end-to-end pipeline: multiple patients, multiple scores,
combined DataFrame, config/schema serialization.
"""

from __future__ import annotations

import numpy as np

from ml.p1_clinical.baselines.baseline_pipeline import (
    BaselinePipeline,
    run_baseline_pipeline,
)
from ml.p1_clinical.baselines.baseline_schema import build_default_schema
from ml.p1_clinical.baselines.clinical_score_config import ClinicalScoreConfig
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
    n_hours: int = 4,
    sepsis_label: int = 0,
    source_dataset: str = "set_a",
    seed: int = 42,
) -> ProcessedPatient:
    rng = np.random.default_rng(seed)
    hours = list(range(1, n_hours + 1))
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
            clinical[var] = [float(-50)] * n_hours
        elif var in ("Age", "Gender", "Unit1", "Unit2"):
            clinical[var] = [float(rng.integers(0, 2))] * n_hours
        else:
            clinical[var] = rng.normal(0, 1, size=n_hours).tolist()
    # Make some SIRS-positive hours
    clinical["Temp"] = _expand_values([39.0, 37.0, 39.0, 37.0], n_hours, 37.0)
    clinical["HR"] = _expand_values([95.0, 80.0, 95.0, 80.0], n_hours, 80.0)
    clinical["Resp"] = _expand_values([18.0, 18.0, 18.0, 18.0], n_hours, 18.0)
    clinical["WBC"] = _expand_values([7.0, 7.0, 7.0, 7.0], n_hours, 7.0)
    return ProcessedPatient(
        patient_id=patient_id,
        hospital_id=source_dataset,
        hour=hours,
        clinical=clinical,
        sepsis_label=[sepsis_label] * n_hours,
        source_file=f"{patient_id}.psv",
        source_dataset=source_dataset,
        n_rows=n_hours,
    )


# ------------------------------------------------------------------
# Pipeline tests
# ------------------------------------------------------------------


def test_pipeline_transform_single_patient():
    patient = _make_patient("p00001", n_hours=4, seed=1)
    pipeline = BaselinePipeline()
    result = pipeline.transform_patient(patient)
    assert "sirs" in result
    assert "qsofa" in result
    assert "news2_style" in result
    assert len(result["sirs"]) == 4
    assert len(result["qsofa"]) == 4
    assert len(result["news2_style"]) == 4


def test_pipeline_multiple_patients():
    p1 = _make_patient("p00001", n_hours=4, seed=1)
    p2 = _make_patient("p00002", n_hours=6, seed=2)
    pipeline = BaselinePipeline()
    result = pipeline.transform_patients([p1, p2])
    assert result.n_patients == 2
    assert result.n_rows == 4 * 3 + 6 * 3  # 3 scores per hour


def test_pipeline_combined_dataframe():
    p1 = _make_patient("p00001", n_hours=4, seed=1)
    pipeline = BaselinePipeline()
    result = pipeline.transform_patients([p1])
    df = result.combined
    assert df is not None
    assert len(df) == 12  # 4 hours x 3 scores
    assert "patient_id" in df.columns
    assert "hour" in df.columns
    assert "sepsis_label" in df.columns
    assert "score_name" in df.columns
    assert "score_value" in df.columns
    assert "is_positive" in df.columns
    assert "score_complete" in df.columns


def test_pipeline_preserves_sepsis_label():
    p1 = _make_patient("p00001", n_hours=4, sepsis_label=1, seed=1)
    pipeline = BaselinePipeline()
    result = pipeline.transform_patients([p1])
    df = result.combined
    assert (df["sepsis_label"] == 1).all()


def test_pipeline_no_score_column_leakage():
    """SepsisLabel must not appear as a score component value."""
    p1 = _make_patient("p00001", n_hours=4, seed=1)
    pipeline = BaselinePipeline()
    result = pipeline.transform_patients([p1])
    df = result.combined
    # SepsisLabel column should be separate, not mixed into components.
    assert "sepsis_label" in df.columns
    # Score values should be small integers (SIRS 0-4, qSOFA 0-3, NEWS2 0-21)
    for sn in df["score_name"].unique():
        sn_df = df[df["score_name"] == sn]
        vals = sn_df["score_value"].dropna()
        if len(vals) > 0:
            assert (vals >= 0).all()


# ------------------------------------------------------------------
# Config/schema serialization
# ------------------------------------------------------------------


def test_config_save_load(tmp_path):
    config = ClinicalScoreConfig()
    path = tmp_path / "config.json"
    config.save(path)
    loaded = ClinicalScoreConfig.load(path)
    assert loaded.sirs_positive_threshold == config.sirs_positive_threshold
    assert loaded.qsofa_positive_threshold == config.qsofa_positive_threshold
    assert loaded.news2_positive_threshold == config.news2_positive_threshold


def test_schema_save_load(tmp_path):
    schema = build_default_schema()
    path = tmp_path / "schema.json"
    schema.save(path)
    from ml.p1_clinical.baselines.baseline_schema import BaselineSchema
    loaded = BaselineSchema.load(path)
    assert loaded.version == schema.version
    assert len(loaded.scores) == len(schema.scores)
    assert loaded.get_score("sirs") is not None
    assert loaded.get_score("qsofa") is not None
    assert loaded.get_score("news2_style") is not None


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------


def test_run_baseline_pipeline(tmp_path):
    p1 = _make_patient("p00001", n_hours=4, seed=1)
    result = run_baseline_pipeline([p1], output_dir=tmp_path)
    assert result.n_patients == 1
    assert (tmp_path / "clinical_score_config.json").exists()
    assert (tmp_path / "baseline_schema.json").exists()
    assert (tmp_path / "combined_baselines.parquet").exists()


# ------------------------------------------------------------------
# Multiple hospitals
# ------------------------------------------------------------------


def test_pipeline_multiple_hospitals():
    p1 = _make_patient("p00001", n_hours=4, source_dataset="set_a", seed=1)
    p2 = _make_patient("p00002", n_hours=4, source_dataset="set_b", seed=2)
    pipeline = BaselinePipeline()
    result = pipeline.transform_patients([p1, p2])
    df = result.combined
    set_a_rows = df[df["source_dataset"] == "set_a"]
    set_b_rows = df[df["source_dataset"] == "set_b"]
    assert len(set_a_rows) > 0
    assert len(set_b_rows) > 0
