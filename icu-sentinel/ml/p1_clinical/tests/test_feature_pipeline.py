"""Tests for feature_pipeline.py.

Uses synthetic ProcessedPatient fixtures. No real patient data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.features.feature_config import FeatureConfig
from ml.p1_clinical.features.feature_pipeline import (
    FeaturePipeline,
    run_pipeline_on_processed_patients,
)
from ml.p1_clinical.features.feature_schema import build_default_schema
from ml.p1_clinical.preprocessing.schemas import ProcessedPatient


def _make_patient(
    patient_id: str = "p00001",
    n_hours: int = 8,
    sepsis_label: int = 0,
    source_dataset: str = "set_a",
    hr_values: list[float] | None = None,
    seed: int = 42,
) -> ProcessedPatient:
    rng = np.random.default_rng(seed)
    hours = list(range(1, n_hours + 1))
    n = len(hours)
    clinical: dict = {}
    for var in [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine",
        "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
        "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets",
        "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
    ]:
        if var == "ICULOS":
            clinical[var] = [float(h) for h in hours]
        elif var == "HospAdmTime":
            clinical[var] = [float(-50)] * n
        elif var in ("Age", "Gender", "Unit1", "Unit2"):
            clinical[var] = [float(rng.integers(0, 2))] * n
        else:
            clinical[var] = rng.normal(0, 1, size=n).tolist()
    if hr_values is not None:
        clinical["HR"] = list(hr_values)
    return ProcessedPatient(
        patient_id=patient_id,
        hospital_id=source_dataset,
        hour=hours,
        clinical=clinical,
        sepsis_label=[sepsis_label] * n,
        source_file=f"{patient_id}.psv",
        source_dataset=source_dataset,
        n_rows=n,
    )


# ------------------------------------------------------------------
# pipeline tests
# ------------------------------------------------------------------


def test_pipeline_transform_single_patient():
    patient = _make_patient(n_hours=6, hr_values=[80.0, 85.0, 90.0, 95.0, 100.0, 105.0])
    pipeline = FeaturePipeline()
    df = pipeline.transform_patient(patient)
    assert len(df) == 6
    assert "patient_id" in df.columns
    assert "source_dataset" in df.columns
    assert "hour" in df.columns
    assert "SepsisLabel" in df.columns
    assert (df["SepsisLabel"] == 0).all()


def test_pipeline_preserves_sepsis_label():
    patient = _make_patient(sepsis_label=1)
    pipeline = FeaturePipeline()
    df = pipeline.transform_patient(patient)
    assert (df["SepsisLabel"] == 1).all()


def test_pipeline_has_feature_columns():
    patient = _make_patient()
    pipeline = FeaturePipeline()
    df = pipeline.transform_patient(patient)
    id_cols = {"patient_id", "source_dataset", "hour", "SepsisLabel"}
    feature_cols = [c for c in df.columns if c not in id_cols]
    assert len(feature_cols) > 0


def test_pipeline_multiple_patients():
    p1 = _make_patient("p00001", n_hours=4, seed=1)
    p2 = _make_patient("p00002", n_hours=6, seed=2)
    pipeline = FeaturePipeline()
    result = pipeline.transform_patients([p1, p2])
    assert result.n_patients == 2
    assert result.n_rows == 10
    assert result.combined is not None


def test_pipeline_no_patient_contamination():
    p1 = _make_patient("p00001", n_hours=4, hr_values=[80.0, 80.0, 80.0, 80.0], seed=1)
    p2 = _make_patient("p00002", n_hours=4, hr_values=[140.0, 140.0, 140.0, 140.0], seed=2)
    pipeline = FeaturePipeline()
    result = pipeline.transform_patients([p1, p2])
    p1_rows = result.combined[result.combined["patient_id"] == "p00001"]
    p2_rows = result.combined[result.combined["patient_id"] == "p00002"]
    assert (p1_rows["HR"] == 80.0).all()
    assert (p2_rows["HR"] == 140.0).all()


def test_pipeline_output_sorted_by_hour():
    patient = _make_patient(n_hours=6)
    pipeline = FeaturePipeline()
    df = pipeline.transform_patient(patient)
    assert df["hour"].is_monotonic_increasing


def test_pipeline_save_and_load_config(tmp_path):
    pipeline = FeaturePipeline()
    config_path = tmp_path / "feature_config.json"
    pipeline.save_config(config_path)
    loaded = FeatureConfig.load(config_path)
    assert loaded.rolling_window_hours == pipeline.config.rolling_window_hours


def test_pipeline_save_and_load_schema(tmp_path):
    pipeline = FeaturePipeline()
    schema_path = tmp_path / "feature_schema.json"
    pipeline.save_schema(schema_path)
    build_default_schema.__wrapped__ if hasattr(build_default_schema, "__wrapped__") else None
    from ml.p1_clinical.features.feature_schema import FeatureSchema
    loaded_schema = FeatureSchema.load(schema_path)
    assert loaded_schema.version == pipeline.schema.version
    assert len(loaded_schema.features) == len(pipeline.schema.features)


def test_pipeline_reproducible():
    patient = _make_patient()
    p1 = FeaturePipeline(config=FeatureConfig(random_state=42) if False else FeatureConfig())
    df1 = p1.transform_patient(patient)
    p2 = FeaturePipeline(config=FeatureConfig())
    df2 = p2.transform_patient(patient)
    pd.testing.assert_frame_equal(df1, df2)


# ------------------------------------------------------------------
# convenience function tests
# ------------------------------------------------------------------


def test_run_pipeline_on_processed_patients(tmp_path):
    p1 = _make_patient("p00001", n_hours=4, seed=1)
    result = run_pipeline_on_processed_patients([p1], output_dir=tmp_path)
    assert result.n_patients == 1
    assert (tmp_path / "feature_config.json").exists()
    assert (tmp_path / "feature_schema.json").exists()
    assert (tmp_path / "combined_features.parquet").exists()


# ------------------------------------------------------------------
# schema integration tests
# ------------------------------------------------------------------


def test_schema_has_all_feature_categories():
    schema = build_default_schema()
    cats = {f.category for f in schema.features}
    assert "current" in cats
    assert "missingness" in cats
    assert "rolling" in cats
    assert "trend" in cats
    assert "clinical" in cats
    assert "identifier" in cats
    assert "target" in cats


def test_schema_target_not_tabular():
    schema = build_default_schema()
    target = schema.get_feature("SepsisLabel")
    assert target is not None
    assert target.tabular_appropriate is False
    assert target.sequential_appropriate is False


def test_schema_all_features_causal():
    schema = build_default_schema()
    non_causal = [f.feature_name for f in schema.features if not f.causal]
    assert len(non_causal) == 0, f"Non-causal features: {non_causal}"
