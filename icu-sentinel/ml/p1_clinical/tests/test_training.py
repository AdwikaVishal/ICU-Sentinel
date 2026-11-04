"""Tests for training.py.

Tests the end-to-end training protocol, split usage, target isolation,
feature selection, and model training mechanics.
Uses synthetic fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.classical_ml.data_loader import (
    load_split_aware_dataset,
)
from ml.p1_clinical.classical_ml.model_config import ModelConfig
from ml.p1_clinical.classical_ml.training import (
    evaluate_on_split,
    run_training_protocol,
)
from ml.p1_clinical.preprocessing.patient_splits import (
    PatientSplitEntry,
    SplitManifest,
)

# ------------------------------------------------------------------
# Small manifest helper
# ------------------------------------------------------------------


def _make_small_manifest(tmp_path, n_patients=6, seed=42) -> SplitManifest:
    rng = np.random.default_rng(seed)
    entries = []
    for i in range(n_patients):
        pid = f"p{i:06d}"
        n_hours = int(rng.integers(5, 12))
        n_pos = int(rng.integers(0, n_hours + 1))
        has_sepsis = n_pos > 0
        if i < 3:
            split = "train"
        elif i < 5:
            split = "validation"
        else:
            split = "test"
        entries.append(
            PatientSplitEntry(
                patient_id=pid,
                hospital_id="set_a" if i % 2 == 0 else "set_b",
                split=split,
                n_records=n_hours,
                n_sepsis_positive_hours=n_pos,
                n_sepsis_negative_hours=n_hours - n_pos,
                has_sepsis=has_sepsis,
                source_file=f"{pid}.parquet",
                source_dataset="set_a" if i % 2 == 0 else "set_b",
            )
        )
    return SplitManifest(
        strategy="stratified_random",
        random_state=42,
        test_size=0.2,
        validation_size=0.2,
        stratify_by_label=True,
        n_patients_total=n_patients,
        n_patients_train=3,
        n_patients_validation=2,
        n_patients_test=1,
        n_records_train=sum(e.n_records for e in entries if e.split == "train"),
        n_records_validation=sum(e.n_records for e in entries if e.split == "validation"),
        n_records_test=sum(e.n_records for e in entries if e.split == "test"),
        n_sepsis_patients_train=sum(1 for e in entries if e.split == "train" and e.has_sepsis),
        n_sepsis_patients_validation=sum(1 for e in entries if e.split == "validation" and e.has_sepsis),
        n_sepsis_patients_test=sum(1 for e in entries if e.split == "test" and e.has_sepsis),
        n_sepsis_positive_hours_train=sum(e.n_sepsis_positive_hours for e in entries if e.split == "train"),
        n_sepsis_positive_hours_validation=sum(e.n_sepsis_positive_hours for e in entries if e.split == "validation"),
        n_sepsis_positive_hours_test=sum(e.n_sepsis_positive_hours for e in entries if e.split == "test"),
        patients=entries,
    )


def _write_synthetic_patients(tmp_path, manifest):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for entry in manifest.patients:
        n = entry.n_records
        clinical = {}
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
                clinical[var] = [float(h + 1) for h in range(n)]
            elif var == "HospAdmTime":
                clinical[var] = [float(-50)] * n
            elif var in ("Age", "Gender", "Unit1", "Unit2"):
                clinical[var] = [float(rng.integers(0, 2))] * n
            else:
                vals = rng.normal(0, 1, size=n)
                vals[rng.random(n) < 0.2] = np.nan
                clinical[var] = vals.tolist()
        labels = [1 if h >= n - 3 else 0 for h in range(n)]
        df = pd.DataFrame(clinical)
        df["patient_id"] = entry.patient_id
        df["source_dataset"] = entry.source_dataset
        df["hour"] = list(range(1, n + 1))
        df["SepsisLabel"] = labels
        df.to_parquet(processed_dir / f"{entry.patient_id}.parquet", index=False)
    return processed_dir


# ------------------------------------------------------------------
# Training protocol tests
# ------------------------------------------------------------------


def test_run_training_protocol(tmp_path):
    manifest = _make_small_manifest(tmp_path, n_patients=6)
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    processed_dir = _write_synthetic_patients(tmp_path, manifest)
    config = ModelConfig(
        rf_n_estimators=5,
        svm_max_iter=500,
        train_logistic_regression=True,
        train_random_forest=True,
        train_svm=True,
    )
    result = run_training_protocol(
        processed_dir=processed_dir,
        split_manifest_path=manifest_path,
        config=config,
        artifact_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        source="synthetic",
    )
    assert result.config is not None
    assert result.dataset_source == "synthetic"
    assert result.split_overlap_check["leakage_detected"] is False
    assert result.target_isolation_check["target_isolation_valid"] is True
    assert len(result.models) >= 1


def test_run_training_protocol_with_manifest(tmp_path):
    manifest = _make_small_manifest(tmp_path, n_patients=6)
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    processed_dir = _write_synthetic_patients(tmp_path, manifest)
    config = ModelConfig(
        rf_n_estimators=5,
        svm_max_iter=500,
    )
    result = run_training_protocol(
        processed_dir=processed_dir,
        split_manifest_path=manifest_path,
        config=config,
        artifact_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        source="synthetic",
    )
    assert result.config is not None
    assert result.dataset_source == "synthetic"
    assert result.split_overlap_check["leakage_detected"] is False
    assert result.target_isolation_check["target_isolation_valid"] is True
    assert len(result.models) >= 1
    assert result.selected_features is not None
    assert len(result.selected_features) > 0


def test_feature_selection_reduce_features(tmp_path):
    manifest = _make_small_manifest(tmp_path, n_patients=4)
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    processed_dir = _write_synthetic_patients(tmp_path, manifest)
    config = ModelConfig(rf_n_estimators=5, svm_max_iter=500)
    result = run_training_protocol(
        processed_dir=processed_dir,
        split_manifest_path=manifest_path,
        config=config,
        artifact_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        source="synthetic",
    )
    assert result.feature_selection_report is not None
    assert result.feature_selection_report.n_features_output > 0
    assert result.feature_selection_report.fit_on_train_only is True


# ------------------------------------------------------------------
# Evaluate on split
# ------------------------------------------------------------------


def test_evaluate_on_split(tmp_path):
    manifest = _make_small_manifest(tmp_path, n_patients=4)
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    processed_dir = _write_synthetic_patients(tmp_path, manifest)
    config = ModelConfig(rf_n_estimators=5, svm_max_iter=500)
    result = run_training_protocol(
        processed_dir=processed_dir,
        split_manifest_path=manifest_path,
        config=config,
        artifact_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        source="synthetic",
    )

    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        source="synthetic",
    )
    val_split = dataset.validation
    assert val_split is not None

    eval_results = evaluate_on_split(
        models=result.models,
        selected_features=result.selected_features,
        split_data=val_split,
        threshold=0.5,
        source="synthetic",
    )
    assert len(eval_results) == len(result.models)
    for eval_result in eval_results.values():
        assert "metrics" in eval_result
        assert "predictions" in eval_result
        assert "auroc" in eval_result["metrics"]
        assert "f1" in eval_result["metrics"]
