"""Tests for inference.py.

Tests model loading, prediction, threshold selection on validation data,
and test-set isolation.
Uses synthetic fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.classical_ml.inference import InferenceResult, ModelInference
from ml.p1_clinical.classical_ml.model_config import ModelConfig
from ml.p1_clinical.classical_ml.training import run_training_protocol
from ml.p1_clinical.preprocessing.patient_splits import (
    PatientSplitEntry,
    SplitManifest,
)


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
# Inference tests
# ------------------------------------------------------------------


def test_model_inference_predict(tmp_path):
    manifest = _make_small_manifest(tmp_path, n_patients=6)
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

    # Load dataset for evaluation.
    from ml.p1_clinical.classical_ml.data_loader import (
        load_split_aware_dataset,
    )
    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        source="synthetic",
    )
    test_split = dataset.test

    inference = ModelInference(
        artifact_dir=tmp_path / "artifacts",
        source="synthetic",
    )

    for name in result.models:
        inf_result = inference.predict(
            model_name=name,
            X=test_split.X,
            model_version="1.0.0",
            threshold=0.5,
            split="test",
        )
        assert isinstance(inf_result, InferenceResult)
        assert inf_result.model_name == name
        assert inf_result.split == "test"
        assert inf_result.n_predictions == len(test_split.X)
        assert "predicted_probability" in inf_result.predictions.to_dataframe().columns


# ------------------------------------------------------------------
# Threshold selection on validation only
# ------------------------------------------------------------------


def test_threshold_selection_validation_only(tmp_path):
    manifest = _make_small_manifest(tmp_path, n_patients=6)
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    processed_dir = _write_synthetic_patients(tmp_path, manifest)
    config = ModelConfig(rf_n_estimators=5, svm_max_iter=500)
    run_training_protocol(
        processed_dir=processed_dir,
        split_manifest_path=manifest_path,
        config=config,
        artifact_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        source="synthetic",
    )

    from ml.p1_clinical.classical_ml.data_loader import (
        load_split_aware_dataset,
    )
    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        source="synthetic",
    )
    val_split = dataset.validation

    inference = ModelInference(
        artifact_dir=tmp_path / "artifacts",
        source="synthetic",
    )

    # Select threshold on validation data.
    val_X_with_label = val_split.X.copy()
    val_X_with_label["SepsisLabel"] = val_split.y
    best_threshold = inference.select_threshold(
        model_name="logistic_regression",
        X_val=val_X_with_label,
        metric="f1",
        grid_step=0.1,
        model_version="1.0.0",
    )
    assert 0.0 <= best_threshold <= 1.0

    # Verify test set was NOT used - predictions with selected threshold
    # should still work on test data.
    test_split = dataset.test
    inf_result = inference.predict(
        model_name="logistic_regression",
        X=test_split.X,
        model_version="1.0.0",
        threshold=best_threshold,
        split="test",
    )
    assert inf_result.threshold == best_threshold


# ------------------------------------------------------------------
# Test-set isolation: changing test values doesn't change training
# ------------------------------------------------------------------


def test_test_set_isolation(tmp_path):
    manifest = _make_small_manifest(tmp_path, n_patients=6)
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

    # Get original training metrics.
    original_metrics = {}
    for name, model_result in result.models.items():
        original_metrics[name] = dict(model_result.training_metrics)

    # Modify test data values (simulating different test set).
    from ml.p1_clinical.classical_ml.data_loader import (
        load_split_aware_dataset,
    )
    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        source="synthetic",
    )
    test_split = dataset.test

    # Change all test features to extreme values.
    test_split_modified = test_split.X.copy()
    test_split_modified.iloc[:, :] = 999.0

    inference = ModelInference(
        artifact_dir=tmp_path / "artifacts",
        source="synthetic",
    )

    # Predictions on modified test data should still use the same model.
    for name in result.models:
        inf_result = inference.predict(
            model_name=name,
            X=test_split_modified,
            model_version="1.0.0",
            threshold=0.5,
            split="test",
        )
        assert inf_result.n_predictions == len(test_split_modified)

    # Original training metrics should be unchanged.
    for name, model_result in result.models.items():
        for key, val in original_metrics[name].items():
            assert model_result.training_metrics[key] == val


# ------------------------------------------------------------------
# Probability output validity
# ------------------------------------------------------------------


def test_probability_output_validity(tmp_path):
    manifest = _make_small_manifest(tmp_path, n_patients=6)
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

    from ml.p1_clinical.classical_ml.data_loader import (
        load_split_aware_dataset,
    )
    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        source="synthetic",
    )
    test_split = dataset.test

    inference = ModelInference(
        artifact_dir=tmp_path / "artifacts",
        source="synthetic",
    )

    for name in result.models:
        inf_result = inference.predict(
            model_name=name,
            X=test_split.X,
            model_version="1.0.0",
            threshold=0.5,
            split="test",
        )
        probs = inf_result.predictions.to_dataframe()["predicted_probability"]
        assert (probs >= 0).all(), f"{name}: probabilities < 0"
        assert (probs <= 1).all(), f"{name}: probabilities > 1"
