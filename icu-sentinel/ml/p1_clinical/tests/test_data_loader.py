"""Tests for data_loader.py.

Tests split-aware data loading, patient overlap prevention, and target
isolation. Uses synthetic fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.classical_ml.data_loader import (
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    check_split_overlap,
    load_split_aware_dataset,
    validate_target_isolation,
)
from ml.p1_clinical.preprocessing.patient_splits import (
    PatientSplitEntry,
    SplitManifest,
)


def _make_split_manifest(tmp_path, n_patients=6, seed=42) -> SplitManifest:
    """Create a small split manifest with train/validation/test."""
    rng = np.random.default_rng(seed)
    entries = []
    for i in range(n_patients):
        pid = f"p{i:06d}"
        n_hours = int(rng.integers(5, 15))
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
    manifest = SplitManifest(
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
    return manifest


def _write_processed_patients(tmp_path, manifest: SplitManifest):
    """Write synthetic processed parquet files for each patient."""
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
        # SepsisLabel: mix of 0 and 1
        labels = [1 if h >= n - 3 else 0 for h in range(n)]
        df = pd.DataFrame(clinical)
        df["patient_id"] = entry.patient_id
        df["source_dataset"] = entry.source_dataset
        df["hour"] = list(range(1, n + 1))
        df["SepsisLabel"] = labels
        df.to_parquet(processed_dir / f"{entry.patient_id}.parquet", index=False)
    return processed_dir


# ------------------------------------------------------------------
# Load tests
# ------------------------------------------------------------------


def test_load_split_aware_dataset(tmp_path):
    manifest = _make_split_manifest(tmp_path, n_patients=6)
    processed_dir = _write_processed_patients(tmp_path, manifest)
    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        source="synthetic",
    )
    assert dataset.n_patients_total == 6
    assert dataset.n_features > 0
    assert dataset.train is not None
    assert dataset.validation is not None
    assert dataset.test is not None


def test_split_no_overlap(tmp_path):
    manifest = _make_split_manifest(tmp_path, n_patients=6)
    processed_dir = _write_processed_patients(tmp_path, manifest)
    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        source="synthetic",
    )
    check = check_split_overlap(dataset)
    assert check["leakage_detected"] is False
    assert check["n_overlapping_patients"] == 0
    assert check["every_patient_in_exactly_one_split"] is True


def test_target_isolation(tmp_path):
    manifest = _make_split_manifest(tmp_path, n_patients=6)
    processed_dir = _write_processed_patients(tmp_path, manifest)
    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        source="synthetic",
    )
    check = validate_target_isolation(dataset)
    assert check["target_isolation_valid"] is True
    assert len(check["issues"]) == 0


def test_feature_columns_exclude_identifiers(tmp_path):
    manifest = _make_split_manifest(tmp_path, n_patients=3)
    processed_dir = _write_processed_patients(tmp_path, manifest)
    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        source="synthetic",
    )
    train = dataset.train
    assert train is not None
    for col in IDENTIFIER_COLUMNS:
        assert col not in train.X.columns, f"Identifier {col} leaked into features"
    assert TARGET_COLUMN not in train.X.columns, "Target leaked into features"


def test_class_imbalance_present(tmp_path):
    manifest = _make_split_manifest(tmp_path, n_patients=6)
    processed_dir = _write_processed_patients(tmp_path, manifest)
    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        source="synthetic",
    )
    train = dataset.train
    assert train is not None
    # Both classes should be present in training.
    assert train.n_positive > 0
    assert train.n_negative > 0
    assert 0.0 < train.positive_rate < 1.0


def test_dataset_summary(tmp_path):
    manifest = _make_split_manifest(tmp_path, n_patients=6)
    processed_dir = _write_processed_patients(tmp_path, manifest)
    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        source="synthetic",
    )
    summary = dataset.summary()
    assert summary["n_patients_total"] == 6
    assert summary["n_features"] > 0
    assert "splits" in summary
    for split in ("train", "validation", "test"):
        assert split in summary["splits"]
        s = summary["splits"][split]
        assert s is not None
        assert "n_patients" in s
        assert "n_rows" in s
        assert "positive_rate" in s
