"""Tests for the P1 patient-level splitting module.

All tests use synthetic fixtures. No real patient data is used.
"""

from __future__ import annotations

import json

import numpy as np

from ml.p1_clinical.preprocessing.patient_splits import (
    PatientSplitter,
    SplitManifest,
    check_split_leakage,
    save_split_manifests,
)
from ml.p1_clinical.preprocessing.schemas import CLINICAL_VARIABLES, ProcessedPatient


def _make_processed_patient(
    patient_id: str,
    n_hours: int,
    sepsis_label: int,
    hospital_id: str = "set_a",
    seed: int = 42,
) -> ProcessedPatient:
    rng = np.random.default_rng(seed)
    clinical = {}
    for var in CLINICAL_VARIABLES:
        if var in ("Age", "Gender", "Unit1", "Unit2"):
            clinical[var] = [float(rng.integers(0, 2))] * n_hours
        elif var == "ICULOS":
            clinical[var] = [float(h + 1) for h in range(n_hours)]
        elif var == "HospAdmTime":
            clinical[var] = [float(-50)] * n_hours
        else:
            clinical[var] = rng.normal(0, 1, size=n_hours).tolist()
    labels = [sepsis_label] * n_hours
    return ProcessedPatient(
        patient_id=patient_id,
        hospital_id=hospital_id,
        hour=list(range(1, n_hours + 1)),
        clinical=clinical,
        sepsis_label=labels,
        source_file=f"{patient_id}.psv",
        source_dataset=hospital_id,
        n_rows=n_hours,
    )


# ----------------------------------------------------------------------
# PatientSplitter tests
# ----------------------------------------------------------------------


def test_split_assigns_every_patient():
    patients = [
        _make_processed_patient(f"p{i:06d}", 10, i % 2, seed=i)
        for i in range(10)
    ]
    splitter = PatientSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    manifest = splitter.split(patients)
    assert manifest.n_patients_total == 10
    assigned = [
        p for p in manifest.patients if p.split in ("train", "validation", "test")
    ]
    assert len(assigned) == 10


def test_split_no_patient_overlap():
    patients = [
        _make_processed_patient(f"p{i:06d}", 10, i % 2, seed=i)
        for i in range(20)
    ]
    splitter = PatientSplitter(test_size=0.25, validation_size=0.15, random_state=7)
    manifest = splitter.split(patients)
    leakage = check_split_leakage(manifest)
    assert leakage["leakage_detected"] is False
    assert leakage["n_overlapping_patients"] == 0
    assert leakage["every_patient_in_exactly_one_split"] is True


def test_split_reproducible():
    patients = [
        _make_processed_patient(f"p{i:06d}", 10, i % 2, seed=i)
        for i in range(15)
    ]
    splitter1 = PatientSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    splitter2 = PatientSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    m1 = splitter1.split(patients)
    m2 = splitter2.split(patients)
    ids1 = [p.patient_id for p in m1.patients if p.split == "train"]
    ids2 = [p.patient_id for p in m2.patients if p.split == "train"]
    assert ids1 == ids2


def test_split_all_observations_in_one_split():
    patients = [
        _make_processed_patient(f"p{i:06d}", 10, i % 2, seed=i)
        for i in range(10)
    ]
    splitter = PatientSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    manifest = splitter.split(patients)
    for entry in manifest.patients:
        assert entry.n_records == 10


def test_split_preserves_class_distribution_info():
    patients = [
        _make_processed_patient(f"p{i:06d}", 10, 1 if i < 5 else 0, seed=i)
        for i in range(10)
    ]
    splitter = PatientSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    manifest = splitter.split(patients)
    assert manifest.n_sepsis_patients_train + manifest.n_sepsis_patients_validation + manifest.n_sepsis_patients_test == 5


def test_split_manifest_save_and_load(tmp_path):
    patients = [
        _make_processed_patient(f"p{i:06d}", 10, i % 2, seed=i)
        for i in range(8)
    ]
    splitter = PatientSplitter(test_size=0.25, validation_size=0.125, random_state=42)
    manifest = splitter.split(patients)
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    loaded = SplitManifest.load(manifest_path)
    assert loaded.n_patients_total == manifest.n_patients_total
    assert loaded.n_patients_train == manifest.n_patients_train
    assert [p.patient_id for p in loaded.patients] == [p.patient_id for p in manifest.patients]


def test_split_leakage_detected_when_overlap():
    # Manually create a manifest with overlap using PatientSplitEntry.
    from ml.p1_clinical.preprocessing.patient_splits import PatientSplitEntry

    entries = [
        PatientSplitEntry(
            patient_id="p000001",
            hospital_id="set_a",
            split="train",
            n_records=10,
            n_sepsis_positive_hours=0,
            n_sepsis_negative_hours=10,
            has_sepsis=False,
            source_file="p000001.psv",
            source_dataset="set_a",
        ),
        PatientSplitEntry(
            patient_id="p000001",
            hospital_id="set_a",
            split="test",
            n_records=10,
            n_sepsis_positive_hours=0,
            n_sepsis_negative_hours=10,
            has_sepsis=False,
            source_file="p000001.psv",
            source_dataset="set_a",
        ),
    ]
    manifest = SplitManifest(
        strategy="test",
        random_state=0,
        test_size=0.0,
        validation_size=0.0,
        stratify_by_label=False,
        n_patients_total=1,
        n_patients_train=1,
        n_patients_validation=0,
        n_patients_test=1,
        n_records_train=10,
        n_records_validation=0,
        n_records_test=10,
        n_sepsis_patients_train=0,
        n_sepsis_patients_validation=0,
        n_sepsis_patients_test=0,
        n_sepsis_positive_hours_train=0,
        n_sepsis_positive_hours_validation=0,
        n_sepsis_positive_hours_test=0,
        patients=entries,
    )
    leakage = check_split_leakage(manifest)
    assert leakage["leakage_detected"] is True
    assert "train_vs_test" in leakage["overlaps"]


def test_split_preserves_hospital_ids():
    patients = [
        _make_processed_patient(f"p{i:06d}", 10, i % 2, hospital_id="set_b" if i % 2 == 0 else "set_a", seed=i)
        for i in range(10)
    ]
    splitter = PatientSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    manifest = splitter.split(patients)
    for entry in manifest.patients:
        assert entry.hospital_id is not None
        assert entry.hospital_id in ("set_a", "set_b")


def test_split_empty_patient_list():
    splitter = PatientSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    manifest = splitter.split([])
    assert manifest.n_patients_total == 0
    assert manifest.n_patients_train == 0
    assert manifest.n_patients_validation == 0
    assert manifest.n_patients_test == 0


def test_save_split_manifests(tmp_path):
    patients = [
        _make_processed_patient(f"p{i:06d}", 10, i % 2, seed=i)
        for i in range(8)
    ]
    splitter = PatientSplitter(test_size=0.25, validation_size=0.125, random_state=42)
    manifest = splitter.split(patients)
    saved = save_split_manifests(manifest, tmp_path)
    assert "manifest" in saved
    assert "train" in saved
    assert "validation" in saved
    assert "test" in saved
    assert saved["manifest"].exists()
    train_ids = json.loads(saved["train"].read_text())
    assert train_ids["n_patients"] == manifest.n_patients_train
