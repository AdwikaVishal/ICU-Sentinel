"""Tests for the P1 preprocessing pipeline (end-to-end).

All tests use synthetic fixtures. No real patient data is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.p1_clinical.ingestion.dataset_loader import (
    build_synthetic_patient_file,
    write_synthetic_psv,
)
from ml.p1_clinical.preprocessing.pipeline import run_pipeline
from ml.p1_clinical.preprocessing.schemas import OUTPUT_COLUMNS


def _write_synthetic_dataset(tmp_path: Path, n_patients: int = 6, n_hours: int = 12) -> Path:
    root = tmp_path / "raw"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    for i in range(n_patients):
        pid = f"p{i:06d}"
        sepsis_onset = None
        if n_hours > 12 and i % 2 == 1:
            sepsis_onset = int(rng.integers(6, n_hours - 6))
        df = build_synthetic_patient_file(
            patient_id=pid, n_hours=n_hours, seed=123 + i,
            sepsis_onset_hour=sepsis_onset,
        )
        write_synthetic_psv(df, root / f"{pid}.psv")
    return root


def test_pipeline_runs_on_synthetic_data(tmp_path):
    raw_dir = _write_synthetic_dataset(tmp_path, n_patients=4, n_hours=10)
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    reports_dir = tmp_path / "reports"

    result = run_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        reports_dir=reports_dir,
        test_size=0.25,
        validation_size=0.125,
        random_state=42,
        limit=None,
    )

    assert result.n_patients_processed == 4
    assert result.split_manifest is not None
    assert result.split_manifest.n_patients_total == 4
    assert result.leakage_report is not None
    assert result.leakage_report["leakage_detected"] is False


def test_pipeline_labels_unchanged(tmp_path):
    raw_dir = _write_synthetic_dataset(tmp_path, n_patients=3, n_hours=10)
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    reports_dir = tmp_path / "reports"

    result = run_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        reports_dir=reports_dir,
        random_state=42,
    )

    for processed in result.processed_patients:
        # Verify all labels are 0 or 1.
        labels = processed.sepsis_label
        assert all(l in (0, 1) for l in labels)


def test_pipeline_output_schema(tmp_path):
    raw_dir = _write_synthetic_dataset(tmp_path, n_patients=2, n_hours=8)
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    reports_dir = tmp_path / "reports"

    result = run_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        reports_dir=reports_dir,
        random_state=42,
    )

    assert len(result.processed_patients) > 0
    df_out = result.processed_patients[0].to_dataframe()
    assert list(df_out.columns) == OUTPUT_COLUMNS


def test_pipeline_no_patient_in_multiple_splits(tmp_path):
    raw_dir = _write_synthetic_dataset(tmp_path, n_patients=8, n_hours=10)
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    reports_dir = tmp_path / "reports"

    result = run_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        reports_dir=reports_dir,
        random_state=42,
    )

    leakage = result.leakage_report
    assert leakage["leakage_detected"] is False
    assert leakage["every_patient_in_exactly_one_split"] is True


def test_pipeline_patient_records_preserved(tmp_path):
    raw_dir = _write_synthetic_dataset(tmp_path, n_patients=3, n_hours=10)
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    reports_dir = tmp_path / "reports"

    result = run_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        reports_dir=reports_dir,
        random_state=42,
    )

    total_records_in = 3 * 10
    total_records_out = sum(p.n_rows for p in result.processed_patients)
    assert total_records_out == total_records_in


def test_pipeline_reports_written(tmp_path):
    raw_dir = _write_synthetic_dataset(tmp_path, n_patients=3, n_hours=8)
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    reports_dir = tmp_path / "reports"

    run_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        reports_dir=reports_dir,
        random_state=42,
    )

    assert (reports_dir / "preprocessing_summary.json").exists()
    assert (reports_dir / "split_summary.json").exists()
    assert (reports_dir / "data_quality_report.json").exists()

    summary = json.loads((reports_dir / "preprocessing_summary.json").read_text())
    assert summary["no_imputation_applied"] is True
    assert summary["future_information_used"] is False
    assert summary["raw_files_unchanged"] is True


def test_pipeline_handles_malformed_file(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Write a malformed .psv file (not pipe-delimited properly).
    (raw_dir / "p000001.psv").write_text("this,is,not,pipe,delimited\n1,2,3\n")

    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    reports_dir = tmp_path / "reports"

    result = run_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        reports_dir=reports_dir,
        random_state=42,
    )

    # Should not crash; malformed file results in failed validation.
    assert len(result.validation_results) == 1
    assert not result.validation_results[0].passed
    assert result.n_patients_processed == 0


def test_pipeline_does_not_modify_raw_files(tmp_path):
    raw_dir = _write_synthetic_dataset(tmp_path, n_patients=2, n_hours=8)
    import hashlib

    def file_hash(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    hashes_before = {
        f.name: file_hash(f) for f in raw_dir.iterdir() if f.suffix == ".psv"
    }

    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    reports_dir = tmp_path / "reports"

    run_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        reports_dir=reports_dir,
        random_state=42,
    )

    hashes_after = {
        f.name: file_hash(f) for f in raw_dir.iterdir() if f.suffix == ".psv"
    }
    assert hashes_before == hashes_after


def test_pipeline_preserves_source_dataset_identity(tmp_path):
    raw_dir = _write_synthetic_dataset(tmp_path, n_patients=3, n_hours=8)
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    reports_dir = tmp_path / "reports"

    result = run_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        reports_dir=reports_dir,
        random_state=42,
    )

    for processed in result.processed_patients:
        assert processed.source_dataset is not None
        assert len(processed.source_dataset) > 0
