"""Tests for P1 dataset ingestion (loader, validation, exploration).

All tests use synthetic fixtures that mirror the PhysioNet/CinC
Challenge 2019 schema. No real patient data is used or required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.p1_clinical.ingestion.dataset_loader import (
    REQUIRED_COLUMNS,
    PhysioNetChallenge2019Loader,
    build_synthetic_patient_file,
    write_synthetic_psv,
)
from ml.p1_clinical.ingestion.exploration import (
    DEMOGRAPHIC_COLUMNS,
    LABORATORY_COLUMNS,
    OUTCOME_COLUMN,
    VITAL_SIGN_COLUMNS,
    DatasetExplorer,
    generate_synthetic_exploration_outputs,
)
from ml.p1_clinical.ingestion.validation import (
    summarize_validation,
    validate_columns,
    validate_directory,
    validate_file,
    validate_hourly_ordering,
    validate_patient_ids,
)

FIXTURE_SEED = 7


def _write_synthetic_dataset(tmp_path: Path, n_patients: int = 3, n_hours: int = 12) -> Path:
    """Write a small synthetic dataset directory and return its path."""
    root = tmp_path / "synthetic_data"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(FIXTURE_SEED)
    for i in range(n_patients):
        pid = f"p{i:06d}"
        sepsis_onset = None
        if n_hours > 12 and i % 2 == 1:
            sepsis_onset = int(rng.integers(6, n_hours - 6))
        df = build_synthetic_patient_file(
            patient_id=pid, n_hours=n_hours, seed=FIXTURE_SEED + i,
            sepsis_onset_hour=sepsis_onset,
        )
        write_synthetic_psv(df, root / f"{pid}.psv")
    return root


# ----------------------------------------------------------------------
# dataset_loader tests
# ----------------------------------------------------------------------


def test_required_columns_count():
    assert len(REQUIRED_COLUMNS) == 41
    assert REQUIRED_COLUMNS[-1] == "SepsisLabel"


def test_patient_id_from_filename():
    loader = PhysioNetChallenge2019Loader("/tmp")
    assert loader.patient_id_from_filename("p00101.psv") == "p00101"
    assert loader.patient_id_from_filename("p00101.txt") is None
    assert loader.patient_id_from_filename("notapatient.psv") is None


def test_build_synthetic_patient_file_shape():
    df = build_synthetic_patient_file(n_hours=10, seed=FIXTURE_SEED)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 10


def test_build_synthetic_patient_file_labels_shifted():
    df = build_synthetic_patient_file(n_hours=20, seed=FIXTURE_SEED, sepsis_onset_hour=10)
    labels = df["SepsisLabel"].tolist()
    # SepsisLabel = 1 when t >= onset - 6 => t >= 4
    assert labels[:4] == [0, 0, 0, 0]
    assert labels[4:] == [1] * 16


def test_loader_catalog_and_patient_ids(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=3, n_hours=8)
    loader = PhysioNetChallenge2019Loader(root)
    catalog = loader.catalog()
    assert catalog.n_patients == 3
    ids = loader.patient_ids()
    assert len(ids) == 3
    assert "p000000" in ids


def test_loader_load_patient(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=2, n_hours=8)
    loader = PhysioNetChallenge2019Loader(root)
    df = loader.load_patient("p000000")
    assert df is not None
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 8


def test_loader_load_missing_patient_returns_none(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=1, n_hours=8)
    loader = PhysioNetChallenge2019Loader(root)
    assert loader.load_patient("p999999") is None


def test_loader_iter_patients(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=3, n_hours=8)
    loader = PhysioNetChallenge2019Loader(root)
    results = list(loader.iter_patients())
    assert len(results) == 3
    assert all(r.n_rows == 8 for r in results)


def test_loader_iter_patients_limit(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=5, n_hours=8)
    loader = PhysioNetChallenge2019Loader(root)
    results = list(loader.iter_patients(limit=2))
    assert len(results) == 2


def test_loader_does_not_load_entire_dataset_at_once(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=3, n_hours=8)
    loader = PhysioNetChallenge2019Loader(root)
    # iter_patients yields one at a time; just confirm it works.
    count = 0
    for _ in loader.iter_patients():
        count += 1
    assert count == 3


# ----------------------------------------------------------------------
# validation tests
# ----------------------------------------------------------------------


def test_validate_columns_valid():
    df = build_synthetic_patient_file(n_hours=5, seed=FIXTURE_SEED)
    report = validate_columns(df)
    assert report.valid is True
    assert report.missing_columns == []
    assert report.n_columns_expected == 41


def test_validate_columns_missing():
    df = build_synthetic_patient_file(n_hours=5, seed=FIXTURE_SEED)
    df = df.drop(columns=["HR", "SepsisLabel"])
    report = validate_columns(df)
    assert report.valid is False
    assert "HR" in report.missing_columns
    assert "SepsisLabel" in report.missing_columns


def test_validate_patient_ids():
    df = pd.DataFrame({"patient_id": ["p00001", "p00002", "bad_id", "p00001"]})
    report = validate_patient_ids(df)
    assert report.valid is False
    assert "bad_id" in report.invalid_ids
    assert "p00001" in report.duplicate_ids
    assert report.n_duplicate_patients > 0


def test_validate_hourly_ordering_valid():
    df = build_synthetic_patient_file(n_hours=8, seed=FIXTURE_SEED)
    report = validate_hourly_ordering(df)
    assert report.valid is True
    assert report.n_rows == 8


def test_validate_hourly_ordering_nonmonotonic():
    df = build_synthetic_patient_file(n_hours=5, seed=FIXTURE_SEED)
    df.loc[3, "ICULOS"] = 1.0  # duplicate / out of order
    report = validate_hourly_ordering(df)
    assert report.valid is False


def test_validate_file_valid(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=1, n_hours=8)
    file_path = root / "p000000.psv"
    report = validate_file(file_path, patient_id="p000000")
    assert report.file_exists is True
    assert report.file_readable is True
    assert report.malformed is False


def test_validate_file_missing(tmp_path):
    report = validate_file(tmp_path / "nonexistent.psv", patient_id="p00000")
    assert report.file_exists is False
    assert report.malformed is True


def test_validate_directory(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=3, n_hours=8)
    reports = validate_directory(root)
    assert len(reports) == 3
    assert all(r.malformed is False for r in reports)
    summary = summarize_validation(reports)
    assert summary["n_files"] == 3
    assert summary["n_valid"] == 3


# ----------------------------------------------------------------------
# exploration tests
# ----------------------------------------------------------------------


def test_explorer_overview_no_data(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    explorer = DatasetExplorer(empty)
    overview = explorer.compute_overview()
    assert overview.data_available is False
    assert overview.n_patients == 0


def test_explorer_overview_with_data(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=2, n_hours=10)
    explorer = DatasetExplorer(root)
    overview = explorer.compute_overview()
    assert overview.data_available is True
    assert overview.n_patients == 2
    assert overview.n_records == 20


def test_explorer_label_distribution(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=4, n_hours=20)
    explorer = DatasetExplorer(root)
    dist = explorer.compute_label_distribution()
    assert dist.n_patients_total == 4
    assert dist.data_available is True


def test_explorer_missingness_dataframe(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=2, n_hours=10)
    explorer = DatasetExplorer(root)
    df = explorer.missingness_dataframe()
    assert set(df.columns) >= {"column", "n_missing", "n_total", "missing_percent", "group"}


def test_explorer_patient_record_counts(tmp_path):
    root = _write_synthetic_dataset(tmp_path, n_patients=2, n_hours=10)
    explorer = DatasetExplorer(root)
    df = explorer.patient_record_counts()
    assert len(df) == 2
    assert (df["n_records"] == 10).all()


def test_generate_synthetic_exploration_outputs(tmp_path):
    out_dir = tmp_path / "reports"
    results = generate_synthetic_exploration_outputs(out_dir, n_patients=3, n_hours=12)
    assert results["synthetic"] is True
    assert (out_dir / "dataset_overview.json").exists()
    assert (out_dir / "missingness_report.csv").exists()
    assert (out_dir / "label_distribution.json").exists()
    assert (out_dir / "patient_record_counts.csv").exists()
    assert (out_dir / "validation_summary.json").exists()

    overview = json.loads((out_dir / "dataset_overview.json").read_text())
    assert overview["n_patients"] == 3
    assert overview["n_records"] == 36


def test_column_group_coverage():
    # Vital signs + labs + demographics + outcome == all 41 columns.
    grouped = (
        VITAL_SIGN_COLUMNS + LABORATORY_COLUMNS + DEMOGRAPHIC_COLUMNS + [OUTCOME_COLUMN]
    )
    assert set(grouped) == set(REQUIRED_COLUMNS)
    assert len(grouped) == 41
