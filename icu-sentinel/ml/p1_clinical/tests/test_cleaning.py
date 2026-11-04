"""Tests for the P1 preprocessing cleaning module.

All tests use synthetic fixtures. No real patient data is used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.p1_clinical.ingestion.dataset_loader import (
    REQUIRED_COLUMNS,
    build_synthetic_patient_file,
    write_synthetic_psv,
)
from ml.p1_clinical.preprocessing.cleaning import (
    clean_patient_dataframe,
    clean_patient_file,
    drops_future_information,
)
from ml.p1_clinical.preprocessing.schemas import OUTPUT_COLUMNS, ProcessedPatient


def _write_tmp_psv(tmp_path: Path, df: pd.DataFrame, name: str = "p00001.psv") -> Path:
    path = tmp_path / name
    write_synthetic_psv(df, path)
    return path


# ----------------------------------------------------------------------
# clean_patient_dataframe tests
# ----------------------------------------------------------------------


def test_clean_preserves_patient_id():
    df = build_synthetic_patient_file(n_hours=10, seed=42, sepsis_onset_hour=8)
    processed, _report = clean_patient_dataframe(
        df, patient_id="p00001", hospital_id="set_a", source_file="p00001.psv"
    )
    assert processed is not None
    assert processed.patient_id == "p00001"
    assert processed.hospital_id == "set_a"


def test_clean_preserves_hourly_ordering():
    df = build_synthetic_patient_file(n_hours=10, seed=42)
    # Shuffle rows to test re-sorting.
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    processed, _ = clean_patient_dataframe(
        df, patient_id="p00001", source_file="p00001.psv"
    )
    assert processed is not None
    assert processed.hour == sorted(processed.hour)


def test_clean_preserves_sepsis_label():
    df = build_synthetic_patient_file(
        n_hours=20, seed=42, sepsis_onset_hour=10
    )
    original_labels = df["SepsisLabel"].tolist()
    processed, _ = clean_patient_dataframe(
        df, patient_id="p00001", source_file="p00001.psv"
    )
    assert processed is not None
    assert processed.sepsis_label == original_labels


def test_clean_no_imputation_preserves_nan():
    df = build_synthetic_patient_file(n_hours=10, seed=42)
    # Force some NaNs.
    df.loc[0, "HR"] = np.nan
    df.loc[3, "Temp"] = np.nan
    processed, report = clean_patient_dataframe(
        df, patient_id="p00001", source_file="p00001.psv"
    )
    assert processed is not None
    assert np.isnan(processed.clinical["HR"][0])
    assert np.isnan(processed.clinical["Temp"][3])
    assert report.n_missing_preserved >= 2


def test_clean_no_rows_dropped():
    df = build_synthetic_patient_file(n_hours=10, seed=42)
    processed, report = clean_patient_dataframe(
        df, patient_id="p00001", source_file="p00001.psv"
    )
    assert processed is not None
    assert report.n_rows_dropped == 0
    assert report.n_rows_input == report.n_rows_output


def test_clean_output_schema_columns():
    df = build_synthetic_patient_file(n_hours=5, seed=42)
    processed, _ = clean_patient_dataframe(
        df, patient_id="p00001", source_file="p00001.psv"
    )
    assert processed is not None
    df_out = processed.to_dataframe()
    assert list(df_out.columns) == OUTPUT_COLUMNS


def test_clean_empty_dataframe():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    processed, report = clean_patient_dataframe(
        df, patient_id="p00001", source_file="p00001.psv"
    )
    assert processed is None
    assert report.n_rows_input == 0


def test_clean_no_future_information():
    df = build_synthetic_patient_file(n_hours=10, seed=42)
    processed, _ = clean_patient_dataframe(
        df, patient_id="p00001", source_file="p00001.psv"
    )
    assert processed is not None
    df_out = processed.to_dataframe()
    assert drops_future_information(df_out)


def test_clean_integer_columns_coerced():
    df = build_synthetic_patient_file(n_hours=5, seed=42)
    # Put a string in Gender column.
    df["Gender"] = "male"
    processed, report = clean_patient_dataframe(
        df, patient_id="p00001", source_file="p00001.psv"
    )
    assert processed is not None
    assert report.n_values_coerced >= 5  # all "male" values coerced to NaN


# ----------------------------------------------------------------------
# clean_patient_file tests
# ----------------------------------------------------------------------


def test_clean_patient_file(tmp_path):
    df = build_synthetic_patient_file(n_hours=8, seed=42, sepsis_onset_hour=6)
    psv_path = _write_tmp_psv(tmp_path, df, "p00001.psv")
    processed, _report = clean_patient_file(
        psv_path, patient_id="p00001", hospital_id="set_a"
    )
    assert processed is not None
    assert processed.patient_id == "p00001"
    assert processed.hospital_id == "set_a"
    assert processed.n_rows == 8


def test_clean_patient_file_nonexistent(tmp_path):
    processed, report = clean_patient_file(
        tmp_path / "nonexistent.psv", patient_id="p99999"
    )
    assert processed is None
    assert report.n_rows_input == 0


# ----------------------------------------------------------------------
# drops_future_information tests
# ----------------------------------------------------------------------


def test_drops_future_information_unsorted_returns_false():
    df = pd.DataFrame(
        {
            "ICULOS": [3.0, 1.0, 2.0],
            "HR": [80.0, 85.0, 90.0],
        }
    )
    assert drops_future_information(df) is False


def test_drops_future_information_sorted_returns_true():
    df = pd.DataFrame(
        {
            "ICULOS": [1.0, 2.0, 3.0],
            "HR": [80.0, 85.0, 90.0],
        }
    )
    assert drops_future_information(df) is True


def test_processed_patient_is_dataclass():
    df = build_synthetic_patient_file(n_hours=3, seed=42)
    processed, _ = clean_patient_dataframe(
        df, patient_id="p00001", source_file="p00001.psv"
    )
    assert processed is not None
    assert isinstance(processed, ProcessedPatient)
    assert processed.n_rows == 3
