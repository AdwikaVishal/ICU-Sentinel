"""Validation utilities for the PhysioNet/CinC Challenge 2019 dataset.

Checks file integrity, required columns, patient identifiers, hourly
ordering, and duplicate detection. Never silently discards malformed
records; all issues are reported explicitly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .dataset_loader import REQUIRED_COLUMNS

logger = logging.getLogger("icu_sentinel.p1_clinical.ingestion.validation")


@dataclass
class ColumnValidationReport:
    """Result of validating that a DataFrame has required columns."""

    valid: bool
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    present_columns: list[str] = field(default_factory=list)
    n_columns_expected: int = 0
    n_columns_present: int = 0


@dataclass
class PatientIdValidationReport:
    """Result of validating patient identifiers."""

    valid: bool
    invalid_ids: list[str] = field(default_factory=list)
    valid_ids: list[str] = field(default_factory=list)
    n_unique_patients: int = 0
    n_duplicate_patients: int = 0
    duplicate_ids: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class HourlyOrderingReport:
    """Result of validating hourly (ICULOS) ordering within a patient."""

    valid: bool = True
    n_rows: int = 0
    iculos_min: float | None = None
    iculos_max: float | None = None
    non_monotonic_rows: list[int] = field(default_factory=list)
    missing_iculos_rows: list[int] = field(default_factory=list)
    duplicate_iculos_rows: list[int] = field(default_factory=list)
    expected_start: float | None = None


@dataclass
class FileValidationReport:
    """Aggregated validation report for a single patient file."""

    patient_id: str
    filepath: str
    file_exists: bool = False
    file_readable: bool = False
    n_rows: int = 0
    n_columns: int = 0
    column_report: ColumnValidationReport | None = None
    patient_id_report: PatientIdValidationReport | None = None
    hourly_report: HourlyOrderingReport | None = None
    malformed: bool = False
    issues: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return (
            self.file_exists
            and self.file_readable
            and self.column_report is not None
            and self.column_report.valid
            and not self.malformed
        )


def validate_columns(df: pd.DataFrame) -> ColumnValidationReport:
    """Check that ``df`` contains all required Challenge columns."""
    present = set(df.columns)
    expected = set(REQUIRED_COLUMNS)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    return ColumnValidationReport(
        valid=len(missing) == 0,
        missing_columns=missing,
        extra_columns=extra,
        present_columns=sorted(present),
        n_columns_expected=len(REQUIRED_COLUMNS),
        n_columns_present=len(present),
    )


def validate_patient_ids(df: pd.DataFrame, patient_id_column: str = "patient_id") -> PatientIdValidationReport:
    """Validate patient identifiers in a DataFrame.

    Expects a column (default ``patient_id``) holding string patient IDs
    matching the official pattern ``p`` followed by six digits.
    """
    if patient_id_column not in df.columns:
        return PatientIdValidationReport(
            valid=False,
            invalid_ids=[],
            valid_ids=[],
            n_unique_patients=0,
            n_duplicate_patients=0,
            duplicate_ids=[],
        )
    ids = df[patient_id_column].dropna().astype(str)
    valid_mask = ids.str.fullmatch(r"p\d{6}")
    invalid_ids = sorted(set(ids[~valid_mask]))
    valid_ids = sorted(set(ids[valid_mask]))
    dup_mask = ids.duplicated(keep=False)
    duplicate_ids = sorted(set(ids[dup_mask]))
    n_duplicate = int(dup_mask.sum())

    issues: list[str] = []
    if invalid_ids:
        issues.append(f"{len(invalid_ids)} invalid patient ID(s): {invalid_ids[:5]}")
    if n_duplicate:
        issues.append(f"{n_duplicate} duplicate patient ID occurrence(s)")

    return PatientIdValidationReport(
        valid=not invalid_ids and n_duplicate == 0,
        invalid_ids=invalid_ids,
        valid_ids=valid_ids,
        n_unique_patients=ids.nunique(),
        n_duplicate_patients=n_duplicate,
        duplicate_ids=duplicate_ids,
        issues=issues,
    )


def validate_hourly_ordering(
    df: pd.DataFrame,
    icolumn: str = "ICULOS",
) -> HourlyOrderingReport:
    """Validate that ICULOS values increase by exactly 1 per row.

    The Challenge data represents one row per ICU hour. ICULOS should be
    monotonically increasing without gaps or duplicates within a patient
    file.
    """
    report = HourlyOrderingReport()
    if icolumn not in df.columns:
        report.valid = False
        report.n_rows = len(df)
        return report

    icol = pd.to_numeric(df[icolumn], errors="coerce")
    report.n_rows = len(df)
    report.icolumn_min = float(icol.min()) if icol.notna().any() else None
    report.icolumn_max = float(icol.max()) if icol.notna().any() else None

    missing_mask = icol.isna()
    report.missing_iculos_rows = list(df.index[missing_mask])

    valid_vals = icol.dropna()
    if len(valid_vals) > 1:
        diffs = valid_vals.diff().dropna()
        non_mono = diffs[diffs <= 0]
        report.non_monotonic_rows = list(valid_vals.index[non_mono.index])
        dup_mask = valid_vals.duplicated(keep=False)
        report.duplicate_iculos_rows = list(valid_vals.index[dup_mask])

    if len(valid_vals) > 0:
        report.expected_start = float(valid_vals.iloc[0])

    issues = []
    if report.missing_iculos_rows:
        issues.append(f"{len(report.missing_iculos_rows)} row(s) with missing ICULOS")
    if report.non_monotonic_rows:
        issues.append(f"{len(report.non_monotonic_rows)} row(s) with non-monotonic ICULOS")
    if report.duplicate_iculos_rows:
        issues.append(f"{len(report.duplicate_iculos_rows)} duplicate ICULOS value(s)")

    report.valid = (
        not report.missing_iculos_rows
        and not report.non_monotonic_rows
        and not report.duplicate_iculos_rows
    )
    return report


def validate_file(
    filepath: str | Path,
    patient_id: str | None = None,
    loader=None,
) -> FileValidationReport:
    """Run all validation checks on a single patient file.

    If ``loader`` is provided it is used to read the file; otherwise the
    file is read directly using the standard Challenge separator.
    """
    path = Path(filepath)
    if patient_id is None:
        from .dataset_loader import PhysioNetChallenge2019Loader

        tmp_loader = PhysioNetChallenge2019Loader(path.parent)
        patient_id = tmp_loader.patient_id_from_filename(path.name)

    report = FileValidationReport(
        patient_id=patient_id or path.stem,
        filepath=str(path),
    )

    if not path.exists():
        report.issues.append("File does not exist")
        report.malformed = True
        return report
    report.file_exists = True

    if not path.is_file() or not os.access(path, os.R_OK):
        report.issues.append("File is not readable")
        report.malformed = True
        return report
    report.file_readable = True

    try:
        if loader is not None:
            df = loader.load_file(path)
        else:
            from .dataset_loader import PhysioNetChallenge2019Loader

            df = PhysioNetChallenge2019Loader(path.parent).load_file(path)
    except Exception as exc:  # noqa: BLE001 - capture and report parse errors
        report.issues.append(f"Failed to parse file: {exc}")
        report.malformed = True
        return report

    if df is None:
        report.issues.append("File returned None (unreadable or malformed)")
        report.malformed = True
        return report

    report.n_rows = len(df)
    report.n_columns = df.shape[1]

    col_report = validate_columns(df)
    report.column_report = col_report
    if not col_report.valid:
        report.issues.extend(
            f"Missing column: {c}" for c in col_report.missing_columns
        )
        if col_report.extra_columns:
            report.issues.append(f"Extra column(s): {col_report.extra_columns}")

    # Hourly ordering on ICULOS.
    hourly = validate_hourly_ordering(df)
    report.hourly_report = hourly
    if hourly.missing_iculos_rows:
        report.issues.append(
            f"Missing ICULOS at row(s): {hourly.missing_iculos_rows[:5]}"
        )
    if hourly.non_monotonic_rows:
        report.issues.append(
            f"Non-monotonic ICULOS at row(s): {hourly.non_monotonic_rows[:5]}"
        )
    if hourly.duplicate_iculos_rows:
        report.issues.append(
            f"Duplicate ICULOS at row(s): {hourly.duplicate_iculos_rows[:5]}"
        )

    report.malformed = (
        not col_report.valid
        or not hourly.valid
        or len(report.issues) > 0
    )
    return report


def validate_directory(
    root: str | Path,
    limit: int | None = None,
) -> list[FileValidationReport]:
    """Validate every patient file under ``root``.

    Returns a list of FileValidationReport, one per patient file.
    Optionally cap the number of files validated (useful for smoke tests).
    """
    from .dataset_loader import PhysioNetChallenge2019Loader

    loader = PhysioNetChallenge2019Loader(root)
    catalog = loader.catalog()
    reports: list[FileValidationReport] = []
    for count, ref in enumerate(catalog.files):
        rep = validate_file(ref.filepath, patient_id=ref.patient_id, loader=loader)
        reports.append(rep)
        if limit is not None and count + 1 >= limit:
            break
    return reports


def summarize_validation(reports: list[FileValidationReport]) -> dict:
    """Aggregate a list of FileValidationReport into a summary dict."""
    total = len(reports)
    valid = sum(1 for r in reports if r.valid)
    malformed = sum(1 for r in reports if r.malformed)
    unreadable = sum(1 for r in reports if not r.file_readable)
    missing_cols = sum(
        1 for r in reports if r.column_report is not None and not r.column_report.valid
    )
    bad_ordering = sum(
        1 for r in reports if r.hourly_report is not None and not r.hourly_report.valid
    )
    total_rows = sum(r.n_rows for r in reports)
    return {
        "n_files": total,
        "n_valid": valid,
        "n_malformed": malformed,
        "n_unreadable": unreadable,
        "n_missing_columns": missing_cols,
        "n_bad_hourly_ordering": bad_ordering,
        "n_total_rows": total_rows,
        "valid_rate": (valid / total) if total else 0.0,
    }
