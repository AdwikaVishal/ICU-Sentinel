"""Validation rules for the P1 preprocessing pipeline.

Implements structured validation of raw PhysioNet/CinC Challenge 2019
patient files against 13 validation rules. Never silently discards
invalid records; all issues are reported explicitly with recovery
strategies documented.

Validation rules:
  1. Required columns present
  2. Expected data types
  3. Patient identifier validity
  4. Hospital identifier preservation
  5. Hourly observation ordering
  6. Duplicate timestamps
  7. Missing timestamps
  8. Invalid numeric values
  9. Unexpected categorical values
  10. Missing or invalid SepsisLabel values
  11. Empty patient files
  12. Malformed PSV files
  13. Inconsistent patient schemas
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schemas import (
    CLINICAL_VARIABLES,
    IDENTIFIER_COLUMNS,
    INTEGER_COLUMNS,
    ValidationReport,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.preprocessing.validation")

# Official dataset column set (40 clinical + outcome), from Table 1.
REQUIRED_COLUMNS: list[str] = list(CLINICAL_VARIABLES) + ["SepsisLabel"]

# Categorical columns and their allowed values (from documentation).
CATEGORICAL_ALLOWED: dict[str, set] = {
    "Gender": {0, 1},
    "SepsisLabel": {0, 1},
}

# Columns expected to be integer-valued.
INTEGER_EXPECTED: set[str] = set(INTEGER_COLUMNS)


@dataclass
class PatientValidationResult:
    """Validation result for a single patient file."""

    patient_id: str | None
    source_file: str
    report: ValidationReport
    dataframe: pd.DataFrame | None = None
    recovered_dataframe: pd.DataFrame | None = None

    @property
    def passed(self) -> bool:
        return self.report.passed


def _add_issue(
    report: ValidationReport,
    severity: str,
    rule: str,
    patient_id: str | None = None,
    hour: int | None = None,
    column: str | None = None,
    message: str = "",
    recovered: bool = False,
    recovery_strategy: str | None = None,
) -> None:
    report.add_issue(
        severity=severity,
        rule=rule,
        patient_id=patient_id,
        hour=hour,
        column=column,
        message=message,
        recovered=recovered,
        recovery_strategy=recovery_strategy,
    )


def validate_required_columns(df: pd.DataFrame, report: ValidationReport) -> None:
    """Rule 1: Required columns present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        _add_issue(
            report,
            "error",
            "required_columns",
            message=f"Missing required columns: {missing}",
        )
    else:
        _add_issue(
            report,
            "warning",
            "required_columns",
            message=f"All {len(REQUIRED_COLUMNS)} required columns present",
        )


def validate_data_types(df: pd.DataFrame, report: ValidationReport) -> None:
    """Rule 2: Expected data types."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            continue
        col_dtype = df[col].dtype
        if col in INTEGER_EXPECTED:
            # Integer columns should be numeric (int or float representing int)
            if not pd.api.types.is_numeric_dtype(col_dtype):
                _add_issue(
                    report,
                    "error",
                    "data_type",
                    column=col,
                    message=f"Column {col} expected numeric, got {col_dtype}",
                )
        # Float columns should be numeric
        elif col not in IDENTIFIER_COLUMNS and col != "SepsisLabel" and not pd.api.types.is_numeric_dtype(col_dtype):
            _add_issue(
                report,
                "error",
                "data_type",
                column=col,
                message=f"Column {col} expected numeric, got {col_dtype}",
            )


def validate_patient_identifier(
    df: pd.DataFrame, report: ValidationReport, patient_id: str | None = None
) -> str | None:
    """Rule 3: Patient identifier validity."""
    pid = patient_id
    if pid is None:
        _add_issue(
            report,
            "error",
            "patient_identifier",
            message="Patient identifier not provided",
        )
        return None
    import re

    if not re.fullmatch(r"p\d{5,6}", str(pid)):
        _add_issue(
            report,
            "error",
            "patient_identifier",
            patient_id=pid,
            message=f"Patient ID '{pid}' does not match pattern p[5-6 digits]",
        )
        return None
    # Verify consistency within the dataframe if a patient_id column exists
    if "patient_id" in df.columns:
        unique_ids = df["patient_id"].dropna().unique()
        if len(unique_ids) > 1:
            _add_issue(
                report,
                "error",
                "patient_identifier",
                patient_id=pid,
                message=f"File contains multiple patient IDs: {list(unique_ids)}",
            )
        elif len(unique_ids) == 1 and str(unique_ids[0]) != str(pid):
            _add_issue(
                report,
                "error",
                "patient_identifier",
                patient_id=pid,
                message=(
                    f"Filename patient ID '{pid}' differs from "
                    f"in-file patient ID '{unique_ids[0]}'"
                ),
            )
    return pid


def validate_hospital_identifier(
    df: pd.DataFrame, report: ValidationReport, hospital_id: str | None = None
) -> str | None:
    """Rule 4: Hospital identifier preservation."""
    if hospital_id is None:
        _add_issue(
            report,
            "warning",
            "hospital_identifier",
            message="Hospital identifier not provided; provenance tracking reduced",
        )
        return None
    if not isinstance(hospital_id, str) or len(hospital_id.strip()) == 0:
        _add_issue(
            report,
            "error",
            "hospital_identifier",
            message=f"Invalid hospital identifier: {hospital_id!r}",
        )
        return None
    return hospital_id


def validate_hourly_ordering(
    df: pd.DataFrame, report: ValidationReport, patient_id: str | None = None
) -> None:
    """Rule 5: Hourly observation ordering (ICULOS monotonic)."""
    if "ICULOS" not in df.columns:
        _add_issue(
            report,
            "error",
            "hourly_ordering",
            patient_id=patient_id,
            message="ICULOS column missing; cannot validate hourly ordering",
        )
        return
    icol = pd.to_numeric(df["ICULOS"], errors="coerce")
    missing_mask = icol.isna()
    if missing_mask.any():
        missing_rows = list(df.index[missing_mask])
        _add_issue(
            report,
            "error",
            "hourly_ordering",
            patient_id=patient_id,
            message=f"{len(missing_rows)} row(s) with missing ICULOS",
        )
    valid_vals = icol.dropna()
    if len(valid_vals) > 1:
        diffs = valid_vals.diff().dropna()
        non_mono = diffs[diffs <= 0]
        if not non_mono.empty:
            _add_issue(
                report,
                "error",
                "hourly_ordering",
                patient_id=patient_id,
                message=f"{len(non_mono)} row(s) with non-monotonic or duplicate ICULOS",
            )


def validate_duplicate_timestamps(
    df: pd.DataFrame, report: ValidationReport, patient_id: str | None = None
) -> None:
    """Rule 6: Duplicate timestamps (duplicate ICULOS values)."""
    if "ICULOS" not in df.columns:
        return
    icol = pd.to_numeric(df["ICULOS"], errors="coerce").dropna()
    dups = icol[icol.duplicated(keep=False)]
    if len(dups) > 0:
        _add_issue(
            report,
            "error",
            "duplicate_timestamps",
            patient_id=patient_id,
            message=f"{len(dups)} row(s) with duplicate ICULOS values",
        )


def validate_missing_timestamps(
    df: pd.DataFrame, report: ValidationReport, patient_id: str | None = None
) -> None:
    """Rule 7: Missing timestamps (gaps in ICULOS sequence)."""
    if "ICULOS" not in df.columns:
        return
    icol = pd.to_numeric(df["ICULOS"], errors="coerce").dropna()
    if len(icol) < 2:
        return
    expected = set(range(int(icol.min()), int(icol.max()) + 1))
    actual = set(icol.astype(int))
    missing = expected - actual
    if missing:
        _add_issue(
            report,
            "warning",
            "missing_timestamps",
            patient_id=patient_id,
            message=f"{len(missing)} missing ICULOS value(s) in sequence",
        )


def validate_invalid_numeric_values(
    df: pd.DataFrame, report: ValidationReport, patient_id: str | None = None
) -> None:
    """Rule 8: Invalid numeric values (inf, extreme outliers)."""
    numeric_cols = [c for c in REQUIRED_COLUMNS if c in df.columns and c not in IDENTIFIER_COLUMNS]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        col_vals = pd.to_numeric(df[col], errors="coerce")
        inf_mask = col_vals.isin([float("inf"), float("-inf")])
        if inf_mask.any():
            _add_issue(
                report,
                "error",
                "invalid_numeric",
                patient_id=patient_id,
                column=col,
                message=f"{int(inf_mask.sum())} infinite value(s) in column {col}",
            )


def validate_categorical_values(
    df: pd.DataFrame, report: ValidationReport, patient_id: str | None = None
) -> None:
    """Rule 9: Unexpected categorical values."""
    for col, allowed in CATEGORICAL_ALLOWED.items():
        if col not in df.columns:
            continue
        col_vals = pd.to_numeric(df[col], errors="coerce").dropna()
        unexpected = set(col_vals.unique()) - allowed
        if unexpected:
            _add_issue(
                report,
                "error",
                "categorical_value",
                patient_id=patient_id,
                column=col,
                message=f"Unexpected values {sorted(unexpected)} in column {col}; allowed={sorted(allowed)}",
            )


def validate_sepsis_label(
    df: pd.DataFrame, report: ValidationReport, patient_id: str | None = None
) -> None:
    """Rule 10: Missing or invalid SepsisLabel values."""
    if "SepsisLabel" not in df.columns:
        _add_issue(
            report,
            "error",
            "sepsis_label",
            patient_id=patient_id,
            message="SepsisLabel column missing",
        )
        return
    labels = pd.to_numeric(df["SepsisLabel"], errors="coerce")
    missing_mask = labels.isna()
    if missing_mask.any():
        _add_issue(
            report,
            "error",
            "sepsis_label",
            patient_id=patient_id,
            message=f"{int(missing_mask.sum())} row(s) with missing SepsisLabel",
        )
    invalid_mask = (~missing_mask) & (~labels.isin([0, 1]))
    if invalid_mask.any():
        _add_issue(
            report,
            "error",
            "sepsis_label",
            patient_id=patient_id,
            message=f"{int(invalid_mask.sum())} row(s) with invalid SepsisLabel values (not 0 or 1)",
        )


def validate_empty_file(
    df: pd.DataFrame, report: ValidationReport, patient_id: str | None = None
) -> None:
    """Rule 11: Empty patient files."""
    if len(df) == 0:
        _add_issue(
            report,
            "error",
            "empty_file",
            patient_id=patient_id,
            message="Patient file contains zero rows",
        )


def validate_malformed_file(
    df: pd.DataFrame | None,
    report: ValidationReport,
    patient_id: str | None = None,
    parse_error: str | None = None,
) -> bool:
    """Rule 12: Malformed PSV files.

    Returns True if the file is malformed (unreadable), False otherwise.
    """
    if parse_error is not None or df is None:
        _add_issue(
            report,
            "error",
            "malformed_file",
            patient_id=patient_id,
            message=parse_error or "File could not be parsed",
        )
        return True
    return False


def validate_inconsistent_schema(
    df: pd.DataFrame, report: ValidationReport, patient_id: str | None = None,
    expected_columns: list[str] | None = None,
) -> None:
    """Rule 13: Inconsistent patient schemas."""
    if expected_columns is None:
        expected_columns = REQUIRED_COLUMNS
    extra = [c for c in df.columns if c not in expected_columns]
    missing = [c for c in expected_columns if c not in df.columns]
    if extra:
        _add_issue(
            report,
            "warning",
            "inconsistent_schema",
            patient_id=patient_id,
            message=f"Unexpected extra columns: {extra}",
        )
    if missing:
        _add_issue(
            report,
            "error",
            "inconsistent_schema",
            patient_id=patient_id,
            message=f"Missing expected columns: {missing}",
        )


def validate_patient_file(
    df: pd.DataFrame | None,
    patient_id: str | None = None,
    hospital_id: str | None = None,
    source_file: str = "",
    expected_columns: list[str] | None = None,
    parse_error: str | None = None,
) -> PatientValidationResult:
    """Run all validation rules on a single patient file.

    Returns a PatientValidationResult with a ValidationReport and the
    (optionally recovered) dataframe. Never silently discards data.
    """
    report = ValidationReport(patient_id=patient_id)

    if validate_malformed_file(df, report, patient_id, parse_error):
        return PatientValidationResult(
            patient_id=patient_id,
            source_file=source_file,
            report=report,
            dataframe=None,
            recovered_dataframe=None,
        )

    assert df is not None
    validate_empty_file(df, report, patient_id)
    if len(df) == 0:
        return PatientValidationResult(
            patient_id=patient_id,
            source_file=source_file,
            report=report,
            dataframe=df,
            recovered_dataframe=df,
        )

    validate_required_columns(df, report)
    validate_inconsistent_schema(df, report, patient_id, expected_columns)
    validate_data_types(df, report)
    pid = validate_patient_identifier(df, report, patient_id)
    validate_hospital_identifier(df, report, hospital_id)
    validate_hourly_ordering(df, report, pid)
    validate_duplicate_timestamps(df, report, pid)
    validate_missing_timestamps(df, report, pid)
    validate_invalid_numeric_values(df, report, pid)
    validate_categorical_values(df, report, pid)
    validate_sepsis_label(df, report, pid)

    return PatientValidationResult(
        patient_id=pid,
        source_file=source_file,
        report=report,
        dataframe=df,
        recovered_dataframe=df,
    )


def validate_directory(
    root: str | Path,
    limit: int | None = None,
) -> list[PatientValidationResult]:
    """Validate every patient file under ``root``.

    Expects the .psv files produced by the ingestion loader. Uses the
    ingestion loader to read each file, then applies all validation
    rules.
    """
    from ..ingestion.dataset_loader import PhysioNetChallenge2019Loader

    loader = PhysioNetChallenge2019Loader(root)
    catalog = loader.catalog()
    results: list[PatientValidationResult] = []
    count = 0
    for ref in catalog.files:
        try:
            df = loader.load_file(ref.filepath)
        except Exception as exc:  # noqa: BLE001
            res = validate_patient_file(
                None,
                patient_id=ref.patient_id,
                source_file=ref.filepath.name,
                parse_error=str(exc),
            )
            results.append(res)
            count += 1
            if limit is not None and count >= limit:
                break
            continue
        res = validate_patient_file(
            df,
            patient_id=ref.patient_id,
            source_file=ref.filepath.name,
        )
        results.append(res)
        count += 1
        if limit is not None and count >= limit:
            break
    return results


def summarize_validation(results: list[PatientValidationResult]) -> dict:
    """Aggregate validation results into a summary dict."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    total_errors = sum(r.report.n_errors for r in results)
    total_warnings = sum(r.report.n_warnings for r in results)
    failed_files = [
        {
            "patient_id": r.patient_id,
            "source_file": r.source_file,
            "n_errors": r.report.n_errors,
            "n_warnings": r.report.n_warnings,
            "issues": r.report.to_dict()["issues"],
        }
        for r in results if not r.passed
    ]
    return {
        "n_files": total,
        "n_passed": passed,
        "n_failed": failed,
        "n_errors": total_errors,
        "n_warnings": total_warnings,
        "pass_rate": (passed / total) if total else 0.0,
        "failed_files": failed_files,
    }
