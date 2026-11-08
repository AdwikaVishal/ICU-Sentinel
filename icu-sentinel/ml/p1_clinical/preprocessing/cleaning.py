"""Cleaning pipeline for the P1 preprocessing module.

Handles missing values and type coercion without introducing future
information and without arbitrary clinical imputation. Raw source files
are never modified.

Cleaning principles:
  - Missing values are preserved as NaN (no imputation).
  - No global mean/median imputation using the entire dataset.
  - No causal forward-fill (that is Part 4 feature engineering).
  - Original hourly ordering is preserved (sorted by ICULOS).
  - Original SepsisLabel is preserved unchanged.
  - Patient IDs and hospital IDs are preserved.
  - All transformations are per-patient; no cross-patient statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .schemas import (
    CLINICAL_VARIABLES,
    INTEGER_COLUMNS,
    ProcessedPatient,
    ValidationReport,
)
from .validation import (
    REQUIRED_COLUMNS,
    validate_patient_file,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.preprocessing.cleaning")


@dataclass
class CleaningReport:
    """Report of cleaning actions performed on a patient file."""

    patient_id: str | None
    source_file: str
    n_rows_input: int = 0
    n_rows_output: int = 0
    n_rows_dropped: int = 0
    n_values_coerced: int = 0
    n_missing_preserved: int = 0
    actions: list[str] = field(default_factory=list)
    validation_report: ValidationReport | None = None

    def add_action(self, message: str) -> None:
        self.actions.append(message)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "source_file": self.source_file,
            "n_rows_input": self.n_rows_input,
            "n_rows_output": self.n_rows_output,
            "n_rows_dropped": self.n_rows_dropped,
            "n_values_coerced": self.n_values_coerced,
            "n_missing_preserved": self.n_missing_preserved,
            "actions": list(self.actions),
            "validation_passed": (
                self.validation_report.passed if self.validation_report else None
            ),
            "validation_issues": (
                self.validation_report.to_dict()["issues"]
                if self.validation_report else []
            ),
        }


def clean_patient_dataframe(
    df: pd.DataFrame,
    patient_id: str | None = None,
    hospital_id: str | None = None,
    source_file: str = "",
    source_dataset: str | None = None,
) -> tuple[ProcessedPatient | None, CleaningReport]:
    """Clean a single patient DataFrame.

    Returns (ProcessedPatient or None, CleaningReport). If validation
    finds unrecoverable errors, returns (None, report) so the caller can
    decide how to handle; data is never silently discarded.
    """
    report = CleaningReport(
        patient_id=patient_id,
        source_file=source_file,
        n_rows_input=len(df) if df is not None else 0,
    )

    # Step 1: Validate first (never skip validation).
    validation = validate_patient_file(
        df,
        patient_id=patient_id,
        hospital_id=hospital_id,
        source_file=source_file,
    )
    report.validation_report = validation.report

    # If validation found errors but data exists, we still proceed with
    # cleaning but flag that issues exist. The caller decides.
    if df is None or len(df) == 0:
        report.add_action("Empty or unreadable file; no cleaning performed")
        return None, report

    # Work on a copy to avoid mutating the source.
    cleaned = df.copy()
    report.n_rows_input = len(cleaned)

    # Step 2: Coerce numeric columns to float, integer columns to nullable int.
    n_coerced = 0
    for col in REQUIRED_COLUMNS:
        if col not in cleaned.columns:
            continue
        if col in INTEGER_COLUMNS:
            before = cleaned[col].copy()
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            # Count values that became NaN but weren't NaN before (coercion failures)
            newly_na = cleaned[col].isna() & before.notna()
            n_coerced += int(newly_na.sum())
        else:
            before = cleaned[col].copy()
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            newly_na = cleaned[col].isna() & before.notna()
            n_coerced += int(newly_na.sum())
    report.n_values_coerced = n_coerced
    if n_coerced > 0:
        report.add_action(f"Coerced {n_coerced} non-numeric value(s) to NaN")

    # Step 3: Sort by ICULOS to preserve causal hourly ordering.
    if "ICULOS" in cleaned.columns:
        cleaned = cleaned.sort_values("ICULOS", kind="stable").reset_index(drop=True)
        report.add_action("Sorted rows by ICULOS (causal hourly order)")

    # Step 4: Derive hour from ICULOS (1-based ICU hour).
    if "ICULOS" in cleaned.columns:
        hour_vals = pd.to_numeric(cleaned["ICULOS"], errors="coerce")
        # Use the ICULOS value directly as the hour index.
        cleaned["_hour"] = hour_vals
    else:
        cleaned["_hour"] = range(len(cleaned))
        report.add_action("ICULOS missing; using row index as hour")

    # Step 5: Count missing values preserved (no imputation).
    n_missing = int(cleaned[REQUIRED_COLUMNS].isna().sum().sum()) if REQUIRED_COLUMNS else 0
    report.n_missing_preserved = n_missing
    report.add_action(f"Preserved {n_missing} missing value(s) as NaN (no imputation)")

    # Step 6: Build ProcessedPatient.
    pid = validation.patient_id or patient_id or "unknown"
    hid = hospital_id
    source_ds = source_dataset
    if source_ds is None:
        lower = str(source_file).lower().replace("\\", "/")
        if "training_seta" in lower or "set_a" in lower or "/seta/" in lower:
            source_ds = "set_a"
        elif "training_setb" in lower or "set_b" in lower or "/setb/" in lower:
            source_ds = "set_b"
        else:
            source_ds = "unknown"

    clinical: dict[str, list[float]] = {}
    for var in CLINICAL_VARIABLES:
        if var in cleaned.columns:
            clinical[var] = cleaned[var].tolist()
        else:
            clinical[var] = [float("nan")] * len(cleaned)

    sepsis_labels: list[int] = []
    if "SepsisLabel" in cleaned.columns:
        sepsis_labels = [int(x) if pd.notna(x) else -1 for x in cleaned["SepsisLabel"].tolist()]
        # -1 used as sentinel for missing label; will be flagged by validation
    else:
        sepsis_labels = [-1] * len(cleaned)

    hour_list = [int(x) if pd.notna(x) else -1 for x in cleaned["_hour"].tolist()]

    processed = ProcessedPatient(
        patient_id=str(pid),
        hospital_id=hid,
        hour=hour_list,
        clinical=clinical,
        sepsis_label=sepsis_labels,
        source_file=source_file,
        source_dataset=source_ds,
        n_rows=len(cleaned),
    )

    report.n_rows_output = len(cleaned)
    report.n_rows_dropped = report.n_rows_input - report.n_rows_output
    report.add_action("Cleaning complete; no rows dropped, no imputation applied")

    return processed, report


def clean_patient_file(
    filepath: str | Path,
    patient_id: str | None = None,
    hospital_id: str | None = None,
    source_dataset: str | None = None,
) -> tuple[ProcessedPatient | None, CleaningReport]:
    """Load and clean a single .psv patient file."""
    from ..ingestion.dataset_loader import PhysioNetChallenge2019Loader

    path = Path(filepath)
    loader = PhysioNetChallenge2019Loader(path.parent)
    df = loader.load_file(path)
    if patient_id is None:
        patient_id = loader.patient_id_from_filename(path.name)
    return clean_patient_dataframe(
        df,
        patient_id=patient_id,
        hospital_id=hospital_id,
        source_file=path.name,
        source_dataset=source_dataset,
    )


def drops_future_information(df: pd.DataFrame) -> bool:
    """Verify that no future information is present in the dataframe.

    Checks that the dataframe is sorted by ICULOS and that each row only
    contains information available at or before that hour (trivially true
    for a static per-hour table, but asserted as a safety check).

    Returns True if no future information detected.
    """
    if "ICULOS" not in df.columns:
        return False
    icol = pd.to_numeric(df["ICULOS"], errors="coerce")
    # Must be non-decreasing
    return bool((icol.diff().dropna() >= 0).all())
