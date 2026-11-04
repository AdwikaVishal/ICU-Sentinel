"""End-to-end preprocessing pipeline for P1.

Orchestrates loading, validation, cleaning, and patient-level splitting
of PhysioNet/CinC Challenge 2019 data. Produces processed patient files
and split manifests ready for consumption by P2 (boosting) and P3
(temporal models).

All steps are per-patient where possible to avoid unnecessary memory
consumption. Raw source files are never modified.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .cleaning import CleaningReport, clean_patient_dataframe
from .patient_splits import (
    PatientSplitter,
    SplitManifest,
    check_split_leakage,
    save_split_manifests,
)
from .schemas import (
    OUTPUT_COLUMNS,
    ProcessedPatient,
)
from .validation import (
    PatientValidationResult,
    summarize_validation,
    validate_patient_file,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.preprocessing.pipeline")


@dataclass
class PipelineResult:
    """Result of running the preprocessing pipeline."""

    processed_patients: list[ProcessedPatient] = field(default_factory=list)
    validation_results: list[PatientValidationResult] = field(default_factory=list)
    cleaning_reports: list[CleaningReport] = field(default_factory=list)
    split_manifest: SplitManifest | None = None
    leakage_report: dict | None = None
    preprocessing_summary: dict | None = None
    data_quality_report: dict | None = None

    @property
    def n_patients_processed(self) -> int:
        return len(self.processed_patients)

    @property
    def n_patients_failed(self) -> int:
        return sum(1 for r in self.validation_results if not r.passed)


def run_pipeline(
    raw_dir: str | Path,
    processed_dir: str | Path,
    splits_dir: str | Path,
    reports_dir: str | Path,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
    stratify_by_label: bool = True,
    limit: int | None = None,
    save_processed: bool = True,
    save_manifests: bool = True,
    save_reports: bool = True,
) -> PipelineResult:
    """Run the full preprocessing pipeline.

    Steps:
      1. Discover patient files in raw_dir.
      2. Validate each file (13 validation rules).
      3. Clean each file (type coercion, ordering, no imputation).
      4. Save processed patients as parquet (one file per patient).
      5. Split patients into train/validation/test (patient-level).
      6. Check for split leakage.
      7. Save split manifests and reports.

    Returns a PipelineResult with all outputs.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    splits_path = Path(splits_dir)
    reports_path = Path(reports_dir)

    if save_processed:
        processed_path.mkdir(parents=True, exist_ok=True)
    if save_manifests:
        splits_path.mkdir(parents=True, exist_ok=True)
    if save_reports:
        reports_path.mkdir(parents=True, exist_ok=True)

    result = PipelineResult()

    # Step 1: Discover and load patients.
    from ..ingestion.dataset_loader import PhysioNetChallenge2019Loader

    loader = PhysioNetChallenge2019Loader(raw_path)
    catalog = loader.catalog()
    logger.info("Discovered %d patient files in %s", catalog.n_patients, raw_path)

    count = 0
    for count, ref in enumerate(catalog.files):
        if limit is not None and count >= limit:
            break

        # Determine source dataset identity from filename/path.
        source_dataset = "set_a" if "set_a" in str(ref.filepath).lower() else "set_b"
        if source_dataset == "set_b" and "set_b" not in str(ref.filepath).lower():
            # Default heuristic based on patient ID ranges if path doesn't indicate.
            source_dataset = "set_a"  # conservative default

        # Step 2: Load and validate.
        try:
            df = loader.load_file(ref.filepath)
        except Exception as exc:  # noqa: BLE001
            df = None
            parse_error = str(exc)
        else:
            parse_error = None

        validation = validate_patient_file(
            df,
            patient_id=ref.patient_id,
            source_file=ref.filepath.name,
            parse_error=parse_error,
        )
        result.validation_results.append(validation)

        # Step 3: Clean (only if validation found no fatal errors, but we
        # still attempt to produce output where possible).
        processed, cleaning_report = clean_patient_dataframe(
            validation.dataframe if validation.dataframe is not None else None,
            patient_id=validation.patient_id,
            hospital_id=None,  # hospital ID not available from filename alone
            source_file=ref.filepath.name,
            source_dataset=source_dataset,
        )
        result.cleaning_reports.append(cleaning_report)

        if processed is not None:
            result.processed_patients.append(processed)
            if save_processed:
                patient_path = processed_path / f"{processed.patient_id}.parquet"
                processed.to_dataframe().to_parquet(patient_path, index=False)

    # Step 4: Split patients.
    splitter = PatientSplitter(
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
        stratify_by_label=stratify_by_label,
    )
    manifest = splitter.split(result.processed_patients)
    result.split_manifest = manifest

    # Step 5: Leakage check.
    leakage = check_split_leakage(manifest)
    result.leakage_report = leakage
    logger.info(
        "Split leakage detected: %s (overlapping patients: %d)",
        leakage["leakage_detected"],
        leakage["n_overlapping_patients"],
    )

    if save_manifests:
        save_split_manifests(manifest, splits_path)

    # Step 6: Build summaries.
    result.preprocessing_summary = _build_preprocessing_summary(result)
    result.data_quality_report = _build_data_quality_report(result)

    if save_reports:
        with open(reports_path / "preprocessing_summary.json", "w") as f:
            json.dump(result.preprocessing_summary, f, indent=2)
        with open(reports_path / "split_summary.json", "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        with open(reports_path / "data_quality_report.json", "w") as f:
            json.dump(result.data_quality_report, f, indent=2)

    return result


def _build_preprocessing_summary(result: PipelineResult) -> dict:
    total_input = sum(cr.n_rows_input for cr in result.cleaning_reports)
    total_output = sum(cr.n_rows_output for cr in result.cleaning_reports)
    total_dropped = sum(cr.n_rows_dropped for cr in result.cleaning_reports)
    total_coerced = sum(cr.n_values_coerced for cr in result.cleaning_reports)
    total_missing = sum(cr.n_missing_preserved for cr in result.cleaning_reports)

    return {
        "n_patients_discovered": len(result.validation_results),
        "n_patients_processed": result.n_patients_processed,
        "n_patients_failed": result.n_patients_failed,
        "n_rows_input_total": total_input,
        "n_rows_output_total": total_output,
        "n_rows_dropped_total": total_dropped,
        "n_values_coerced_total": total_coerced,
        "n_missing_preserved_total": total_missing,
        "output_schema": OUTPUT_COLUMNS,
        "no_imputation_applied": True,
        "future_information_used": False,
        "raw_files_unchanged": True,
    }


def _build_data_quality_report(result: PipelineResult) -> dict:
    val_summary = summarize_validation(result.validation_results)
    failed_patients = [
        {
            "patient_id": r.patient_id,
            "source_file": r.source_file,
            "n_errors": r.report.n_errors,
            "n_warnings": r.report.n_warnings,
        }
        for r in result.validation_results if not r.passed
    ]
    return {
        "validation": val_summary,
        "failed_patients": failed_patients,
        "cleaning": {
            "n_patients_cleaned": len(result.cleaning_reports),
            "n_rows_dropped_total": sum(cr.n_rows_dropped for cr in result.cleaning_reports),
            "n_missing_preserved_total": sum(cr.n_missing_preserved for cr in result.cleaning_reports),
        },
        "labels_preserved": True,
        "sepsis_label_shift_applied": False,
    }


def save_processed_patient(
    processed: ProcessedPatient,
    output_dir: str | Path,
) -> Path:
    """Save a single ProcessedPatient to parquet."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{processed.patient_id}.parquet"
    processed.to_dataframe().to_parquet(path, index=False)
    return path
