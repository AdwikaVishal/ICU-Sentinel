"""P1 clinical preprocessing package."""

from .cleaning import (
    CleaningReport,
    clean_patient_dataframe,
    clean_patient_file,
    drops_future_information,
)
from .patient_splits import (
    PatientSplitEntry,
    PatientSplitter,
    SplitManifest,
    check_split_leakage,
    save_split_manifests,
)
from .pipeline import (
    PipelineResult,
    run_pipeline,
    save_processed_patient,
)
from .schemas import (
    CLINICAL_VARIABLES,
    IDENTIFIER_COLUMNS,
    INTEGER_COLUMNS,
    OUTPUT_COLUMNS,
    REQUIRED_OUTPUT_COLUMNS,
    ProcessedPatient,
    ValidationIssue,
    ValidationReport,
)
from .validation import (
    PatientValidationResult,
    summarize_validation,
    validate_directory,
    validate_patient_file,
)
from .validation import (
    ValidationReport as _ValReport,  # noqa: F401  (re-export alias avoided)
)

__all__ = [
    "CLINICAL_VARIABLES",
    "IDENTIFIER_COLUMNS",
    "INTEGER_COLUMNS",
    "OUTPUT_COLUMNS",
    "REQUIRED_OUTPUT_COLUMNS",
    "CleaningReport",
    "PatientSplitEntry",
    "PatientSplitter",
    "PatientValidationResult",
    "PipelineResult",
    "ProcessedPatient",
    "SplitManifest",
    "ValidationIssue",
    "ValidationReport",
    "check_split_leakage",
    "clean_patient_dataframe",
    "clean_patient_file",
    "drops_future_information",
    "run_pipeline",
    "save_processed_patient",
    "save_split_manifests",
    "summarize_validation",
    "validate_directory",
    "validate_patient_file",
]
