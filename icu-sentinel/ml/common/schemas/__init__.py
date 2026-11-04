# Shared schemas for ICU Sentinel P1.
# These define contracts between P1 data engineering, clinical baselines,
# classical ML, evaluation, and patient APIs so that P2/P3 can integrate later.

from dataclasses import dataclass


@dataclass
class PatientRecord:
    """A single patient ICU stay record."""
    patient_id: str
    hospital_id: str
    icu_id: str
    gender: str | None = None
    age: int | None = None
    height: float | None = None
    weight: float | None = None
    admission_type: str | None = None
    admitted_from: str | None = None
    died_in_hospital: int | None = None
    sepsis_label: int | None = None


@dataclass
class Observation:
    """A single hourly observation for a patient."""
    patient_id: str
    hour: int
    parameter: str
    value: float | None
    unit: str | None = None


@dataclass
class SplitMetadata:
    """Patient-level split metadata."""
    split_name: str  # train / val / test
    patient_ids: list[str]
    n_patients: int
    n_positive: int
    n_negative: int
    random_state: int


@dataclass
class FeatureSet:
    """Schema for engineered feature data."""
    patient_id: str
    hour: int
    features: dict
    sepsis_label: int


@dataclass
class ClinicalBaselineResult:
    """Result from a clinical baseline score (SIRS/qSOFA/NEWS)."""
    patient_id: str
    hour: int
    score: int
    positive: bool  # whether threshold met


@dataclass
class MetricResult:
    name: str
    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    bootstrap: bool = False
