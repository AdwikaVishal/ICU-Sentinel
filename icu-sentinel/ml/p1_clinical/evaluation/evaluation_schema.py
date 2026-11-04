"""Shared evaluation schema for P1 research evaluation.

Defines the contract for all evaluation outputs including classification
metrics, calibration, lead time, alert burden, feature ablation, subgroup
analysis, bootstrap confidence intervals, cross-hospital results, and model
comparison.

All schemas include a `source` field to distinguish synthetic verification
from real-data results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ClassificationMetricEntry:
    """A single classification metric result."""

    metric_name: str
    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    bootstrap: bool = False
    n_patients: int = 0
    n_rows: int = 0
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassificationMetricsResult:
    """Complete classification metrics result for one model."""

    model_name: str
    model_version: str
    split: str
    metrics: list[ClassificationMetricEntry]
    confusion_matrix: dict[str, int]  # tp, tn, fp, fn
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationEntry:
    """A single calibration bin result."""

    bin_index: int
    bin_lower: float
    bin_upper: float
    bin_center: float
    n_samples: int
    n_positive: int
    mean_predicted_probability: float
    observed_frequency: float
    calibration_error: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationResult:
    """Complete calibration result for one model."""

    model_name: str
    model_version: str
    split: str
    expected_calibration_error: float
    max_calibration_error: float
    bins: list[CalibrationEntry]
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeadTimeEntry:
    """A single lead-time result for one patient."""

    patient_id: str
    source_dataset: str | None
    first_alert_hour: int | None
    first_positive_label_hour: int | None
    lead_time_hours: float | None
    alert_before_onset: bool  # alert at or before first positive label
    alert_after_onset: bool  # alert after first positive label
    missed: bool  # no alert at all

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeadTimeResult:
    """Complete lead-time result."""

    model_name: str
    model_version: str
    split: str
    n_patients: int
    n_with_alert: int
    n_with_onset: int
    n_alert_before_onset: int
    n_alert_after_onset: int
    n_missed: int
    mean_lead_time_hours: float | None
    median_lead_time_hours: float | None
    detection_rate_before_onset: float
    detection_rate_at_or_before_onset: float
    detection_rate_after_onset: float
    missed_patient_rate: float
    patient_details: list[LeadTimeEntry]
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlertEpisode:
    """A single alert episode for one patient."""

    patient_id: str
    source_dataset: str | None
    episode_start_hour: int
    episode_end_hour: int
    episode_duration_hours: float
    n_positive_predictions: int
    is_true_alert: bool  # patient has sepsis
    is_false_alert: bool  # patient does not have sepsis

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlertBurdenResult:
    """Complete alert burden result."""

    model_name: str
    model_version: str
    split: str
    n_patients: int
    n_patient_days: float
    n_alert_episodes: int
    n_true_alert_episodes: int
    n_false_alert_episodes: int
    n_patients_with_false_alert: int
    false_alerts_per_patient_day: float
    alerts_per_patient: float
    mean_alert_duration_hours: float | None
    episodes: list[AlertEpisode]
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureAblationResult:
    """Result of ablating one feature group."""

    feature_group: str
    n_features_removed: int
    metrics_full: dict[str, float]
    metrics_ablated: dict[str, float]
    metric_changes: dict[str, float]  # ablated - full
    metric_changes_relative: dict[str, float]  # (ablated - full) / full
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureAblationSummary:
    """Complete feature ablation summary."""

    model_name: str
    model_version: str
    split: str
    n_features_full: int
    ablations: list[FeatureAblationResult]
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SubgroupResult:
    """Subgroup evaluation result for one subgroup."""

    subgroup_name: str
    subgroup_value: str
    n_patients: int
    n_positive_patients: int
    n_rows: int
    n_positive_rows: int
    positive_rate: float
    metrics: dict[str, float]
    ci_lower: dict[str, float | None]
    ci_upper: dict[str, float | None]
    bootstrap: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SubgroupAnalysisResult:
    """Complete subgroup analysis result."""

    model_name: str
    model_version: str
    split: str
    subgroup_dimension: str  # e.g., "age_group", "gender", "source_dataset"
    subgroups: list[SubgroupResult]
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result for one metric."""

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_iterations: int
    n_patients: int
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BootstrapSummary:
    """Complete bootstrap confidence interval summary."""

    model_name: str
    model_version: str
    split: str
    metrics: list[BootstrapResult]
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrossHospitalResult:
    """Cross-hospital evaluation result."""

    train_hospital: str
    test_hospital: str
    model_name: str
    model_version: str
    n_patients_train: int
    n_patients_test: int
    n_rows_test: int
    metrics: dict[str, float]
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelComparisonEntry:
    """A single model's comparison entry."""

    model_name: str
    model_version: str
    model_type: str  # classical_ml / clinical_baseline / boosting / temporal
    split: str
    metrics: dict[str, float]
    n_patients: int
    n_rows: int
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelComparisonResult:
    """Complete model comparison result."""

    split: str
    models: list[ModelComparisonEntry]
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation result aggregating all components."""

    config_version: str = "1.0.0"
    evaluation_timestamp: str = ""
    source: str = "synthetic"

    classification_metrics: list[ClassificationMetricsResult] = field(
        default_factory=list
    )
    calibration_results: list[CalibrationResult] = field(default_factory=list)
    lead_time_results: list[LeadTimeResult] = field(default_factory=list)
    alert_burden_results: list[AlertBurdenResult] = field(default_factory=list)
    feature_ablation_results: list[FeatureAblationSummary] = field(
        default_factory=list
    )
    subgroup_results: list[SubgroupAnalysisResult] = field(default_factory=list)
    bootstrap_results: list[BootstrapSummary] = field(default_factory=list)
    cross_hospital_results: list[CrossHospitalResult] = field(
        default_factory=list
    )
    model_comparison_results: list[ModelComparisonResult] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> EvaluationResult:
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> EvaluationResult:
        return cls(**data)


# Module-level constant
EVALUATION_SCHEMA_VERSION: str = "1.0.0"
