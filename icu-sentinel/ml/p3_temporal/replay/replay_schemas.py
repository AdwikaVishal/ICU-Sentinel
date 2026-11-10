"""API and service schemas for the P3 patient replay engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ml.p3_temporal.alerts.alert_schemas import AlertRecord
from ml.p3_temporal.common.prediction import PredictionResult

DISCLAIMER = (
    "Retrospective research prototype. Predictions and alerts are not a "
    "diagnosis, do not confirm sepsis, and must not be used for clinical "
    "decisions. Synthetic demonstration data and predictions are labelled."
)


class ObservationValue(BaseModel):
    """One observation in the replay panel.

    ``available=False`` means the measurement was not recorded. It is not
    a normal value and must not be rendered as one.
    """

    parameter: str
    display_name: str
    category: str
    value: float | None = None
    unit: str | None = None
    available: bool


class PatientSummary(BaseModel):
    patient_id: str
    n_hours: int = Field(ge=0, description="Number of recorded hourly rows")
    total_hours: int = Field(ge=0, description="Timeline span including missing hours")
    recorded_hours: int = Field(ge=0)
    missing_hours: int = Field(ge=0)
    observation_count: int = Field(ge=0)
    data_source: str
    source_dataset: str | None = None
    is_synthetic: bool
    has_retrospective_label: bool = False


class TimelineHourSummary(BaseModel):
    hour: int
    available_observation_count: int = Field(ge=0)
    unavailable_observation_count: int = Field(ge=0)
    has_observations: bool
    retrospective_label: int | None = None
    label_available: bool = False
    prediction_status: str = "not_evaluated"


class ReplaySession(BaseModel):
    patient_id: str
    current_hour: int
    min_hour: int
    max_hour: int
    total_hours: int
    at_start: bool
    at_end: bool


class CausalityMetadata(BaseModel):
    max_input_hour: int
    prefix_row_count: int = Field(ge=0)
    input_observation_count: int = Field(ge=0)
    future_data_used: bool = False
    retrospective_label_used_as_input: bool = False
    model_input_hash: str | None = None
    feature_schema_version: str | None = None


class ReplayState(BaseModel):
    session: ReplaySession
    patient: PatientSummary
    hour: int
    timeline: TimelineHourSummary
    observations: list[ObservationValue]
    prediction: PredictionResult
    predictions: list[PredictionResult] = Field(default_factory=list)
    alert: AlertRecord | None = None
    alert_status: str = "not_evaluated"
    alert_error: str | None = None
    retrospective_label: int | None = None
    retrospective_label_available: bool = False
    label_disclosure: str = (
        "Retrospective demonstration label shown for evaluation only; never "
        "used as a model input or alert trigger."
    )
    data_source: str
    is_synthetic: bool
    prediction_status: str
    explanation_available: bool = False
    causality: CausalityMetadata
    disclaimer: str = DISCLAIMER


class PatientListResponse(BaseModel):
    patients: list[PatientSummary]
    count: int
    data_source: str
    is_synthetic: bool
    prediction_source: str
    provider: str
    disclaimer: str = DISCLAIMER


class TimelineResponse(BaseModel):
    patient: PatientSummary
    hours: list[TimelineHourSummary]
    count: int
    missing_hours: int
    data_source: str
    is_synthetic: bool
    disclaimer: str = DISCLAIMER


class ReplayStepRequest(BaseModel):
    hour: int = Field(ge=0)
    direction: Literal["forward", "backward"] = "forward"
    evaluate_alerts: bool = True


class ReplayJumpRequest(BaseModel):
    hour: int = Field(ge=0)
    evaluate_alerts: bool = True


class ReplayEvaluationRequest(BaseModel):
    """Optional request flags for reset/end navigation."""

    model_config = ConfigDict(extra="forbid")
    evaluate_alerts: bool = False
