"""Alert schemas for the ICU Sentinel P3 alert engine.

Alert statuses are application workflow states. They are not clinical
diagnoses and do not confirm sepsis.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AlertStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    CLEARED = "CLEARED"


class AlertRecord(BaseModel):
    """A persisted alert episode."""

    model_config = ConfigDict(use_enum_values=False)

    alert_id: str
    patient_id: str
    rule_id: str
    hour: int = Field(ge=0)
    first_hour: int = Field(ge=0)
    last_hour: int = Field(ge=0)
    probability: float
    peak_probability: float
    threshold: float
    threshold_source: str
    threshold_version: str
    model_name: str
    model_version: str
    prediction_source: str
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    acknowledged_at: datetime | None = None
    cleared_at: datetime | None = None
    cleared_reason: str | None = None
    occurrence_count: int = Field(default=1, ge=1)
    is_synthetic: bool = True

    @field_validator("probability", "peak_probability", "threshold")
    @classmethod
    def _finite_probability(cls, value: float) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("probability and threshold must be finite")
        if not 0.0 <= number <= 1.0:
            raise ValueError("probability and threshold must be between 0.0 and 1.0")
        return number


class AlertEvaluationRequest(BaseModel):
    """Request to evaluate one prediction against the alert rule."""

    patient_id: str = Field(min_length=1)
    hour: int = Field(ge=0)
    probability: float
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prediction_source: str = Field(min_length=1)
    threshold: float | None = None
    rule_id: str | None = None
    is_synthetic: bool = True

    @field_validator("probability", "threshold")
    @classmethod
    def _validate_probability(cls, value: float | None) -> float | None:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("probability and threshold must be finite")
        if not 0.0 <= number <= 1.0:
            raise ValueError("probability and threshold must be between 0.0 and 1.0")
        return number


class AlertEvaluationResult(BaseModel):
    """Outcome of evaluating one prediction.

    ``action`` is one of: created, updated, cleared, none, unchanged,
    unavailable. ``deduplicated`` is true when an existing ongoing
    episode absorbed the prediction instead of creating a duplicate.
    """

    action: str
    alert: AlertRecord | None = None
    reason: str | None = None
    deduplicated: bool = False


class AlertListResponse(BaseModel):
    alerts: list[AlertRecord]
    count: int
    filters: dict[str, str | int | None] = Field(default_factory=dict)


class AlertThresholdInfo(BaseModel):
    """Public, configuration-only threshold metadata."""

    rule_id: str
    threshold: float
    source: str
    version: str
    dedup_window_hours: int
    clear_on_below_threshold: bool
    clinically_validated: bool = False
    disclosure: str = (
        "Configured demonstration threshold. Not a clinically validated "
        "decision threshold and not a diagnosis."
    )
