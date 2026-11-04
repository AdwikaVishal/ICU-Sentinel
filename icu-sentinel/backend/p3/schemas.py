"""Backend-only request/response helpers for the P3 API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ml.p3_temporal.common.prediction import safe_number


class AlertClearRequest(BaseModel):
    reason: str = Field(default="manual_clear", min_length=1, max_length=200)


class CausalInputResponse(BaseModel):
    """The exact causal model input used for a prediction.

    ``retrospective_label`` is included for retrospective display only. It
    is not part of ``feature_values`` and is not used to build the input.
    """

    patient_id: str
    hour: int = Field(ge=0)
    feature_schema_version: str
    input_hash: str
    feature_names: list[str]
    feature_values: dict[str, float | None]
    data_source: str
    is_synthetic_data: bool
    prefix_row_count: int = Field(ge=0)
    observed_hours_used: list[int]
    retrospective_label: int | None = None
    label_disclosure: str = (
        "Retrospective label shown for evaluation only; it is excluded from "
        "the feature vector and from the input hash."
    )

    @classmethod
    def from_context(cls, context: Any) -> "CausalInputResponse":
        values = {
            str(name): safe_number(value)
            for name, value in dict(context.feature_values or {}).items()
        }
        return cls(
            patient_id=context.patient_id,
            hour=context.hour,
            feature_schema_version=context.feature_schema_version,
            input_hash=context.input_hash,
            feature_names=list(context.feature_names),
            feature_values=values,
            data_source=context.data_source,
            is_synthetic_data=context.is_synthetic_data,
            prefix_row_count=context.prefix_row_count,
            observed_hours_used=list(context.observed_hours_used),
            retrospective_label=context.retrospective_label,
        )


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "icu-sentinel-p3"
    version: str = "3.0.0"
