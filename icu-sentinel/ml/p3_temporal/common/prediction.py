"""Shared prediction contracts for the ICU Sentinel P3 modules.

A single :class:`PredictionResult` representation is used by replay,
explainability, and the alert engine so that the same patient-hour
prediction cannot be redefined inconsistently in different modules.

This module contains no clinical logic. It deliberately keeps the
retrospective ground-truth label out of the prediction input contract:
labels are attached only to replay/UI metadata for retrospective display.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionSource(str, Enum):
    """Where a prediction came from.

    ``SYNTHETIC_HEURISTIC`` is a hand-written demonstration score and is
    never explained with SHAP. ``SYNTHETIC_DEMO_MODEL`` is a genuinely
    fitted model trained on synthetic demonstration data; it is not a
    clinically validated model. The P1/P2/P3 sources are reserved for
    real research artifacts.
    """

    SYNTHETIC_HEURISTIC = "synthetic_heuristic"
    SYNTHETIC_DEMO_MODEL = "synthetic_demo_model"
    P1_CLASSICAL_MODEL = "p1_classical_model"
    P2_BOOSTING_MODEL = "p2_boosting_model"
    P3_TEMPORAL_MODEL = "p3_temporal_model"
    UNAVAILABLE = "unavailable"


SYNTHETIC_SOURCES = {
    PredictionSource.SYNTHETIC_HEURISTIC,
    PredictionSource.SYNTHETIC_DEMO_MODEL,
}


class ModelIdentity(BaseModel):
    """Machine-readable identity of the prediction model or provider."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source: PredictionSource
    display_name: str | None = None
    is_synthetic: bool = False
    feature_schema_version: str | None = None


class ProviderStatus(BaseModel):
    """Availability metadata for a prediction provider."""

    available: bool
    provider_name: str
    model: ModelIdentity | None = None
    reason: str | None = None
    required_features: list[str] | None = None


class PredictionResult(BaseModel):
    """One prediction for one patient-hour.

    ``probability`` is ``None`` only when the prediction is unavailable.
    ``available`` is always explicit so callers never infer availability
    from a missing/zero probability.
    """

    model_config = ConfigDict(use_enum_values=False)

    patient_id: str
    hour: int = Field(ge=0)
    probability: float | None = None
    predicted_class: int | None = None
    threshold: float | None = None
    model: ModelIdentity
    source: PredictionSource
    available: bool
    unavailable_reason: str | None = None
    prediction_id: str
    feature_schema_version: str | None = None
    model_input_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_synthetic: bool = False

    @field_validator("probability")
    @classmethod
    def _validate_probability(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(float(value)):
            raise ValueError("probability must be finite")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("probability must be between 0.0 and 1.0")
        return float(value)


@dataclass(frozen=True)
class ObservationSnapshot:
    """A single observation at the selected replay hour.

    ``available=False`` with ``value=None`` means the measurement was not
    recorded. It must never be interpreted as a normal clinical value.
    """

    parameter: str
    value: float | None
    unit: str | None = None
    available: bool = False


@dataclass(frozen=True)
class PredictionContext:
    """The exact, causal model input for one patient-hour.

    ``observations`` and ``feature_values`` contain information at or
    before ``hour`` only. The builder that creates this object is
    responsible for enforcing that invariant.
    """

    patient_id: str
    hour: int
    observations: tuple[ObservationSnapshot, ...]
    feature_values: Mapping[str, float] | None
    feature_names: tuple[str, ...]
    feature_schema_version: str
    input_hash: str
    data_source: str
    is_synthetic_data: bool
    retrospective_label: int | None = None
    prefix_row_count: int = 0
    observed_hours_used: tuple[int, ...] = field(default_factory=tuple)


@runtime_checkable
class PredictionProvider(Protocol):
    """Interface implemented by every prediction provider."""

    @property
    def identity(self) -> ModelIdentity:
        ...

    @property
    def required_features(self) -> list[str] | None:
        ...

    def predict(self, context: PredictionContext) -> PredictionResult:
        ...

    def status(self) -> ProviderStatus:
        ...


def canonical_json(value: Any) -> str:
    """Stable JSON encoding used for deterministic input hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def safe_number(value: Any) -> float | None:
    """Convert a value to a JSON-safe float.

    NaN and infinities become ``None`` at the API boundary. Internal
    feature values preserve infinity for P1's ``hours_since_last``
    semantics, but the model input builder sanitizes those values before
    passing them to a scikit-learn estimator.
    """

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def sanitize_feature_values(values: Mapping[str, Any]) -> dict[str, float]:
    """Return a model-ready float mapping with non-finite values as NaN."""

    out: dict[str, float] = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        if not math.isfinite(number):
            number = float("nan")
        out[str(key)] = number
    return out


def hash_prediction_input(
    *,
    patient_id: str,
    hour: int,
    feature_values: Mapping[str, Any] | None,
    observations: tuple[ObservationSnapshot, ...],
) -> str:
    """Hash only causal inputs used for a prediction.

    The retrospective label is deliberately not part of the hash. The
    hash changes if any observation at or before ``hour`` changes, and is
    unaffected by later observations.
    """

    if feature_values is not None:
        payload: dict[str, Any] = {
            "kind": "features",
            "patient_id": patient_id,
            "hour": hour,
            "features": {str(k): safe_number(v) for k, v in sorted(feature_values.items())},
        }
    else:
        payload = {
            "kind": "observations",
            "patient_id": patient_id,
            "hour": hour,
            "observations": [
                {
                    "parameter": obs.parameter,
                    "value": safe_number(obs.value),
                    "available": obs.available,
                }
                for obs in observations
            ],
        }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def make_prediction_id(
    *,
    patient_id: str,
    hour: int,
    model_name: str,
    model_version: str,
    input_hash: str,
) -> str:
    """Deterministic prediction identifier for one model + input pair."""

    payload = {
        "patient_id": patient_id,
        "hour": hour,
        "model_name": model_name,
        "model_version": model_version,
        "input_hash": input_hash,
    }
    return "pred_" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:24]


def unavailable_prediction(
    *,
    patient_id: str,
    hour: int,
    identity: ModelIdentity,
    reason: str,
    feature_schema_version: str | None = None,
    input_hash: str | None = None,
) -> PredictionResult:
    """Build an explicit, honest unavailable prediction record."""

    return PredictionResult(
        patient_id=patient_id,
        hour=hour,
        probability=None,
        predicted_class=None,
        threshold=None,
        model=identity,
        source=PredictionSource.UNAVAILABLE,
        available=False,
        unavailable_reason=reason,
        prediction_id=make_prediction_id(
            patient_id=patient_id,
            hour=hour,
            model_name=identity.name,
            model_version=identity.version,
            input_hash=input_hash or "unavailable",
        ),
        feature_schema_version=feature_schema_version,
        model_input_hash=input_hash,
        is_synthetic=identity.is_synthetic,
    )
