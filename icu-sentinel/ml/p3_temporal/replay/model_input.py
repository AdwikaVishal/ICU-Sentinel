"""Causal model-input construction for replay.

The central rule of this module: for a prediction at hour ``t``, build a
new ``ProcessedPatient`` containing only rows with ``hour <= t`` and run
P1's feature pipeline on that prefix. Features are never computed on the
full stay and then sliced, because rolling windows, forward filling and
trend features must not see future rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import pandas as pd

from ml.p1_clinical.features.feature_pipeline import FeaturePipeline
from ml.p1_clinical.preprocessing.schemas import CLINICAL_VARIABLES, ProcessedPatient

from ml.p3_temporal.common.prediction import (
    ObservationSnapshot,
    PredictionContext,
    hash_prediction_input,
)

from .demo_data import PatientRepository
from .timeline import TimelineIndex, TimelineService

TARGET_COLUMN = "SepsisLabel"
IDENTIFIER_COLUMNS = {
    "patient_id",
    "hospital_id",
    "hour",
    "source_file",
    "source_dataset",
}
NON_FEATURE_COLUMNS = IDENTIFIER_COLUMNS | {TARGET_COLUMN}
FEATURE_SCHEMA_VERSION = "p1-feature-schema-1.0.0"


class ModelInputError(ValueError):
    """Raised when a causal model input cannot be constructed."""


@dataclass(frozen=True)
class PrefixModelInput:
    """Causal model input plus metadata for one patient-hour."""

    context: PredictionContext
    has_observation_at_hour: bool
    feature_frame: pd.DataFrame | None
    timeline_index: TimelineIndex
    label_at_hour: int | None


class CausalFeatureBuilder:
    """Build and cache causal model inputs from patient prefixes."""

    def __init__(
        self,
        repository: PatientRepository,
        *,
        timeline_service: TimelineService | None = None,
        pipeline: FeaturePipeline | None = None,
    ) -> None:
        self.repository = repository
        self.timeline = timeline_service or TimelineService(repository)
        self.pipeline = pipeline or FeaturePipeline()
        self._cache: dict[tuple[str, int, tuple[str, ...] | None], PrefixModelInput] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def build(
        self,
        patient_id: str,
        hour: int,
        *,
        selected_features: list[str] | None = None,
    ) -> PrefixModelInput:
        hour = int(hour)
        feature_key = tuple(selected_features) if selected_features is not None else None
        cache_key = (patient_id, hour, feature_key)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        index = self.timeline.index(patient_id)
        if hour < index.min_hour or hour > index.max_hour:
            raise ModelInputError(
                f"Hour {hour} outside timeline [{index.min_hour}, {index.max_hour}] "
                f"for patient {patient_id}"
            )

        observations = self._observations_at(index, hour)
        has_observation_at_hour = index.is_observed(hour)
        label_at_hour = self._label_at(index, hour)
        if not has_observation_at_hour:
            context = PredictionContext(
                patient_id=patient_id,
                hour=hour,
                observations=tuple(),
                feature_values=None,
                feature_names=tuple(selected_features or ()),
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                input_hash=hash_prediction_input(
                    patient_id=patient_id,
                    hour=hour,
                    feature_values=None,
                    observations=tuple(),
                ),
                data_source=self.repository.data_source,
                is_synthetic_data=self.repository.is_synthetic,
                retrospective_label=label_at_hour,
                prefix_row_count=sum(1 for h in index.observed_hours if h <= hour),
                observed_hours_used=tuple(h for h in index.observed_hours if h <= hour),
            )
            result = PrefixModelInput(
                context=context,
                has_observation_at_hour=False,
                feature_frame=None,
                timeline_index=index,
                label_at_hour=label_at_hour,
            )
            self._cache[cache_key] = result
            return result

        prefix = self._prefix_patient(index.patient, hour)
        feature_frame = self.pipeline.transform_patient(prefix)
        if feature_frame is None or len(feature_frame) == 0:
            raise ModelInputError(f"No causal features produced for {patient_id} hour {hour}")
        exact_rows = feature_frame[feature_frame["hour"] == hour]
        if len(exact_rows) == 0:
            raise ModelInputError(
                f"Feature pipeline produced no row for {patient_id} hour {hour}"
            )
        row = exact_rows.iloc[-1]

        if selected_features is not None:
            missing = [feature for feature in selected_features if feature not in row.index]
            if missing:
                raise ModelInputError(
                    f"Model features are not produced by the P1 feature pipeline: {missing[:8]}"
                )
            feature_values = {
                feature: _as_float(row[feature]) for feature in selected_features
            }
            feature_names = tuple(selected_features)
        else:
            feature_values = {}
            for column in feature_frame.columns:
                if column in NON_FEATURE_COLUMNS:
                    continue
                feature_values[str(column)] = _as_float(row[column])
            feature_names = tuple(feature_values)

        context = PredictionContext(
            patient_id=patient_id,
            hour=hour,
            observations=tuple(observations),
            feature_values=feature_values,
            feature_names=feature_names,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            input_hash=hash_prediction_input(
                patient_id=patient_id,
                hour=hour,
                feature_values=feature_values,
                observations=tuple(observations),
            ),
            data_source=self.repository.data_source,
            is_synthetic_data=self.repository.is_synthetic,
            retrospective_label=label_at_hour,
            prefix_row_count=len(prefix.hour),
            observed_hours_used=tuple(h for h in index.observed_hours if h <= hour),
        )
        result = PrefixModelInput(
            context=context,
            has_observation_at_hour=True,
            feature_frame=exact_rows.copy(),
            timeline_index=index,
            label_at_hour=label_at_hour,
        )
        self._cache[cache_key] = result
        return result

    def _prefix_patient(self, patient: ProcessedPatient, hour: int) -> ProcessedPatient:
        row_indices = [
            index
            for index, value in enumerate(patient.hour)
            if int(value) >= 0 and int(value) <= hour
        ]
        if not row_indices:
            raise ModelInputError(f"No observations at or before hour {hour}")
        sliced_clinical: dict[str, list[float]] = {}
        for variable in CLINICAL_VARIABLES:
            values = patient.clinical.get(variable, [])
            sliced_clinical[variable] = [
                values[index] if index < len(values) else float("nan")
                for index in row_indices
            ]
        return ProcessedPatient(
            patient_id=patient.patient_id,
            hospital_id=patient.hospital_id,
            hour=[int(patient.hour[index]) for index in row_indices],
            clinical=sliced_clinical,
            sepsis_label=[
                int(patient.sepsis_label[index])
                if index < len(patient.sepsis_label)
                else -1
                for index in row_indices
            ],
            source_file=patient.source_file,
            source_dataset=patient.source_dataset,
            n_rows=len(row_indices),
        )

    def _observations_at(self, index: TimelineIndex, hour: int) -> list[ObservationSnapshot]:
        row = index.row_by_hour.get(hour)
        if row is None:
            return []
        observations: list[ObservationSnapshot] = []
        for variable in CLINICAL_VARIABLES:
            values = index.patient.clinical.get(variable, [])
            value = values[row] if row < len(values) else None
            available = _is_finite(value)
            observations.append(
                ObservationSnapshot(
                    parameter=variable,
                    value=float(value) if available and value is not None else None,
                    unit=None,
                    available=available,
                )
            )
        return observations

    def _label_at(self, index: TimelineIndex, hour: int) -> int | None:
        row = index.row_by_hour.get(hour)
        if row is None or row >= len(index.patient.sepsis_label):
            return None
        label = index.patient.sepsis_label[row]
        return int(label) if label in (0, 1) else None


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def input_fingerprint(frame: pd.DataFrame, features: list[str]) -> str:
    """Deterministic fingerprint of an ordered model-input frame."""

    values = {feature: _as_float(frame.iloc[0][feature]) for feature in features}
    payload = {feature: (None if not math.isfinite(value) else value) for feature, value in values.items()}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
