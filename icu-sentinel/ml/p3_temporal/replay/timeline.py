"""Chronological patient timeline construction for replay."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ml.p1_clinical.preprocessing.schemas import CLINICAL_VARIABLES, ProcessedPatient

from ml.p3_temporal.common.clinical_catalog import category_for, display_name, unit_for

from .demo_data import PatientRepository, summarize_patient
from .replay_schemas import ObservationValue, PatientSummary, TimelineHourSummary


class HourOutOfRangeError(ValueError):
    """Raised when an hour is outside the patient's timeline."""


@dataclass(frozen=True)
class TimelineIndex:
    patient: ProcessedPatient
    min_hour: int
    max_hour: int
    observed_hours: tuple[int, ...]
    row_by_hour: dict[int, int]
    summary: PatientSummary

    @property
    def total_hours(self) -> int:
        return self.max_hour - self.min_hour + 1

    @property
    def missing_hours(self) -> int:
        return self.total_hours - len(self.observed_hours)

    def clamp(self, hour: int) -> int:
        return max(self.min_hour, min(self.max_hour, int(hour)))

    def is_observed(self, hour: int) -> bool:
        return int(hour) in self.row_by_hour


class TimelineService:
    """Build causal, chronologically ordered patient timelines."""

    def __init__(self, repository: PatientRepository) -> None:
        self.repository = repository
        self._index_cache: dict[str, TimelineIndex] = {}

    def patient_summary(self, patient_id: str) -> PatientSummary:
        return summarize_patient(
            self.repository.get_patient(patient_id),
            is_synthetic=self.repository.is_synthetic,
        )

    def index(self, patient_id: str) -> TimelineIndex:
        cached = self._index_cache.get(patient_id)
        if cached is not None:
            return cached
        patient = self.repository.get_patient(patient_id)
        valid_pairs = [
            (int(hour), index)
            for index, hour in enumerate(patient.hour)
            if int(hour) >= 0
        ]
        if not valid_pairs:
            raise HourOutOfRangeError(f"Patient {patient_id} has no valid hourly records")
        valid_pairs.sort(key=lambda pair: pair[0])
        hours = [hour for hour, _ in valid_pairs]
        # Duplicate hours keep the first deterministic occurrence. They are not
        # silently averaged, because averaging could fabricate a measurement.
        row_by_hour: dict[int, int] = {}
        for hour, index in valid_pairs:
            row_by_hour.setdefault(hour, index)
        min_hour = hours[0]
        max_hour = hours[-1]
        observed = tuple(sorted(row_by_hour))
        result = TimelineIndex(
            patient=patient,
            min_hour=min_hour,
            max_hour=max_hour,
            observed_hours=observed,
            row_by_hour=row_by_hour,
            summary=self.patient_summary(patient_id),
        )
        self._index_cache[patient_id] = result
        return result

    def timeline(self, patient_id: str) -> list[TimelineHourSummary]:
        index = self.index(patient_id)
        return [
            self.hour_summary(index, hour)
            for hour in range(index.min_hour, index.max_hour + 1)
        ]

    def hour_summary(
        self,
        index_or_patient: TimelineIndex | str,
        hour: int,
        *,
        prediction_status: str = "not_evaluated",
    ) -> TimelineHourSummary:
        index = (
            index_or_patient
            if isinstance(index_or_patient, TimelineIndex)
            else self.index(index_or_patient)
        )
        hour = int(hour)
        if hour < index.min_hour or hour > index.max_hour:
            raise HourOutOfRangeError(
                f"Hour {hour} outside timeline [{index.min_hour}, {index.max_hour}] "
                f"for patient {index.patient.patient_id}"
            )
        row = index.row_by_hour.get(hour)
        if row is None:
            return TimelineHourSummary(
                hour=hour,
                available_observation_count=0,
                unavailable_observation_count=0,
                has_observations=False,
                retrospective_label=None,
                label_available=False,
                prediction_status="missing_hour",
            )
        available = 0
        unavailable = 0
        for variable in CLINICAL_VARIABLES:
            if variable == "ICULOS":
                # ICULOS is time-in-ICU metadata, not a clinical observation.
                continue
            values = index.patient.clinical.get(variable, [])
            value = values[row] if row < len(values) else None
            if _finite(value):
                available += 1
            else:
                unavailable += 1
        label = (
            int(index.patient.sepsis_label[row])
            if row < len(index.patient.sepsis_label)
            and index.patient.sepsis_label[row] in (0, 1)
            else None
        )
        return TimelineHourSummary(
            hour=hour,
            available_observation_count=available,
            unavailable_observation_count=unavailable,
            has_observations=available > 0,
            retrospective_label=label,
            label_available=label is not None,
            prediction_status="available" if available > 0 else prediction_status,
        )

    def observations(self, patient_id: str, hour: int) -> list[ObservationValue]:
        index = self.index(patient_id)
        hour = int(hour)
        if hour < index.min_hour or hour > index.max_hour:
            raise HourOutOfRangeError(
                f"Hour {hour} outside timeline [{index.min_hour}, {index.max_hour}] "
                f"for patient {patient_id}"
            )
        row = index.row_by_hour.get(hour)
        if row is None:
            return []
        observations: list[ObservationValue] = []
        for variable in CLINICAL_VARIABLES:
            if variable == "ICULOS":
                # ICULOS is time-in-ICU metadata, not a clinical observation.
                continue
            values = index.patient.clinical.get(variable, [])
            value = values[row] if row < len(values) else None
            available = _finite(value)
            observations.append(
                ObservationValue(
                    parameter=variable,
                    display_name=display_name(variable),
                    category=category_for(variable),
                    value=float(value) if available and value is not None else None,
                    unit=unit_for(variable),
                    available=available,
                )
            )
        return observations

    def retrospective_label(self, patient_id: str, hour: int) -> int | None:
        index = self.index(patient_id)
        row = index.row_by_hour.get(int(hour))
        if row is None:
            return None
        if row >= len(index.patient.sepsis_label):
            return None
        label = index.patient.sepsis_label[row]
        return int(label) if label in (0, 1) else None


def _finite(value: float | None) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
