"""Test helpers for building controlled patient repositories."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from typing import Any

from ml.p1_clinical.preprocessing.schemas import ProcessedPatient

from ml.p3_temporal.replay.demo_data import PatientNotFoundError, summarize_patient
from ml.p3_temporal.replay.replay_schemas import PatientSummary


class SinglePatientRepository:
    """Minimal read-only repository over one or more ProcessedPatient objects."""

    data_source = "synthetic_test_fixture"
    is_synthetic = True
    source_dataset = "synthetic_test_fixture"

    def __init__(self, patients: Iterable[ProcessedPatient]) -> None:
        self._patients = {patient.patient_id: patient for patient in patients}

    def list_patients(self) -> list[PatientSummary]:
        return [
            summarize_patient(patient, is_synthetic=self.is_synthetic)
            for _, patient in sorted(self._patients.items())
        ]

    def get_patient(self, patient_id: str) -> ProcessedPatient:
        try:
            return self._patients[patient_id]
        except KeyError as exc:
            raise PatientNotFoundError(f"Patient not found: {patient_id}") from exc

    def patient_exists(self, patient_id: str) -> bool:
        return patient_id in self._patients


def clone_patient(patient: ProcessedPatient) -> ProcessedPatient:
    return copy.deepcopy(patient)


def mutate_hours_after(
    patient: ProcessedPatient,
    *,
    after_hour: int,
    value: float,
) -> ProcessedPatient:
    """Set every observed value strictly after ``after_hour`` to ``value``."""

    mutated = clone_patient(patient)
    indices = [index for index, hour in enumerate(mutated.hour) if int(hour) > after_hour]
    for variable, values in mutated.clinical.items():
        for index in indices:
            if index < len(values):
                values[index] = value
    return mutated


def mutate_label(patient: ProcessedPatient, *, value: int) -> ProcessedPatient:
    mutated = clone_patient(patient)
    mutated.sepsis_label = [value] * len(mutated.sepsis_label)
    return mutated


def append_future_row(
    patient: ProcessedPatient,
    *,
    hour: int,
    value: float = 999.0,
) -> ProcessedPatient:
    """Append an extreme observation well after the current stay."""

    mutated = clone_patient(patient)
    mutated.hour.append(int(hour))
    for variable, values in mutated.clinical.items():
        values.append(value)
    mutated.sepsis_label.append(1)
    mutated.n_rows = len(mutated.hour)
    return mutated


def mutate_past_hour(
    patient: ProcessedPatient,
    *,
    hour: int,
    variable: str,
    value: float,
) -> ProcessedPatient:
    mutated = clone_patient(patient)
    for index, observed_hour in enumerate(mutated.hour):
        if int(observed_hour) == hour:
            mutated.clinical[variable][index] = value
    return mutated


def feature_values_equal(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> bool:
    """Compare feature mappings treating NaN as equal to NaN."""

    if left is None or right is None:
        return left is right
    if set(left) != set(right):
        return False
    for key in left:
        a = left[key]
        b = right[key]
        try:
            a_float = float(a)
            b_float = float(b)
        except (TypeError, ValueError):
            if a != b:
                return False
            continue
        if math.isnan(a_float) and math.isnan(b_float):
            continue
        if a_float != b_float:
            return False
    return True
