"""Mandatory temporal leakage tests for the P3 prediction path.

The key invariant: a prediction at hour ``t`` must be a pure function of
data at or before ``t``. These tests alter future observations, future
labels and future rows, then compare the complete causal input and the
prediction at an earlier hour.
"""

from __future__ import annotations

from ml.p3_temporal.replay.prediction_provider import ModelPredictionProvider
from ml.p3_temporal.replay.replay_engine import ReplayEngine

from .support import (
    SinglePatientRepository,
    append_future_row,
    clone_patient,
    feature_values_equal,
    mutate_hours_after,
    mutate_label,
)


def _engine(patient, bundle) -> ReplayEngine:
    return ReplayEngine(SinglePatientRepository([patient]), ModelPredictionProvider(bundle))


def _truncate_patient(patient, *, max_hour: int):
    clone = clone_patient(patient)
    indices = [index for index, hour in enumerate(clone.hour) if int(hour) <= max_hour]
    clone.hour = [int(clone.hour[index]) for index in indices]
    clone.clinical = {
        variable: [values[index] for index in indices]
        for variable, values in clone.clinical.items()
    }
    clone.sepsis_label = [clone.sepsis_label[index] for index in indices]
    clone.n_rows = len(clone.hour)
    return clone


def test_hour_24_input_matches_prefix_only_recomputation(demo_repository, demo_bundle):
    patient = demo_repository.get_patient("DEMO-0002")
    full_context = _engine(patient, demo_bundle).get_model_input("DEMO-0002", 24)

    prefix_only = _truncate_patient(patient, max_hour=24)
    prefix_context = _engine(prefix_only, demo_bundle).get_model_input("DEMO-0002", 24)

    assert full_context.input_hash == prefix_context.input_hash
    assert feature_values_equal(full_context.feature_values, prefix_context.feature_values)
    assert full_context.observed_hours_used == prefix_context.observed_hours_used
    assert full_context.prefix_row_count == 24


def test_every_future_mutation_leaves_hour_24_unchanged(demo_repository, demo_bundle):
    patient = demo_repository.get_patient("DEMO-0002")
    baseline = _engine(patient, demo_bundle).get_state("DEMO-0002", 24)

    for value in (-999.0, 0.0, 250.0, 1e9):
        mutated_patient = mutate_hours_after(patient, after_hour=24, value=value)
        state = _engine(mutated_patient, demo_bundle).get_state("DEMO-0002", 24)
        assert state.prediction.probability == baseline.prediction.probability
        assert state.causality.model_input_hash == baseline.causality.model_input_hash
        assert dict(state.prediction.model_dump()) | {"created_at": None} == dict(
            baseline.prediction.model_dump()
        ) | {"created_at": None}
        assert state.causality.max_input_hour == 24


def test_future_rows_do_not_enter_the_causal_input(demo_repository, demo_bundle):
    patient = demo_repository.get_patient("DEMO-0002")
    baseline = _engine(patient, demo_bundle).get_model_input("DEMO-0002", 24)

    changed = append_future_row(patient, hour=300, value=-500.0)
    mutated = _engine(changed, demo_bundle).get_model_input("DEMO-0002", 24)

    assert mutated.input_hash == baseline.input_hash
    assert mutated.observed_hours_used == baseline.observed_hours_used
    assert max(mutated.observed_hours_used) == 24
    assert mutated.prefix_row_count == 24


def test_future_labels_do_not_enter_the_causal_input(demo_repository, demo_bundle):
    patient = demo_repository.get_patient("DEMO-0004")
    baseline = _engine(patient, demo_bundle).get_model_input("DEMO-0004", 16)

    positive = _engine(mutate_label(patient, value=1), demo_bundle).get_model_input(
        "DEMO-0004", 16
    )
    negative = _engine(mutate_label(patient, value=0), demo_bundle).get_model_input(
        "DEMO-0004", 16
    )

    assert baseline.input_hash == positive.input_hash == negative.input_hash
    assert feature_values_equal(baseline.feature_values, positive.feature_values)


def test_gap_patient_only_uses_observed_prefix_rows(demo_repository, demo_bundle):
    context = _engine(
        demo_repository.get_patient("DEMO-0003"), demo_bundle
    ).get_model_input("DEMO-0003", 21)
    expected = tuple(hour for hour in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21))
    assert context.observed_hours_used == expected
    assert context.prefix_row_count == len(expected)
    assert all(hour <= 21 for hour in context.observed_hours_used)
