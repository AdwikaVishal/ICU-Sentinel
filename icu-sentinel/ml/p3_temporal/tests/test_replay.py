"""Replay engine behaviour: navigation, gaps, determinism and isolation."""

from __future__ import annotations

import pytest

from ml.p3_temporal.replay.prediction_provider import (
    ModelPredictionProvider,
    SyntheticHeuristicProvider,
    UnavailableProvider,
)
from ml.p3_temporal.replay.replay_engine import ReplayEngine
from ml.p3_temporal.replay.timeline import HourOutOfRangeError

from .support import (
    SinglePatientRepository,
    append_future_row,
    mutate_hours_after,
    mutate_label,
    mutate_past_hour,
)


def _engine(patient, bundle) -> ReplayEngine:
    return ReplayEngine(SinglePatientRepository([patient]), ModelPredictionProvider(bundle))


def test_patient_listing_is_explicitly_synthetic(demo_repository, replay_engine):
    listing = replay_engine.list_patients()
    assert listing.count == 6
    assert listing.is_synthetic is True
    assert listing.data_source == "synthetic_demo"
    assert listing.prediction_source == "synthetic_demo_model"
    assert all(patient.is_synthetic for patient in listing.patients)
    assert "not a diagnosis" in listing.disclaimer.lower()


def test_timeline_reports_missing_hours(replay_engine):
    timeline = replay_engine.get_timeline("DEMO-0003")
    assert timeline.missing_hours == 3
    assert {hour.hour for hour in timeline.hours if not hour.has_observations} == {12, 13, 22}


def test_navigation_respects_boundaries(replay_engine):
    start = replay_engine.start("DEMO-0002")
    assert start.hour == 1
    assert start.session.at_start is True

    forward = replay_engine.step_forward("DEMO-0002", 1)
    assert forward.hour == 2

    backward_at_start = replay_engine.step_backward("DEMO-0002", 1)
    assert backward_at_start.hour == 1

    end = replay_engine.end("DEMO-0002")
    assert end.hour == 48
    assert end.session.at_end is True
    assert replay_engine.step_forward("DEMO-0002", 48).hour == 48

    jump = replay_engine.jump_to_hour("DEMO-0002", 24)
    assert jump.hour == 24
    assert replay_engine.reset("DEMO-0002").hour == 1


def test_out_of_range_hours_are_rejected(replay_engine):
    with pytest.raises(HourOutOfRangeError):
        replay_engine.get_state("DEMO-0002", 0)
    with pytest.raises(HourOutOfRangeError):
        replay_engine.get_state("DEMO-0002", 49)


def test_missing_hour_has_explicit_unavailable_prediction(replay_engine):
    state = replay_engine.get_state("DEMO-0003", 12)
    assert state.prediction_status == "missing_hour"
    assert state.prediction.available is False
    assert state.prediction.probability is None
    assert state.prediction.unavailable_reason
    assert state.observations == []
    assert state.timeline.has_observations is False
    assert state.alert_status == "not_available"


def test_available_prediction_has_no_zero_placeholder(replay_engine):
    state = replay_engine.get_state("DEMO-0002", 24)
    assert state.prediction.available is True
    assert state.prediction.probability is not None
    assert 0.0 <= state.prediction.probability <= 1.0
    assert state.prediction.source.value == "synthetic_demo_model"
    assert state.prediction.is_synthetic is True


def test_prediction_is_deterministic(demo_repository, demo_bundle):
    first = ReplayEngine(
        demo_repository, ModelPredictionProvider(demo_bundle)
    ).get_state("DEMO-0004", 18)
    second = ReplayEngine(
        demo_repository, ModelPredictionProvider(demo_bundle)
    ).get_state("DEMO-0004", 18)
    assert first.prediction.prediction_id == second.prediction.prediction_id
    assert first.prediction.probability == second.prediction.probability
    assert first.causality.model_input_hash == second.causality.model_input_hash


def test_causality_metadata_only_covers_observed_prefix(replay_engine):
    state = replay_engine.get_state("DEMO-0002", 24)
    assert state.causality.max_input_hour == 24
    assert state.causality.future_data_used is False
    assert state.causality.retrospective_label_used_as_input is False
    assert state.causality.prefix_row_count == 24
    assert state.causality.input_observation_count == 40


def test_future_observations_do_not_change_past_prediction(demo_repository, demo_bundle):
    patient = demo_repository.get_patient("DEMO-0002")
    baseline = _engine(patient, demo_bundle).get_state("DEMO-0002", 24)

    changed = mutate_hours_after(patient, after_hour=24, value=999.0)
    mutated = _engine(changed, demo_bundle).get_state("DEMO-0002", 24)

    assert mutated.prediction.probability == baseline.prediction.probability
    assert mutated.prediction.prediction_id == baseline.prediction.prediction_id
    assert mutated.causality.model_input_hash == baseline.causality.model_input_hash
    assert mutated.causality.prefix_row_count == 24


def test_appended_future_row_does_not_change_past_prediction(demo_repository, demo_bundle):
    patient = demo_repository.get_patient("DEMO-0002")
    baseline = _engine(patient, demo_bundle).get_state("DEMO-0002", 24)

    changed = append_future_row(patient, hour=200, value=999.0)
    mutated = _engine(changed, demo_bundle).get_state("DEMO-0002", 24)

    assert mutated.prediction.probability == baseline.prediction.probability
    assert mutated.causality.model_input_hash == baseline.causality.model_input_hash
    assert mutated.causality.max_input_hour == 24


def test_future_labels_do_not_change_past_prediction(demo_repository, demo_bundle):
    patient = demo_repository.get_patient("DEMO-0004")
    baseline = _engine(patient, demo_bundle).get_state("DEMO-0004", 16)

    changed = mutate_label(patient, value=1)
    mutated = _engine(changed, demo_bundle).get_state("DEMO-0004", 16)

    assert mutated.prediction.probability == baseline.prediction.probability
    assert mutated.causality.model_input_hash == baseline.causality.model_input_hash
    assert mutated.retrospective_label == 1


def test_retrospective_label_is_not_a_model_input(replay_engine):
    state = replay_engine.get_state("DEMO-0002", 30)
    context = replay_engine.get_model_input("DEMO-0002", 30)
    assert "SepsisLabel" not in context.feature_names
    assert "SepsisLabel" not in context.feature_values
    assert "SepsisLabel" not in {observation.parameter for observation in context.observations}
    assert state.retrospective_label is not None
    assert "never used as a model input" in state.label_disclosure.lower()


def test_past_observation_change_changes_input_hash(demo_repository, demo_bundle):
    patient = demo_repository.get_patient("DEMO-0002")
    baseline = _engine(patient, demo_bundle).get_state("DEMO-0002", 24)

    changed = mutate_past_hour(patient, hour=23, variable="HR", value=250.0)
    mutated = _engine(changed, demo_bundle).get_state("DEMO-0002", 24)

    assert mutated.causality.model_input_hash != baseline.causality.model_input_hash


def test_heuristic_provider_is_labelled_and_not_explainable(demo_repository):
    engine = ReplayEngine(demo_repository, SyntheticHeuristicProvider())
    state = engine.get_state("DEMO-0002", 24)
    assert state.prediction.source.value == "synthetic_heuristic"
    assert state.prediction.is_synthetic is True
    assert state.explanation_available is False
    assert engine.provider_status().available is True


def test_unavailable_provider_returns_explicit_status(demo_repository):
    provider = UnavailableProvider("No model artifact configured.")
    engine = ReplayEngine(demo_repository, provider)
    state = engine.get_state("DEMO-0002", 24)
    assert state.prediction.available is False
    assert state.prediction.source.value == "unavailable"
    assert state.prediction.probability is None
    assert state.prediction.unavailable_reason == "No model artifact configured."
    assert provider.status().available is False


def test_alerts_are_not_evaluated_during_plain_navigation(replay_engine):
    state = replay_engine.get_state("DEMO-0002", 44)
    assert state.alert is None
    assert state.alert_status == "not_evaluated"


def test_alerts_are_created_when_requested(replay_engine):
    state = replay_engine.get_state("DEMO-0002", 44, evaluate_alerts=True)
    assert state.alert_status == "created"
    assert state.alert is not None
    assert state.alert.status.value == "ACTIVE"
    assert state.alert.is_synthetic is True
