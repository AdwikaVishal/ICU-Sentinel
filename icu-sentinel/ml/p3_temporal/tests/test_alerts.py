"""Alert lifecycle, deduplication and persistence tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ml.p3_temporal.alerts.alert_engine import AlertEngine
from ml.p3_temporal.alerts.alert_rules import (
    AlertThresholdConfig,
    is_same_episode,
    is_triggered,
)
from ml.p3_temporal.alerts.alert_schemas import (
    AlertEvaluationRequest,
    AlertStatus,
)
from ml.p3_temporal.alerts.alert_store import (
    AlertNotFoundError,
    AlertStateError,
    AlertStore,
)
from ml.p3_temporal.common.prediction import (
    ModelIdentity,
    PredictionSource,
    unavailable_prediction,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _engine(store, **overrides) -> AlertEngine:
    config = AlertThresholdConfig(**overrides)
    return AlertEngine(store, config, clock=lambda: NOW)


def _request(hour: int, probability: float, **overrides) -> AlertEvaluationRequest:
    payload = {
        "patient_id": "DEMO-0002",
        "hour": hour,
        "probability": probability,
        "model_name": "demo_logistic_regression",
        "model_version": "demo-1.0.0",
        "prediction_source": "synthetic_demo_model",
        "is_synthetic": True,
    }
    payload.update(overrides)
    return AlertEvaluationRequest(**payload)


# ----------------------------------------------------------------------
# Configuration and pure rules
# ----------------------------------------------------------------------
def test_threshold_configuration_is_validated():
    with pytest.raises(ValueError):
        AlertThresholdConfig(threshold=-0.1)
    with pytest.raises(ValueError):
        AlertThresholdConfig(threshold=1.1)
    with pytest.raises(ValueError):
        AlertThresholdConfig(dedup_window_hours=-1)
    with pytest.raises(ValueError):
        AlertThresholdConfig(source="  ")


def test_threshold_is_configurable_and_not_clinically_validated():
    config = AlertThresholdConfig(threshold=0.42, source="test_config")
    info = config.to_public()
    assert info.threshold == 0.42
    assert info.source == "test_config"
    assert info.clinically_validated is False
    assert "not a clinically validated" in info.disclosure.lower()


def test_trigger_rule_uses_configured_threshold_inclusive():
    assert is_triggered(0.60, 0.60) is True
    assert is_triggered(0.59, 0.60) is False
    with pytest.raises(ValueError):
        is_triggered(1.2, 0.6)


def test_episode_window_logic():
    assert is_same_episode(existing_last_hour=5, current_hour=5, dedup_window_hours=6)
    assert is_same_episode(existing_last_hour=5, current_hour=11, dedup_window_hours=6)
    assert not is_same_episode(existing_last_hour=5, current_hour=12, dedup_window_hours=6)
    # Out-of-order replay belongs to the same episode and must not roll back.
    assert is_same_episode(existing_last_hour=10, current_hour=4, dedup_window_hours=6)


def test_alert_request_has_no_ground_truth_label_field():
    assert "sepsis_label" not in AlertEvaluationRequest.model_fields
    assert "SepsisLabel" not in AlertEvaluationRequest.model_fields


# ----------------------------------------------------------------------
# Creation and deduplication
# ----------------------------------------------------------------------
def test_first_trigger_creates_one_active_alert(alert_store):
    engine = _engine(alert_store)
    result = engine.evaluate(_request(10, 0.80))
    assert result.action == "created"
    assert result.deduplicated is False
    assert result.alert is not None
    assert result.alert.status == AlertStatus.ACTIVE
    assert result.alert.first_hour == 10
    assert result.alert.last_hour == 10
    assert result.alert.occurrence_count == 1
    assert result.alert.threshold == 0.60
    assert result.alert.is_synthetic is True
    assert len(engine.list_active()) == 1


def test_repeated_trigger_updates_episode_without_duplicate(alert_store):
    engine = _engine(alert_store)
    first = engine.evaluate(_request(10, 0.70)).alert
    second = engine.evaluate(_request(11, 0.90))
    assert second.action == "updated"
    assert second.deduplicated is True
    assert second.alert is not None
    assert second.alert.alert_id == first.alert_id
    assert second.alert.last_hour == 11
    assert second.alert.occurrence_count == 2
    assert second.alert.peak_probability == pytest.approx(0.90)
    assert len(engine.list_active()) == 1


def test_out_of_order_trigger_does_not_roll_back_or_duplicate(alert_store):
    engine = _engine(alert_store)
    first = engine.evaluate(_request(10, 0.70)).alert
    result = engine.evaluate(_request(8, 0.95))
    assert result.action == "unchanged"
    assert result.deduplicated is True
    assert result.alert is not None
    assert result.alert.alert_id == first.alert_id
    assert result.alert.last_hour == 10
    assert result.alert.occurrence_count == 1
    assert len(engine.list_active()) == 1


def test_gap_beyond_window_supersedes_episode(alert_store):
    engine = _engine(alert_store)
    first = engine.evaluate(_request(2, 0.70)).alert
    result = engine.evaluate(_request(12, 0.70))
    assert result.action == "created"
    assert result.alert is not None
    assert result.alert.alert_id != first.alert_id
    assert len(engine.list_active()) == 1

    old = alert_store.get_alert(first.alert_id)
    assert old.status == AlertStatus.CLEARED
    assert old.cleared_reason == "superseded_by_new_episode"
    assert len(engine.list_history()) == 1


def test_threshold_override_is_recorded(alert_store):
    engine = _engine(alert_store)
    result = engine.evaluate(_request(4, 0.40, threshold=0.35))
    assert result.action == "created"
    assert result.alert.threshold == 0.35
    assert result.alert.threshold_source == "request_override"


# ----------------------------------------------------------------------
# Clearing, acknowledgement and re-trigger
# ----------------------------------------------------------------------
def test_below_threshold_clears_active_episode(alert_store):
    engine = _engine(alert_store)
    created = engine.evaluate(_request(10, 0.80)).alert
    result = engine.evaluate(_request(11, 0.20))
    assert result.action == "cleared"
    assert result.alert is not None
    assert result.alert.alert_id == created.alert_id
    assert result.alert.status == AlertStatus.CLEARED
    assert result.alert.cleared_reason == "probability_below_threshold"
    assert engine.list_active() == []
    assert len(engine.list_history()) == 1


def test_below_threshold_keeps_episode_when_clear_policy_disabled(alert_store):
    engine = _engine(alert_store, clear_on_below_threshold=False)
    engine.evaluate(_request(10, 0.80))
    result = engine.evaluate(_request(11, 0.20))
    assert result.action == "none"
    assert result.alert is not None
    assert result.alert.status == AlertStatus.ACTIVE
    assert len(engine.list_active()) == 1


def test_acknowledged_episode_can_be_updated_and_cleared(alert_store):
    engine = _engine(alert_store)
    created = engine.evaluate(_request(10, 0.80)).alert
    acknowledged = engine.acknowledge(created.alert_id)
    assert acknowledged.status == AlertStatus.ACKNOWLEDGED
    assert acknowledged.acknowledged_at is not None

    updated = engine.evaluate(_request(11, 0.85))
    assert updated.action == "updated"
    assert updated.alert is not None
    assert updated.alert.status == AlertStatus.ACKNOWLEDGED

    cleared = engine.clear(created.alert_id, reason="clinician_review_complete")
    assert cleared.status == AlertStatus.CLEARED
    assert cleared.cleared_reason == "clinician_review_complete"


def test_acknowledge_is_idempotent_and_cleared_cannot_be_acknowledged(alert_store):
    engine = _engine(alert_store)
    created = engine.evaluate(_request(10, 0.80)).alert
    first = engine.acknowledge(created.alert_id)
    second = engine.acknowledge(created.alert_id)
    assert first.acknowledged_at == second.acknowledged_at

    engine.clear(created.alert_id)
    with pytest.raises(AlertStateError):
        engine.acknowledge(created.alert_id)


def test_clear_is_idempotent(alert_store):
    engine = _engine(alert_store)
    created = engine.evaluate(_request(10, 0.80)).alert
    first = engine.clear(created.alert_id, reason="manual_clear")
    second = engine.clear(created.alert_id, reason="another_reason")
    assert first.cleared_reason == "manual_clear"
    assert second.cleared_reason == "manual_clear"


def test_re_trigger_after_clear_creates_new_episode(alert_store):
    engine = _engine(alert_store)
    created = engine.evaluate(_request(10, 0.80)).alert
    engine.clear(created.alert_id)
    result = engine.evaluate(_request(11, 0.90))
    assert result.action == "created"
    assert result.alert is not None
    assert result.alert.alert_id != created.alert_id
    assert result.alert.status == AlertStatus.ACTIVE


def test_acknowledge_unknown_alert_raises(alert_store):
    engine = _engine(alert_store)
    with pytest.raises(AlertNotFoundError):
        engine.acknowledge("alert_missing")


# ----------------------------------------------------------------------
# Persistence and listing
# ----------------------------------------------------------------------
def test_alerts_persist_across_store_reopen(tmp_path):
    path = tmp_path / "persistent.db"
    store = AlertStore(path)
    engine = _engine(store)
    created = engine.evaluate(_request(10, 0.80)).alert
    engine.acknowledge(created.alert_id)
    store.close()

    reopened = AlertStore(path)
    try:
        persisted = reopened.get_alert(created.alert_id)
        assert persisted.status == AlertStatus.ACKNOWLEDGED
        assert persisted.patient_id == "DEMO-0002"
        assert persisted.probability == pytest.approx(0.80)
        assert persisted.is_synthetic is True
        assert len(reopened.list_alerts(patient_id="DEMO-0002")) == 1
    finally:
        reopened.close()


def test_list_filters_by_status_and_patient(alert_store):
    engine = _engine(alert_store)
    first = engine.evaluate(_request(10, 0.80)).alert
    engine.evaluate(_request(30, 0.80))
    other = engine.evaluate(_request(10, 0.80, patient_id="DEMO-0004")).alert

    assert len(engine.list_active()) == 2
    assert len(engine.list_for_patient("DEMO-0002")) == 2
    assert engine.list_for_patient("DEMO-0004")[0].alert_id == other.alert_id

    engine.clear(first.alert_id)
    assert len(engine.list_history()) == 1
    assert len(engine.list_all(status=AlertStatus.CLEARED)) == 1


def test_unavailable_prediction_does_not_create_alert(alert_store):
    engine = _engine(alert_store)
    identity = ModelIdentity(
        name="demo_logistic_regression",
        version="demo-1.0.0",
        source=PredictionSource.SYNTHETIC_DEMO_MODEL,
        is_synthetic=True,
    )
    prediction = unavailable_prediction(
        patient_id="DEMO-0002",
        hour=10,
        identity=identity,
        reason="No model input at this hour.",
    )
    result = engine.evaluate_prediction(prediction)
    assert result.action == "unavailable"
    assert result.alert is None
    assert engine.list_active() == []
