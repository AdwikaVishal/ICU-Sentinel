"""End-to-end P3 integration: replay -> explanation -> alert."""

from __future__ import annotations

from ml.p3_temporal.alerts.alert_engine import AlertEngine
from ml.p3_temporal.alerts.alert_rules import AlertThresholdConfig
from ml.p3_temporal.alerts.alert_schemas import AlertStatus
from ml.p3_temporal.alerts.alert_store import AlertStore
from ml.p3_temporal.explainability.explanation_schemas import ExplanationStatus
from ml.p3_temporal.replay.prediction_provider import ModelPredictionProvider
from ml.p3_temporal.replay.replay_engine import ReplayEngine


def test_replay_explanation_and_alert_share_the_same_model_input(
    replay_engine, explanation_service, alert_store
):
    state = replay_engine.get_state("DEMO-0004", 20, evaluate_alerts=True)
    assert state.prediction.available is True
    assert state.prediction.probability > 0.6
    assert state.alert is not None
    assert state.alert_status == "created"

    explanation = explanation_service.explain("DEMO-0004", 20)
    assert explanation.status == ExplanationStatus.AVAILABLE
    assert explanation.input_hash == state.prediction.model_input_hash
    assert explanation.prediction.prediction_id == state.prediction.prediction_id

    # Explanation/replay reads never create additional alerts.
    before = len(alert_store.list_active())
    explanation_service.explain("DEMO-0004", 20)
    replay_engine.get_state("DEMO-0004", 20)
    assert len(alert_store.list_active()) == before == 1


def test_alert_centre_workflow_across_acknowledge_and_clear(
    replay_engine, alert_store
):
    state = replay_engine.get_state("DEMO-0004", 20, evaluate_alerts=True)
    alert_id = state.alert.alert_id

    acknowledged = replay_engine.alert_engine.acknowledge(alert_id)
    assert acknowledged.status == AlertStatus.ACKNOWLEDGED
    assert alert_store.list_active() == []
    assert alert_store.get_active_for_episode("DEMO-0004", "sepsis_risk_threshold") is not None

    cleared = replay_engine.alert_engine.clear(alert_id, reason="review_complete")
    assert cleared.status == AlertStatus.CLEARED
    assert alert_store.list_active() == []
    assert len(alert_store.list_history()) == 1


def test_configured_threshold_changes_replay_alert_outcome(demo_repository, demo_bundle):
    engine = ReplayEngine(demo_repository, ModelPredictionProvider(demo_bundle))

    low_store = AlertStore(":memory:")
    try:
        low_engine = ReplayEngine(
            demo_repository,
            ModelPredictionProvider(demo_bundle),
            alert_engine=AlertEngine(
                low_store, AlertThresholdConfig(threshold=0.50)
            ),
        )
        state = low_engine.get_state("DEMO-0004", 20, evaluate_alerts=True)
        assert state.alert_status == "created"
        assert state.alert.threshold == 0.50
    finally:
        low_store.close()

    high_store = AlertStore(":memory:")
    try:
        high_engine = ReplayEngine(
            demo_repository,
            ModelPredictionProvider(demo_bundle),
            alert_engine=AlertEngine(
                high_store, AlertThresholdConfig(threshold=0.999999999)
            ),
        )
        state = high_engine.get_state("DEMO-0004", 20, evaluate_alerts=True)
        assert state.alert_status == "none"
        assert state.alert is None
    finally:
        high_store.close()

    # The unconfigured engine is not used for alerts here; this keeps the
    # fixture explicit about which threshold produced which outcome.
    assert engine.alert_engine is None


def test_public_metadata_carries_honest_disclaimers(
    replay_engine, explanation_service, alert_engine
):
    state = replay_engine.get_state("DEMO-0004", 20)
    assert "not a diagnosis" in state.disclaimer.lower()
    assert state.is_synthetic is True

    explanation = explanation_service.explain("DEMO-0004", 20)
    assert "not causal medical explanations" in explanation.disclaimer.lower()

    info = alert_engine.threshold_info()
    assert info.clinically_validated is False
    assert "not a clinically validated" in info.disclosure.lower()
