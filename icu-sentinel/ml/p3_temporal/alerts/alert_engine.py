"""Alert decision engine.

The engine is independent of replay. Replay calls it, but the same
engine is exposed directly through the alert API. It never receives or
uses a ground-truth label.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from typing import Callable

from ml.p3_temporal.common.prediction import PredictionResult

from .alert_rules import AlertThresholdConfig, is_same_episode, is_triggered
from .alert_schemas import (
    AlertEvaluationRequest,
    AlertEvaluationResult,
    AlertRecord,
    AlertStatus,
    AlertThresholdInfo,
)
from .alert_store import AlertNotFoundError, AlertStateError, AlertStore


def _new_alert_id() -> str:
    return "alert_" + uuid.uuid4().hex


class AlertEngine:
    """Evaluate configured rules and persist alert episodes."""

    def __init__(
        self,
        store: AlertStore,
        config: AlertThresholdConfig | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.config = config or AlertThresholdConfig.from_env()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, request: AlertEvaluationRequest) -> AlertEvaluationResult:
        with self._lock:
            now = self._clock()
            threshold = (
                float(request.threshold)
                if request.threshold is not None
                else float(self.config.threshold)
            )
            threshold_source = (
                "request_override" if request.threshold is not None else self.config.source
            )
            threshold_version = (
                "request-override" if request.threshold is not None else self.config.version
            )
            rule_id = request.rule_id or self.config.rule_id
            triggered = is_triggered(request.probability, threshold)
            existing = self.store.get_active_for_episode(request.patient_id, rule_id)

            if not triggered:
                if existing is not None and self.config.clear_on_below_threshold:
                    cleared = self.store.clear_alert(
                        existing.alert_id,
                        at=now,
                        reason="probability_below_threshold",
                    )
                    return AlertEvaluationResult(
                        action="cleared",
                        alert=cleared,
                        reason=(
                            "Probability fell below the configured threshold; "
                            "the active episode was cleared by the documented "
                            "clear_on_below_threshold policy."
                        ),
                        deduplicated=False,
                    )
                return AlertEvaluationResult(
                    action="none",
                    alert=existing,
                    reason="Probability is below the configured threshold.",
                    deduplicated=False,
                )

            if existing is not None:
                same_episode = is_same_episode(
                    existing_last_hour=existing.last_hour,
                    current_hour=request.hour,
                    dedup_window_hours=self.config.dedup_window_hours,
                )
                if not same_episode:
                    self.store.clear_alert(
                        existing.alert_id,
                        at=now,
                        reason="superseded_by_new_episode",
                    )
                    existing = None
                elif request.hour < existing.last_hour:
                    # Navigating replay backwards must not roll an episode
                    # backwards or create another alert.
                    return AlertEvaluationResult(
                        action="unchanged",
                        alert=existing,
                        reason=(
                            "Replay hour is earlier than the latest hour of the "
                            "existing episode; no new alert was created."
                        ),
                        deduplicated=True,
                    )
                else:
                    updated = self.store.update_alert(
                        existing.alert_id,
                        hour=request.hour,
                        last_hour=max(existing.last_hour, request.hour),
                        probability=float(request.probability),
                        peak_probability=max(existing.peak_probability, float(request.probability)),
                        threshold=threshold,
                        threshold_source=threshold_source,
                        threshold_version=threshold_version,
                        model_name=request.model_name,
                        model_version=request.model_version,
                        prediction_source=request.prediction_source,
                        updated_at=now,
                        occurrence_count=existing.occurrence_count + 1,
                    )
                    return AlertEvaluationResult(
                        action="updated",
                        alert=updated,
                        reason=(
                            "Prediction continues the existing alert episode within "
                            "the configured deduplication window; no duplicate "
                            "active alert was created."
                        ),
                        deduplicated=True,
                    )

            record = AlertRecord(
                alert_id=_new_alert_id(),
                patient_id=request.patient_id,
                rule_id=rule_id,
                hour=request.hour,
                first_hour=request.hour,
                last_hour=request.hour,
                probability=float(request.probability),
                peak_probability=float(request.probability),
                threshold=threshold,
                threshold_source=threshold_source,
                threshold_version=threshold_version,
                model_name=request.model_name,
                model_version=request.model_version,
                prediction_source=request.prediction_source,
                status=AlertStatus.ACTIVE,
                created_at=now,
                updated_at=now,
                occurrence_count=1,
                is_synthetic=request.is_synthetic,
            )
            created = self.store.create_alert(record)
            return AlertEvaluationResult(
                action="created",
                alert=created,
                reason="Probability met the configured alert threshold.",
                deduplicated=False,
            )

    def evaluate_prediction(
        self,
        prediction: PredictionResult,
        *,
        is_synthetic: bool | None = None,
    ) -> AlertEvaluationResult:
        if not prediction.available or prediction.probability is None:
            return AlertEvaluationResult(
                action="unavailable",
                alert=None,
                reason="No available prediction for this patient-hour.",
                deduplicated=False,
            )
        request = AlertEvaluationRequest(
            patient_id=prediction.patient_id,
            hour=prediction.hour,
            probability=float(prediction.probability),
            model_name=prediction.model.name,
            model_version=prediction.model.version,
            prediction_source=(
                prediction.source.value
                if hasattr(prediction.source, "value")
                else str(prediction.source)
            ),
            is_synthetic=prediction.is_synthetic if is_synthetic is None else is_synthetic,
        )
        return self.evaluate(request)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def acknowledge(self, alert_id: str) -> AlertRecord:
        return self.store.acknowledge_alert(alert_id, at=self._clock())

    def clear(self, alert_id: str, reason: str = "manual_clear") -> AlertRecord:
        return self.store.clear_alert(alert_id, at=self._clock(), reason=reason)

    def get(self, alert_id: str) -> AlertRecord:
        return self.store.get_alert(alert_id)

    def list_active(self, limit: int = 200) -> list[AlertRecord]:
        return self.store.list_active(limit=limit)

    def list_history(self, limit: int = 200) -> list[AlertRecord]:
        return self.store.list_history(limit=limit)

    def list_all(
        self,
        *,
        status: AlertStatus | str | None = None,
        patient_id: str | None = None,
        limit: int = 200,
    ) -> list[AlertRecord]:
        return self.store.list_alerts(status=status, patient_id=patient_id, limit=limit)

    def list_for_patient(self, patient_id: str, limit: int = 200) -> list[AlertRecord]:
        return self.store.list_alerts(patient_id=patient_id, limit=limit)

    def threshold_info(self) -> AlertThresholdInfo:
        return self.config.to_public()


__all__ = [
    "AlertEngine",
    "AlertNotFoundError",
    "AlertStateError",
]
