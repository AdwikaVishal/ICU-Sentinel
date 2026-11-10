"""Reusable, stateless patient replay engine.

Navigation methods take the current hour explicitly and return a new
replay state. No module-level mutable replay cursor exists, so separate
users/sessions cannot interfere with one another.
"""

from __future__ import annotations

from typing import Any

from ml.p3_temporal.alerts.alert_engine import AlertEngine
from ml.p3_temporal.alerts.alert_schemas import AlertRecord, AlertThresholdInfo
from ml.p3_temporal.common.prediction import (
    PredictionContext,
    PredictionProvider,
    PredictionResult,
    PredictionSource,
    ProviderStatus,
)

from .demo_data import PatientNotFoundError, PatientRepository
from .model_input import CausalFeatureBuilder, ModelInputError
from .replay_schemas import (
    CausalityMetadata,
    PatientListResponse,
    PatientSummary,
    ReplaySession,
    ReplayState,
    TimelineResponse,
    TimelineHourSummary,
)
from .timeline import HourOutOfRangeError, TimelineIndex, TimelineService


class ReplayEngine:
    """Patient listing, timeline, prediction and navigation service."""

    def __init__(
        self,
        repository: PatientRepository,
        provider: PredictionProvider,
        *,
        feature_builder: CausalFeatureBuilder | None = None,
        timeline_service: TimelineService | None = None,
        alert_engine: AlertEngine | None = None,
    ) -> None:
        self.repository = repository
        self.provider = provider
        self.timeline = timeline_service or TimelineService(repository)
        self.feature_builder = feature_builder or CausalFeatureBuilder(
            repository, timeline_service=self.timeline
        )
        self.alert_engine = alert_engine

    # ------------------------------------------------------------------
    # Inventory and timeline
    # ------------------------------------------------------------------
    def list_patients(self) -> PatientListResponse:
        patients = self.repository.list_patients()
        return PatientListResponse(
            patients=patients,
            count=len(patients),
            data_source=self.repository.data_source,
            is_synthetic=self.repository.is_synthetic,
            prediction_source=self.provider.identity.source.value,
            provider=self.provider.identity.name,
        )

    def get_patient(self, patient_id: str) -> PatientSummary:
        return self.timeline.patient_summary(patient_id)

    def get_timeline(self, patient_id: str) -> TimelineResponse:
        index = self.timeline.index(patient_id)
        hours = self.timeline.timeline(patient_id)
        return TimelineResponse(
            patient=index.summary,
            hours=hours,
            count=len(hours),
            missing_hours=index.missing_hours,
            data_source=self.repository.data_source,
            is_synthetic=self.repository.is_synthetic,
        )

    def timeline_index(self, patient_id: str) -> TimelineIndex:
        return self.timeline.index(patient_id)

    # ------------------------------------------------------------------
    # Replay state
    # ------------------------------------------------------------------
    def get_state(
        self,
        patient_id: str,
        hour: int,
        *,
        evaluate_alerts: bool = False,
    ) -> ReplayState:
        index = self.timeline.index(patient_id)
        hour = self._validate_hour(index, hour)
        timeline_hour = self.timeline.hour_summary(index, hour)
        observations = self.timeline.observations(patient_id, hour)
        context = self._build_context(patient_id, hour)
        prediction = self.provider.predict(context)

        # Collect predictions from all wrapped models (real mode) or just
        # the single provider (synthetic mode).
        predictions: list = [prediction]
        if hasattr(self.provider, "predict_all"):
            try:
                all_preds = self.provider.predict_all(context)
                if all_preds:
                    predictions = list(all_preds)
            except Exception:  # noqa: BLE001 - never break replay for extra models
                predictions = [prediction]

        alert: AlertRecord | None = None
        alert_status = "not_available" if not prediction.available else "not_evaluated"
        alert_error: str | None = None
        if prediction.available and evaluate_alerts and self.alert_engine is not None:
            try:
                evaluation = self.alert_engine.evaluate_prediction(prediction)
                alert = evaluation.alert
                alert_status = evaluation.action
            except Exception as exc:  # noqa: BLE001 - alert failure must not break replay
                alert_status = "unavailable"
                alert_error = f"{type(exc).__name__}: {exc}"
        elif prediction.available and evaluate_alerts and self.alert_engine is None:
            alert_status = "unavailable"
            alert_error = "Alert engine is not configured."

        prediction_status = (
            "missing_hour"
            if not context.observations
            else ("available" if prediction.available else "unavailable")
        )
        if alert_status == "not_evaluated" and prediction_status == "missing_hour":
            alert_status = "not_available"

        explanation_available = bool(
            prediction.available
            and prediction.source != PredictionSource.SYNTHETIC_HEURISTIC
            and prediction.source != PredictionSource.UNAVAILABLE
        )
        session = self._session(index, hour)
        return ReplayState(
            session=session,
            patient=index.summary,
            hour=hour,
            timeline=timeline_hour,
            observations=observations,
            prediction=prediction,
            predictions=predictions,
            alert=alert,
            alert_status=alert_status,
            alert_error=alert_error,
            retrospective_label=context.retrospective_label,
            retrospective_label_available=context.retrospective_label is not None,
            data_source=self.repository.data_source,
            is_synthetic=self.repository.is_synthetic,
            prediction_status=prediction_status,
            explanation_available=explanation_available,
            causality=CausalityMetadata(
                max_input_hour=hour,
                prefix_row_count=context.prefix_row_count,
                input_observation_count=len(context.observations),
                model_input_hash=context.input_hash,
                feature_schema_version=context.feature_schema_version,
            ),
        )

    def start(
        self,
        patient_id: str,
        hour: int | None = None,
        *,
        evaluate_alerts: bool = False,
    ) -> ReplayState:
        index = self.timeline.index(patient_id)
        return self.get_state(
            patient_id,
            index.min_hour if hour is None else hour,
            evaluate_alerts=evaluate_alerts,
        )

    def step_forward(
        self,
        patient_id: str,
        hour: int,
        *,
        evaluate_alerts: bool = True,
    ) -> ReplayState:
        index = self.timeline.index(patient_id)
        current = self._validate_hour(index, hour)
        return self.get_state(
            patient_id,
            min(index.max_hour, current + 1),
            evaluate_alerts=evaluate_alerts,
        )

    def step_backward(
        self,
        patient_id: str,
        hour: int,
        *,
        evaluate_alerts: bool = True,
    ) -> ReplayState:
        index = self.timeline.index(patient_id)
        current = self._validate_hour(index, hour)
        return self.get_state(
            patient_id,
            max(index.min_hour, current - 1),
            evaluate_alerts=evaluate_alerts,
        )

    def jump_to_hour(
        self,
        patient_id: str,
        hour: int,
        *,
        evaluate_alerts: bool = True,
    ) -> ReplayState:
        return self.get_state(patient_id, hour, evaluate_alerts=evaluate_alerts)

    def reset(self, patient_id: str, *, evaluate_alerts: bool = False) -> ReplayState:
        index = self.timeline.index(patient_id)
        return self.get_state(patient_id, index.min_hour, evaluate_alerts=evaluate_alerts)

    def end(self, patient_id: str, *, evaluate_alerts: bool = False) -> ReplayState:
        index = self.timeline.index(patient_id)
        return self.get_state(patient_id, index.max_hour, evaluate_alerts=evaluate_alerts)

    # ------------------------------------------------------------------
    # Model input / metadata
    # ------------------------------------------------------------------
    def get_model_input(self, patient_id: str, hour: int) -> PredictionContext:
        return self._build_context(patient_id, hour)

    def get_prediction(self, patient_id: str, hour: int) -> PredictionResult:
        return self.provider.predict(self._build_context(patient_id, hour))

    def provider_status(self) -> ProviderStatus:
        return self.provider.status()

    def threshold_info(self) -> AlertThresholdInfo | None:
        if self.alert_engine is None:
            return None
        return self.alert_engine.threshold_info()

    def alerts_for_patient(self, patient_id: str, limit: int = 200) -> list[AlertRecord]:
        if self.alert_engine is None:
            return []
        return self.alert_engine.list_for_patient(patient_id, limit=limit)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_context(self, patient_id: str, hour: int) -> PredictionContext:
        required = self.provider.required_features
        try:
            return self.feature_builder.build(
                patient_id,
                hour,
                selected_features=required,
            ).context
        except ModelInputError:
            raise
        except PatientNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelInputError(
                f"Could not construct causal model input for {patient_id} hour {hour}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _validate_hour(self, index: TimelineIndex, hour: int) -> int:
        hour = int(hour)
        if hour < index.min_hour or hour > index.max_hour:
            raise HourOutOfRangeError(
                f"Hour {hour} outside timeline [{index.min_hour}, {index.max_hour}] "
                f"for patient {index.patient.patient_id}"
            )
        return hour

    def _session(self, index: TimelineIndex, hour: int) -> ReplaySession:
        return ReplaySession(
            patient_id=index.patient.patient_id,
            current_hour=hour,
            min_hour=index.min_hour,
            max_hour=index.max_hour,
            total_hours=index.total_hours,
            at_start=hour == index.min_hour,
            at_end=hour == index.max_hour,
        )
