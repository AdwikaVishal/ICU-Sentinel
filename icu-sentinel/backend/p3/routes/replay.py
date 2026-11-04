"""Patient replay API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ml.p3_temporal.alerts.alert_schemas import AlertThresholdInfo
from ml.p3_temporal.common.prediction import ProviderStatus
from ml.p3_temporal.replay.replay_schemas import (
    PatientListResponse,
    PatientSummary,
    ReplayJumpRequest,
    ReplayState,
    ReplayStepRequest,
    TimelineResponse,
)

from backend.p3.dependencies import get_container
from backend.p3.schemas import CausalInputResponse

router = APIRouter(prefix="/api/v3", tags=["replay"])


@router.get("/patients", response_model=PatientListResponse)
def list_patients(container=Depends(get_container)) -> PatientListResponse:
    return container.replay_engine.list_patients()


@router.get("/patients/{patient_id}", response_model=PatientSummary)
def get_patient(patient_id: str, container=Depends(get_container)) -> PatientSummary:
    return container.replay_engine.get_patient(patient_id)


@router.get("/patients/{patient_id}/timeline", response_model=TimelineResponse)
def get_timeline(patient_id: str, container=Depends(get_container)) -> TimelineResponse:
    return container.replay_engine.get_timeline(patient_id)


@router.get("/provider/status", response_model=ProviderStatus)
def provider_status(container=Depends(get_container)) -> ProviderStatus:
    return container.replay_engine.provider_status()


@router.get("/provider/threshold", response_model=AlertThresholdInfo | None)
def provider_threshold(container=Depends(get_container)):
    return container.replay_engine.threshold_info()


@router.get("/replay/{patient_id}", response_model=ReplayState)
def get_replay_state(
    patient_id: str,
    hour: int | None = Query(default=None, ge=0),
    evaluate_alerts: bool = Query(default=False),
    container=Depends(get_container),
) -> ReplayState:
    if hour is None:
        return container.replay_engine.start(
            patient_id, evaluate_alerts=evaluate_alerts
        )
    return container.replay_engine.get_state(
        patient_id, hour, evaluate_alerts=evaluate_alerts
    )


@router.get("/replay/{patient_id}/model-input", response_model=CausalInputResponse)
def get_model_input(
    patient_id: str,
    hour: int = Query(ge=0),
    container=Depends(get_container),
) -> CausalInputResponse:
    context = container.replay_engine.get_model_input(patient_id, hour)
    return CausalInputResponse.from_context(context)


@router.post("/replay/{patient_id}/step", response_model=ReplayState)
def step_replay(
    patient_id: str,
    request: ReplayStepRequest,
    container=Depends(get_container),
) -> ReplayState:
    if request.direction == "forward":
        return container.replay_engine.step_forward(
            patient_id,
            request.hour,
            evaluate_alerts=request.evaluate_alerts,
        )
    return container.replay_engine.step_backward(
        patient_id,
        request.hour,
        evaluate_alerts=request.evaluate_alerts,
    )


@router.post("/replay/{patient_id}/jump", response_model=ReplayState)
def jump_replay(
    patient_id: str,
    request: ReplayJumpRequest,
    container=Depends(get_container),
) -> ReplayState:
    return container.replay_engine.jump_to_hour(
        patient_id,
        request.hour,
        evaluate_alerts=request.evaluate_alerts,
    )


@router.post("/replay/{patient_id}/reset", response_model=ReplayState)
def reset_replay(
    patient_id: str,
    evaluate_alerts: bool = Query(default=False),
    container=Depends(get_container),
) -> ReplayState:
    return container.replay_engine.reset(
        patient_id, evaluate_alerts=evaluate_alerts
    )


@router.post("/replay/{patient_id}/end", response_model=ReplayState)
def end_replay(
    patient_id: str,
    evaluate_alerts: bool = Query(default=False),
    container=Depends(get_container),
) -> ReplayState:
    return container.replay_engine.end(patient_id, evaluate_alerts=evaluate_alerts)
