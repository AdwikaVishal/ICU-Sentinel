"""Alert Centre API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ml.p3_temporal.alerts.alert_schemas import (
    AlertEvaluationRequest,
    AlertEvaluationResult,
    AlertListResponse,
    AlertRecord,
    AlertThresholdInfo,
)

from backend.p3.dependencies import get_container
from backend.p3.schemas import AlertClearRequest

router = APIRouter(prefix="/api/v3/alerts", tags=["alerts"])


@router.get("", response_model=AlertListResponse)
def list_alerts(
    status: str | None = Query(default=None),
    patient_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    container=Depends(get_container),
) -> AlertListResponse:
    alerts = container.alert_engine.list_all(
        status=status, patient_id=patient_id, limit=limit
    )
    return AlertListResponse(
        alerts=alerts,
        count=len(alerts),
        filters={"status": status, "patient_id": patient_id, "limit": limit},
    )


@router.get("/active", response_model=AlertListResponse)
def list_active_alerts(
    limit: int = Query(default=200, ge=1, le=1000),
    container=Depends(get_container),
) -> AlertListResponse:
    alerts = container.alert_engine.list_active(limit=limit)
    return AlertListResponse(
        alerts=alerts,
        count=len(alerts),
        filters={"status": "ACTIVE", "limit": limit},
    )


@router.get("/history", response_model=AlertListResponse)
def list_alert_history(
    limit: int = Query(default=200, ge=1, le=1000),
    container=Depends(get_container),
) -> AlertListResponse:
    alerts = container.alert_engine.list_history(limit=limit)
    return AlertListResponse(
        alerts=alerts,
        count=len(alerts),
        filters={"status": "ACKNOWLEDGED_OR_CLEARED", "limit": limit},
    )


@router.get("/threshold", response_model=AlertThresholdInfo)
def alert_threshold(container=Depends(get_container)) -> AlertThresholdInfo:
    return container.alert_engine.threshold_info()


@router.post("/evaluate", response_model=AlertEvaluationResult)
def evaluate_alert(
    request: AlertEvaluationRequest,
    container=Depends(get_container),
) -> AlertEvaluationResult:
    return container.alert_engine.evaluate(request)


@router.get("/{alert_id}", response_model=AlertRecord)
def get_alert(alert_id: str, container=Depends(get_container)) -> AlertRecord:
    return container.alert_engine.get(alert_id)


@router.post("/{alert_id}/acknowledge", response_model=AlertRecord)
def acknowledge_alert(alert_id: str, container=Depends(get_container)) -> AlertRecord:
    return container.alert_engine.acknowledge(alert_id)


@router.post("/{alert_id}/clear", response_model=AlertRecord)
def clear_alert(
    alert_id: str,
    request: AlertClearRequest | None = None,
    container=Depends(get_container),
) -> AlertRecord:
    reason = request.reason if request is not None else "manual_clear"
    return container.alert_engine.clear(alert_id, reason=reason)
