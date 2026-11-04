"""Explainable AI API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ml.p3_temporal.explainability.explanation_schemas import (
    FeatureMetadata,
    GlobalExplanation,
    HourlyExplanationComparison,
    ModelCatalogResponse,
    PatientExplanation,
)

from backend.p3.dependencies import get_container

router = APIRouter(prefix="/api/v3/explanations", tags=["explanations"])


@router.get("/models", response_model=ModelCatalogResponse)
def list_explainable_models(container=Depends(get_container)) -> ModelCatalogResponse:
    return container.explanation_service.catalog()


@router.get("/global", response_model=GlobalExplanation)
def global_explanation(
    model_name: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    container=Depends(get_container),
) -> GlobalExplanation:
    return container.explanation_service.global_explanation(
        model_name=model_name, model_version=model_version
    )


@router.get("/features/{feature_name}", response_model=FeatureMetadata)
def feature_metadata(
    feature_name: str,
    container=Depends(get_container),
) -> FeatureMetadata:
    return container.explanation_service.mapper.describe(feature_name)


@router.get("/{patient_id}/compare", response_model=HourlyExplanationComparison)
def compare_hours(
    patient_id: str,
    hour_a: int = Query(ge=0),
    hour_b: int = Query(ge=0),
    container=Depends(get_container),
) -> HourlyExplanationComparison:
    return container.explanation_service.compare(patient_id, hour_a, hour_b)


@router.get("/{patient_id}/{hour}", response_model=PatientExplanation)
def explain_hour(
    patient_id: str,
    hour: int,
    container=Depends(get_container),
) -> PatientExplanation:
    return container.explanation_service.explain(patient_id, hour)
