"""Schemas for model explanations (SHAP and explicit unavailable states)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ml.p3_temporal.common.prediction import ModelIdentity, PredictionResult


class ExplanationStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class ExplanationMethod(str, Enum):
    LINEAR_SHAP = "linear_shap"
    TREE_SHAP = "tree_shap"
    KERNEL_SHAP = "kernel_shap"
    NOT_APPLICABLE = "not_applicable"


class FeatureMetadata(BaseModel):
    """Readable metadata for a model feature.

    Unknown features keep their original name and are marked unavailable;
    no clinical meaning, unit or direction is invented for them.
    """

    feature_name: str
    display_name: str
    category: str
    description: str | None = None
    unit: str | None = None
    feature_type: str | None = None
    is_missing_indicator: bool = False
    is_engineered: bool = False
    metadata_available: bool = False


class FeatureContribution(BaseModel):
    feature_name: str
    display_name: str
    category: str
    description: str | None = None
    unit: str | None = None
    is_missing_indicator: bool = False
    value: float | None = None
    shap_value: float
    magnitude: float = Field(ge=0)
    direction: str
    rank: int = Field(ge=1)


class ExplanationValidation(BaseModel):
    additive_error: float | None = None
    tolerance: float | None = None
    passed: bool
    checked_output_space: str
    explanation_output: float | None = None
    prediction_probability: float | None = None
    note: str | None = None


class ExplanationModelMetadata(BaseModel):
    model: ModelIdentity
    explanation_method: ExplanationMethod
    output_space: str
    feature_schema_version: str
    background_rows: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PatientExplanation(BaseModel):
    status: ExplanationStatus
    patient_id: str
    hour: int
    prediction: PredictionResult | None = None
    model: ModelIdentity | None = None
    explanation_method: ExplanationMethod = ExplanationMethod.NOT_APPLICABLE
    output_space: str | None = None
    base_value: float | None = None
    model_output: float | None = None
    prediction_probability: float | None = None
    contributions: list[FeatureContribution] = Field(default_factory=list)
    positive_contributions: list[FeatureContribution] = Field(default_factory=list)
    negative_contributions: list[FeatureContribution] = Field(default_factory=list)
    validation: ExplanationValidation | None = None
    input_hash: str | None = None
    explanation_id: str | None = None
    reason: str | None = None
    metadata: ExplanationModelMetadata | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    disclaimer: str = (
        "SHAP contributions describe how this fitted model's output changes "
        "for this input. They are not causal medical explanations and do not "
        "confirm sepsis."
    )


class GlobalFeatureImportance(BaseModel):
    rank: int = Field(ge=1)
    feature_name: str
    display_name: str
    category: str
    description: str | None = None
    unit: str | None = None
    is_missing_indicator: bool = False
    mean_abs_shap: float = Field(ge=0)
    mean_shap: float


class GlobalFeatureGroup(BaseModel):
    category: str
    mean_abs_shap: float = Field(ge=0)
    n_features: int = Field(ge=1)


class GlobalExplanation(BaseModel):
    status: ExplanationStatus
    model: ModelIdentity | None = None
    explanation_method: ExplanationMethod = ExplanationMethod.NOT_APPLICABLE
    output_space: str | None = None
    features: list[GlobalFeatureImportance] = Field(default_factory=list)
    feature_groups: list[GlobalFeatureGroup] = Field(default_factory=list)
    background_rows: int = 0
    reason: str | None = None
    metadata: ExplanationModelMetadata | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    disclaimer: str = (
        "Global importance ranks mean absolute SHAP contributions for the "
        "selected fitted model and background dataset. It is not evidence of "
        "clinical causality."
    )


class FeatureChange(BaseModel):
    feature_name: str
    display_name: str
    category: str
    shap_a: float
    shap_b: float
    delta: float
    magnitude: float = Field(ge=0)
    direction: str


class HourlyExplanationComparison(BaseModel):
    status: ExplanationStatus
    patient_id: str
    hour_a: int
    hour_b: int
    explanation_a: PatientExplanation | None = None
    explanation_b: PatientExplanation | None = None
    changes: list[FeatureChange] = Field(default_factory=list)
    largest_changes: list[FeatureChange] = Field(default_factory=list)
    reason: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    disclaimer: str = (
        "Changes in SHAP contributions describe model behaviour between two "
        "hours. They do not establish a clinical cause or a causal relationship."
    )


class ModelCatalogEntry(BaseModel):
    model: ModelIdentity
    explainable: bool
    explanation_method: ExplanationMethod | None = None
    reason: str | None = None
    background_rows: int = 0
    feature_schema_version: str
    notes: str | None = None


class ModelCatalogResponse(BaseModel):
    models: list[ModelCatalogEntry]
    count: int
