"""Explainable AI (SHAP) for ICU Sentinel P3.

The package explains fitted models only. It never generates stand-in
values for an unsupported model and never describes SHAP contributions as
clinical causes.
"""

from .demo_model import DEMO_FEATURE_COLUMNS, build_demo_model_bundle
from .explainer_factory import (
    ExplainerFactory,
    ExplainerNotAvailable,
    ExplainerPlan,
    ExplanationOutput,
    FittedExplainer,
    plan_status,
)
from .feature_mapping import FeatureMapper
from .global_explainer import GlobalExplainer
from .patient_explainer import (
    ExplanationService,
    PatientExplainer,
    validate_additivity,
)
from .explanation_schemas import (
    ExplanationMethod,
    ExplanationStatus,
    FeatureContribution,
    FeatureMetadata,
    GlobalExplanation,
    HourlyExplanationComparison,
    ModelCatalogEntry,
    ModelCatalogResponse,
    PatientExplanation,
)

__all__ = [
    "DEMO_FEATURE_COLUMNS",
    "ExplainerFactory",
    "ExplainerNotAvailable",
    "ExplainerPlan",
    "ExplanationMethod",
    "ExplanationOutput",
    "ExplanationService",
    "ExplanationStatus",
    "FeatureContribution",
    "FeatureMapper",
    "FeatureMetadata",
    "FittedExplainer",
    "GlobalExplanation",
    "GlobalExplainer",
    "HourlyExplanationComparison",
    "ModelCatalogEntry",
    "ModelCatalogResponse",
    "PatientExplanation",
    "PatientExplainer",
    "build_demo_model_bundle",
    "plan_status",
    "validate_additivity",
]
