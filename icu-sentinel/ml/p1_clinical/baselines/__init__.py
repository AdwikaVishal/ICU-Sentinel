"""P1 clinical baselines package."""

from .baseline_pipeline import (
    BaselinePipeline,
    BaselinePipelineResult,
    run_baseline_pipeline,
)
from .baseline_schema import (
    BaselineSchema,
    ComponentResult,
    ScoreEntry,
    ScoreResult,
    build_default_schema,
)
from .clinical_score_config import ClinicalScoreConfig
from .news_style import (
    NEWS2_VERSION,
    compute_news2_style,
    validate_news2_results,
)
from .qsofa import QSOFA_VERSION, compute_qsofa, validate_qsofa_results
from .score_completeness import (
    CompletenessLevel,
    ComponentAvailability,
    ScoreCompleteness,
    assess_completeness,
    is_score_complete,
    is_score_none,
    is_score_partial,
)
from .sirs import SIRS_VERSION, compute_sirs_for_patient, validate_sirs_results

__all__ = [
    "NEWS2_VERSION",
    "QSOFA_VERSION",
    "SIRS_VERSION",
    "BaselinePipeline",
    "BaselinePipelineResult",
    "BaselineSchema",
    "ClinicalScoreConfig",
    "CompletenessLevel",
    "ComponentAvailability",
    "ComponentResult",
    "ScoreCompleteness",
    "ScoreEntry",
    "ScoreResult",
    "assess_completeness",
    "build_default_schema",
    "compute_news2_style",
    "compute_qsofa",
    "compute_sirs_for_patient",
    "is_score_complete",
    "is_score_none",
    "is_score_partial",
    "run_baseline_pipeline",
    "validate_news2_results",
    "validate_qsofa_results",
    "validate_sirs_results",
]
