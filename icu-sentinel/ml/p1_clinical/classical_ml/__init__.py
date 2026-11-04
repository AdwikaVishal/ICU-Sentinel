"""P1 classical ML models package.

Implements Logistic Regression, Random Forest, and SVM for early sepsis
prediction. All models use the shared Part 4 feature schema and Part 3
patient-level splits. Preprocessing is fitted on training data only.
"""

from . import inference, logistic_regression, random_forest, svm, training
from .data_loader import (
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    LoadedDataset,
    SplitData,
    check_split_overlap,
    load_from_manifest_and_features,
    load_split_aware_dataset,
    validate_target_isolation,
)
from .feature_selector import FeatureSelectionReport, FeatureSelector
from .model_config import ModelConfig
from .model_registry import ModelMetadata, ModelRegistry
from .model_serialization import (
    load_feature_list,
    load_model_artifact,
    load_preprocessing_pipeline,
    save_feature_list,
    save_model_artifact,
    save_preprocessing_pipeline,
)
from .prediction_schema import (
    MODEL_VERSION,
    PredictionBatch,
    PredictionRecord,
    PredictionSchema,
)

__all__ = [
    "IDENTIFIER_COLUMNS",
    "MODEL_VERSION",
    "TARGET_COLUMN",
    "FeatureSelectionReport",
    "FeatureSelector",
    "LoadedDataset",
    "ModelConfig",
    "ModelMetadata",
    "ModelRegistry",
    "PredictionBatch",
    "PredictionRecord",
    "PredictionSchema",
    "SplitData",
    "check_split_overlap",
    "inference",
    "load_feature_list",
    "load_from_manifest_and_features",
    "load_model_artifact",
    "load_preprocessing_pipeline",
    "load_split_aware_dataset",
    "logistic_regression",
    "random_forest",
    "save_feature_list",
    "save_model_artifact",
    "save_preprocessing_pipeline",
    "svm",
    "training",
    "validate_target_isolation",
]
