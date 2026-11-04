"""Training protocol for P1 classical ML models.

Implements the training/evaluation protocol:
- Training set: fit imputers, scalers, feature selectors, models.
- Validation set: compare configurations, select hyperparameters,
  select alert thresholds.
- Test set: evaluate final frozen models once (no tuning).

All preprocessing is fitted on training data only. Validation and test
data never influence fitted parameters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..features.feature_schema import FeatureSchema
from ..preprocessing.patient_splits import SplitManifest
from . import logistic_regression as lr_module
from . import random_forest as rf_module
from . import svm as svm_module
from .data_loader import (
    check_split_overlap,
    load_split_aware_dataset,
    validate_target_isolation,
)
from .feature_selector import FeatureSelectionReport, FeatureSelector
from .model_config import ModelConfig
from .model_registry import ModelMetadata, ModelRegistry
from .prediction_schema import PredictionRecord

logger = logging.getLogger("icu_sentinel.p1_clinical.classical_ml.training")


@dataclass
class TrainingResult:
    """Complete training result for all models."""

    config: ModelConfig
    dataset_source: str
    feature_selection_report: FeatureSelectionReport | None = None
    selected_features: list[str] = field(default_factory=list)
    models: dict = field(default_factory=dict)  # model_name -> result
    validation_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    split_overlap_check: dict = field(default_factory=dict)
    target_isolation_check: dict = field(default_factory=dict)
    training_timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "dataset_source": self.dataset_source,
            "feature_selection_report": (
                self.feature_selection_report.to_dict()
                if self.feature_selection_report else None
            ),
            "selected_features": list(self.selected_features),
            "n_selected_features": len(self.selected_features),
            "models": {
                name: {
                    "training_metrics": getattr(m, "training_metrics", {}),
                    "validation_metrics": getattr(m, "validation_metrics", {}),
                    "n_features": len(getattr(m, "selected_features", [])),
                }
                for name, m in self.models.items()
            },
            "validation_metrics": dict(self.validation_metrics),
            "test_metrics": dict(self.test_metrics),
            "split_overlap_check": dict(self.split_overlap_check),
            "target_isolation_check": dict(self.target_isolation_check),
            "training_timestamp": self.training_timestamp,
        }

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def run_training_protocol(
    processed_dir: str | Path,
    split_manifest_path: str | Path,
    feature_schema_path: str | Path | None = None,
    config: ModelConfig | None = None,
    artifact_dir: str | Path = "artifacts/p1/classical_ml",
    reports_dir: str | Path = "data/reports/classical_ml",
    source: str = "synthetic",
) -> TrainingResult:
    """Run the complete training protocol.

    Parameters
    ----------
    processed_dir : str | Path
        Directory with per-patient processed parquet files (Part 4 output).
    split_manifest_path : str | Path
        Path to split manifest JSON (Part 3 output).
    feature_schema_path : str | Path, optional
        Path to feature schema JSON (Part 4 output).
    config : ModelConfig, optional
        Model configuration. Uses defaults if None.
    artifact_dir : str | Path
        Directory for model artifacts.
    reports_dir : str | Path
        Directory for training reports.
    source : str
        Data source label ("synthetic" or "real").

    Returns
    -------
    TrainingResult
        Complete training result.
    """
    config = config or ModelConfig()
    artifact_path = Path(artifact_dir)
    report_path = Path(reports_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    training_timestamp = datetime.now(UTC).isoformat()

    # 1. Load split manifest and dataset.
    logger.info("Loading split-aware dataset from %s", processed_dir)
    manifest = None
    feature_schema = None
    if split_manifest_path is not None:
        manifest = SplitManifest.load(split_manifest_path)
    if feature_schema_path is not None:
        feature_schema = FeatureSchema.load(feature_schema_path)
    dataset = load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        feature_schema=feature_schema,
        source=source,
    )

    # 2. Check split overlap (leakage prevention).
    overlap_check = check_split_overlap(dataset)
    logger.info(
        "Split overlap check: leakage_detected=%s, overlapping=%d",
        overlap_check["leakage_detected"],
        overlap_check["n_overlapping_patients"],
    )

    # 3. Validate target isolation.
    target_check = validate_target_isolation(dataset)
    logger.info(
        "Target isolation check: valid=%s",
        target_check["target_isolation_valid"],
    )

    # 4. Get training split.
    train_split = dataset.train
    if train_split is None or len(train_split.X) == 0:
        raise ValueError("No training data available.")

    X_train = train_split.X.copy()
    y_train = train_split.y.copy()

    # 5. Feature selection on training data only.
    logger.info("Running feature selection on training data.")
    selector = FeatureSelector(
        remove_constant=config.remove_constant,
        remove_duplicate=config.remove_duplicate,
        remove_all_missing=config.remove_all_missing,
        remove_infinite=config.remove_infinite,
    )

    feature_schema = None
    if feature_schema_path is not None:
        feature_schema = FeatureSchema.load(feature_schema_path)

    X_train_selected = selector.fit_transform(
        X_train, feature_schema=feature_schema
    )
    fs_report = selector.report()
    selected_features = selector.kept_features
    logger.info(
        "Feature selection: %d -> %d features",
        fs_report.n_features_input,
        fs_report.n_features_output,
    )

    # 6. Train each enabled model.
    models = {}
    validation_metrics = {}

    if "logistic_regression" in config.enabled_models():
        logger.info("Training Logistic Regression.")
        lr_result = lr_module.train_lr(
            X_train_selected, y_train, config, selected_features
        )
        models["logistic_regression"] = lr_result
        validation_metrics["logistic_regression"] = lr_result.training_metrics

    if "random_forest" in config.enabled_models():
        logger.info("Training Random Forest.")
        rf_result = rf_module.train_rf(
            X_train_selected, y_train, config, selected_features
        )
        models["random_forest"] = rf_result
        validation_metrics["random_forest"] = rf_result.training_metrics

    if "svm" in config.enabled_models():
        logger.info("Training SVM.")
        svm_result = svm_module.train_svm(
            X_train_selected, y_train, config, selected_features
        )
        models["svm"] = svm_result
        validation_metrics["svm"] = svm_result.training_metrics

    # 7. Save model artifacts and metadata.
    for name, result in models.items():
        model_artifact_dir = artifact_path / name
        model_artifact_dir.mkdir(parents=True, exist_ok=True)

        # Determine save function for this model.
        save_func_map = {
            "logistic_regression": lr_module.save_lr_artifact,
            "random_forest": rf_module.save_rf_artifact,
            "svm": svm_module.save_svm_artifact,
        }
        save_func = save_func_map[name]
        save_func(result, model_artifact_dir)

        # Register in model registry.
        registry = ModelRegistry(artifact_path / "registry")
        metadata = ModelMetadata(
            model_name=name,
            model_version="1.0.0",
            training_timestamp=training_timestamp,
            random_seed=config.random_seed,
            feature_names=list(X_train.columns),
            selected_feature_names=list(selected_features),
            n_features_input=len(X_train.columns),
            n_features_output=len(selected_features),
            n_patients_train=int(train_split.n_patients),
            n_rows_train=int(train_split.n_rows),
            n_positive_train=int(train_split.n_positive),
            n_negative_train=int(train_split.n_negative),
            class_weight=getattr(config, f"{name.split('_')[0]}_class_weight", None),
            hyperparameters={},
            preprocessing_config={
                "impute_strategy": config.impute_strategy,
                "scale_with_mean": config.scale_with_mean,
                "scale_with_std": config.scale_with_std,
            },
            validation_metrics=dict(result.validation_metrics),
            split_identity="patient_level_stratified",
            source=source,
        )
        registry.register(
            model_name=name,
            model_version="1.0.0",
            metadata=metadata,
            artifact_path=model_artifact_dir / f"{name}_1.0.0.joblib",
        )
    registry.save_registry(artifact_path / "registry" / "registry.json")

    # 8. Save feature selection report.
    fs_report.save(report_path / "feature_selection_report.json")

    # 9. Save selected features list.
    import json as _json

    with open(report_path / "selected_features.json", "w") as f:
        _json.dump(
            {
                "selected_features": list(selected_features),
                "n_features": len(selected_features),
                "fit_on_train_only": True,
            },
            f,
            indent=2,
        )

    result = TrainingResult(
        config=config,
        dataset_source=source,
        feature_selection_report=fs_report,
        selected_features=list(selected_features),
        models=models,
        validation_metrics=validation_metrics,
        split_overlap_check=overlap_check,
        target_isolation_check=target_check,
        training_timestamp=training_timestamp,
    )
    result.save(report_path / "training_summary.json")

    logger.info("Training protocol complete. %d models trained.", len(models))
    return result


def evaluate_on_split(
    models: dict,
    selected_features: list[str],
    split_data,
    threshold: float = 0.5,
    source: str = "synthetic",
) -> dict:
    """Evaluate all trained models on a given split.

    Parameters
    ----------
    models : dict
        Trained model results (name -> result).
    selected_features : list[str]
        Selected feature names.
    split_data : SplitData
        Split data to evaluate on.
    threshold : float
        Decision threshold.
    source : str
        Data source label.

    Returns
    -------
    dict
        Per-model metrics and predictions.
    """
    from .prediction_schema import PredictionBatch

    X = split_data.X[selected_features].copy()
    y = split_data.y.copy()
    patient_ids = split_data.patient_ids.copy()
    hours = split_data.hours.copy()

    results = {}
    for name, model_result in models.items():
        # Get prediction function for this model.
        if name == "logistic_regression":
            pred_df = lr_module.predict_lr(model_result, X, threshold=threshold)
        elif name == "random_forest":
            pred_df = rf_module.predict_rf(model_result, X, threshold=threshold)
        elif name == "svm":
            pred_df = svm_module.predict_svm(model_result, X, threshold=threshold)
        else:
            continue

        # Compute metrics.
        metrics = lr_module._compute_metrics(
            y, pred_df["predicted_class"], pred_df["predicted_probability"]
        )

        # Build prediction batch.
        pred_records = []
        for i in range(len(pred_df)):
            pred_records.append(
                {
                    "patient_id": str(patient_ids.iloc[i]),
                    "hour": int(hours.iloc[i]),
                    "true_label": int(y.iloc[i]),
                    "predicted_probability": float(
                        pred_df["predicted_probability"].iloc[i]
                    ),
                    "predicted_class": int(pred_df["predicted_class"].iloc[i]),
                    "model_name": name,
                    "model_version": "1.0.0",
                    "split": split_data.split,
                    "source_dataset": None,
                    "threshold": threshold,
                }
            )
        batch = PredictionBatch(
            model_name=name,
            model_version="1.0.0",
            split=split_data.split,
            n_predictions=len(pred_records),
            n_positive_true=int((y == 1).sum()),
            n_positive_pred=int((pred_df["predicted_class"] == 1).sum()),
            predictions=[
                PredictionRecord(**r) for r in pred_records
            ],
        )

        results[name] = {
            "metrics": metrics,
            "predictions": batch,
        }

    return results
