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
from typing import Any, Callable

import numpy as np

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
    thresholds: dict = field(default_factory=dict)
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
                name:                 {
                    "training_metrics": getattr(m, "training_metrics", {}),
                    "validation_metrics": getattr(m, "validation_metrics", {}),
                    "n_features": len(getattr(m, "selected_features", [])),
                    "n_train_rows_used": getattr(m, "n_train_rows_used", None),
                    "subsampled": getattr(m, "subsampled", False),
                }
                for name, m in self.models.items()
            },
            "validation_metrics": dict(self.validation_metrics),
            "test_metrics": dict(self.test_metrics),
            "thresholds": dict(self.thresholds),
            "split_overlap_check": dict(self.split_overlap_check),
            "target_isolation_check": dict(self.target_isolation_check),
            "training_timestamp": self.training_timestamp,
        }

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _predict_model(
    name: str,
    model_result,
    X,
    threshold: float = 0.5,
):
    if name == "logistic_regression":
        return lr_module.predict_lr(model_result, X, threshold=threshold)
    if name == "random_forest":
        return rf_module.predict_rf(model_result, X, threshold=threshold)
    if name == "svm":
        return svm_module.predict_svm(model_result, X, threshold=threshold)
    raise ValueError(f"Unsupported model: {name}")


def select_threshold_on_validation(
    y_true,
    y_prob,
    metric: str = "f1",
    grid_step: float = 0.01,
) -> float:
    """Select a decision threshold using validation data only."""
    if grid_step <= 0 or grid_step > 1:
        raise ValueError("grid_step must be in (0, 1]")

    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    metric_funcs = {
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score,
    }
    if metric not in metric_funcs:
        raise ValueError(
            f"Unsupported threshold metric '{metric}'; "
            f"expected one of {sorted(metric_funcs)}"
        )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be non-empty and equal length")

    score_func = metric_funcs[metric]
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in np.arange(0.0, 1.0 + grid_step, grid_step):
        y_pred = (y_prob >= threshold).astype(int)
        try:
            score = float(score_func(y_true, y_pred))
        except ValueError:
            score = float("nan")
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_threshold = float(round(threshold, 10))

    return best_threshold


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

    X_train = train_split.X
    y_train = train_split.y.copy()
    original_feature_names = list(X_train.columns)

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

    # Drop unused columns in place to avoid a second full-matrix copy.
    X_train_selected = selector.fit_transform(
        X_train, feature_schema=feature_schema, copy=False
    )
    fs_report = selector.report()
    selected_features = selector.kept_features
    logger.info(
        "Feature selection: %d -> %d features",
        fs_report.n_features_input,
        fs_report.n_features_output,
    )

    # 6. Train each enabled model.
    models: dict[str, Any] = {}

    if "logistic_regression" in config.enabled_models():
        logger.info("Training Logistic Regression.")
        lr_result = lr_module.train_lr(
            X_train_selected, y_train, config, selected_features
        )
        models["logistic_regression"] = lr_result

    if "random_forest" in config.enabled_models():
        logger.info("Training Random Forest.")
        rf_result = rf_module.train_rf(
            X_train_selected, y_train, config, selected_features
        )
        models["random_forest"] = rf_result

    if "svm" in config.enabled_models():
        logger.info("Training SVM.")
        svm_result = svm_module.train_svm(
            X_train_selected, y_train, config, selected_features
        )
        models["svm"] = svm_result

    # 7. Evaluate on validation, select thresholds on validation only,
    # then evaluate the frozen models once on the test split.
    validation_split = dataset.validation
    if validation_split is None or len(validation_split.X) == 0:
        raise ValueError(
            "No validation data available; cannot select thresholds or "
            "report validation metrics."
        )

    validation_metrics: dict = {}
    test_metrics: dict = {}
    thresholds: dict = {}

    for name, model_result in models.items():
        X_val = validation_split.X[selected_features]
        val_pred = _predict_model(name, model_result, X_val, threshold=0.5)
        threshold = select_threshold_on_validation(
            validation_split.y,
            val_pred["predicted_probability"],
            metric=config.threshold_selection_metric,
            grid_step=config.threshold_grid_step,
        )
        thresholds[name] = threshold

        val_eval = evaluate_on_split(
            models={name: model_result},
            selected_features=selected_features,
            split_data=validation_split,
            threshold=threshold,
            source=source,
            build_prediction_batch=False,
        )
        validation_metrics[name] = val_eval[name]["metrics"]
        model_result.validation_metrics = dict(validation_metrics[name])

        test_split = dataset.test
        if test_split is not None and len(test_split.X) > 0:
            test_eval = evaluate_on_split(
                models={name: model_result},
                selected_features=selected_features,
                split_data=test_split,
                threshold=threshold,
                source=source,
                build_prediction_batch=False,
            )
            test_metrics[name] = test_eval[name]["metrics"]

        logger.info(
            "Validation eval %s: threshold=%.3f, AUROC=%.4f, AUPRC=%.4f",
            name,
            threshold,
            validation_metrics[name].get("auroc", float("nan")),
            validation_metrics[name].get("auprc", float("nan")),
        )

    # 8. Save model artifacts and metadata.
    for name, trained in models.items():
        model_artifact_dir = artifact_path / name
        model_artifact_dir.mkdir(parents=True, exist_ok=True)

        # Determine save function for this model.
        save_func_map: dict[str, Callable[[Any, Path], Any]] = {
            "logistic_regression": lr_module.save_lr_artifact,
            "random_forest": rf_module.save_rf_artifact,
            "svm": svm_module.save_svm_artifact,
        }
        save_func = save_func_map[name]
        save_func(trained, model_artifact_dir)

        # Register in model registry.
        registry = ModelRegistry(artifact_path / "registry")
        metadata = ModelMetadata(
            model_name=name,
            model_version="1.0.0",
            training_timestamp=training_timestamp,
            random_seed=config.random_seed,
            feature_names=list(original_feature_names),
            selected_feature_names=list(selected_features),
            n_features_input=len(original_feature_names),
            n_features_output=len(selected_features),
            n_patients_train=int(train_split.n_patients),
            n_rows_train=int(
                getattr(trained, "n_train_rows_used", 0) or train_split.n_rows
            ),
            n_positive_train=int(train_split.n_positive),
            n_negative_train=int(train_split.n_negative),
            class_weight=getattr(config, f"{name.split('_')[0]}_class_weight", None),
            hyperparameters={},
            preprocessing_config={
                "impute_strategy": config.impute_strategy,
                "scale_with_mean": config.scale_with_mean,
                "scale_with_std": config.scale_with_std,
            },
            validation_metrics=dict(trained.validation_metrics),
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

    # 9. Save feature selection report.
    fs_report.save(report_path / "feature_selection_report.json")

    # 10. Save validation-selected thresholds.
    with open(report_path / "thresholds.json", "w") as f:
        json.dump(
            {
                "thresholds": dict(thresholds),
                "selection_split": "validation",
                "selection_metric": config.threshold_selection_metric,
                "grid_step": config.threshold_grid_step,
                "test_set_used_for_selection": False,
            },
            f,
            indent=2,
        )

    # 11. Save selected features list.
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
        test_metrics=test_metrics,
        thresholds=thresholds,
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
    build_prediction_batch: bool = True,
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

    X = (
        split_data.X
        if list(split_data.X.columns) == list(selected_features)
        else split_data.X[selected_features]
    )
    y = split_data.y
    patient_ids = split_data.patient_ids
    hours = split_data.hours

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

        # Build prediction batch only when requested. On the full hourly
        # dataset, materialising one PredictionRecord per row is a large
        # memory cost that training does not need.
        batch = None
        if build_prediction_batch:
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
