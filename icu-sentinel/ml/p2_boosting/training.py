"""Training orchestration for P2 models.

The protocol mirrors P1: fit on training data, use validation only for
threshold selection, and evaluate the frozen model once on test.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ml.p1_clinical.classical_ml.data_loader import (
    LoadedDataset,
    check_split_overlap,
    load_from_manifest_and_features,
    validate_target_isolation,
)
from ml.p1_clinical.preprocessing.targets import DERIVED_TARGET_COLUMN

from .base import P2ModelResult, predict_probabilities
from .checkpoint import (
    MANIFEST_FILENAME,
    RunManifest,
    load_manifest,
    manifest_for_run,
    manifest_matches_config,
    record_model_complete,
    save_manifest,
)
from .config import P2Config
from .evaluation import evaluate_predictions, select_validation_threshold
from .experiment import json_safe, save_experiment_record
from .feature_contract import (
    P2_TARGET_COLUMNS,
    resolve_feature_names,
    sanitize_infinities,
    validate_feature_frame,
    validate_feature_names,
)
from .pipelines import SUPPORTED_MODELS, train_model
from .serialization import load_p2_model, save_p2_model


@dataclass
class P2TrainingResult:
    """Complete P2 run with split-specific metrics."""

    config: P2Config
    models: dict[str, P2ModelResult] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    target_column: str = DERIVED_TARGET_COLUMN
    source: str = "synthetic"
    thresholds: dict[str, float] = field(default_factory=dict)
    training_metrics: dict[str, dict] = field(default_factory=dict)
    validation_metrics: dict[str, dict] = field(default_factory=dict)
    test_metrics: dict[str, dict] = field(default_factory=dict)
    split_check: dict = field(default_factory=dict)
    target_check: dict = field(default_factory=dict)
    dataset_summary: dict = field(default_factory=dict)
    training_timestamp: str = ""

    @property
    def n_features_total(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "feature_names": list(self.feature_names),
            "target_column": self.target_column,
            "source": self.source,
            "thresholds": dict(self.thresholds),
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "split_check": self.split_check,
            "target_check": self.target_check,
            "dataset_summary": self.dataset_summary,
            "training_timestamp": self.training_timestamp,
            "effective_threads": {
                "extra_trees": int(self.config.extra_trees.n_jobs),
                "xgboost": int(self.config.xgboost.n_jobs),
                "cpu_count": int(os.cpu_count() or 1),
            },
            "models": {
                name: {
                    "model_version": model.model_version,
                    "n_features": len(model.selected_features),
                    "threshold": model.threshold,
                    "random_seed": model.random_seed,
                }
                for name, model in self.models.items()
            },
        }

    def save(self, filepath: str | Path, overwrite: bool = False) -> Path:
        path = Path(filepath)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing training summary: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(json_safe(self.to_dict()), f, indent=2, allow_nan=False)
        return path


def _require_split(dataset: LoadedDataset, name: str):
    split_data = dataset.get_split(name)
    if split_data is None or len(split_data.X) == 0:
        raise ValueError(f"No {name} data available for P2 training/evaluation.")
    return split_data


def _resolve_features(dataset: LoadedDataset, train) -> list[str]:
    if dataset.feature_names:
        names = list(dataset.feature_names)
    else:
        names = resolve_feature_names(train.X.columns)
    return validate_feature_names(names)


def train_p2_models(
    dataset: LoadedDataset,
    config: P2Config | None = None,
    *,
    artifact_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    source: str | None = None,
    overwrite: bool = False,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
) -> P2TrainingResult:
    """Train, select thresholds on validation, and evaluate on test."""
    config = config or P2Config()
    enabled = config.enabled_models()
    if not enabled:
        raise ValueError("No P2 models are enabled in the configuration.")
    unknown = [name for name in enabled if name not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(f"Unsupported enabled model(s): {unknown}.")

    # Checkpoint / resume wiring.
    checkpoint_path: Path | None = None
    manifest: RunManifest | None = None
    completed_models: set[str] = set()
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir) / MANIFEST_FILENAME
    if resume:
        if checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_dir.")
        manifest = load_manifest(checkpoint_path)
        if manifest is None:
            raise FileNotFoundError(
                f"No checkpoint manifest found at {checkpoint_path}; "
                "cannot resume."
            )
        if not manifest_matches_config(manifest, config):
            raise ValueError(
                "Checkpoint config fingerprint mismatch; refusing to combine "
                "artifacts from different configurations."
            )
        completed_models = set(manifest.completed)
    if dataset.target_column != DERIVED_TARGET_COLUMN:
        raise ValueError(
            "P2 must use the approved target "
            f"'{DERIVED_TARGET_COLUMN}', got '{dataset.target_column}'."
        )
    if dataset.target_column not in P2_TARGET_COLUMNS:
        raise ValueError(
            f"Target '{dataset.target_column}' is not an approved target column."
        )

    train = _require_split(dataset, "train")
    validation = _require_split(dataset, "validation")
    test = dataset.get_split("test")

    split_check = check_split_overlap(dataset)
    if split_check.get("leakage_detected"):
        raise ValueError(
            "Patient-level split leakage detected; refusing to train."
        )
    target_check = validate_target_isolation(dataset)
    if not target_check.get("target_isolation_valid", False):
        raise ValueError(
            "Target/identifier isolation failed; refusing to train: "
            f"{target_check.get('issues', [])}"
        )

    feature_names = _resolve_features(dataset, train)
    # Validate and sanitise each split exactly once. The cleaned frames are
    # reused for every model and for train/validation/test prediction, avoiding
    # repeated full-frame copies. sanitize_infinities is idempotent so values
    # are unchanged versus sanitising inside each model/prediction call.
    X_train = sanitize_infinities(
        validate_feature_frame(train.X, feature_names), inplace=False
    )
    y_train = train.y
    X_validation = sanitize_infinities(
        validate_feature_frame(validation.X, feature_names), inplace=False
    )
    y_validation = validation.y

    run_source = source if source is not None else dataset.source
    if checkpoint_dir is not None and manifest is None:
        manifest = manifest_for_run(
            config,
            run_source,
            dataset.target_column,
            feature_names,
            enabled,
        )
    if manifest is not None and list(manifest.feature_names) != list(
        feature_names
    ):
        raise ValueError(
            "Checkpoint feature set mismatch; refusing to combine artifacts "
            "from different configurations."
        )
    timestamp = datetime.now(UTC).isoformat()
    result = P2TrainingResult(
        config=config,
        feature_names=feature_names,
        target_column=dataset.target_column,
        source=run_source,
        split_check=split_check,
        target_check=target_check,
        dataset_summary=dataset.summary(),
        training_timestamp=timestamp,
    )

    for name in enabled:
        if name in completed_models and manifest is not None:
            # Resume: load the previously checkpointed artifact; never retrain.
            entry = manifest.models[name]
            artifact_path = (
                Path(entry.artifact_path) if entry.artifact_path else None
            )
            if artifact_path is None or not artifact_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint artifact missing for model '{name}': "
                    f"{artifact_path}"
                )
            model_result = load_p2_model(artifact_path)
            model_result.threshold = entry.threshold
            model_result.training_metrics = dict(entry.training_metrics)
            model_result.validation_metrics = dict(entry.validation_metrics)
            model_result.test_metrics = dict(entry.test_metrics)
            model_result.n_patients_train = train.n_patients
            threshold = entry.threshold
        else:
            model_result = train_model(
                name,
                X_train,
                y_train,
                config,
                feature_names=feature_names,
                source=run_source,
                sanitize=False,
            )
            model_result.n_patients_train = train.n_patients

            train_predictions = predict_probabilities(
                model_result, X_train, threshold=0.5, already_sanitized=True
            )
            model_result.training_metrics = evaluate_predictions(
                y_train,
                train_predictions["predicted_probability"],
                threshold=0.5,
                model_name=name,
                split="train",
            )

            validation_predictions = predict_probabilities(
                model_result, X_validation, threshold=0.5, already_sanitized=True
            )
            threshold = select_validation_threshold(
                y_validation,
                validation_predictions["predicted_probability"],
                metric=config.threshold_selection_metric,
                grid_step=config.threshold_grid_step,
                partition="validation",
            )
            model_result.threshold = threshold
            model_result.validation_metrics = evaluate_predictions(
                y_validation,
                validation_predictions["predicted_probability"],
                threshold=threshold,
                model_name=name,
                split="validation",
            )

            if test is not None and len(test.X) > 0:
                X_test = sanitize_infinities(
                    validate_feature_frame(test.X, feature_names), inplace=False
                )
                test_predictions = predict_probabilities(
                    model_result, X_test, threshold=threshold, already_sanitized=True
                )
                model_result.test_metrics = evaluate_predictions(
                    test.y,
                    test_predictions["predicted_probability"],
                    threshold=threshold,
                    model_name=name,
                    split="test",
                )

            # Persist model artifacts and the checkpoint manifest atomically
            # only for models actually trained in this run.
            model_save_dirs = [d for d in {artifact_dir, checkpoint_dir} if d is not None]
            for save_dir in model_save_dirs:
                save_p2_model(
                    model_result,
                    Path(save_dir) / name,
                    overwrite=overwrite,
                )
            if checkpoint_path is not None and manifest is not None:
                record_model_complete(
                    manifest,
                    name,
                    Path(checkpoint_dir) / name / f"{name}_1.0.0.joblib",
                    threshold,
                    model_result.training_metrics,
                    model_result.validation_metrics,
                    model_result.test_metrics,
                )
                save_manifest(checkpoint_path, manifest)

        result.models[name] = model_result
        result.thresholds[name] = threshold
        result.training_metrics[name] = model_result.training_metrics
        result.validation_metrics[name] = model_result.validation_metrics
        if model_result.test_metrics:
            result.test_metrics[name] = model_result.test_metrics

    if artifact_dir is not None:
        config.save(Path(artifact_dir) / "p2_config.json", overwrite=overwrite)

    if reports_dir is not None:
        report_path = Path(reports_dir)
        thresholds_path = report_path / "p2_thresholds.json"
        if thresholds_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing thresholds: {thresholds_path}"
            )
        report_path.mkdir(parents=True, exist_ok=True)
        with open(thresholds_path, "w") as f:
            json.dump(
                {
                    "thresholds": dict(result.thresholds),
                    "selection_split": "validation",
                    "selection_metric": config.threshold_selection_metric,
                    "grid_step": config.threshold_grid_step,
                    "test_set_used_for_selection": False,
                    "source": run_source,
                },
                f,
                indent=2,
            )
        result.save(
            report_path / "p2_training_summary.json", overwrite=overwrite
        )
        save_experiment_record(
            result.to_dict(),
            report_path / "p2_experiment.json",
            overwrite=overwrite,
        )

    return result


def train_p2_from_artifacts(
    processed_dir: str | Path,
    split_manifest_path: str | Path,
    feature_schema_path: str | Path | None = None,
    config: P2Config | None = None,
    *,
    artifact_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    source: str = "real",
    overwrite: bool = False,
) -> P2TrainingResult:
    """Load P1 artefacts and run the P2 protocol."""
    dataset = load_from_manifest_and_features(
        manifest_path=split_manifest_path,
        processed_dir=processed_dir,
        schema_path=feature_schema_path,
        source=source,
    )
    return train_p2_models(
        dataset,
        config,
        artifact_dir=artifact_dir,
        reports_dir=reports_dir,
        source=source,
        overwrite=overwrite,
    )
