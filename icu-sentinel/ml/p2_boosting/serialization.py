"""Artifact persistence for P2 models using the P1 serialization contract."""

from __future__ import annotations

from pathlib import Path

from ml.p1_clinical.classical_ml.model_registry import ModelMetadata
from ml.p1_clinical.classical_ml.model_serialization import (
    load_model_artifact,
    save_feature_list,
    save_model_artifact,
)

from .base import P2ModelResult


def _artifact_path(result: P2ModelResult, artifact_dir: str | Path) -> Path:
    return (
        Path(artifact_dir)
        / f"{result.model_name}_{result.model_version}.joblib"
    )


def save_p2_model(
    result: P2ModelResult,
    artifact_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a fitted P2 model without overwriting existing artifacts."""
    model_path = _artifact_path(result, artifact_dir)
    feature_path = (
        Path(artifact_dir)
        / f"{result.model_name}_{result.model_version}_features.json"
    )
    if model_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing model artifact: {model_path}"
        )
    if feature_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing feature list: {feature_path}"
        )
    if result.model is None:
        raise ValueError("Cannot save an unfitted P2 model.")

    metadata = ModelMetadata(
        model_name=result.model_name,
        model_version=result.model_version,
        training_timestamp=result.training_timestamp,
        random_seed=result.random_seed,
        feature_names=list(result.selected_features),
        selected_feature_names=list(result.selected_features),
        n_features_input=len(result.selected_features),
        n_features_output=len(result.selected_features),
        n_patients_train=result.n_patients_train,
        n_rows_train=result.n_rows_train,
        n_positive_train=result.n_positive_train,
        n_negative_train=result.n_negative_train,
        class_weight=result.class_weight,
        hyperparameters=dict(result.hyperparameters),
        preprocessing_config={
            "native_nan_handling": result.model_name != "extra_trees",
            "imputer": (
                "median" if result.model_name == "extra_trees" else None
            ),
        },
        validation_metrics=dict(result.validation_metrics),
        split_identity="patient_level_stratified",
        source=result.source,
    )
    save_model_artifact(
        model=result.model,
        preprocessing_pipeline=result.preprocessing_pipeline,
        metadata=metadata,
        artifact_dir=artifact_dir,
    )
    save_feature_list(
        result.selected_features,
        artifact_dir,
        result.model_name,
        result.model_version,
    )
    return model_path


def load_p2_model(model_path: str | Path) -> P2ModelResult:
    """Load a P2 model artifact back into a prediction-ready result."""
    artifact = load_model_artifact(model_path)
    metadata = artifact["metadata"]
    return P2ModelResult(
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
        model=artifact["model"],
        preprocessing_pipeline=artifact.get("preprocessing_pipeline"),
        selected_features=list(metadata["selected_feature_names"]),
        hyperparameters=dict(metadata.get("hyperparameters", {})),
        validation_metrics=dict(metadata.get("validation_metrics", {})),
        random_seed=int(metadata.get("random_seed", 42)),
        training_timestamp=metadata.get("training_timestamp", ""),
        n_rows_train=int(metadata.get("n_rows_train", 0)),
        n_positive_train=int(metadata.get("n_positive_train", 0)),
        n_negative_train=int(metadata.get("n_negative_train", 0)),
        class_weight=metadata.get("class_weight"),
        source=metadata.get("source", "synthetic"),
    )
