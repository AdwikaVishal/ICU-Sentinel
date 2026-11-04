"""Model serialization for P1 classical ML models.

Saves and loads trained model artifacts along with preprocessing
pipelines and metadata using joblib.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

from .model_registry import ModelMetadata

logger = logging.getLogger("icu_sentinel.p1_clinical.classical_ml.model_serialization")


def save_model_artifact(
    model: Any,
    preprocessing_pipeline: Any,
    metadata: ModelMetadata,
    artifact_dir: str | Path,
) -> Path:
    """Save a trained model, preprocessing pipeline, and metadata.

    Parameters
    ----------
    model : Any
        Fitted scikit-learn model (or pipeline).
    preprocessing_pipeline : Any
        Fitted preprocessing (imputer + scaler) or None.
    metadata : ModelMetadata
        Model metadata.
    artifact_dir : str | Path
        Directory to save the artifact.

    Returns
    -------
    Path
        Path to the saved model file.
    """
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    model_filename = f"{metadata.model_name}_{metadata.model_version}.joblib"
    model_path = artifact_path / model_filename

    artifact = {
        "model": model,
        "preprocessing_pipeline": preprocessing_pipeline,
        "metadata": metadata.to_dict(),
    }
    joblib.dump(artifact, model_path)
    metadata.artifact_path = str(model_path)
    metadata.save(artifact_path / f"{metadata.model_name}_{metadata.model_version}_metadata.json")

    logger.info("Saved model artifact to %s", model_path)
    return model_path


def load_model_artifact(
    model_path: str | Path,
) -> dict:
    """Load a saved model artifact.

    Parameters
    ----------
    model_path : str | Path
        Path to the .joblib artifact file.

    Returns
    -------
    dict
        Dictionary with keys: "model", "preprocessing_pipeline", "metadata".
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    artifact = joblib.load(path)
    logger.info("Loaded model artifact from %s", path)
    return artifact


def save_preprocessing_pipeline(
    pipeline: Any,
    artifact_dir: str | Path,
    model_name: str,
    model_version: str,
) -> Path:
    """Save a preprocessing pipeline separately."""
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    filename = f"{model_name}_{model_version}_preprocessing.joblib"
    path = artifact_path / filename
    joblib.dump(pipeline, path)
    logger.info("Saved preprocessing pipeline to %s", path)
    return path


def load_preprocessing_pipeline(
    path: str | Path,
) -> Any:
    """Load a preprocessing pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing pipeline not found: {path}")
    return joblib.load(path)


def save_feature_list(
    features: list[str],
    artifact_dir: str | Path,
    model_name: str,
    model_version: str,
) -> Path:
    """Save the selected feature list as JSON."""
    import json

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    filename = f"{model_name}_{model_version}_features.json"
    path = artifact_path / filename
    with open(path, "w") as f:
        json.dump({"selected_features": list(features)}, f, indent=2)
    logger.info("Saved feature list to %s", path)
    return path


def load_feature_list(
    path: str | Path,
) -> list[str]:
    """Load the selected feature list."""
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return list(data["selected_features"])
