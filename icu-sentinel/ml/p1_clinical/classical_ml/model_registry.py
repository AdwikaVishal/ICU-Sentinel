"""Model registry for P1 classical ML models.

Tracks model artifacts, training metadata, and versioning for
reproducibility and P2 handoff.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("icu_sentinel.p1_clinical.classical_ml.model_registry")


@dataclass
class ModelMetadata:
    """Metadata for a trained model artifact."""

    model_name: str  # logistic_regression / random_forest / svm
    model_version: str
    training_timestamp: str  # ISO 8601
    random_seed: int
    feature_names: list[str]
    selected_feature_names: list[str]
    n_features_input: int
    n_features_output: int
    n_patients_train: int
    n_rows_train: int
    n_positive_train: int
    n_negative_train: int
    class_weight: str | None
    hyperparameters: dict
    preprocessing_config: dict
    validation_metrics: dict
    split_identity: str  # e.g. "patient_level_stratified"
    source: str = "synthetic"
    artifact_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "training_timestamp": self.training_timestamp,
            "random_seed": self.random_seed,
            "feature_names": list(self.feature_names),
            "selected_feature_names": list(self.selected_feature_names),
            "n_features_input": self.n_features_input,
            "n_features_output": self.n_features_output,
            "n_patients_train": self.n_patients_train,
            "n_rows_train": self.n_rows_train,
            "n_positive_train": self.n_positive_train,
            "n_negative_train": self.n_negative_train,
            "class_weight": self.class_weight,
            "hyperparameters": dict(self.hyperparameters),
            "preprocessing_config": dict(self.preprocessing_config),
            "validation_metrics": dict(self.validation_metrics),
            "split_identity": self.split_identity,
            "source": self.source,
            "artifact_path": self.artifact_path,
        }

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> ModelMetadata:
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ModelRegistryEntry:
    """A single entry in the model registry."""

    model_name: str
    model_version: str
    metadata: ModelMetadata
    artifact_path: Path

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "metadata": self.metadata.to_dict(),
            "artifact_path": str(self.artifact_path),
        }


class ModelRegistry:
    """Registry of trained model artifacts."""

    def __init__(self, registry_dir: str | Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, ModelRegistryEntry] = {}
        self.logger = logging.getLogger(
            "icu_sentinel.p1_clinical.classical_ml.model_registry.ModelRegistry"
        )

    def register(
        self,
        model_name: str,
        model_version: str,
        metadata: ModelMetadata,
        artifact_path: str | Path,
    ) -> None:
        """Register a trained model artifact."""
        key = f"{model_name}_{model_version}"
        self._entries[key] = ModelRegistryEntry(
            model_name=model_name,
            model_version=model_version,
            metadata=metadata,
            artifact_path=Path(artifact_path),
        )
        self.logger.info(
            "Registered model %s version %s at %s",
            model_name,
            model_version,
            artifact_path,
        )

    def get(self, model_name: str, model_version: str) -> ModelRegistryEntry | None:
        """Retrieve a registry entry."""
        key = f"{model_name}_{model_version}"
        return self._entries.get(key)

    def list_models(self) -> list[dict]:
        """List all registered models."""
        return [e.to_dict() for e in self._entries.values()]

    def save_registry(self, filepath: str | Path) -> None:
        """Save the registry to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "registry_dir": str(self.registry_dir),
            "models": self.list_models(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_registry(cls, filepath: str | Path) -> ModelRegistry:
        """Load a registry from a JSON file (metadata only)."""
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)
        registry = cls(registry_dir=data.get("registry_dir", "."))
        # Note: artifact files must exist on disk to be loadable via
        # model_serialization.
        return registry

    def latest_version(self, model_name: str) -> str | None:
        """Return the latest version string for a model name."""
        versions = [
            e.model_version
            for e in self._entries.values()
            if e.model_name == model_name
        ]
        if not versions:
            return None
        return max(versions)
