"""Model bundle and registry shared by P3 replay and explainability.

A bundle is the smallest consistent description of a fitted model:
model object, ordered feature list, optional SHAP background dataset,
and model identity metadata. Replay prediction providers and SHAP
explainers resolve the same bundle, which keeps the explained input
identical to the predicted input.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .prediction import ModelIdentity, PredictionSource


@dataclass
class ModelBundle:
    """A fitted model with its ordered feature contract."""

    identity: ModelIdentity
    model: Any
    selected_features: list[str]
    background: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_schema_version: str = "1.0.0"
    notes: str | None = None

    @property
    def name(self) -> str:
        return self.identity.name

    @property
    def version(self) -> str:
        return self.identity.version

    def public_metadata(self) -> dict[str, Any]:
        return {
            "model": self.identity.model_dump(mode="json"),
            "feature_schema_version": self.feature_schema_version,
            "n_features": len(self.selected_features),
            "background_rows": int(len(self.background)) if self.background is not None else 0,
            "notes": self.notes,
            **dict(self.metadata),
        }


class ModelRegistry:
    """Thread-safe in-memory registry of model bundles."""

    def __init__(self) -> None:
        self._bundles: dict[tuple[str, str], ModelBundle] = {}
        self._lock = threading.RLock()

    def register(self, bundle: ModelBundle) -> None:
        with self._lock:
            self._bundles[(bundle.name, bundle.version)] = bundle

    def get(self, model_name: str, model_version: str) -> ModelBundle | None:
        with self._lock:
            return self._bundles.get((model_name, model_version))

    def require(self, model_name: str, model_version: str) -> ModelBundle:
        bundle = self.get(model_name, model_version)
        if bundle is None:
            raise KeyError(f"Model not registered: {model_name} v{model_version}")
        return bundle

    def list_bundles(self) -> list[ModelBundle]:
        with self._lock:
            return sorted(self._bundles.values(), key=lambda b: (b.name, b.version))

    def list_public(self) -> list[dict[str, Any]]:
        return [bundle.public_metadata() for bundle in self.list_bundles()]

    def latest(self, model_name: str | None = None) -> ModelBundle | None:
        bundles = [
            bundle
            for bundle in self.list_bundles()
            if model_name is None or bundle.name == model_name
        ]
        return bundles[-1] if bundles else None


def load_p1_model_bundle(
    artifact_dir: str | Path,
    model_name: str,
    model_version: str = "1.0.0",
    *,
    feature_schema_version: str = "p1-1.0.0",
) -> ModelBundle:
    """Load a P1 classical model artifact into a P3 model bundle.

    A background dataset is optional. It is required for SHAP linear
    explainers; if no background file exists, patient-level explanations
    for that artifact return an explicit unavailable status rather than
    inventing values.
    """

    from ml.p1_clinical.classical_ml.model_serialization import (
        load_feature_list,
        load_model_artifact,
    )

    root = Path(artifact_dir)
    model_dir = root / model_name
    artifact_path = model_dir / f"{model_name}_{model_version}.joblib"
    feature_path = model_dir / f"{model_name}_{model_version}_features.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"P1 model artifact not found: {artifact_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"P1 feature list not found: {feature_path}")

    artifact = load_model_artifact(artifact_path)
    features = load_feature_list(feature_path)
    background = _load_optional_background(model_dir, features)

    metadata = dict(artifact.get("metadata") or {})
    source = PredictionSource.P1_CLASSICAL_MODEL
    if str(metadata.get("source", "")).lower() == "synthetic":
        # Training source does not change the model family, but it is
        # surfaced honestly in metadata/UI.
        metadata["training_source"] = "synthetic"
    else:
        metadata["training_source"] = metadata.get("source", "unknown")

    identity = ModelIdentity(
        name=model_name,
        version=model_version,
        source=source,
        display_name=f"P1 {model_name}",
        is_synthetic=bool(metadata.get("training_source") == "synthetic"),
        feature_schema_version=feature_schema_version,
    )
    return ModelBundle(
        identity=identity,
        model=artifact["model"],
        selected_features=list(features),
        background=background,
        metadata=metadata,
        feature_schema_version=feature_schema_version,
        notes=(
            "Loaded from P1 classical ML artifact. Explanation availability "
            "depends on the model family and on an optional background dataset."
        ),
    )


def _load_optional_background(
    model_dir: Path,
    features: list[str],
) -> pd.DataFrame | None:
    for filename in ("background.parquet", "background.csv"):
        path = model_dir / filename
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        missing = [feature for feature in features if feature not in frame.columns]
        if missing:
            raise ValueError(
                f"Background dataset {path} is missing {len(missing)} model features"
            )
        return frame[features].copy()
    return None
