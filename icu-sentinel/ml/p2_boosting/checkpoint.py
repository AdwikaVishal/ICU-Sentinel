"""Checkpoint / run-manifest support for P2 training.

A run manifest records the configuration fingerprint and per-model progress
so an interrupted run can be resumed safely. The manifest is written
atomically (temp file + os.replace) and verified before resume: artifacts
from a different configuration are never combined with a new run.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import P2Config, config_fingerprint


MANIFEST_FILENAME = "p2_run_manifest.json"


@dataclass
class ModelEntry:
    """Recorded result for a single completed model."""

    model_name: str
    status: str = "done"
    artifact_path: str | None = None
    threshold: float | None = None
    training_metrics: dict = field(default_factory=dict)
    validation_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    fingerprint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelEntry":
        return cls(**data)


@dataclass
class RunManifest:
    """Checkpoint state for one P2 training run."""

    config_fingerprint: str
    source: str
    target_column: str
    feature_names: list[str]
    n_features: int
    enabled_models: list[str]
    models: dict[str, ModelEntry] = field(default_factory=dict)
    completed: list[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "config_fingerprint": self.config_fingerprint,
            "source": self.source,
            "target_column": self.target_column,
            "feature_names": list(self.feature_names),
            "n_features": int(self.n_features),
            "enabled_models": list(self.enabled_models),
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "completed": list(self.completed),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunManifest":
        return cls(
            config_fingerprint=data["config_fingerprint"],
            source=data["source"],
            target_column=data["target_column"],
            feature_names=list(data["feature_names"]),
            n_features=int(data["n_features"]),
            enabled_models=list(data["enabled_models"]),
            models={
                k: ModelEntry.from_dict(v) for k, v in data.get("models", {}).items()
            },
            completed=list(data.get("completed", [])),
            updated_at=data.get("updated_at", ""),
        )


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def save_manifest(path: str | Path, manifest: RunManifest) -> Path:
    """Atomically write a run manifest (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return path


def load_manifest(path: str | Path) -> RunManifest | None:
    """Load a run manifest, or None if it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def manifest_matches_config(manifest: RunManifest, config: P2Config) -> bool:
    """Return True if the manifest was produced by the same configuration."""
    return manifest.config_fingerprint == config_fingerprint(config)


def manifest_for_run(
    config: P2Config,
    source: str,
    target_column: str,
    feature_names: list[str],
    enabled_models: list[str],
) -> RunManifest:
    """Create a fresh run manifest for a configuration."""
    return RunManifest(
        config_fingerprint=config_fingerprint(config),
        source=source,
        target_column=target_column,
        feature_names=list(feature_names),
        n_features=len(feature_names),
        enabled_models=list(enabled_models),
        updated_at=_now_iso(),
    )


def record_model_complete(
    manifest: RunManifest,
    model_name: str,
    artifact_path: str | Path | None,
    threshold: float | None,
    training_metrics: dict,
    validation_metrics: dict,
    test_metrics: dict,
) -> None:
    """Mark one model as complete in the manifest (in place)."""
    manifest.models[model_name] = ModelEntry(
        model_name=model_name,
        status="done",
        artifact_path=str(artifact_path) if artifact_path is not None else None,
        threshold=threshold,
        training_metrics=dict(training_metrics),
        validation_metrics=dict(validation_metrics),
        test_metrics=dict(test_metrics),
        fingerprint=manifest.config_fingerprint,
    )
    if model_name not in manifest.completed:
        manifest.completed.append(model_name)
    manifest.updated_at = _now_iso()
