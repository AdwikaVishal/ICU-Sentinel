"""Configuration for P2 advanced tabular models.

Defaults are reproducible software defaults. They have not been tuned on
real data and must not be presented as an optimised research configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

RANDOM_SEED: int = 42


def cpu_count() -> int:
    """Return the logical CPU count, falling back to 1 if unavailable."""
    return os.cpu_count() or 1


def safe_thread_default() -> int:
    """Safe default thread count that leaves headroom for the OS on Windows.

    Capped at 12 so a single P2 run cannot starve the machine; leaves at
    least 2 logical cores free.
    """
    return min(12, max(1, cpu_count() - 2))


@dataclass
class ExtraTreesConfig:
    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | float | None = "sqrt"
    class_weight: str | None = "balanced"
    n_jobs: int = 1


@dataclass
class LightGBMConfig:
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    class_weight: str | None = "balanced"
    n_jobs: int = 1
    verbosity: int = -1


@dataclass
class XGBoostConfig:
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 4
    min_child_weight: float = 1.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    n_jobs: int = 1
    tree_method: str = "hist"


@dataclass
class P2Config:
    """Serializable configuration for the P2 model suite."""

    random_seed: int = RANDOM_SEED
    threshold_selection_metric: str = "f1"
    threshold_grid_step: float = 0.01
    calibration_method: str = "sigmoid"

    train_extra_trees: bool = True
    train_lightgbm: bool = True
    train_xgboost: bool = True

    extra_trees: ExtraTreesConfig = field(default_factory=ExtraTreesConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)

    def enabled_models(self) -> list[str]:
        models: list[str] = []
        if self.train_extra_trees:
            models.append("extra_trees")
        if self.train_lightgbm:
            models.append("lightgbm")
        if self.train_xgboost:
            models.append("xgboost")
        return models

    def to_dict(self) -> dict:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> P2Config:
        payload = dict(data)
        payload["extra_trees"] = ExtraTreesConfig(
            **payload.get("extra_trees", {})
        )
        payload["lightgbm"] = LightGBMConfig(**payload.get("lightgbm", {}))
        payload["xgboost"] = XGBoostConfig(**payload.get("xgboost", {}))
        return cls(**payload)

    def save(self, filepath: str | Path, overwrite: bool = False) -> Path:
        path = Path(filepath)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite config: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, filepath: str | Path) -> P2Config:
        path = Path(filepath)
        with open(path) as f:
            return cls.from_dict(json.load(f))


def config_fingerprint(config: P2Config) -> str:
    """Stable fingerprint of a P2Config for checkpoint/config-mismatch detection."""
    payload = json.dumps(config.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
