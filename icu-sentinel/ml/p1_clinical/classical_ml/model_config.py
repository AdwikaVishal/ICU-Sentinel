"""Centralized configuration for P1 classical ML models.

All model hyperparameters, preprocessing settings, and reproducibility
configuration live here. Serializable to/from JSON for experiment
tracking and reproducibility.

Defaults are reasonable starting points for local execution and are NOT
tuned on any real dataset (real data is unavailable).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Random seed for reproducibility.
RANDOM_SEED: int = 42

# Logistic Regression defaults.
LR_C: float = 1.0
LR_PENALTY: str = "l2"
LR_SOLVER: str = "lbfgs"
LR_MAX_ITER: int = 1000
LR_CLASS_WEIGHT: str | None = "balanced"
LR_N_JOBS: int = 1

# Random Forest defaults.
RF_N_ESTIMATORS: int = 100
RF_MAX_DEPTH: int | None = 10
RF_MIN_SAMPLES_SPLIT: int = 2
RF_MIN_SAMPLES_LEAF: int = 1
RF_CLASS_WEIGHT: str | None = "balanced"
RF_N_JOBS: int = 1

# SVM defaults.
SVM_C: float = 1.0
SVM_KERNEL: str = "rbf"
SVM_GAMMA: str = "scale"
SVM_CLASS_WEIGHT: str | None = "balanced"
SVM_PROBABILITY: bool = True
SVM_MAX_ITER: int = -1

# Imputation defaults.
IMPUTE_STRATEGY: str = "median"
IMPUTE_ADD_INDICATOR: bool = False

# Scaling defaults.
SCALE_WITH_STD: bool = True
SCALE_WITH_MEAN: bool = True

# Feature selection defaults.
REMOVE_CONSTANT: bool = True
REMOVE_DUPLICATE: bool = True
REMOVE_ALL_MISSING: bool = True
REMOVE_INFINITE: bool = True

# Class imbalance strategy.
CLASS_IMBALANCE_STRATEGY: str = "class_weight"  # class_weight | none

# Threshold selection.
THRESHOLD_SELECTION_METRIC: str = "f1"  # f1 | precision | recall | balanced_accuracy
THRESHOLD_GRID_STEP: float = 0.01


@dataclass
class ModelConfig:
    """Serializable classical ML configuration."""

    # Reproducibility.
    random_seed: int = RANDOM_SEED

    # Logistic Regression.
    lr_c: float = LR_C
    lr_penalty: str = LR_PENALTY
    lr_solver: str = LR_SOLVER
    lr_max_iter: int = LR_MAX_ITER
    lr_class_weight: str | None = LR_CLASS_WEIGHT
    lr_n_jobs: int = LR_N_JOBS

    # Random Forest.
    rf_n_estimators: int = RF_N_ESTIMATORS
    rf_max_depth: int | None = RF_MAX_DEPTH
    rf_min_samples_split: int = RF_MIN_SAMPLES_SPLIT
    rf_min_samples_leaf: int = RF_MIN_SAMPLES_LEAF
    rf_class_weight: str | None = RF_CLASS_WEIGHT
    rf_n_jobs: int = RF_N_JOBS

    # SVM.
    svm_c: float = SVM_C
    svm_kernel: str = SVM_KERNEL
    svm_gamma: str = SVM_GAMMA
    svm_class_weight: str | None = SVM_CLASS_WEIGHT
    svm_probability: bool = SVM_PROBABILITY
    svm_max_iter: int = SVM_MAX_ITER

    # Preprocessing.
    impute_strategy: str = IMPUTE_STRATEGY
    impute_add_indicator: bool = IMPUTE_ADD_INDICATOR
    scale_with_std: bool = SCALE_WITH_STD
    scale_with_mean: bool = SCALE_WITH_MEAN

    # Feature selection.
    remove_constant: bool = REMOVE_CONSTANT
    remove_duplicate: bool = REMOVE_DUPLICATE
    remove_all_missing: bool = REMOVE_ALL_MISSING
    remove_infinite: bool = REMOVE_INFINITE

    # Class imbalance.
    class_imbalance_strategy: str = CLASS_IMBALANCE_STRATEGY

    # Threshold selection.
    threshold_selection_metric: str = THRESHOLD_SELECTION_METRIC
    threshold_grid_step: float = THRESHOLD_GRID_STEP

    # Model set to train.
    train_logistic_regression: bool = True
    train_random_forest: bool = True
    train_svm: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ModelConfig:
        return cls(**data)

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> ModelConfig:
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def enabled_models(self) -> list[str]:
        """Return list of model names enabled for training."""
        models: list[str] = []
        if self.train_logistic_regression:
            models.append("logistic_regression")
        if self.train_random_forest:
            models.append("random_forest")
        if self.train_svm:
            models.append("svm")
        return models
