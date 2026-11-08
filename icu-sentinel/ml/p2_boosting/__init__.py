"""P2 advanced tabular ML package.

Provides Extra Trees, LightGBM, and XGBoost classifiers on top of the
verified P1 feature and split contracts, plus unified evaluation,
probability calibration, and lightweight experiment records.

All defaults are software defaults, not tuned scientific results.
"""

from .config import P2Config
from .evaluation import evaluate_predictions, select_validation_threshold
from .training import P2TrainingResult, train_p2_models

__all__ = [
    "P2Config",
    "P2TrainingResult",
    "evaluate_predictions",
    "select_validation_threshold",
    "train_p2_models",
]
