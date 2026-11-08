"""LightGBM classifier for P2."""

from __future__ import annotations

import pandas as pd

from .base import P2ModelResult, predict_probabilities
from .config import P2Config
from .pipelines import build_pipeline, train_model


def build_lightgbm_pipeline(config: P2Config, y_train: pd.Series):
    return build_pipeline("lightgbm", config, y_train)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: P2Config,
    feature_names: list[str] | None = None,
    source: str = "synthetic",
) -> P2ModelResult:
    return train_model(
        "lightgbm",
        X_train,
        y_train,
        config,
        feature_names=feature_names,
        source=source,
    )


def predict_lightgbm(
    result: P2ModelResult,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    return predict_probabilities(result, X, threshold=threshold)
