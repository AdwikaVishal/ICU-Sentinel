"""Shared estimator construction and fitting for P2 models."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .base import P2ModelResult
from .config import P2Config
from .feature_contract import (
    sanitize_infinities,
    validate_feature_frame,
    validate_feature_names,
    validate_training_target,
)

SUPPORTED_MODELS = ("extra_trees", "lightgbm", "xgboost")


def _scale_pos_weight(y: pd.Series) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("XGBoost requires both classes in the training target.")
    return float(negatives / positives)


def build_pipeline(
    model_name: str,
    config: P2Config,
    y_train: pd.Series,
) -> Pipeline:
    """Build a configured scikit-learn-compatible pipeline."""
    if model_name == "extra_trees":
        params = config.extra_trees
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=False),
                ),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=params.n_estimators,
                        max_depth=params.max_depth,
                        min_samples_split=params.min_samples_split,
                        min_samples_leaf=params.min_samples_leaf,
                        max_features=params.max_features,
                        class_weight=params.class_weight,
                        n_jobs=params.n_jobs,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )

    if model_name == "lightgbm":
        params = config.lightgbm
        return Pipeline(
            [
                (
                    "classifier",
                    LGBMClassifier(
                        n_estimators=params.n_estimators,
                        learning_rate=params.learning_rate,
                        num_leaves=params.num_leaves,
                        max_depth=params.max_depth,
                        min_child_samples=params.min_child_samples,
                        subsample=params.subsample,
                        colsample_bytree=params.colsample_bytree,
                        reg_alpha=params.reg_alpha,
                        reg_lambda=params.reg_lambda,
                        class_weight=params.class_weight,
                        n_jobs=params.n_jobs,
                        verbosity=params.verbosity,
                        random_state=config.random_seed,
                        deterministic=True,
                        force_col_wise=True,
                        subsample_freq=1,
                    ),
                )
            ]
        )

    if model_name == "xgboost":
        params = config.xgboost
        return Pipeline(
            [
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=params.n_estimators,
                        learning_rate=params.learning_rate,
                        max_depth=params.max_depth,
                        min_child_weight=params.min_child_weight,
                        subsample=params.subsample,
                        colsample_bytree=params.colsample_bytree,
                        reg_alpha=params.reg_alpha,
                        reg_lambda=params.reg_lambda,
                        scale_pos_weight=_scale_pos_weight(y_train),
                        n_jobs=params.n_jobs,
                        tree_method=params.tree_method,
                        random_state=config.random_seed,
                        objective="binary:logistic",
                        eval_metric="logloss",
                    ),
                )
            ]
        )

    raise ValueError(
        f"Unsupported model '{model_name}'; expected {SUPPORTED_MODELS}."
    )


def hyperparameters_for(model_name: str, config: P2Config) -> dict:
    if model_name == "extra_trees":
        return config.extra_trees.__dict__.copy()
    if model_name == "lightgbm":
        return config.lightgbm.__dict__.copy()
    if model_name == "xgboost":
        return config.xgboost.__dict__.copy()
    raise ValueError(f"Unsupported model '{model_name}'.")


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: P2Config,
    feature_names: list[str] | None = None,
    source: str = "synthetic",
    *,
    sanitize: bool = True,
) -> P2ModelResult:
    """Fit one P2 model on training data only.

    When ``sanitize=False`` the caller guarantees ``X_train`` has already been
    validated and infinities replaced with NaN exactly once; no additional
    copy or sanitisation pass is made. This avoids repeated full-frame copies
    when the same cleaned training frame is reused across models.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'; expected {SUPPORTED_MODELS}."
        )
    names = validate_feature_names(
        feature_names if feature_names is not None else list(X_train.columns)
    )
    # copy=False: we do not mutate X_train; the caller owns this frame.
    X = validate_feature_frame(X_train, names, copy=False)
    if sanitize:
        X = sanitize_infinities(X, inplace=False)
    y = validate_training_target(y_train, "training target")
    if len(X) != len(y):
        raise ValueError("X_train and y_train must have the same length.")

    pipeline = build_pipeline(model_name, config, y)
    pipeline.fit(X, y)

    preprocessing = (
        pipeline.named_steps["imputer"]
        if "imputer" in pipeline.named_steps
        else None
    )
    class_weight = getattr(
        pipeline.named_steps["classifier"], "class_weight", None
    )
    if isinstance(class_weight, dict):
        class_weight = "balanced"

    result = P2ModelResult(
        model_name=model_name,
        model=pipeline,
        preprocessing_pipeline=preprocessing,
        selected_features=names,
        hyperparameters=hyperparameters_for(model_name, config),
        random_seed=config.random_seed,
        training_timestamp=datetime.now(UTC).isoformat(),
        n_rows_train=len(X),
        n_positive_train=int((y == 1).sum()),
        n_negative_train=int((y == 0).sum()),
        class_weight=class_weight if isinstance(class_weight, str) else None,
        source=source,
    )
    return result
