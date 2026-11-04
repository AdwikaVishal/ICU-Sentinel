"""Feature ablation framework for P1 research evaluation.

Implements feature ablation by removing feature groups one at a time,
retraining models using training data only, and evaluating on the same
frozen test population.

Feature groups are defined by the Part 4 feature schema categories:
- current: Current observation features
- missingness: Missingness indicators, time-since-last-measurement, availability
- forward_fill: Causal forward-filled features
- rolling: Rolling 6/12/24-hour statistics
- trend: Temporal slopes and changes
- clinical: Derived clinical features (shock index, SIRS count)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..features.feature_schema import (
    CATEGORY_CLINICAL,
    CATEGORY_CURRENT,
    CATEGORY_FORWARD_FILL,
    CATEGORY_MISSINGNESS,
    CATEGORY_ROLLING,
    CATEGORY_TREND,
)
from .classification_metrics import compute_binary_metrics
from .evaluation_schema import FeatureAblationResult, FeatureAblationSummary

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.feature_ablation")


def get_features_by_category(
    feature_schema,
    category: str,
) -> list[str]:
    """Get feature names for a given category from the feature schema.

    Parameters
    ----------
    feature_schema : FeatureSchema
        The feature schema object.
    category : str
        Feature category name.

    Returns
    -------
    list[str]
        List of feature names in the category.
    """
    features = feature_schema.by_category(category)
    return [f.feature_name for f in features]


def compute_feature_ablation(
    model_train_func,
    model_predict_func,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_schema,
    config,
    threshold: float = 0.5,
    source: str = "synthetic",
) -> FeatureAblationSummary:
    """Compute feature ablation results for all feature groups.

    Parameters
    ----------
    model_train_func : callable
        Function that trains a model: train_func(X, y, config) -> model_result.
    model_predict_func : callable
        Function that predicts: predict_func(model_result, X, threshold) -> pd.DataFrame with 'predicted_probability' and 'predicted_class'.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    feature_schema : FeatureSchema
        Feature schema with category definitions.
    config : EvaluationConfig
        Evaluation configuration.
    threshold : float
        Decision threshold.
    source : str
        Data source label.

    Returns
    -------
    FeatureAblationSummary
        Complete feature ablation summary.
    """
    # Get all feature categories from schema
    all_categories = [
        CATEGORY_CURRENT,
        CATEGORY_MISSINGNESS,
        CATEGORY_FORWARD_FILL,
        CATEGORY_ROLLING,
        CATEGORY_TREND,
        CATEGORY_CLINICAL,
    ]

    # Filter to only categories that have features in the schema
    available_categories = []
    for cat in all_categories:
        feats = get_features_by_category(feature_schema, cat)
        if feats:
            available_categories.append(cat)

    # Get full feature set
    all_features = list(X_train.columns)

    # Train and evaluate full model
    logger.info("Training full model with %d features", len(all_features))
    full_result = model_train_func(X_train, y_train, config)
    full_pred = model_predict_func(full_result, X_test, threshold)
    full_metrics = compute_binary_metrics(
        y_test, full_pred["predicted_class"], full_pred["predicted_probability"]
    )

    # Perform ablation for each feature group
    ablations: list[FeatureAblationResult] = []

    for category in available_categories:
        features_to_remove = get_features_by_category(feature_schema, category)
        features_to_keep = [f for f in all_features if f not in features_to_remove]

        if not features_to_keep:
            logger.warning(
                "No features remaining after removing category %s", category
            )
            continue

        X_train_ablated = X_train[features_to_keep].copy()
        X_test_ablated = X_test[features_to_keep].copy()

        # Train ablated model
        ablated_result = model_train_func(X_train_ablated, y_train, config)
        ablated_pred = model_predict_func(
            ablated_result, X_test_ablated, threshold
        )
        ablated_metrics = compute_binary_metrics(
            y_test, ablated_pred["predicted_class"], ablated_pred["predicted_probability"]
        )

        # Compute metric changes
        metric_changes: dict[str, float] = {}
        metric_changes_relative: dict[str, float] = {}

        for metric_name in full_metrics:
            if metric_name in (
                "tp", "tn", "fp", "fn",
                "n_positive_pred", "n_negative_pred",
                "n_positive_true", "n_negative_true",
            ):
                continue  # Skip confusion matrix counts

            full_val = full_metrics.get(metric_name, float("nan"))
            ablated_val = ablated_metrics.get(metric_name, float("nan"))

            if np.isfinite(full_val) and np.isfinite(ablated_val):
                metric_changes[metric_name] = ablated_val - full_val
                if abs(full_val) > 1e-10:
                    metric_changes_relative[metric_name] = (
                        (ablated_val - full_val) / abs(full_val)
                    )
                else:
                    metric_changes_relative[metric_name] = float("nan")
            else:
                metric_changes[metric_name] = float("nan")
                metric_changes_relative[metric_name] = float("nan")

        ablations.append(
            FeatureAblationResult(
                feature_group=category,
                n_features_removed=len(features_to_remove),
                metrics_full={k: float(v) for k, v in full_metrics.items()
                             if k not in ("tp", "tn", "fp", "fn",
                                          "n_positive_pred", "n_negative_pred",
                                          "n_positive_true", "n_negative_true")},
                metrics_ablated={k: float(v) for k, v in ablated_metrics.items()
                                 if k not in ("tp", "tn", "fp", "fn",
                                              "n_positive_pred", "n_negative_pred",
                                              "n_positive_true", "n_negative_true")},
                metric_changes=metric_changes,
                metric_changes_relative=metric_changes_relative,
                source=source,
            )
        )

    return FeatureAblationSummary(
        model_name=getattr(full_result, "model_name", "unknown"),
        model_version=getattr(full_result, "model_version", "1.0.0"),
        split="test",
        n_features_full=len(all_features),
        ablations=ablations,
        source=source,
    )


def validate_feature_ablation_result(
    result: FeatureAblationSummary,
) -> dict:
    """Validate a FeatureAblationSummary.

    Returns a dict with validation results.
    """
    issues: list[str] = []

    if result.n_features_full <= 0:
        issues.append("n_features_full is non-positive")

    total_removed = sum(a.n_features_removed for a in result.ablations)
    if total_removed > result.n_features_full:
        issues.append("Total removed features exceeds n_features_full")

    for ablation in result.ablations:
        if ablation.n_features_removed <= 0:
            issues.append(
                f"Ablation {ablation.feature_group}: n_features_removed is non-positive"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_ablations": len(result.ablations),
    }