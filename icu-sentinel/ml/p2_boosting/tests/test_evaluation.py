"""Tests for unified P2 evaluation and validation-only selection."""

from __future__ import annotations

import numpy as np
import pytest

from ml.p2_boosting.evaluation import (
    evaluate_predictions,
    select_validation_threshold,
)


def test_evaluation_output_schema_and_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    result = evaluate_predictions(
        y_true, y_prob, threshold=0.5, model_name="m", split="validation"
    )
    assert result["model_name"] == "m"
    assert result["split"] == "validation"
    assert result["threshold"] == 0.5
    assert result["n_rows"] == 4
    assert result["n_positive"] == 2
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    metrics = result["metrics"]
    for key in (
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "brier",
        "tp",
        "tn",
        "fp",
        "fn",
    ):
        assert key in metrics


def test_evaluation_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        evaluate_predictions([0, 1], [0.2], threshold=0.5)


def test_evaluation_rejects_non_finite_probability():
    with pytest.raises(ValueError, match="finite"):
        evaluate_predictions([0, 1], [0.2, np.inf], threshold=0.5)


def test_threshold_selection_uses_validation_split():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = select_validation_threshold(
        y_true, y_prob, metric="f1", grid_step=0.01, partition="validation"
    )
    assert 0.2 < threshold <= 0.8


def test_threshold_selection_on_test_is_forbidden():
    with pytest.raises(ValueError, match="test split is forbidden"):
        select_validation_threshold(
            [0, 1], [0.2, 0.8], partition="test"
        )


def test_threshold_selection_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        select_validation_threshold([0, 0], [0.1, 0.2])
