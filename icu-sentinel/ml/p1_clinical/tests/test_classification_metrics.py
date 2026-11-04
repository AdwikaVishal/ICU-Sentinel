"""Tests for classification_metrics.py.

Tests metric calculations against known examples, single-class edge cases,
and missing prediction values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.evaluation.classification_metrics import (
    compute_all_metrics,
    compute_binary_metrics,
    metrics_result_from_dict,
    validate_metrics_result,
)


def _make_binary_data(n=100, seed=42, pos_rate=0.3):
    """Generate synthetic binary classification data."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < pos_rate).astype(int)
    # Generate probabilities correlated with truth
    y_prob = y_true * 0.7 + rng.random(n) * 0.3
    y_prob = np.clip(y_prob, 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


def test_compute_binary_metrics_basic():
    """Test basic metric computation with known data."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.1, 0.7])

    metrics = compute_binary_metrics(y_true, y_pred, y_prob)

    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "precision" in metrics
    assert "f1" in metrics
    assert "brier" in metrics
    assert "accuracy" in metrics
    assert "tp" in metrics
    assert "tn" in metrics
    assert "fp" in metrics
    assert "fn" in metrics

    # Check confusion matrix
    assert metrics["tp"] == 3  # indices 2, 3, 7
    assert metrics["tn"] == 4  # indices 0, 1, 4, 6
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1  # index 5

    # Sensitivity = 3/4 = 0.75
    assert abs(metrics["sensitivity"] - 0.75) < 1e-10
    # Specificity = 4/4 = 1.0
    assert abs(metrics["specificity"] - 1.0) < 1e-10


def test_compute_binary_metrics_single_class():
    """Test single-class edge case (all negative)."""
    y_true = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    y_prob = np.zeros(10, dtype=float)

    metrics = compute_binary_metrics(y_true, y_pred, y_prob)

    # Sensitivity undefined (no positives) -> NaN
    assert np.isnan(metrics["sensitivity"])
    # Precision undefined (no positive predictions) -> NaN
    assert np.isnan(metrics["precision"])
    # F1 undefined -> NaN
    assert np.isnan(metrics["f1"])
    # Specificity = 10/10 = 1.0
    assert abs(metrics["specificity"] - 1.0) < 1e-10
    # Accuracy = 1.0
    assert abs(metrics["accuracy"] - 1.0) < 1e-10


def test_compute_binary_metrics_all_positive():
    """Test all positive edge case."""
    y_true = np.ones(10, dtype=int)
    y_pred = np.ones(10, dtype=int)
    y_prob = np.ones(10, dtype=float)

    metrics = compute_binary_metrics(y_true, y_pred, y_prob)

    # Sensitivity = 10/10 = 1.0
    assert abs(metrics["sensitivity"] - 1.0) < 1e-10
    # Specificity undefined (no negatives) -> NaN
    assert np.isnan(metrics["specificity"])
    # Precision = 10/10 = 1.0
    assert abs(metrics["precision"] - 1.0) < 1e-10


def test_compute_binary_metrics_no_probabilities():
    """Test metrics without probabilities (threshold-dependent only)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    metrics = compute_binary_metrics(y_true, y_pred, None)

    # AUROC, AUPRC, Brier should be NaN (no probabilities)
    assert np.isnan(metrics["auroc"])
    assert np.isnan(metrics["auprc"])
    assert np.isnan(metrics["brier"])
    # Threshold-dependent metrics should work
    assert abs(metrics["sensitivity"] - 1.0) < 1e-10
    assert abs(metrics["specificity"] - 1.0) < 1e-10


def test_compute_all_metrics_structured():
    """Test compute_all_metrics returns structured result."""
    y_true, y_pred, y_prob = _make_binary_data(n=100, seed=42)

    result = compute_all_metrics(
        y_true, y_pred, y_prob,
        model_name="test_model",
        model_version="1.0.0",
        split="test",
        source="synthetic",
    )

    assert result.model_name == "test_model"
    assert result.model_version == "1.0.0"
    assert result.split == "test"
    assert result.source == "synthetic"
    assert len(result.metrics) > 0

    # Check AUROC metric exists
    auroc_entry = [m for m in result.metrics if m.metric_name == "auroc"]
    assert len(auroc_entry) == 1
    assert 0 <= auroc_entry[0].value <= 1 or np.isnan(auroc_entry[0].value)


def test_metrics_result_from_dict():
    """Test metrics_result_from_dict helper."""
    metrics_dict = {
        "auroc": 0.85,
        "auprc": 0.60,
        "sensitivity": 0.75,
        "specificity": 0.80,
        "tp": 50, "tn": 40, "fp": 5, "fn": 5,
    }

    result = metrics_result_from_dict(
        "test_model", "1.0.0", "test", metrics_dict, source="synthetic"
    )

    assert result.model_name == "test_model"
    assert result.split == "test"
    assert result.source == "synthetic"

    auroc_entry = [m for m in result.metrics if m.metric_name == "auroc"]
    assert len(auroc_entry) == 1
    assert abs(auroc_entry[0].value - 0.85) < 1e-10


def test_validate_metrics_result_valid():
    """Test validation of valid metrics result."""
    y_true, y_pred, y_prob = _make_binary_data(n=50, seed=42)
    result = compute_all_metrics(
        y_true, y_pred, y_prob,
        "test_model", "1.0.0", "test", source="synthetic",
    )

    validation = validate_metrics_result(result)
    assert validation["valid"] is True
    assert len(validation["issues"]) == 0


def test_validate_metrics_result_invalid():
    """Test validation of invalid metrics result."""
    y_true, y_pred, y_prob = _make_binary_data(n=50, seed=42)
    result = compute_all_metrics(
        y_true, y_pred, y_prob,
        "test_model", "1.0.0", "test", source="synthetic",
    )

    # Corrupt a metric value
    result.metrics[0].value = float("inf")

    validation = validate_metrics_result(result)
    assert validation["valid"] is False
    assert len(validation["issues"]) > 0


def test_compute_binary_metrics_pandas_input():
    """Test with pandas Series input."""
    y_true = pd.Series([0, 0, 1, 1, 0])
    y_pred = pd.Series([0, 0, 1, 1, 0])
    y_prob = pd.Series([0.1, 0.2, 0.8, 0.9, 0.3])

    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    assert "auroc" in metrics
    assert "f1" in metrics
    assert abs(metrics["accuracy"] - 1.0) < 1e-10  # all correct


def test_prevalence_calculation():
    """Test prevalence calculation."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1])

    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    # 3 positives out of 8 = 0.375
    assert abs(metrics["prevalence"] - 0.375) < 1e-10