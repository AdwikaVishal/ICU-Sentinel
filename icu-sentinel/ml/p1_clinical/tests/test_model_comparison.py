"""Tests for model_comparison.py.

Tests model comparison, pairwise comparison, and validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.evaluation.model_comparison import (
    _classify_model_type,
    compute_model_comparison,
    compute_pairwise_comparison,
    validate_model_comparison_result,
)


def test_classify_model_type():
    """Test model type classification."""
    assert _classify_model_type("logistic_regression") == "classical_ml"
    assert _classify_model_type("random_forest") == "classical_ml"
    assert _classify_model_type("svm") == "classical_ml"
    assert _classify_model_type("sirs") == "clinical_baseline"
    assert _classify_model_type("qsofa") == "clinical_baseline"
    assert _classify_model_type("news2_style") == "clinical_baseline"
    assert _classify_model_type("unknown_model") == "unknown"


def test_compute_model_comparison_basic():
    """Test basic model comparison."""
    models = {}
    for name, auroc in [("logistic_regression", 0.85),
                        ("random_forest", 0.88),
                        ("svm", 0.82),
                        ("sirs", 0.65),
                        ("qsofa", 0.58)]:
        rows = []
        for pid in ["P001", "P002"]:
            for hour in range(1, 6):
                label = int(hour >= 3)
                prob = 0.8 if (label == 1 and name in ["logistic_regression", "random_forest"]) else 0.1
                rows.append({"patient_id": pid, "hour": hour,
                             "true_label": label,
                             "predicted_probability": prob,
                             "predicted_class": int(prob >= 0.5)})
        models[name] = pd.DataFrame(rows)

    result = compute_model_comparison(
        models, "test", threshold=0.5, source="synthetic",
    )

    assert result.split == "test"
    assert result.source == "synthetic"
    assert len(result.models) == 5

    # Check models are sorted by AUROC descending
    aurocs = [e.metrics.get("auroc", float("-inf")) for e in result.models]
    assert all(aurocs[i] >= aurocs[i+1] for i in range(len(aurocs)-1))


def test_compute_model_comparison_validation():
    """Test validation of model comparison result."""
    models = {}
    for name in ["logistic_regression", "random_forest"]:
        rows = []
        for pid in ["P001"]:
            for hour in range(1, 6):
                label = int(hour >= 3)
                prob = 0.8 if label == 1 else 0.1
                rows.append({"patient_id": pid, "hour": hour,
                             "true_label": label,
                             "predicted_probability": prob,
                             "predicted_class": int(prob >= 0.5)})
        models[name] = pd.DataFrame(rows)

    result = compute_model_comparison(
        models, "test", threshold=0.5, source="synthetic",
    )

    validation = validate_model_comparison_result(result)
    assert validation["valid"] is True
    assert len(validation["issues"]) == 0
    assert validation["n_models"] == 2


def test_compute_pairwise_comparison():
    """Test pairwise comparison between two models."""
    from ml.p1_clinical.evaluation.evaluation_schema import ModelComparisonEntry

    entry_a = ModelComparisonEntry(
        model_name="model_a", model_version="1.0.0",
        model_type="classical_ml", split="test",
        metrics={"auroc": 0.8, "f1": 0.7},
        n_patients=10, n_rows=50, source="synthetic",
    )
    entry_b = ModelComparisonEntry(
        model_name="model_b", model_version="1.0.0",
        model_type="classical_ml", split="test",
        metrics={"auroc": 0.85, "f1": 0.75},
        n_patients=10, n_rows=50, source="synthetic",
    )

    comparison = compute_pairwise_comparison(entry_a, entry_b, metric="auroc")

    assert comparison["model_a"] == "model_a"
    assert comparison["model_b"] == "model_b"
    assert comparison["metric"] == "auroc"
    assert abs(comparison["value_a"] - 0.8) < 1e-10
    assert abs(comparison["value_b"] - 0.85) < 1e-10
    assert abs(comparison["difference"] - 0.05) < 1e-10
    assert comparison["model_b_better"] is True


def test_compute_pairwise_comparison_nan():
    """Test pairwise comparison with NaN values."""
    from ml.p1_clinical.evaluation.evaluation_schema import ModelComparisonEntry

    entry_a = ModelComparisonEntry(
        model_name="model_a", model_version="1.0.0",
        model_type="classical_ml", split="test",
        metrics={"auroc": float("nan")},
        n_patients=10, n_rows=50, source="synthetic",
    )
    entry_b = ModelComparisonEntry(
        model_name="model_b", model_version="1.0.0",
        model_type="classical_ml", split="test",
        metrics={"auroc": 0.8},
        n_patients=10, n_rows=50, source="synthetic",
    )

    comparison = compute_pairwise_comparison(entry_a, entry_b, metric="auroc")
    assert np.isnan(comparison["difference"])
    assert np.isnan(comparison["relative_change"])