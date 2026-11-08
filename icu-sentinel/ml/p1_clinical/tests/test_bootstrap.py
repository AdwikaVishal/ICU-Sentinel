"""Tests for bootstrap.py.

Tests patient-level bootstrap confidence intervals with synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.p1_clinical.evaluation.bootstrap import (
    SUPPORTED_METRICS,
    compute_bootstrap_confidence_intervals,
    validate_bootstrap_result,
)


def _make_bootstrap_data(n_patients=50, n_hours=5, seed=42):
    """Generate synthetic patient-hour data for bootstrap testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        pid = f"P{i:03d}"
        base_prob = rng.random()
        for hour in range(1, n_hours + 1):
            label = int(base_prob > 0.5)
            prob = label * 0.7 + rng.random() * 0.3
            prob = np.clip(prob, 0, 1)
            rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label, "predicted_probability": prob,
                "predicted_class": int(prob >= 0.5),
            })
    return pd.DataFrame(rows)


def test_compute_bootstrap_confidence_intervals_basic():
    """Test basic bootstrap confidence interval computation."""
    df = _make_bootstrap_data(n_patients=100, n_hours=5, seed=42)

    result = compute_bootstrap_confidence_intervals(
        df, "test_model", "1.0.0", "test",
        metric_names=["auroc", "f1", "sensitivity"],
        n_iterations=100,
        confidence_level=0.95,
        random_seed=42,
        source="synthetic",
    )

    assert result.model_name == "test_model"
    assert result.model_version == "1.0.0"
    assert result.split == "test"
    assert result.source == "synthetic"
    assert len(result.metrics) > 0

    # Check that each metric has CI bounds
    for br in result.metrics:
        assert br.n_iterations == 100
        assert br.confidence_level == 0.95
        assert br.n_patients == 100
        if br.ci_lower is not None and br.ci_upper is not None:
            assert br.ci_lower <= br.point_estimate <= br.ci_upper or \
                   (np.isnan(br.ci_lower) and np.isnan(br.ci_upper))


def test_bootstrap_single_patient():
    """Test bootstrap with single patient."""
    rows = []
    for hour in range(1, 6):
        rows.append({"patient_id": "P001", "hour": hour,
                     "true_label": 0, "predicted_probability": 0.1,
                     "predicted_class": 0})

    df = pd.DataFrame(rows)
    result = compute_bootstrap_confidence_intervals(
        df, "test_model", "1.0.0", "test",
        metric_names=["auroc"],
        n_iterations=10,
        random_seed=42, source="synthetic",
    )

    # With single patient, bootstrap may produce no valid metrics
    # because all bootstrap samples have the same label
    if len(result.metrics) > 0:
        assert result.metrics[0].n_patients == 1


def test_validate_bootstrap_result_valid():
    """Test validation of valid bootstrap result."""
    rng = np.random.default_rng(42)
    rows = []
    for pid in [f"P{i:03d}" for i in range(10)]:
        for hour in range(1, 6):
            label = int(rng.random() < 0.3)
            prob = label * 0.7 + rng.random() * 0.3
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob,
                         "predicted_class": int(prob >= 0.5)})

    df = pd.DataFrame(rows)
    result = compute_bootstrap_confidence_intervals(
        df, "test_model", "1.0.0", "test",
        metric_names=["auroc", "f1"],
        n_iterations=20, random_seed=42, source="synthetic",
    )

    validation = validate_bootstrap_result(result)
    assert validation["valid"] is True
    assert len(validation["issues"]) == 0


def test_validate_bootstrap_result_invalid():
    """Test validation of invalid bootstrap result."""
    df = _make_bootstrap_data(n_patients=50, n_hours=5, seed=42)
    result = compute_bootstrap_confidence_intervals(
        df, "test_model", "1.0.0", "test",
        metric_names=["auroc"],
        n_iterations=100, random_seed=42, source="synthetic",
    )

    # Corrupt result
    result.metrics[0].ci_lower = 0.9
    result.metrics[0].ci_upper = 0.1  # ci_lower > ci_upper

    validation = validate_bootstrap_result(result)
    assert validation["valid"] is False
    assert len(validation["issues"]) > 0


def test_bootstrap_reproducibility():
    """Test that bootstrap is reproducible with same seed."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(50):
        pid = f"P{i:03d}"
        base_prob = rng.random()
        for hour in range(1, 6):
            label = int(base_prob > 0.5)
            prob = label * 0.7 + rng.random() * 0.3
            prob = np.clip(prob, 0, 1)
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob,
                         "predicted_class": int(prob >= 0.5)})

    df = pd.DataFrame(rows)

    result1 = compute_bootstrap_confidence_intervals(
        df, "test_model", "1.0.0", "test",
        metric_names=["auroc"], n_iterations=50,
        random_seed=42, source="synthetic",
    )
    result2 = compute_bootstrap_confidence_intervals(
        df, "test_model", "1.0.0", "test",
        metric_names=["auroc"], n_iterations=50,
        random_seed=42, source="synthetic",
    )

    # Same seed should give same results
    assert len(result1.metrics) > 0
    assert len(result2.metrics) > 0
    assert result1.metrics[0].point_estimate == result2.metrics[0].point_estimate
    assert result1.metrics[0].ci_lower == result2.metrics[0].ci_lower
    assert result1.metrics[0].ci_upper == result2.metrics[0].ci_upper


# ------------------------------------------------------------------
# Regression tests for metric argument handling and error behavior
# ------------------------------------------------------------------


def test_all_supported_metrics_are_computed():
    """Every declared bootstrap metric must be populated with valid data."""
    df = _make_bootstrap_data(n_patients=60, n_hours=5, seed=11)
    result = compute_bootstrap_confidence_intervals(
        df, "test_model", "1.0.0", "test",
        metric_names=list(SUPPORTED_METRICS),
        n_iterations=20, random_seed=42, source="synthetic",
    )

    computed = {br.metric_name for br in result.metrics}
    assert computed == set(SUPPORTED_METRICS)
    for br in result.metrics:
        assert np.isfinite(br.point_estimate)
        assert np.isfinite(br.ci_lower)
        assert np.isfinite(br.ci_upper)
        assert br.ci_lower <= br.ci_upper


def test_empty_input_returns_empty_summary():
    """An empty predictions frame yields an empty, well-formed summary."""
    df = pd.DataFrame(
        columns=[
            "patient_id",
            "true_label",
            "predicted_probability",
            "predicted_class",
        ]
    )
    result = compute_bootstrap_confidence_intervals(
        df, "test_model", "1.0.0", "test",
        metric_names=["auroc"], n_iterations=10,
        random_seed=42, source="synthetic",
    )
    assert result.metrics == []


def test_single_class_resamples_are_omitted_transparently():
    """AUROC/AUPRC/sensitivity are undefined for an all-negative cohort."""
    rows = []
    for i in range(20):
        pid = f"P{i:03d}"
        for hour in range(1, 4):
            rows.append({
                "patient_id": pid,
                "hour": hour,
                "true_label": 0,
                "predicted_probability": 0.2 if i % 2 else 0.8,
                "predicted_class": 0 if i % 2 else 1,
            })
    df = pd.DataFrame(rows)

    result = compute_bootstrap_confidence_intervals(
        df, "test_model", "1.0.0", "test",
        metric_names=["auroc", "auprc", "sensitivity", "specificity"],
        n_iterations=10, random_seed=42, source="synthetic",
    )

    names = {br.metric_name for br in result.metrics}
    assert "auroc" not in names
    assert "auprc" not in names
    assert "sensitivity" not in names
    assert "specificity" in names
    for br in result.metrics:
        assert np.isfinite(br.point_estimate)


@pytest.mark.parametrize(
    "missing_column",
    ["patient_id", "true_label", "predicted_probability", "predicted_class"],
)
def test_missing_required_columns_raise(missing_column):
    """Missing prediction columns must raise a clear ValueError."""
    df = _make_bootstrap_data(n_patients=10, n_hours=2, seed=1)
    df = df.drop(columns=[missing_column])

    with pytest.raises(ValueError, match="Missing required columns"):
        compute_bootstrap_confidence_intervals(
            df, "test_model", "1.0.0", "test",
            metric_names=["auroc"], n_iterations=2,
            random_seed=42, source="synthetic",
        )


def test_unsupported_metric_raises():
    """Unknown metric names must be rejected rather than silently ignored."""
    df = _make_bootstrap_data(n_patients=10, n_hours=2, seed=1)

    with pytest.raises(ValueError, match="Unsupported bootstrap metrics"):
        compute_bootstrap_confidence_intervals(
            df, "test_model", "1.0.0", "test",
            metric_names=["auroc", "not_a_metric"], n_iterations=2,
            random_seed=42, source="synthetic",
        )


def test_invalid_iteration_and_confidence_values_raise():
    df = _make_bootstrap_data(n_patients=10, n_hours=2, seed=1)

    with pytest.raises(ValueError, match="n_iterations must be positive"):
        compute_bootstrap_confidence_intervals(
            df, "test_model", "1.0.0", "test",
            metric_names=["auroc"], n_iterations=0,
            random_seed=42, source="synthetic",
        )

    with pytest.raises(ValueError, match="confidence_level must be in"):
        compute_bootstrap_confidence_intervals(
            df, "test_model", "1.0.0", "test",
            metric_names=["auroc"], n_iterations=2,
            confidence_level=1.5, random_seed=42, source="synthetic",
        )


def test_empty_metric_list_raises():
    df = _make_bootstrap_data(n_patients=10, n_hours=2, seed=1)
    with pytest.raises(ValueError, match="at least one metric"):
        compute_bootstrap_confidence_intervals(
            df, "test_model", "1.0.0", "test",
            metric_names=[], n_iterations=2,
            random_seed=42, source="synthetic",
        )


def test_unexpected_metric_error_is_not_swallowed(monkeypatch):
    """Programming errors must propagate, not be hidden by a broad except."""
    df = _make_bootstrap_data(n_patients=10, n_hours=2, seed=1)

    import ml.p1_clinical.evaluation.bootstrap as bootstrap_module

    def _boom(*_args, **_kwargs):
        raise RuntimeError("unexpected metric failure")

    monkeypatch.setattr(bootstrap_module, "_safe_roc_auc", _boom)

    with pytest.raises(RuntimeError, match="unexpected metric failure"):
        compute_bootstrap_confidence_intervals(
            df, "test_model", "1.0.0", "test",
            metric_names=["auroc"], n_iterations=2,
            random_seed=42, source="synthetic",
        )
