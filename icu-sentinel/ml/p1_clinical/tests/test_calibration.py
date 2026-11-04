"""Tests for calibration.py.

Tests calibration curve computation, ECE, MCE, and calibration result
validation with synthetic data.
"""

from __future__ import annotations

import numpy as np

from ml.p1_clinical.evaluation.calibration import (
    compute_calibration,
    compute_calibration_curve,
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    validate_calibration_result,
)


def test_compute_calibration_curve_basic():
    """Test basic calibration curve computation."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    bins = compute_calibration_curve(y_true, y_prob, n_bins=4)

    assert len(bins) == 4
    for bin_entry in bins:
        assert bin_entry.n_samples > 0
        assert 0 <= bin_entry.bin_lower < bin_entry.bin_upper <= 1
        assert 0 <= bin_entry.bin_center <= 1


def test_compute_calibration_curve_perfect():
    """Test calibration with perfectly calibrated predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])

    bins = compute_calibration_curve(y_true, y_prob, n_bins=2)

    # Perfect calibration: predicted probability matches observed frequency
    for bin_entry in bins:
        if bin_entry.n_samples > 0:
            assert abs(bin_entry.mean_predicted_probability - bin_entry.observed_frequency) < 1e-10


def test_compute_calibration_curve_empty():
    """Test calibration with empty input."""
    y_true = np.array([], dtype=int)
    y_prob = np.array([], dtype=float)

    bins = compute_calibration_curve(y_true, y_prob, n_bins=5)
    assert len(bins) == 0


def test_compute_ece():
    """Test Expected Calibration Error computation."""
    bins = [
        # Perfectly calibrated bin
        type("Bin", (), {"n_samples": 50, "calibration_error": 0.0})(),
        # Biased bin
        type("Bin", (), {"n_samples": 50, "calibration_error": 0.2})(),
    ]

    ece = compute_expected_calibration_error(bins, 100)
    assert abs(ece - 0.1) < 1e-10  # (50/100)*0.0 + (50/100)*0.2 = 0.1


def test_compute_mce():
    """Test Maximum Calibration Error computation."""
    bins = [
        type("Bin", (), {"n_samples": 50, "calibration_error": 0.1})(),
        type("Bin", (), {"n_samples": 50, "calibration_error": 0.3})(),
        type("Bin", (), {"n_samples": 50, "calibration_error": 0.05})(),
    ]

    mce = compute_maximum_calibration_error(bins)
    assert abs(mce - 0.3) < 1e-10


def test_compute_calibration_full():
    """Test full calibration computation."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = (rng.random(n) < 0.3).astype(int)
    y_prob = y_true * 0.6 + rng.random(n) * 0.4
    y_prob = np.clip(y_prob, 0, 1)

    result = compute_calibration(
        y_true, y_prob, "test_model", "1.0.0", "test",
        n_bins=10, source="synthetic",
    )

    assert result.model_name == "test_model"
    assert result.model_version == "1.0.0"
    assert result.split == "test"
    assert result.source == "synthetic"
    assert len(result.bins) > 0
    assert np.isfinite(result.expected_calibration_error) or np.isnan(result.expected_calibration_error)
    assert np.isfinite(result.max_calibration_error) or np.isnan(result.max_calibration_error)


def test_validate_calibration_result_valid():
    """Test validation of valid calibration result."""
    rng = np.random.default_rng(42)
    y_true = (rng.random(100) < 0.3).astype(int)
    y_prob = np.clip(y_true * 0.7 + rng.random(100) * 0.3, 0, 1)

    result = compute_calibration(
        y_true, y_prob, "test_model", "1.0.0", "test",
        source="synthetic",
    )

    validation = validate_calibration_result(result)
    assert validation["valid"] is True
    assert len(validation["issues"]) == 0


def test_validate_calibration_result_invalid():
    """Test validation of invalid calibration result."""
    rng = np.random.default_rng(42)
    y_true = (rng.random(100) < 0.3).astype(int)
    y_prob = np.clip(y_true * 0.7 + rng.random(100) * 0.3, 0, 1)

    result = compute_calibration(
        y_true, y_prob, "test_model", "1.0.0", "test",
        source="synthetic",
    )

    # Corrupt ECE
    result.expected_calibration_error = float("inf")

    validation = validate_calibration_result(result)
    assert validation["valid"] is False
    assert len(validation["issues"]) > 0


def test_calibration_bin_empty():
    """Test calibration with empty bins."""
    y_true = np.array([0, 1])
    y_prob = np.array([0.1, 0.9])

    bins = compute_calibration_curve(y_true, y_prob, n_bins=10)

    # Many bins will be empty
    empty_bins = [b for b in bins if b.n_samples == 0]
    non_empty_bins = [b for b in bins if b.n_samples > 0]

    assert len(empty_bins) > 0
    assert len(non_empty_bins) == 2

    # Empty bins should have NaN values
    for b in empty_bins:
        assert np.isnan(b.mean_predicted_probability)
        assert np.isnan(b.observed_frequency)
        assert np.isnan(b.calibration_error)