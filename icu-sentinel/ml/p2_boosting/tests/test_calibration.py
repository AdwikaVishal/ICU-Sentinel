"""Tests for P2 probability calibration foundation."""

from __future__ import annotations

import numpy as np
import pytest

from ml.p2_boosting.calibration import (
    ProbabilityCalibrator,
    apply_probability_calibrator,
    fit_probability_calibrator,
)


def _calibration_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    y = (rng.random(200) < 0.3).astype(int)
    probabilities = np.clip(0.3 * y + rng.random(200) * 0.7, 0, 1)
    return y, probabilities


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_fit_and_apply_calibrator(method):
    y, probabilities = _calibration_data()
    calibrator = fit_probability_calibrator(
        y, probabilities, method=method, partition="validation"
    )
    calibrated = apply_probability_calibrator(calibrator, probabilities)
    assert calibrated.shape == probabilities.shape
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)
    assert calibrator.partition == "validation"
    assert calibrator.n_samples == len(y)


def test_calibration_on_test_is_forbidden():
    y, probabilities = _calibration_data()
    with pytest.raises(ValueError, match="never be fitted on the test"):
        fit_probability_calibrator(y, probabilities, partition="test")


def test_calibration_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        fit_probability_calibrator(
            np.zeros(10, dtype=int),
            np.linspace(0.1, 0.9, 10),
        )


def test_calibration_rejects_invalid_probabilities():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fit_probability_calibrator(y, [0.1, 1.2, 0.3, 0.4])


def test_calibrator_serialization_roundtrip(tmp_path):
    y, probabilities = _calibration_data()
    calibrator = fit_probability_calibrator(y, probabilities, method="sigmoid")
    path = calibrator.save(tmp_path / "calibrator.joblib")
    loaded = ProbabilityCalibrator.load(path)
    np.testing.assert_allclose(
        apply_probability_calibrator(calibrator, probabilities),
        apply_probability_calibrator(loaded, probabilities),
    )


def test_calibrator_refuses_overwrite(tmp_path):
    y, probabilities = _calibration_data()
    calibrator = fit_probability_calibrator(y, probabilities)
    path = tmp_path / "calibrator.joblib"
    calibrator.save(path)
    with pytest.raises(FileExistsError):
        calibrator.save(path)
