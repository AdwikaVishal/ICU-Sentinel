"""Probability calibration foundation for P2.

Calibrators are fitted from probability/label pairs on an explicit
validation or dedicated calibration partition. Fitting on the test split is
rejected. This module does not automatically calibrate test predictions;
callers must choose when to apply the frozen calibrator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .feature_contract import validate_prediction_target

SUPPORTED_CALIBRATION_METHODS = ("sigmoid", "isotonic")
_EPS = 1e-6


@dataclass
class ProbabilityCalibrator:
    """A fitted probability calibrator and its provenance."""

    method: str
    partition: str
    n_samples: int
    calibrator: Any

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "partition": self.partition,
            "n_samples": self.n_samples,
        }

    def save(self, filepath: str | Path, overwrite: bool = False) -> Path:
        path = Path(filepath)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite calibrator: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "method": self.method,
                "partition": self.partition,
                "n_samples": self.n_samples,
                "calibrator": self.calibrator,
            },
            path,
        )
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite calibrator metadata: {metadata_path}"
            )
        with open(metadata_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, filepath: str | Path) -> ProbabilityCalibrator:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator not found: {path}")
        payload = joblib.load(path)
        return cls(
            method=payload["method"],
            partition=payload["partition"],
            n_samples=int(payload["n_samples"]),
            calibrator=payload["calibrator"],
        )


def _validate_probabilities(y_prob) -> np.ndarray:
    probabilities = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(probabilities) == 0:
        raise ValueError("Probabilities are empty.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probabilities must lie in [0, 1].")
    return probabilities


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_probability_calibrator(
    y_true,
    y_prob,
    *,
    method: str = "sigmoid",
    partition: str = "validation",
) -> ProbabilityCalibrator:
    """Fit Platt (sigmoid) or isotonic calibration.

    Only validation or a dedicated calibration partition is accepted.
    """
    if partition == "test":
        raise ValueError("Calibration must never be fitted on the test split.")
    if partition not in {"validation", "calibration"}:
        raise ValueError(
            "partition must be 'validation' or 'calibration'; "
            f"got '{partition}'."
        )
    if method not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(
            f"Unsupported calibration method '{method}'; expected "
            f"{SUPPORTED_CALIBRATION_METHODS}."
        )

    y = validate_prediction_target(y_true, "calibration target")
    probabilities = _validate_probabilities(y_prob)
    if len(y) != len(probabilities):
        raise ValueError("y_true and y_prob must have the same length.")
    if len(y) < 3:
        raise ValueError("Calibration requires at least three samples.")

    if method == "sigmoid":
        calibrator = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        calibrator.fit(_logit(probabilities), y)
    else:
        calibrator = IsotonicRegression(
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0,
        )
        calibrator.fit(probabilities, y)

    return ProbabilityCalibrator(
        method=method,
        partition=partition,
        n_samples=len(y),
        calibrator=calibrator,
    )


def apply_probability_calibrator(
    calibrator: ProbabilityCalibrator,
    y_prob,
) -> np.ndarray:
    """Apply a frozen calibrator to probabilities."""
    probabilities = _validate_probabilities(y_prob)
    if calibrator.method == "sigmoid":
        calibrated = calibrator.calibrator.predict_proba(
            _logit(probabilities)
        )[:, 1]
    elif calibrator.method == "isotonic":
        calibrated = calibrator.calibrator.predict(probabilities)
    else:
        raise ValueError(
            f"Unsupported calibration method '{calibrator.method}'."
        )

    calibrated = np.asarray(calibrated, dtype=float).reshape(-1)
    if len(calibrated) != len(probabilities):
        raise ValueError("Calibrator returned an unexpected number of rows.")
    if not np.all(np.isfinite(calibrated)):
        raise ValueError("Calibrator returned non-finite probabilities.")
    return np.clip(calibrated, 0.0, 1.0)
