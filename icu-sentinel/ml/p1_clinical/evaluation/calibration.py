"""Calibration metrics for P1 research evaluation.

Implements calibration curve computation and calibration error metrics
(Expected Calibration Error, Maximum Calibration Error).

Calibration assesses whether predicted probabilities match observed
frequencies. A perfectly calibrated model has predicted probability p
corresponding to an observed frequency of p.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .evaluation_schema import CalibrationEntry, CalibrationResult

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.calibration")


def compute_calibration_curve(
    y_true: np.ndarray | list[int] | pd.Series,
    y_prob: np.ndarray | list[float] | pd.Series,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> list[CalibrationEntry]:
    """Compute calibration curve bins.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int
        Number of calibration bins.
    strategy : str
        Binning strategy: "uniform" (equal-width bins) or "quantile"
        (equal-frequency bins).

    Returns
    -------
    list[CalibrationEntry]
        List of calibration bin entries.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    n_samples = len(y_true)
    if n_samples == 0:
        return []

    # Determine bin edges
    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.unique(
            np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        )
        if len(bin_edges) < 2:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    bins: list[CalibrationEntry] = []

    for i in range(len(bin_edges) - 1):
        bin_lower = float(bin_edges[i])
        bin_upper = float(bin_edges[i + 1])

        # Determine which samples fall in this bin
        if i < len(bin_edges) - 2:
            # Not the last bin: [lower, upper)
            mask = (y_prob >= bin_lower) & (y_prob < bin_upper)
        else:
            # Last bin: [lower, upper] (inclusive)
            mask = (y_prob >= bin_lower) & (y_prob <= bin_upper)

        n_samples_in_bin = int(mask.sum())
        if n_samples_in_bin == 0:
            # Empty bin
            bins.append(
                CalibrationEntry(
                    bin_index=i,
                    bin_lower=bin_lower,
                    bin_upper=bin_upper,
                    bin_center=float((bin_lower + bin_upper) / 2),
                    n_samples=0,
                    n_positive=0,
                    mean_predicted_probability=float("nan"),
                    observed_frequency=float("nan"),
                    calibration_error=float("nan"),
                )
            )
            continue

        bin_y_true = y_true[mask]
        bin_y_prob = y_prob[mask]

        n_positive = int(bin_y_true.sum())
        mean_pred_prob = float(bin_y_prob.mean())
        observed_freq = float(n_positive / n_samples_in_bin)
        calib_error = abs(mean_pred_prob - observed_freq)

        bins.append(
            CalibrationEntry(
                bin_index=i,
                bin_lower=bin_lower,
                bin_upper=bin_upper,
                bin_center=float((bin_lower + bin_upper) / 2),
                n_samples=n_samples_in_bin,
                n_positive=n_positive,
                mean_predicted_probability=mean_pred_prob,
                observed_frequency=observed_freq,
                calibration_error=calib_error,
            )
        )

    return bins


def compute_expected_calibration_error(
    bins: list[CalibrationEntry],
    n_samples: int,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE is the weighted average of |accuracy - confidence| across bins,
    weighted by the number of samples in each bin.

    Parameters
    ----------
    bins : list[CalibrationEntry]
        Calibration bin entries.
    n_samples : int
        Total number of samples.

    Returns
    -------
    float
        Expected Calibration Error.
    """
    if n_samples == 0:
        return float("nan")

    ece = 0.0
    for bin_entry in bins:
        if bin_entry.n_samples > 0 and not np.isnan(bin_entry.calibration_error):
            ece += (bin_entry.n_samples / n_samples) * bin_entry.calibration_error

    return float(ece)


def compute_maximum_calibration_error(
    bins: list[CalibrationEntry],
) -> float:
    """Compute Maximum Calibration Error (MCE).

    MCE is the maximum |accuracy - confidence| across all bins.

    Parameters
    ----------
    bins : list[CalibrationEntry]
        Calibration bin entries.

    Returns
    -------
    float
        Maximum Calibration Error.
    """
    errors = [
        b.calibration_error
        for b in bins
        if b.n_samples > 0 and not np.isnan(b.calibration_error)
    ]
    if not errors:
        return float("nan")
    return float(max(errors))


def compute_calibration(
    y_true: np.ndarray | list[int] | pd.Series,
    y_prob: np.ndarray | list[float] | pd.Series,
    model_name: str,
    model_version: str,
    split: str,
    n_bins: int = 10,
    strategy: str = "uniform",
    source: str = "synthetic",
) -> CalibrationResult:
    """Compute complete calibration result for one model.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    model_name : str
        Name of the model.
    model_version : str
        Version of the model.
    split : str
        Split identity.
    n_bins : int
        Number of calibration bins.
    strategy : str
        Binning strategy.
    source : str
        Data source label.

    Returns
    -------
    CalibrationResult
        Structured calibration result.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n_samples = len(y_true)

    bins = compute_calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    ece = compute_expected_calibration_error(bins, n_samples)
    mce = compute_maximum_calibration_error(bins)

    return CalibrationResult(
        model_name=model_name,
        model_version=model_version,
        split=split,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        bins=bins,
        source=source,
    )


def validate_calibration_result(
    result: CalibrationResult,
) -> dict:
    """Validate a CalibrationResult.

    Returns a dict with validation results.
    """
    issues: list[str] = []

    if result.expected_calibration_error is not None and not np.isfinite(
        result.expected_calibration_error
    ) and not np.isnan(result.expected_calibration_error):
        issues.append(
            f"ECE {result.expected_calibration_error} is not finite"
        )

    if result.max_calibration_error is not None and not np.isfinite(
        result.max_calibration_error
    ) and not np.isnan(result.max_calibration_error):
        issues.append(
            f"MCE {result.max_calibration_error} is not finite"
        )

    for bin_entry in result.bins:
        if bin_entry.n_samples < 0:
            issues.append(
                f"Bin {bin_entry.bin_index}: n_samples {bin_entry.n_samples} is negative"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_bins": len(result.bins),
    }