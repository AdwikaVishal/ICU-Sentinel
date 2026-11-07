"""Equivalence tests for the indexed patient-level bootstrap.

The reference function below is a direct copy of the pre-optimisation
sampling loop. It is intentionally slow and only used on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.p1_clinical.evaluation import bootstrap as bootstrap_module
from ml.p1_clinical.evaluation.bootstrap import (
    SUPPORTED_METRICS,
    compute_bootstrap_confidence_intervals,
)


def _make_overlapping_bootstrap_data() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    for i in range(40):
        patient_id = f"P{i:03d}"
        patient_effect = rng.normal()
        for hour in range(1, 5):
            latent = patient_effect + rng.normal(0, 1)
            label = int(latent > 0)
            probability = np.clip(
                0.5 + 0.15 * latent + rng.normal(0, 0.15),
                0.01,
                0.99,
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "hour": hour,
                    "true_label": label,
                    "predicted_probability": float(probability),
                    "predicted_class": int(probability >= 0.5),
                }
            )
    return pd.DataFrame(rows)


def _reference_bootstrap(
    predictions_df: pd.DataFrame,
    metric_names: list[str],
    n_iterations: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, tuple[float, float, float]]:
    metric_funcs = {
        "auroc": bootstrap_module._safe_roc_auc,
        "auprc": bootstrap_module._safe_auprc,
        "sensitivity": bootstrap_module._safe_sensitivity,
        "specificity": bootstrap_module._safe_specificity,
        "precision": bootstrap_module._safe_precision,
        "f1": bootstrap_module._safe_f1,
        "accuracy": bootstrap_module._safe_accuracy,
        "brier": bootstrap_module._safe_brier,
    }
    rng = np.random.default_rng(random_seed)
    patient_ids = predictions_df["patient_id"].unique()
    n_patients = len(patient_ids)
    samples: dict[str, list[float]] = {name: [] for name in metric_names}

    for _ in range(n_iterations):
        sampled_patient_ids = rng.choice(
            patient_ids, size=n_patients, replace=True
        )
        sampled_rows = [
            predictions_df[predictions_df["patient_id"] == pid]
            for pid in sampled_patient_ids
        ]
        bootstrap_df = pd.concat(sampled_rows, ignore_index=True)
        y_true = bootstrap_df["true_label"].astype(int).values
        y_pred = bootstrap_df["predicted_class"].astype(int).values
        y_prob = bootstrap_df["predicted_probability"].astype(float).values
        for metric_name in metric_names:
            value = metric_funcs[metric_name](y_true, y_pred, y_prob)
            if np.isfinite(value):
                samples[metric_name].append(value)

    alpha = 1.0 - confidence_level
    result = {}
    for metric_name in metric_names:
        if not samples[metric_name]:
            continue
        values = np.array(samples[metric_name])
        result[metric_name] = (
            float(np.mean(values)),
            float(np.percentile(values, (alpha / 2) * 100)),
            float(np.percentile(values, (1 - alpha / 2) * 100)),
        )
    return result


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_indexed_bootstrap_matches_reference_implementation(seed):
    predictions = _make_overlapping_bootstrap_data()
    metric_names = list(SUPPORTED_METRICS)

    actual = compute_bootstrap_confidence_intervals(
        predictions,
        "test_model",
        "1.0.0",
        "test",
        metric_names=metric_names,
        n_iterations=100,
        confidence_level=0.95,
        random_seed=seed,
        source="synthetic",
    )
    expected = _reference_bootstrap(
        predictions,
        metric_names=metric_names,
        n_iterations=100,
        confidence_level=0.95,
        random_seed=seed,
    )

    actual_by_name = {metric.metric_name: metric for metric in actual.metrics}
    assert set(actual_by_name) == set(expected)
    for metric_name, (point, lower, upper) in expected.items():
        metric = actual_by_name[metric_name]
        assert metric.point_estimate == point
        assert metric.ci_lower == lower
        assert metric.ci_upper == upper
        assert metric.n_iterations == 100
        assert metric.n_patients == 40


def test_indexed_bootstrap_preserves_interleaved_row_order():
    predictions = _make_overlapping_bootstrap_data()
    shuffled = predictions.sample(frac=1.0, random_state=7).reset_index(drop=True)

    actual = compute_bootstrap_confidence_intervals(
        shuffled,
        "test_model",
        "1.0.0",
        "test",
        metric_names=["auroc", "f1", "brier"],
        n_iterations=50,
        random_seed=11,
        source="synthetic",
    )
    expected = _reference_bootstrap(
        shuffled,
        metric_names=["auroc", "f1", "brier"],
        n_iterations=50,
        confidence_level=0.95,
        random_seed=11,
    )

    actual_by_name = {metric.metric_name: metric for metric in actual.metrics}
    for metric_name, (point, lower, upper) in expected.items():
        metric = actual_by_name[metric_name]
        assert metric.point_estimate == point
        assert metric.ci_lower == lower
        assert metric.ci_upper == upper
