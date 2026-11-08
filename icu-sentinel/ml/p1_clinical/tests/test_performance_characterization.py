"""Golden characterization tests for P1 bootstrap and data loading.

These tests lock in the behaviour of the existing synthetic-fixture paths
before performance refactoring. No real patient data is used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.p1_clinical.classical_ml.data_loader import load_split_aware_dataset
from ml.p1_clinical.evaluation.bootstrap import (
    compute_bootstrap_confidence_intervals,
)
from ml.p1_clinical.tests.test_training import (
    _make_small_manifest,
    _write_synthetic_patients,
)

GOLDEN_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "golden"
    / "performance_characterization.json"
)


def _load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _sha_array(values) -> str:
    arr = np.ascontiguousarray(values)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sha_series(values) -> str:
    joined = "|".join(str(value) for value in values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


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


def test_loader_golden_signature(tmp_path):
    golden = _load_golden()["loader"]
    manifest = _make_small_manifest(tmp_path)
    processed_dir = _write_synthetic_patients(tmp_path, manifest)

    dataset = load_split_aware_dataset(
        processed_dir,
        manifest,
        source="synthetic",
    )

    assert list(dataset.feature_names) == golden["feature_names"]
    for split_name, split_data in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        expected = golden["splits"][split_name]
        assert len(split_data.X) == expected["n_rows"]
        assert split_data.patient_ids.nunique() == expected["n_patients"]
        assert int((split_data.y == 1).sum()) == expected["n_positive"]
        assert int((split_data.y == 0).sum()) == expected["n_negative"]
        assert _sha_array(split_data.X.to_numpy()) == expected["x_sha256"]
        assert _sha_array(split_data.y.to_numpy()) == expected["y_sha256"]
        assert (
            _sha_series(split_data.patient_ids.tolist())
            == expected["patient_id_sha256"]
        )
        assert (
            _sha_series(split_data.hours.tolist())
            == expected["hour_sha256"]
        )


def test_bootstrap_golden_values():
    golden = _load_golden()["bootstrap"]
    predictions = _make_overlapping_bootstrap_data()

    result = compute_bootstrap_confidence_intervals(
        predictions,
        "test_model",
        "1.0.0",
        "test",
        metric_names=list(golden),
        n_iterations=100,
        confidence_level=0.95,
        random_seed=42,
        source="synthetic",
    )

    actual = {metric.metric_name: metric for metric in result.metrics}
    assert set(actual) == set(golden)
    for metric_name, expected in golden.items():
        metric = actual[metric_name]
        assert metric.n_iterations == expected["n_iterations"]
        assert metric.n_patients == expected["n_patients"]
        np.testing.assert_allclose(
            metric.point_estimate,
            expected["point_estimate"],
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            metric.ci_lower,
            expected["ci_lower"],
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            metric.ci_upper,
            expected["ci_upper"],
            rtol=0,
            atol=1e-12,
        )


def test_golden_fixture_has_no_real_patient_data():
    golden = _load_golden()["loader"]
    for split in golden["splits"].values():
        assert split["n_patients"] <= 3


def test_golden_file_is_valid_json():
    assert isinstance(_load_golden(), dict)
