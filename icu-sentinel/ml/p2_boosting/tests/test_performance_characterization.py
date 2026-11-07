"""Golden characterization tests for P2 training and prediction paths.

The fixtures are synthetic. These tests lock in current model outputs before
performance refactoring so thread or memory changes can be checked for
equivalence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ml.p2_boosting.base import predict_probabilities
from ml.p2_boosting.extra_trees import train_extra_trees
from ml.p2_boosting.tests.fixtures import fast_config, make_dataset, make_split
from ml.p2_boosting.training import train_p2_models
from ml.p2_boosting.xgboost_model import train_xgboost

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


def _single_thread_config():
    config = fast_config()
    config.extra_trees.n_jobs = 1
    config.xgboost.n_jobs = 1
    return config


def _assert_close(actual, expected):
    if expected is None:
        assert actual is None
        return
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_p2_training_golden_outputs():
    golden = _load_golden()["p2_training"]
    dataset = make_dataset()

    result = train_p2_models(
        dataset,
        _single_thread_config(),
        source="synthetic",
    )

    assert set(result.thresholds) == set(golden["thresholds"])
    for model_name, expected_threshold in golden["thresholds"].items():
        _assert_close(result.thresholds[model_name], expected_threshold)

    for model_name, expected_splits in golden["metrics"].items():
        model = result.models[model_name]
        actual_splits = {
            "train": model.training_metrics,
            "validation": model.validation_metrics,
            "test": model.test_metrics,
        }
        for split_name, expected in expected_splits.items():
            actual = actual_splits[split_name]
            assert actual["n_rows"] == expected["n_rows"]
            _assert_close(actual["threshold"], expected["threshold"])
            for metric_name in ("auroc", "auprc", "f1"):
                _assert_close(
                    actual["metrics"].get(metric_name),
                    expected[metric_name],
                )


def test_p2_prediction_golden_outputs():
    golden = _load_golden()["model_predictions"]
    split = make_split("train", n_rows=120, seed=42)
    X = split.X.copy()
    y = split.y.copy()
    y.iloc[::7] = 0
    y.iloc[0] = 1
    config = _single_thread_config()

    et_result = train_extra_trees(X, y, config, list(X.columns))
    xgb_result = train_xgboost(X, y, config, list(X.columns))

    for model_name, result in (
        ("extra_trees", et_result),
        ("xgboost", xgb_result),
    ):
        predictions = predict_probabilities(result, X, threshold=0.5)
        expected = golden[model_name]
        assert (
            _sha_array(
                predictions["predicted_probability"].to_numpy(
                    dtype=np.float64
                )
            )
            == expected["probability_sha256"]
        )
        assert (
            _sha_array(
                predictions["predicted_class"].to_numpy(dtype=np.int64)
            )
            == expected["class_sha256"]
        )
