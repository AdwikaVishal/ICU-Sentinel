"""Tests for P2 checkpoint / resume functionality.

Synthetic fixtures only. Verifies atomic manifest writes, fingerprint
verification before resume, skip of already-completed models, and refusal to
combine artifacts from different configurations.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.p1_clinical.preprocessing.targets import DERIVED_TARGET_COLUMN
from ml.p2_boosting.checkpoint import (
    MANIFEST_FILENAME,
    manifest_for_run,
    load_manifest,
    save_manifest,
)
from ml.p2_boosting.config import config_fingerprint
from ml.p2_boosting.tests.fixtures import fast_config, make_dataset
from ml.p2_boosting.training import train_p2_models


def _assert_close(actual, expected):
    if isinstance(expected, dict) and isinstance(actual, dict):
        assert set(actual) == set(expected), (set(actual), set(expected))
        for key in expected:
            _assert_close(actual[key], expected[key])
        return
    if isinstance(expected, (str, bool)) or isinstance(actual, (str, bool)):
        assert actual == expected, (actual, expected)
        return
    if expected is None:
        assert actual is None
        return
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def _run_checkpoint_train(tmp_path, config):
    dataset = make_dataset()
    ckpt_dir = tmp_path / "checkpoints"
    result = train_p2_models(
        dataset,
        config,
        checkpoint_dir=ckpt_dir,
        source="synthetic",
    )
    return result, ckpt_dir


def test_checkpoint_writes_manifest_and_artifacts(tmp_path):
    config = fast_config()
    config.extra_trees.n_jobs = 1
    config.xgboost.n_jobs = 1
    result, ckpt_dir = _run_checkpoint_train(tmp_path, config)

    manifest_path = ckpt_dir / MANIFEST_FILENAME
    assert manifest_path.exists()
    # No temp leftovers from atomic writes.
    assert not any(p.suffix == ".tmp" for p in ckpt_dir.rglob("*.tmp"))

    manifest = load_manifest(manifest_path)
    assert manifest is not None
    assert manifest.config_fingerprint == config_fingerprint(config)
    assert manifest.source == "synthetic"
    assert manifest.target_column == DERIVED_TARGET_COLUMN
    assert set(manifest.completed) == set(config.enabled_models())
    for name in config.enabled_models():
        assert manifest.models[name].status == "done"
        assert manifest.models[name].threshold == result.thresholds[name]
        artifact = ckpt_dir / name / f"{name}_1.0.0.joblib"
        assert artifact.exists()


def test_resume_is_equivalent_and_skips_retraining(tmp_path, monkeypatch):
    config = fast_config()
    config.extra_trees.n_jobs = 1
    config.xgboost.n_jobs = 1

    first, ckpt_dir = _run_checkpoint_train(tmp_path, config)

    called = {"train_model": 0}

    def boom_train_model(*args, **kwargs):
        called["train_model"] += 1
        raise AssertionError("train_model must not be called on resume")

    monkeypatch.setattr(
        "ml.p2_boosting.training.train_model", boom_train_model
    )

    second = train_p2_models(
        make_dataset(),
        config,
        checkpoint_dir=ckpt_dir,
        resume=True,
        source="synthetic",
    )
    assert called["train_model"] == 0

    # Equivalent thresholds and metrics.
    assert set(second.thresholds) == set(first.thresholds)
    for name in first.thresholds:
        _assert_close(second.thresholds[name], first.thresholds[name])
        for split_metrics in (
            "training_metrics",
            "validation_metrics",
            "test_metrics",
        ):
            first_split = getattr(first, split_metrics).get(name, {})
            second_split = getattr(second, split_metrics).get(name, {})
            assert set(second_split) == set(first_split)
            for key in first_split:
                _assert_close(second_split[key], first_split[key])


def test_resume_refuses_different_configuration(tmp_path):
    config_a = fast_config()
    config_a.extra_trees.n_jobs = 1
    config_a.xgboost.n_jobs = 1
    _run_checkpoint_train(tmp_path, config_a)

    config_b = fast_config()
    config_b.extra_trees.n_jobs = 1
    config_b.xgboost.n_jobs = 1
    # Different estimator count -> different fingerprint.
    config_b.extra_trees.n_estimators = 5

    with pytest.raises(
        ValueError, match="fingerprint mismatch|different configurations"
    ):
        train_p2_models(
            make_dataset(),
            config_b,
            checkpoint_dir=tmp_path / "checkpoints",
            resume=True,
            source="synthetic",
        )


def test_resume_without_manifest_raises(tmp_path):
    config = fast_config()
    config.extra_trees.n_jobs = 1
    config.xgboost.n_jobs = 1
    empty_ckpt = tmp_path / "empty_checkpoints"
    empty_ckpt.mkdir()
    with pytest.raises(FileNotFoundError, match="No checkpoint manifest"):
        train_p2_models(
            make_dataset(),
            config,
            checkpoint_dir=empty_ckpt,
            resume=True,
            source="synthetic",
        )


def test_save_and_load_manifest_roundtrip(tmp_path):
    config = fast_config()
    manifest = manifest_for_run(
        config,
        "synthetic",
        DERIVED_TARGET_COLUMN,
        ["f0", "f1"],
        ["extra_trees", "xgboost"],
    )
    path = tmp_path / "manifest.json"
    save_manifest(path, manifest)
    assert path.exists()
    loaded = load_manifest(path)
    assert loaded is not None
    assert loaded.config_fingerprint == config_fingerprint(config)
    assert loaded.completed == manifest.completed
