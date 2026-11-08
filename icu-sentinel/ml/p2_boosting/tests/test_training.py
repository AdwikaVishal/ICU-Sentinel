"""Tests for the P2 training and split-evaluation protocol."""

from __future__ import annotations

import pytest

from ml.p1_clinical.classical_ml.data_loader import LoadedDataset
from ml.p2_boosting.config import P2Config
from ml.p2_boosting.tests.fixtures import fast_config, make_dataset
from ml.p2_boosting.training import train_p2_models


def test_train_all_models_keeps_splits_distinct():
    dataset = make_dataset()
    result = train_p2_models(dataset, fast_config(), source="synthetic")

    # LightGBM disabled by default due to Windows DLL issues
    assert set(result.models) == {"extra_trees", "xgboost"}
    for name in result.models:
        assert name in result.thresholds
        assert 0.0 <= result.thresholds[name] <= 1.0
        assert result.models[name].training_metrics["split"] == "train"
        assert (
            result.models[name].validation_metrics["split"] == "validation"
        )
        assert result.models[name].test_metrics["split"] == "test"
        assert result.models[name].training_metrics["threshold"] == 0.5
        assert (
            result.models[name].validation_metrics["threshold"]
            == result.thresholds[name]
        )
        assert (
            result.models[name].test_metrics["threshold"]
            == result.thresholds[name]
        )


def test_training_writes_reports_and_artifacts(tmp_path):
    dataset = make_dataset()
    artifact_dir = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports"
    result = train_p2_models(
        dataset,
        fast_config(),
        artifact_dir=artifact_dir,
        reports_dir=reports_dir,
        source="synthetic",
    )
    assert (artifact_dir / "p2_config.json").exists()
    for name in result.models:
        assert (artifact_dir / name / f"{name}_1.0.0.joblib").exists()
    assert (reports_dir / "p2_thresholds.json").exists()
    assert (reports_dir / "p2_training_summary.json").exists()
    assert (reports_dir / "p2_experiment.json").exists()


def test_training_requires_validation_split():
    dataset = make_dataset()
    dataset.validation = None
    with pytest.raises(ValueError, match="No validation data"):
        train_p2_models(dataset, fast_config())


def test_training_rejects_original_label_target():
    dataset = make_dataset()
    dataset.target_column = "SepsisLabel"
    with pytest.raises(ValueError, match="approved target"):
        train_p2_models(dataset, fast_config())


def test_training_rejects_split_leakage():
    dataset = make_dataset()
    dataset.validation.patient_ids = dataset.train.patient_ids.copy()
    with pytest.raises(ValueError, match="leakage"):
        train_p2_models(dataset, fast_config())


def test_training_rejects_no_enabled_models():
    dataset = make_dataset()
    config = P2Config(
        train_extra_trees=False,
        train_lightgbm=False,
        train_xgboost=False,
    )
    with pytest.raises(ValueError, match="No P2 models"):
        train_p2_models(dataset, config)


def test_training_result_serialization(tmp_path):
    dataset = make_dataset()
    result = train_p2_models(dataset, fast_config())
    path = result.save(tmp_path / "summary.json")
    assert path.exists()
    with pytest.raises(FileExistsError):
        result.save(tmp_path / "summary.json")


def test_training_without_test_split():
    dataset = make_dataset()
    dataset.test = None
    result = train_p2_models(dataset, fast_config())
    assert result.test_metrics == {}
    for model in result.models.values():
        assert model.test_metrics == {}
