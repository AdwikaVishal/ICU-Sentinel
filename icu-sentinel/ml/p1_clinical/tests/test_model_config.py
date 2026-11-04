"""Tests for model_config.py.

Tests configuration serialization, defaults, and enabled models.
"""

from __future__ import annotations

import json

from ml.p1_clinical.classical_ml.model_config import ModelConfig

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------


def test_default_config_values():
    config = ModelConfig()
    assert config.random_seed == 42
    assert config.lr_c == 1.0
    assert config.lr_penalty == "l2"
    assert config.lr_solver == "lbfgs"
    assert config.lr_max_iter == 1000
    assert config.lr_class_weight == "balanced"
    assert config.rf_n_estimators == 100
    assert config.rf_max_depth == 10
    assert config.svm_c == 1.0
    assert config.svm_kernel == "rbf"
    assert config.svm_probability is True
    assert config.impute_strategy == "median"
    assert config.class_imbalance_strategy == "class_weight"
    assert config.threshold_selection_metric == "f1"
    assert config.threshold_grid_step == 0.01


# ------------------------------------------------------------------
# Enabled models
# ------------------------------------------------------------------


def test_enabled_models_all():
    config = ModelConfig()
    models = config.enabled_models()
    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "svm" in models


def test_enabled_models_subset():
    config = ModelConfig(
        train_logistic_regression=True,
        train_random_forest=False,
        train_svm=False,
    )
    models = config.enabled_models()
    assert models == ["logistic_regression"]


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------


def test_config_save_load(tmp_path):
    config = ModelConfig(lr_c=0.5, rf_n_estimators=200, random_seed=123)
    path = tmp_path / "model_config.json"
    config.save(path)
    loaded = ModelConfig.load(path)
    assert loaded.lr_c == 0.5
    assert loaded.rf_n_estimators == 200
    assert loaded.random_seed == 123


def test_config_to_dict_roundtrip():
    config = ModelConfig(lr_c=2.0, svm_kernel="linear")
    d = config.to_dict()
    assert d["lr_c"] == 2.0
    assert d["svm_kernel"] == "linear"
    loaded = ModelConfig.from_dict(d)
    assert loaded.lr_c == 2.0
    assert loaded.svm_kernel == "linear"


def test_config_json_serializable():
    config = ModelConfig()
    # Should not raise.
    json_str = json.dumps(config.to_dict())
    assert isinstance(json_str, str)
    assert "random_seed" in json_str
