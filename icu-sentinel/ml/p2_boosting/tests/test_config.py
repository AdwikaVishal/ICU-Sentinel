"""Tests for P2 configuration and model initialization."""

from __future__ import annotations

import pytest

from ml.p2_boosting.config import (
    ExtraTreesConfig,
    LightGBMConfig,
    P2Config,
    XGBoostConfig,
)


def test_default_models_enabled():
    assert P2Config().enabled_models() == [
        "extra_trees",
        "lightgbm",
        "xgboost",
    ]


def test_config_roundtrip(tmp_path):
    config = P2Config(
        random_seed=7,
        extra_trees=ExtraTreesConfig(n_estimators=12),
        lightgbm=LightGBMConfig(n_estimators=13),
        xgboost=XGBoostConfig(n_estimators=14),
    )
    path = config.save(tmp_path / "config.json")
    loaded = P2Config.load(path)
    assert loaded == config


def test_config_refuses_overwrite(tmp_path):
    config = P2Config()
    path = config.save(tmp_path / "config.json")
    with pytest.raises(FileExistsError):
        config.save(path)
