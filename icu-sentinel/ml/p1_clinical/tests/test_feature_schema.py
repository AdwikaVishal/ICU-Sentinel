"""Tests for feature_schema.py.

Tests the stable feature schema for P2/P3 integration.
"""

from __future__ import annotations

import json

from ml.p1_clinical.features.feature_config import FeatureConfig
from ml.p1_clinical.features.feature_schema import (
    CATEGORY_CLINICAL,
    CATEGORY_CURRENT,
    CATEGORY_MISSINGNESS,
    CATEGORY_ROLLING,
    CATEGORY_TREND,
    FeatureEntry,
    FeatureSchema,
    build_default_schema,
)

# ------------------------------------------------------------------
# FeatureEntry tests
# ------------------------------------------------------------------


def test_feature_entry_creation():
    entry = FeatureEntry(
        feature_name="HR",
        source_variable="HR",
        category=CATEGORY_CURRENT,
        data_type="float",
        formula="Original HR measurement",
        causal=True,
        tabular_appropriate=True,
        sequential_appropriate=True,
    )
    assert entry.feature_name == "HR"
    assert entry.causal is True


def test_feature_entry_to_dict():
    entry = FeatureEntry(
        feature_name="HR",
        source_variable="HR",
        category=CATEGORY_CURRENT,
        data_type="float",
        formula="Original HR measurement",
    )
    d = entry.to_dict()
    assert d["feature_name"] == "HR"
    assert d["category"] == CATEGORY_CURRENT


# ------------------------------------------------------------------
# FeatureSchema tests
# ------------------------------------------------------------------


def test_feature_schema_add_and_get():
    schema = FeatureSchema()
    entry = FeatureEntry(
        feature_name="HR", source_variable="HR", category=CATEGORY_CURRENT,
        data_type="float", formula="Original HR",
    )
    schema.add_feature(entry)
    fetched = schema.get_feature("HR")
    assert fetched is not None
    assert fetched.feature_name == "HR"


def test_feature_schema_feature_names():
    schema = FeatureSchema()
    schema.add_feature(FeatureEntry(
        feature_name="A", source_variable=None, category=CATEGORY_CURRENT,
        data_type="float", formula="A",
    ))
    schema.add_feature(FeatureEntry(
        feature_name="B", source_variable=None, category=CATEGORY_CURRENT,
        data_type="float", formula="B",
    ))
    assert schema.feature_names() == ["A", "B"]


def test_feature_schema_by_category():
    schema = FeatureSchema()
    schema.add_feature(FeatureEntry(
        feature_name="A", source_variable=None, category=CATEGORY_ROLLING,
        data_type="float", formula="A",
    ))
    schema.add_feature(FeatureEntry(
        feature_name="B", source_variable=None, category=CATEGORY_TREND,
        data_type="float", formula="B",
    ))
    rolling = schema.by_category(CATEGORY_ROLLING)
    assert len(rolling) == 1
    assert rolling[0].feature_name == "A"


def test_schema_causal_features_all_causal():
    schema = build_default_schema()
    non_causal = [f.feature_name for f in schema.features if not f.causal]
    assert len(non_causal) == 0


def test_schema_identifier_and_target_preserved():
    schema = build_default_schema()
    names = schema.feature_names()
    assert "patient_id" in names
    assert "source_dataset" in names
    assert "hour" in names
    assert "SepsisLabel" in names


def test_schema_target_not_appropriate():
    schema = build_default_schema()
    target = schema.get_feature("SepsisLabel")
    assert target.tabular_appropriate is False
    assert target.sequential_appropriate is False


# ------------------------------------------------------------------
# build_default_schema tests
# ------------------------------------------------------------------


def test_build_default_schema_has_categories():
    schema = build_default_schema()
    cats = {f.category for f in schema.features}
    assert CATEGORY_CURRENT in cats
    assert CATEGORY_MISSINGNESS in cats
    assert CATEGORY_ROLLING in cats
    assert CATEGORY_TREND in cats
    assert CATEGORY_CLINICAL in cats


def test_build_default_schema_has_expected_features():
    schema = build_default_schema()
    names = schema.feature_names()
    assert "HR" in names
    assert "HR_is_missing" in names
    assert "HR_hours_since_last" in names
    assert "HR_rolling_mean_6h" in names
    assert "HR_abs_change_6h" in names
    assert "shock_index" in names
    assert "sirs_count" in names


# ------------------------------------------------------------------
# serialization tests
# ------------------------------------------------------------------


def test_schema_save_and_load(tmp_path):
    schema = build_default_schema()
    path = tmp_path / "feature_schema.json"
    schema.save(path)
    loaded = FeatureSchema.load(path)
    assert loaded.version == schema.version
    assert len(loaded.features) == len(schema.features)
    assert loaded.get_feature("HR") is not None


def test_schema_serialization_roundtrip(tmp_path):
    schema = build_default_schema()
    path = tmp_path / "schema.json"
    schema.save(path)
    data = json.loads(path.read_text())
    assert "version" in data
    assert "features" in data
    assert "identifier_columns" in data
    assert "target_columns" in data
    assert data["target_columns"] == ["SepsisLabel"]


def test_config_save_and_load(tmp_path):
    config = FeatureConfig()
    path = tmp_path / "feature_config.json"
    config.save(path)
    loaded = FeatureConfig.load(path)
    assert loaded.rolling_window_hours == config.rolling_window_hours
    assert loaded.trend_window_hours == config.trend_window_hours
    assert loaded.compute_missingness_features == config.compute_missingness_features
