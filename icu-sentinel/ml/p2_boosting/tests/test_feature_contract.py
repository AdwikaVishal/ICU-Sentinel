"""Tests for the P2 feature/target contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.p2_boosting.feature_contract import (
    resolve_feature_names,
    sanitize_infinities,
    validate_feature_frame,
    validate_training_target,
)


def test_resolve_feature_names_excludes_identifiers_and_targets():
    columns = [
        "patient_id",
        "source_dataset",
        "hour",
        "source_file",
        "hospital_id",
        "SepsisLabel",
        "sepsis_within_6h",
        "target_eligible",
        "HR",
        "HR_rolling_mean_6h",
    ]
    assert resolve_feature_names(columns) == ["HR", "HR_rolling_mean_6h"]


def test_missing_feature_raises_clear_error():
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_feature_frame(frame, ["a", "b"])


def test_unexpected_columns_rejected_when_strict():
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_feature_frame(frame, ["a"], allow_extra_columns=False)


def test_non_numeric_feature_raises():
    frame = pd.DataFrame({"a": ["x", "y"]})
    with pytest.raises(ValueError, match="must be numeric"):
        validate_feature_frame(frame, ["a"])


def test_target_cannot_be_a_feature():
    frame = pd.DataFrame({"sepsis_within_6h": [0, 1]})
    with pytest.raises(ValueError, match="cannot be model features"):
        validate_feature_frame(frame, ["sepsis_within_6h"])


def test_empty_frame_raises():
    frame = pd.DataFrame({"a": pd.Series([], dtype=float)})
    with pytest.raises(ValueError, match="zero rows"):
        validate_feature_frame(frame, ["a"])


def test_sanitize_infinities_replaces_with_nan():
    frame = pd.DataFrame({"a": [np.inf, -np.inf, 1.0]})
    cleaned = sanitize_infinities(frame)
    assert cleaned["a"].isna().sum() == 2
    assert cleaned["a"].iloc[2] == 1.0


def test_training_target_rejects_non_binary():
    with pytest.raises(ValueError, match="must be binary"):
        validate_training_target(pd.Series([0, 1, 2]), "y")


def test_training_target_rejects_single_class():
    with pytest.raises(ValueError, match="both classes"):
        validate_training_target(pd.Series([0, 0, 0]), "y")


def test_training_target_rejects_missing():
    with pytest.raises(ValueError, match="missing"):
        validate_training_target(pd.Series([0, 1, np.nan]), "y")
