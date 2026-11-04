"""Tests for feature_selector.py.

Tests constant, duplicate, all-missing, infinite column detection.
All selection is fitted on training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.p1_clinical.classical_ml.feature_selector import (
    FeatureSelector,
)


def _make_df(n_rows=20, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_features):
        data[f"feat_{i}"] = rng.normal(0, 1, size=n_rows)
    return pd.DataFrame(data)


# ------------------------------------------------------------------
# Constant feature removal
# ------------------------------------------------------------------


def test_remove_constant_features():
    df = _make_df(n_rows=10, n_features=3, seed=1)
    df["constant_col"] = 5.0
    df["another_constant"] = 3.0
    selector = FeatureSelector(remove_constant=True)
    X_out = selector.fit_transform(df)
    assert "constant_col" not in X_out.columns
    assert "another_constant" not in X_out.columns
    assert "constant_col" in selector.removed_features["constant"]
    assert "another_constant" in selector.removed_features["constant"]


def test_keep_non_constant():
    df = _make_df(n_rows=10, n_features=3, seed=1)
    selector = FeatureSelector(remove_constant=True)
    X_out = selector.fit_transform(df)
    assert len(X_out.columns) == 3


# ------------------------------------------------------------------
# All-missing removal
# ------------------------------------------------------------------


def test_remove_all_missing():
    df = _make_df(n_rows=10, n_features=2, seed=1)
    df["all_nan"] = np.nan
    selector = FeatureSelector(remove_all_missing=True)
    X_out = selector.fit_transform(df)
    assert "all_nan" not in X_out.columns
    assert "all_nan" in selector.removed_features["all_missing"]


# ------------------------------------------------------------------
# Infinite removal
# ------------------------------------------------------------------


def test_remove_infinite():
    df = _make_df(n_rows=10, n_features=2, seed=1)
    df["has_inf"] = [np.inf if i == 0 else 1.0 for i in range(10)]
    selector = FeatureSelector(remove_infinite=True)
    X_out = selector.fit_transform(df)
    assert "has_inf" not in X_out.columns
    assert "has_inf" in selector.removed_features["infinite"]


# ------------------------------------------------------------------
# Duplicate removal
# ------------------------------------------------------------------


def test_remove_duplicate():
    df = _make_df(n_rows=10, n_features=2, seed=1)
    df["dup_of_feat_0"] = df["feat_0"]
    selector = FeatureSelector(remove_duplicate=True)
    X_out = selector.fit_transform(df)
    assert "dup_of_feat_0" not in X_out.columns
    assert "dup_of_feat_0" in selector.removed_features["duplicate"]


# ------------------------------------------------------------------
# Fit on train only - transform doesn't refit
# ------------------------------------------------------------------


def test_fit_on_train_only():
    train_df = _make_df(n_rows=20, n_features=3, seed=1)
    train_df["constant_col"] = 7.0
    test_df = _make_df(n_rows=10, n_features=3, seed=2)
    test_df["constant_col"] = 9.0  # constant in test but different value

    selector = FeatureSelector(remove_constant=True)
    X_train_out = selector.fit_transform(train_df)
    X_test_out = selector.transform(test_df)

    # Train: constant_col removed.
    assert "constant_col" not in X_train_out.columns
    # Test: same columns as train (constant_col removed based on train fit).
    assert "constant_col" not in X_test_out.columns
    assert list(X_train_out.columns) == list(X_test_out.columns)


def test_transform_without_fit_raises():
    df = _make_df(n_rows=10, n_features=3, seed=1)
    selector = FeatureSelector()
    with pytest.raises(ValueError):
        selector.transform(df)


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------


def test_feature_selection_report():
    df = _make_df(n_rows=10, n_features=2, seed=1)
    df["const"] = 1.0
    df["all_nan"] = np.nan
    selector = FeatureSelector()
    selector.fit_transform(df)
    report = selector.report()
    assert report.n_features_input == 4
    assert report.n_features_output == 2
    assert "const" in report.removed_constant
    assert "all_nan" in report.removed_all_missing
    assert report.fit_on_train_only is True
    assert set(report.kept_features) == {"feat_0", "feat_1"}


# ------------------------------------------------------------------
# Highly correlated (optional)
# ------------------------------------------------------------------


def test_remove_highly_correlated():
    df = _make_df(n_rows=100, n_features=2, seed=1)
    # Create a highly correlated copy.
    df["corr_copy"] = df["feat_0"] * 2 + 1.0 + rng_noise(100, seed=2)
    selector = FeatureSelector(correlation_threshold=0.99)
    selector.fit_transform(df)
    # One of the correlated pair should be removed.
    assert "corr_copy" in selector.removed_features["highly_correlated"] or \
           "feat_0" in selector.removed_features["highly_correlated"]


def rng_noise(n, seed):
    return np.random.default_rng(seed).normal(0, 0.001, size=n)
