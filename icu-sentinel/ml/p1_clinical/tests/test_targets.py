"""Tests for preprocessing.targets (strict 6h pre-onset target)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.p1_clinical.preprocessing.targets import (
    DERIVED_TARGET_COLUMN,
    ORIGINAL_LABEL_COLUMN,
    TARGET_ELIGIBLE_COLUMN,
    TargetPolicy,
    assign_sepsis_target,
    assign_target_for_patient,
    ensure_sepsis_target,
    summarize_targets,
)


def _patient_frame(labels, start_hour=1, patient_id="p1"):
    hours = list(range(start_hour, start_hour + len(labels)))
    return pd.DataFrame(
        {
            "patient_id": [patient_id] * len(labels),
            "hour": hours,
            ORIGINAL_LABEL_COLUMN: labels,
        }
    )


def test_positive_window_and_post_onset_exclusion():
    df = _patient_frame([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    out = assign_sepsis_target(df)

    target = out[DERIVED_TARGET_COLUMN].tolist()
    eligible = out[TARGET_ELIGIBLE_COLUMN].tolist()

    # onset at hour 3; positive hours 3..8; ineligible from hour 9.
    assert target[:8] == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert np.isnan(target[8]) and np.isnan(target[9])
    assert eligible[:8] == [1] * 8
    assert eligible[8:] == [0, 0]


def test_non_septic_patient_all_negative_and_eligible():
    df = _patient_frame([0, 0, 0, 0])
    out = assign_sepsis_target(df)
    assert (out[DERIVED_TARGET_COLUMN] == 0).all()
    assert (out[TARGET_ELIGIBLE_COLUMN] == 1).all()


def test_onset_at_first_hour():
    df = _patient_frame([1, 1, 1, 1, 1, 1, 1])
    out = assign_sepsis_target(df)
    assert out[DERIVED_TARGET_COLUMN].iloc[:6].tolist() == [1.0] * 6
    assert np.isnan(out[DERIVED_TARGET_COLUMN].iloc[6])
    assert out[TARGET_ELIGIBLE_COLUMN].tolist() == [1] * 6 + [0]


def test_truncated_positive_window():
    df = _patient_frame([0, 1, 1])
    out = assign_sepsis_target(df)
    assert out[DERIVED_TARGET_COLUMN].tolist() == [0.0, 1.0, 1.0]
    assert out[TARGET_ELIGIBLE_COLUMN].tolist() == [1, 1, 1]


def test_original_label_is_preserved():
    df = _patient_frame([0, 1, 1, 1, 1, 1, 1, 1, 1])
    original = df[ORIGINAL_LABEL_COLUMN].tolist()
    out = assign_sepsis_target(df)
    assert out[ORIGINAL_LABEL_COLUMN].tolist() == original
    # Original frame is not mutated.
    assert DERIVED_TARGET_COLUMN not in df.columns


def test_patients_are_independent():
    df = pd.concat(
        [
            _patient_frame([0, 1, 1], patient_id="p1"),
            _patient_frame([0, 0, 0], patient_id="p2"),
        ],
        ignore_index=True,
    )
    out = assign_sepsis_target(df)
    p1 = out[out["patient_id"] == "p1"]
    p2 = out[out["patient_id"] == "p2"]
    assert p1[DERIVED_TARGET_COLUMN].tolist() == [0.0, 1.0, 1.0]
    assert p2[DERIVED_TARGET_COLUMN].tolist() == [0.0, 0.0, 0.0]
    assert (p2[TARGET_ELIGIBLE_COLUMN] == 1).all()


def test_missing_labels_are_ineligible():
    df = _patient_frame([0, np.nan, 1, 1])
    out = assign_sepsis_target(df)
    assert np.isnan(out[DERIVED_TARGET_COLUMN].iloc[1])
    assert out[TARGET_ELIGIBLE_COLUMN].iloc[1] == 0


def test_invalid_label_raises():
    df = _patient_frame([0, 5, 1])
    with pytest.raises(ValueError, match="invalid values"):
        assign_sepsis_target(df)


def test_missing_required_column_raises():
    df = _patient_frame([0, 1]).drop(columns=[ORIGINAL_LABEL_COLUMN])
    with pytest.raises(ValueError, match="Missing required"):
        assign_sepsis_target(df)


def test_duplicate_derived_columns_raise_without_overwrite():
    df = _patient_frame([0, 1, 1])
    out = assign_sepsis_target(df)
    with pytest.raises(ValueError, match="already exists"):
        assign_sepsis_target(out)
    recomputed = assign_sepsis_target(out, overwrite=True)
    assert recomputed[DERIVED_TARGET_COLUMN].tolist() == out[DERIVED_TARGET_COLUMN].tolist()


def test_custom_horizon():
    policy = TargetPolicy(horizon_hours=3)
    df = _patient_frame([0, 0, 1, 1, 1, 1])
    out = assign_sepsis_target(df, policy=policy)
    assert out[DERIVED_TARGET_COLUMN].iloc[:5].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert np.isnan(out[DERIVED_TARGET_COLUMN].iloc[5])
    assert out[TARGET_ELIGIBLE_COLUMN].tolist() == [1, 1, 1, 1, 1, 0]


def test_invalid_policy_horizon_raises():
    with pytest.raises(ValueError, match="horizon_hours"):
        TargetPolicy(horizon_hours=0)


def test_non_contiguous_hours_are_handled_by_hour_value():
    df = pd.DataFrame(
        {
            "patient_id": ["p1"] * 4,
            "hour": [1, 2, 8, 9],
            ORIGINAL_LABEL_COLUMN: [0, 1, 1, 1],
        }
    )
    out = assign_sepsis_target(df)
    # onset hour 2; positive window 2..7 -> hour 8 is post-onset.
    assert out[DERIVED_TARGET_COLUMN].iloc[:2].tolist() == [0.0, 1.0]
    assert np.isnan(out[DERIVED_TARGET_COLUMN].iloc[2])
    assert np.isnan(out[DERIVED_TARGET_COLUMN].iloc[3])
    assert out[TARGET_ELIGIBLE_COLUMN].tolist() == [1, 1, 0, 0]


def test_ensure_sepsis_target_computes_only_when_absent():
    df = _patient_frame([0, 1, 1])
    once = ensure_sepsis_target(df)
    twice = ensure_sepsis_target(once)
    assert twice[DERIVED_TARGET_COLUMN].tolist() == once[DERIVED_TARGET_COLUMN].tolist()
    assert twice is once


def test_summary_counts():
    df = _patient_frame([0, 1, 1, 1, 1, 1, 1, 1, 1])
    out = assign_sepsis_target(df)
    summary = summarize_targets(out)
    assert summary["available"] is True
    assert summary["n_rows"] == 9
    assert summary["n_eligible"] == 7
    assert summary["n_ineligible"] == 2
    assert summary["n_positive"] == 6
    assert summary["n_negative"] == 1
    assert summary["n_missing_target"] == 2
    assert summary["original_label_modified"] is False


def test_assign_target_for_patient_returns_aligned_index():
    hours = pd.Series([1, 2, 3], index=[10, 11, 12])
    labels = pd.Series([0, 1, 1], index=[10, 11, 12])
    out = assign_target_for_patient(hours, labels)
    assert out.index.tolist() == [10, 11, 12]
    assert out[DERIVED_TARGET_COLUMN].tolist() == [0.0, 1.0, 1.0]
