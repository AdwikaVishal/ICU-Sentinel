"""Sepsis target construction for P1.

The official Challenge ``SepsisLabel`` marks every hour at or after
``t_sepsis - 6`` (and remains 1 after onset), so it cannot be used
directly as a strict 6-hour pre-onset prediction target.

This module derives an explicit, auditable target without modifying the
original label:

- ``sepsis_within_6h``: 1 for hours in ``[onset, onset + horizon - 1]``,
  0 for hours before ``onset`` and for patients who never develop
  sepsis. Hours at or after ``onset + horizon`` are not eligible.
- ``target_eligible``: 1 when the hour has a valid target (before onset,
  inside the positive window, or a non-septic patient), 0 otherwise.

``onset`` is the first hour where ``SepsisLabel == 1`` for a patient.
The original ``SepsisLabel`` column is preserved unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("icu_sentinel.p1_clinical.preprocessing.targets")

ORIGINAL_LABEL_COLUMN = "SepsisLabel"
DERIVED_TARGET_COLUMN = "sepsis_within_6h"
TARGET_ELIGIBLE_COLUMN = "target_eligible"
DEFAULT_HORIZON_HOURS = 6
VALID_LABELS = (0, 1)


@dataclass(frozen=True)
class TargetPolicy:
    """Configuration for the strict pre-onset sepsis target."""

    horizon_hours: int = DEFAULT_HORIZON_HOURS
    target_name: str = DERIVED_TARGET_COLUMN
    eligibility_name: str = TARGET_ELIGIBLE_COLUMN
    description: str = (
        "Strict 6h pre-onset target: positive for onset..onset+5, "
        "negative before onset and for non-septic patients, ineligible "
        "at/after onset+6."
    )

    def __post_init__(self) -> None:
        if self.horizon_hours <= 0:
            raise ValueError(
                f"horizon_hours must be positive, got {self.horizon_hours}"
            )
        if self.target_name == ORIGINAL_LABEL_COLUMN:
            raise ValueError("Derived target name must differ from SepsisLabel")
        if self.eligibility_name == ORIGINAL_LABEL_COLUMN:
            raise ValueError("Eligibility name must differ from SepsisLabel")

    def to_dict(self) -> dict:
        return {
            "horizon_hours": self.horizon_hours,
            "target_name": self.target_name,
            "eligibility_name": self.eligibility_name,
            "description": self.description,
            "original_label_column": ORIGINAL_LABEL_COLUMN,
            "original_label_modified": False,
        }


def _validate_hours(hours: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(hours, errors="coerce")
    if numeric.isna().any():
        raise ValueError("hour column contains missing or non-numeric values")
    return numeric.astype(int)


def _validate_labels(labels: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(labels, errors="coerce")
    observed = labels.notna()
    invalid = observed & ~numeric.isin(VALID_LABELS)
    if invalid.any():
        bad_values = sorted(numeric[invalid].dropna().unique().tolist())
        raise ValueError(
            f"SepsisLabel contains invalid values {bad_values}; "
            "expected 0/1 (missing is allowed)"
        )
    return numeric


def assign_target_for_patient(
    hours: pd.Series,
    labels: pd.Series,
    policy: TargetPolicy | None = None,
) -> pd.DataFrame:
    """Assign derived target columns for one patient.

    Parameters
    ----------
    hours : pd.Series
        ICU hour values for the patient.
    labels : pd.Series
        Original ``SepsisLabel`` values (0/1, missing allowed).
    policy : TargetPolicy, optional
        Target policy; defaults to a 6-hour horizon.

    Returns
    -------
    pd.DataFrame
        Columns ``sepsis_within_6h`` (float, NaN when ineligible) and
        ``target_eligible`` (int 0/1).
    """
    policy = policy or TargetPolicy()
    hours_arr = _validate_hours(hours)
    labels_arr = _validate_labels(labels)

    valid = labels_arr.notna()
    positive = valid & (labels_arr == 1)

    if positive.any():
        onset = int(hours_arr[positive].min())
        window_end = onset + policy.horizon_hours
        in_window = valid & (hours_arr >= onset) & (hours_arr < window_end)
        post_onset = hours_arr >= window_end
        eligible = valid & ~post_onset
        target = pd.Series(
            np.where(in_window, 1.0, 0.0),
            index=hours.index,
            dtype=float,
        )
    else:
        onset = None
        eligible = valid
        target = pd.Series(
            np.where(valid, 0.0, np.nan),
            index=hours.index,
            dtype=float,
        )

    target = target.where(eligible, np.nan)
    return pd.DataFrame(
        {
            policy.target_name: target,
            policy.eligibility_name: eligible.astype(int),
        },
        index=hours.index,
    )


def assign_sepsis_target(
    dataframe: pd.DataFrame,
    policy: TargetPolicy | None = None,
    patient_id_column: str = "patient_id",
    hour_column: str = "hour",
    label_column: str = ORIGINAL_LABEL_COLUMN,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Return a copy of ``dataframe`` with derived target columns.

    The original label column is never modified. Each patient is
    processed independently. Existing derived columns are preserved
    unless ``overwrite`` is True.
    """
    policy = policy or TargetPolicy()
    required = [patient_id_column, hour_column, label_column]
    missing = [c for c in required if c not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required target columns: {missing}")

    if not overwrite:
        if policy.target_name in dataframe.columns:
            raise ValueError(
                f"Column '{policy.target_name}' already exists; "
                "pass overwrite=True to recompute"
            )
        if policy.eligibility_name in dataframe.columns:
            raise ValueError(
                f"Column '{policy.eligibility_name}' already exists; "
                "pass overwrite=True to recompute"
            )

    out = dataframe.copy()
    derived_parts: list[pd.DataFrame] = []

    for patient_id, group in out.groupby(patient_id_column, sort=False):
        part = assign_target_for_patient(
            group[hour_column], group[label_column], policy=policy
        )
        derived_parts.append(part)

    if derived_parts:
        derived = pd.concat(derived_parts).reindex(out.index)
    else:
        derived = pd.DataFrame(
            {
                policy.target_name: pd.Series(dtype=float),
                policy.eligibility_name: pd.Series(dtype=int),
            },
            index=out.index,
        )

    out[policy.target_name] = derived[policy.target_name]
    out[policy.eligibility_name] = derived[policy.eligibility_name]
    return out


def ensure_sepsis_target(
    dataframe: pd.DataFrame,
    policy: TargetPolicy | None = None,
    patient_id_column: str = "patient_id",
    hour_column: str = "hour",
    label_column: str = ORIGINAL_LABEL_COLUMN,
) -> pd.DataFrame:
    """Ensure derived target columns exist, computing them if absent."""
    policy = policy or TargetPolicy()
    if policy.target_name in dataframe.columns and policy.eligibility_name in dataframe.columns:
        return dataframe
    return assign_sepsis_target(
        dataframe,
        policy=policy,
        patient_id_column=patient_id_column,
        hour_column=hour_column,
        label_column=label_column,
        overwrite=True,
    )


def summarize_targets(dataframe: pd.DataFrame, policy: TargetPolicy | None = None) -> dict:
    """Summarize derived target counts for reporting."""
    policy = policy or TargetPolicy()
    if policy.target_name not in dataframe.columns:
        return {
            "target_name": policy.target_name,
            "available": False,
        }

    target = dataframe[policy.target_name]
    eligible = dataframe[policy.eligibility_name].astype(bool)
    return {
        "target_name": policy.target_name,
        "available": True,
        "horizon_hours": policy.horizon_hours,
        "policy": policy.to_dict(),
        "n_rows": int(len(dataframe)),
        "n_eligible": int(eligible.sum()),
        "n_ineligible": int((~eligible).sum()),
        "n_positive": int((target == 1).sum()),
        "n_negative": int((target == 0).sum()),
        "n_missing_target": int(target.isna().sum()),
        "positive_rate_eligible": float(
            (target[eligible] == 1).mean() if eligible.any() else float("nan")
        ),
        "original_label_modified": False,
    }
