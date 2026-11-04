"""Current observation features.

Retains the original clinical observations at each patient-hour without
modification. These are the raw measurements from the Part 3 cleaned
output, carried forward as feature columns.

No imputation, no forward filling, no transformation is applied here.
Original values (including NaN) are preserved exactly.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..preprocessing.schemas import ProcessedPatient
from .feature_config import (
    FEATURE_VARIABLES,
    STATIC_VARIABLES,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.features.current_features")


def extract_current_features(
    patient: ProcessedPatient,
    feature_variables: list[str] | None = None,
    static_variables: list[str] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame of current (raw) observation features for one patient.

    Parameters
    ----------
    patient : ProcessedPatient
        Cleaned patient data from Part 3.
    feature_variables : list[str], optional
        Clinical variables to retain as current features.
    static_variables : list[str], optional
        Static/demographic variables repeated for each hour.

    Returns
    -------
    pd.DataFrame
        One row per patient-hour with columns: patient_id, source_dataset,
        hour, <feature_variables>, <static_variables>, SepsisLabel.
    """
    fvars = feature_variables or FEATURE_VARIABLES
    svars = static_variables or STATIC_VARIABLES

    n = patient.n_rows
    data: dict = {
        "patient_id": [patient.patient_id] * n,
        "source_dataset": [patient.source_dataset] * n,
        "hour": list(patient.hour),
    }

    # Clinical variables (raw, unchanged).
    for var in fvars:
        values = patient.clinical.get(var, [float("nan")] * n)
        data[var] = list(values)

    # Static variables (repeated for each hour).
    for var in svars:
        values = patient.clinical.get(var, [float("nan")] * n)
        data[var] = list(values)

    # Target label (preserved unchanged, target-only).
    data["SepsisLabel"] = list(patient.sepsis_label)

    df = pd.DataFrame(data)

    # Ensure column order: identifiers, clinical, static, target.
    ordered = ["patient_id", "source_dataset", "hour"]
    ordered += [v for v in fvars if v in df.columns]
    ordered += [v for v in svars if v in df.columns]
    ordered += ["SepsisLabel"]
    df = df[ordered]

    return df


def validate_current_features(df: pd.DataFrame) -> list[str]:
    """Validate current features output.

    Returns a list of issue messages (empty if valid).
    """
    issues: list[str] = []
    expected = ["patient_id", "source_dataset", "hour", "SepsisLabel"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")
    if len(df) == 0:
        issues.append("Empty current features DataFrame")
    if "SepsisLabel" in df.columns:
        bad = df["SepsisLabel"].dropna()
        bad = bad[~bad.isin([0, 1])]
        if len(bad) > 0:
            issues.append(f"{len(bad)} row(s) with invalid SepsisLabel values")
    return issues
