"""Subgroup analysis for P1 research evaluation.

Implements configurable subgroup evaluation where reliable metadata exists.
Supports age groups, sex groups, source dataset groups (Set A/Set B as
dataset-cohort comparisons, NOT hospital identities), and other
clinically justified groups supported by available data.

Missing subgroup variables are marked as unavailable rather than
fabricated. Small subgroups are handled safely with appropriate warnings.
"""

from __future__ import annotations

import logging

import pandas as pd

from .classification_metrics import compute_binary_metrics
from .evaluation_schema import SubgroupAnalysisResult, SubgroupResult

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.subgroup_analysis")


def compute_subgroup_analysis(
    predictions_df: pd.DataFrame,
    model_name: str,
    model_version: str,
    split: str,
    subgroup_dimension: str,
    subgroup_column: str,
    threshold: float = 0.5,
    source: str = "synthetic",
    min_subgroup_size: int = 5,
) -> SubgroupAnalysisResult:
    """Compute subgroup analysis for one model on one split.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with prediction columns and subgroup column.
    model_name : str
        Name of the model.
    model_version : str
        Version of the model.
    split : str
        Split identity.
    subgroup_dimension : str
        Name of the subgroup dimension (e.g., "age_group", "gender",
        "source_dataset").
    subgroup_column : str
        Column name in the DataFrame containing subgroup values.
    threshold : float
        Decision threshold for positive prediction.
    source : str
        Data source label.
    min_subgroup_size : int
        Minimum number of patients required for a subgroup to be
        reported. Subgroups with fewer patients are still reported
        but flagged as small.

    Returns
    -------
    SubgroupAnalysisResult
        Subgroup analysis result.
    """
    required_cols = ["patient_id", "hour", "true_label",
                     "predicted_probability", "predicted_class",
                     subgroup_column]
    missing = [c for c in required_cols if c not in predictions_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    subgroups: list[SubgroupResult] = []

    for subgroup_value, group_df in predictions_df.groupby(subgroup_column):
        n_patients = group_df["patient_id"].nunique()
        n_rows = len(group_df)
        n_positive_patients = int(
            group_df.groupby("patient_id")["true_label"].max().sum()
        )
        n_positive_rows = int((group_df["true_label"] == 1).sum())
        positive_rate = n_positive_rows / n_rows if n_rows > 0 else float("nan")

        # Compute metrics
        metrics = compute_binary_metrics(
            group_df["true_label"],
            group_df["predicted_class"],
            group_df["predicted_probability"],
        )

        # Bootstrap not computed here (see bootstrap.py)
        ci_lower: dict[str, float | None] = {}
        ci_upper: dict[str, float | None] = {}
        for metric_name in metrics:
            ci_lower[metric_name] = None
            ci_upper[metric_name] = None

        subgroups.append(
            SubgroupResult(
                subgroup_name=subgroup_dimension,
                subgroup_value=str(subgroup_value),
                n_patients=n_patients,
                n_positive_patients=n_positive_patients,
                n_rows=n_rows,
                n_positive_rows=n_positive_rows,
                positive_rate=float(positive_rate),
                metrics={
                    k: float(v) for k, v in metrics.items()
                    if k not in ("tp", "tn", "fp", "fn",
                                 "n_positive_pred", "n_negative_pred",
                                 "n_positive_true", "n_negative_true")
                },
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                bootstrap=False,
            )
        )

    # Sort subgroups by name for deterministic output
    subgroups.sort(key=lambda s: (s.subgroup_name, s.subgroup_value))

    # Warn about small subgroups
    for sub in subgroups:
        if sub.n_patients < min_subgroup_size:
            logger.warning(
                "Subgroup %s=%s has only %d patients (min: %d)",
                sub.subgroup_name, sub.subgroup_value,
                sub.n_patients, min_subgroup_size
            )

    return SubgroupAnalysisResult(
        model_name=model_name,
        model_version=model_version,
        split=split,
        subgroup_dimension=subgroup_dimension,
        subgroups=subgroups,
        source=source,
    )


def validate_subgroup_analysis_result(
    result: SubgroupAnalysisResult,
) -> dict:
    """Validate a SubgroupAnalysisResult.

    Returns a dict with validation results.
    """
    issues: list[str] = []

    total_patients = sum(s.n_patients for s in result.subgroups)
    total_rows = sum(s.n_rows for s in result.subgroups)

    if result.subgroups and total_patients == 0:
        issues.append("Total patients is zero but subgroups exist")

    for sub in result.subgroups:
        if sub.n_patients <= 0:
            issues.append(
                f"Subgroup {sub.subgroup_name}={sub.subgroup_value}: "
                f"n_patients is non-positive"
            )
        if sub.n_positive_patients > sub.n_patients:
            issues.append(
                f"Subgroup {sub.subgroup_name}={sub.subgroup_value}: "
                f"n_positive_patients exceeds n_patients"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_subgroups": len(result.subgroups),
        "total_patients": total_patients,
        "total_rows": total_rows,
    }