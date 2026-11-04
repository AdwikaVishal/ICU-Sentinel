"""Cross-hospital evaluation for P1 research evaluation.

Prepares a cross-hospital evaluation interface supporting
train-on-hospital-A/test-on-hospital-B style evaluation.

IMPORTANT: Source dataset identity (Set A/Set B) is NOT automatically
a hospital identifier. If real hospital identifiers are unavailable,
cross-hospital evaluation is reported as unavailable.
"""

from __future__ import annotations

import logging

import pandas as pd

from .classification_metrics import compute_binary_metrics
from .evaluation_schema import CrossHospitalResult

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.cross_hospital")


def compute_cross_hospital_evaluation(
    train_predictions_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    train_hospital: str,
    test_hospital: str,
    model_name: str,
    model_version: str,
    threshold: float = 0.5,
    source: str = "synthetic",
) -> CrossHospitalResult:
    """Compute cross-hospital evaluation result.

    Trains on one hospital's data and tests on another's. Since we
    already have predictions (not training data), this function evaluates
    the model trained on train_hospital against test data from test_hospital.

    Parameters
    ----------
    train_predictions_df : pd.DataFrame
        Predictions on training hospital data (for reference).
    test_predictions_df : pd.DataFrame
        Predictions on test hospital data.
    train_hospital : str
        Training hospital identifier.
    test_hospital : str
        Test hospital identifier.
    model_name : str
        Name of the model.
    model_version : str
        Version of the model.
    threshold : float
        Decision threshold for positive prediction.
    source : str
        Data source label.

    Returns
    -------
    CrossHospitalResult
        Cross-hospital evaluation result.
    """
    required_cols = ["patient_id", "hour", "true_label",
                     "predicted_probability", "predicted_class"]
    missing = [c for c in required_cols if c not in test_predictions_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in test data: {missing}")

    # Evaluate on test hospital
    y_true = test_predictions_df["true_label"].astype(int).values
    y_pred = test_predictions_df["predicted_class"].astype(int).values
    y_prob = test_predictions_df["predicted_probability"].astype(float).values

    metrics = compute_binary_metrics(y_true, y_pred, y_prob)

    n_patients_train = int(
        train_predictions_df["patient_id"].nunique()
    ) if "patient_id" in train_predictions_df.columns else 0
    n_patients_test = int(
        test_predictions_df["patient_id"].nunique()
    )
    n_rows_test = len(test_predictions_df)

    return CrossHospitalResult(
        train_hospital=train_hospital,
        test_hospital=test_hospital,
        model_name=model_name,
        model_version=model_version,
        n_patients_train=n_patients_train,
        n_patients_test=n_patients_test,
        n_rows_test=n_rows_test,
        metrics={k: float(v) for k, v in metrics.items()
                 if k not in ("tp", "tn", "fp", "fn",
                              "n_positive_pred", "n_negative_pred",
                              "n_positive_true", "n_negative_true")},
        source=source,
    )


def validate_cross_hospital_result(
    result: CrossHospitalResult,
) -> dict:
    """Validate a CrossHospitalResult.

    Returns a dict with validation results.
    """
    issues: list[str] = []

    if result.train_hospital == result.test_hospital:
        issues.append(
            "Train and test hospital are the same "
            "(not true cross-hospital evaluation)"
        )

    if result.n_patients_test <= 0:
        issues.append("n_patients_test is non-positive")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_patients_train": result.n_patients_train,
        "n_patients_test": result.n_patients_test,
    }


def run_cross_hospital_evaluation(
    predictions_by_hospital: dict[str, pd.DataFrame],
    model_name: str,
    model_version: str,
    threshold: float = 0.5,
    source: str = "synthetic",
) -> list[CrossHospitalResult]:
    """Run cross-hospital evaluation for all hospital pairs.

    Parameters
    ----------
    predictions_by_hospital : dict[str, pd.DataFrame]
        Dictionary mapping hospital name to predictions DataFrame.
    model_name : str
        Name of the model.
    model_version : str
        Version of the model.
    threshold : float
        Decision threshold for positive prediction.
    source : str
        Data source label.

    Returns
    -------
    list[CrossHospitalResult]
        List of cross-hospital evaluation results.
    """
    results: list[CrossHospitalResult] = []

    hospitals = list(predictions_by_hospital.keys())
    for train_hosp in hospitals:
        for test_hosp in hospitals:
            if train_hosp == test_hosp:
                continue  # Skip same-hospital (not cross-hospital)

            train_df = predictions_by_hospital[train_hosp]
            test_df = predictions_by_hospital[test_hosp]

            result = compute_cross_hospital_evaluation(
                train_df, test_df,
                train_hosp, test_hosp,
                model_name, model_version,
                threshold=threshold,
                source=source,
            )
            results.append(result)

    return results