"""Lead-time evaluation for P1 research evaluation.

Implements lead-time evaluation based on the official PhysioNet/CinC
Challenge 2019 label convention.

IMPORTANT: The official SepsisLabel already incorporates a six-hour
shift relative to sepsis onset. Therefore:
- We do NOT shift labels again.
- We do NOT reconstruct sepsis onset independently.
- Lead time is measured relative to the first positive SepsisLabel
  (t >= t_sepsis - 6), NOT relative to actual clinical sepsis onset.
- If actual onset cannot be independently determined, metrics are
  labeled as lead time relative to the official label transition.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .evaluation_schema import LeadTimeEntry, LeadTimeResult

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.lead_time")


def compute_lead_time_per_patient(
    patient_id: str,
    source_dataset: str | None,
    predictions: list[tuple[int, float, int]],  # (hour, prob, true_label)
    threshold: float = 0.5,
) -> LeadTimeEntry:
    """Compute lead-time metrics for a single patient.

    Parameters
    ----------
    patient_id : str
        Patient identifier.
    source_dataset : str | None
        Source dataset identity.
    predictions : list of (hour, prob, true_label)
        List of (hour, predicted_probability, true_label) tuples,
        sorted by hour.
    threshold : float
        Decision threshold for positive prediction.

    Returns
    -------
    LeadTimeEntry
        Lead-time result for the patient.
    """
    if not predictions:
        return LeadTimeEntry(
            patient_id=patient_id,
            source_dataset=source_dataset,
            first_alert_hour=None,
            first_positive_label_hour=None,
            lead_time_hours=None,
            alert_before_onset=False,
            alert_after_onset=False,
            missed=True,
        )

    # Sort by hour
    sorted_preds = sorted(predictions, key=lambda x: x[0])

    # Find first alert hour (first hour with predicted_probability >= threshold)
    first_alert_hour: int | None = None
    for hour, prob, _ in sorted_preds:
        if prob >= threshold:
            first_alert_hour = hour
            break

    # Find first positive label hour
    first_positive_label_hour: int | None = None
    for hour, _, label in sorted_preds:
        if label == 1:
            first_positive_label_hour = hour
            break

    # Determine lead time
    lead_time_hours: float | None = None
    alert_before_onset = False
    alert_after_onset = False
    missed = True

    if first_positive_label_hour is not None:
        if first_alert_hour is not None:
            lead_time_hours = float(first_positive_label_hour - first_alert_hour)
            if first_alert_hour <= first_positive_label_hour:
                alert_before_onset = True
                missed = False
            else:
                alert_after_onset = True
                missed = False
        else:
            missed = True
    else:
        # No positive label - patient is non-sepsis
        if first_alert_hour is not None:
            alert_after_onset = True  # False alert
            missed = False
        else:
            missed = False  # No alert, no sepsis - not missed

    return LeadTimeEntry(
        patient_id=patient_id,
        source_dataset=source_dataset,
        first_alert_hour=first_alert_hour,
        first_positive_label_hour=first_positive_label_hour,
        lead_time_hours=lead_time_hours,
        alert_before_onset=alert_before_onset,
        alert_after_onset=alert_after_onset,
        missed=missed,
    )


def compute_lead_time(
    predictions_df: pd.DataFrame,
    model_name: str,
    model_version: str,
    split: str,
    threshold: float = 0.5,
    source: str = "synthetic",
) -> LeadTimeResult:
    """Compute lead-time metrics for all patients.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with columns: patient_id, source_dataset (optional),
        hour, true_label, predicted_probability.
    model_name : str
        Name of the model.
    model_version : str
        Version of the model.
    split : str
        Split identity.
    threshold : float
        Decision threshold for positive prediction.
    source : str
        Data source label.

    Returns
    -------
    LeadTimeResult
        Lead-time result.
    """
    required_cols = ["patient_id", "hour", "true_label", "predicted_probability"]
    missing = [c for c in required_cols if c not in predictions_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Group by patient
    patient_entries: dict[str, list[tuple[int, float, int]]] = {}
    patient_source: dict[str, str | None] = {}

    for _, row in predictions_df.iterrows():
        pid = str(row["patient_id"])
        hour = int(row["hour"])
        prob = float(row["predicted_probability"])
        label = int(row["true_label"])
        source_ds = (
            str(row["source_dataset"])
            if "source_dataset" in predictions_df.columns
            and pd.notna(row["source_dataset"])
            else None
        )

        if pid not in patient_entries:
            patient_entries[pid] = []
            patient_source[pid] = source_ds

        patient_entries[pid].append((hour, prob, label))

    patient_results: list[LeadTimeEntry] = []
    for pid in sorted(patient_entries.keys()):
        result = compute_lead_time_per_patient(
            pid, patient_source[pid], patient_entries[pid], threshold
        )
        patient_results.append(result)

    # Compute aggregate statistics
    n_patients = len(patient_results)
    n_with_alert = sum(1 for r in patient_results if r.first_alert_hour is not None)
    n_with_onset = sum(1 for r in patient_results if r.first_positive_label_hour is not None)
    n_alert_before_onset = sum(1 for r in patient_results if r.alert_before_onset)
    n_alert_after_onset = sum(1 for r in patient_results if r.alert_after_onset)
    n_missed = sum(1 for r in patient_results if r.missed)

    lead_times = [
        r.lead_time_hours
        for r in patient_results
        if r.lead_time_hours is not None and r.alert_before_onset
    ]

    mean_lead_time = float(np.mean(lead_times)) if lead_times else float("nan")
    median_lead_time = float(np.median(lead_times)) if lead_times else float("nan")

    detection_rate_before_onset = (
        n_alert_before_onset / n_with_onset if n_with_onset > 0 else float("nan")
    )
    detection_rate_at_or_before_onset = detection_rate_before_onset  # Same definition
    detection_rate_after_onset = (
        n_alert_after_onset / n_with_onset if n_with_onset > 0 else float("nan")
    )
    missed_patient_rate = n_missed / n_patients if n_patients > 0 else float("nan")

    return LeadTimeResult(
        model_name=model_name,
        model_version=model_version,
        split=split,
        n_patients=n_patients,
        n_with_alert=n_with_alert,
        n_with_onset=n_with_onset,
        n_alert_before_onset=n_alert_before_onset,
        n_alert_after_onset=n_alert_after_onset,
        n_missed=n_missed,
        mean_lead_time_hours=mean_lead_time,
        median_lead_time_hours=median_lead_time,
        detection_rate_before_onset=detection_rate_before_onset,
        detection_rate_at_or_before_onset=detection_rate_at_or_before_onset,
        detection_rate_after_onset=detection_rate_after_onset,
        missed_patient_rate=missed_patient_rate,
        patient_details=patient_results,
        source=source,
    )


def validate_lead_time_result(
    result: LeadTimeResult,
) -> dict:
    """Validate a LeadTimeResult.

    Returns a dict with validation results.
    """
    issues: list[str] = []

    if result.n_patients < 0:
        issues.append("n_patients is negative")

    if (
        result.n_alert_before_onset
        + result.n_alert_after_onset
        + result.n_missed
        > result.n_patients
    ):
        issues.append(
            "Sum of alert_before_onset, alert_after_onset, and missed exceeds n_patients"
        )

    if result.mean_lead_time_hours is not None and not np.isfinite(
        result.mean_lead_time_hours
    ) and not np.isnan(result.mean_lead_time_hours):
        issues.append("mean_lead_time_hours is not finite")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_patients": result.n_patients,
    }