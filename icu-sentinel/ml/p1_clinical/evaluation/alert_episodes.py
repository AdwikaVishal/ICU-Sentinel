"""Alert episode engine for P1 research evaluation.

Implements configurable alert episode aggregation:
- Configurable alert threshold
- Configurable cooldown period
- Configurable minimum alert duration / consecutive-positive requirement
- Merges repeated positive predictions into a single alert episode

Alert episodes are defined as contiguous periods where the model
predicts sepsis (probability >= threshold), subject to cooldown and
minimum duration constraints.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .evaluation_schema import AlertBurdenResult, AlertEpisode

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.alert_episodes")


def compute_alert_episodes(
    patient_id: str,
    source_dataset: str | None,
    predictions: list[tuple[int, float, int]],  # (hour, prob, true_label)
    threshold: float = 0.5,
    cooldown_hours: int = 6,
    min_duration_hours: int = 1,
    min_consecutive: int = 1,
) -> list[AlertEpisode]:
    """Compute alert episodes for a single patient.

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
    cooldown_hours : int
        Minimum hours between the end of one alert episode and the
        start of the next. Predictions within cooldown of the previous
        episode end are merged into the same episode.
    min_duration_hours : int
        Minimum duration (in hours) for an alert episode to be valid.
        Episodes shorter than this are discarded.
    min_consecutive : int
        Minimum number of consecutive positive predictions required
        to start an alert episode.

    Returns
    -------
    list[AlertEpisode]
        List of alert episodes for the patient.
    """
    if not predictions:
        return []

    # Sort by hour
    sorted_preds = sorted(predictions, key=lambda x: x[0])

    # Determine patient-level sepsis status (any positive label)
    patient_has_sepsis = any(label == 1 for _, _, label in sorted_preds)

    # Identify positive prediction hours
    positive_hours: list[int] = []
    for hour, prob, _ in sorted_preds:
        if prob >= threshold:
            positive_hours.append(hour)

    if not positive_hours:
        return []

    # Group into episodes based on cooldown
    episodes: list[AlertEpisode] = []
    current_episode_hours: list[int] = [positive_hours[0]]

    for i in range(1, len(positive_hours)):
        prev_hour = positive_hours[i - 1]
        curr_hour = positive_hours[i]

        if curr_hour - prev_hour <= cooldown_hours:
            # Within cooldown, extend current episode
            current_episode_hours.append(curr_hour)
        else:
            # Start new episode
            episodes.append(
                AlertEpisode(
                    patient_id=patient_id,
                    source_dataset=source_dataset,
                    episode_start_hour=current_episode_hours[0],
                    episode_end_hour=current_episode_hours[-1],
                    episode_duration_hours=float(
                        current_episode_hours[-1] - current_episode_hours[0] + 1
                    ),
                    n_positive_predictions=len(current_episode_hours),
                    is_true_alert=patient_has_sepsis,
                    is_false_alert=not patient_has_sepsis,
                )
            )
            current_episode_hours = [curr_hour]

    # Don't forget the last episode
    episodes.append(
        AlertEpisode(
            patient_id=patient_id,
            source_dataset=source_dataset,
            episode_start_hour=current_episode_hours[0],
            episode_end_hour=current_episode_hours[-1],
            episode_duration_hours=float(
                current_episode_hours[-1] - current_episode_hours[0] + 1
            ),
            n_positive_predictions=len(current_episode_hours),
            is_true_alert=patient_has_sepsis,
            is_false_alert=not patient_has_sepsis,
        )
    )

    # Filter by minimum duration and minimum consecutive predictions
    filtered_episodes: list[AlertEpisode] = []
    for ep in episodes:
        if ep.episode_duration_hours >= min_duration_hours and ep.n_positive_predictions >= min_consecutive:
            filtered_episodes.append(ep)

    return filtered_episodes


def compute_alert_burden(
    predictions_df: pd.DataFrame,
    model_name: str,
    model_version: str,
    split: str,
    threshold: float = 0.5,
    cooldown_hours: int = 6,
    min_duration_hours: int = 1,
    min_consecutive: int = 1,
    source: str = "synthetic",
) -> AlertBurdenResult:
    """Compute alert burden metrics for all patients.

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
    cooldown_hours : int
        Minimum hours between alert episodes.
    min_duration_hours : int
        Minimum duration for an alert episode.
    min_consecutive : int
        Minimum consecutive positive predictions for an episode.
    source : str
        Data source label.

    Returns
    -------
    AlertBurdenResult
        Alert burden result.
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
        src_ds = (
            str(row["source_dataset"])
            if "source_dataset" in predictions_df.columns
            and pd.notna(row["source_dataset"])
            else None
        )

        if pid not in patient_entries:
            patient_entries[pid] = []
            patient_source[pid] = src_ds

        patient_entries[pid].append((hour, prob, label))

    # Compute episodes for each patient
    all_episodes: list[AlertEpisode] = []
    for pid in sorted(patient_entries.keys()):
        episodes = compute_alert_episodes(
            pid,
            patient_source[pid],
            patient_entries[pid],
            threshold=threshold,
            cooldown_hours=cooldown_hours,
            min_duration_hours=min_duration_hours,
            min_consecutive=min_consecutive,
        )
        all_episodes.extend(episodes)

    # Compute aggregate statistics
    n_patients = len(patient_entries)
    n_alert_episodes = len(all_episodes)
    n_true_alert_episodes = sum(1 for e in all_episodes if e.is_true_alert)
    n_false_alert_episodes = sum(1 for e in all_episodes if e.is_false_alert)
    n_patients_with_false_alert = len(
        {e.patient_id for e in all_episodes if e.is_false_alert}
    )

    # Patient-days: sum of (max_hour - min_hour + 1) / 24 for each patient
    patient_days = 0.0
    for pid, preds in patient_entries.items():
        hours = [p[0] for p in preds]
        patient_days += (max(hours) - min(hours) + 1) / 24.0

    false_alerts_per_patient_day = (
        n_false_alert_episodes / patient_days if patient_days > 0 else float("nan")
    )
    alerts_per_patient = n_alert_episodes / n_patients if n_patients > 0 else float("nan")

    mean_alert_duration = (
        float(np.mean([e.episode_duration_hours for e in all_episodes]))
        if all_episodes
        else float("nan")
    )

    return AlertBurdenResult(
        model_name=model_name,
        model_version=model_version,
        split=split,
        n_patients=n_patients,
        n_patient_days=patient_days,
        n_alert_episodes=n_alert_episodes,
        n_true_alert_episodes=n_true_alert_episodes,
        n_false_alert_episodes=n_false_alert_episodes,
        n_patients_with_false_alert=n_patients_with_false_alert,
        false_alerts_per_patient_day=false_alerts_per_patient_day,
        alerts_per_patient=alerts_per_patient,
        mean_alert_duration_hours=mean_alert_duration,
        episodes=all_episodes,
        source=source,
    )


def validate_alert_burden_result(
    result: AlertBurdenResult,
) -> dict:
    """Validate an AlertBurdenResult.

    Returns a dict with validation results.
    """
    issues: list[str] = []

    if result.n_patients < 0:
        issues.append("n_patients is negative")

    if (
        result.n_true_alert_episodes + result.n_false_alert_episodes
        != result.n_alert_episodes
    ):
        issues.append(
            "n_true_alert_episodes + n_false_alert_episodes != n_alert_episodes"
        )

    if result.n_patients_with_false_alert > result.n_patients:
        issues.append("n_patients_with_false_alert exceeds n_patients")

    if not np.isfinite(result.false_alerts_per_patient_day) and not np.isnan(result.false_alerts_per_patient_day):
            issues.append("false_alerts_per_patient_day is not finite")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_alert_episodes": result.n_alert_episodes,
    }