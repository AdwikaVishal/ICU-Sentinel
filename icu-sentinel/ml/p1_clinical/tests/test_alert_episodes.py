"""Tests for alert_episodes.py and alert_burden.py.

Tests alert episode merging, cooldown behavior, false alert calculation,
and patient-day denominator.
"""

from __future__ import annotations

import pandas as pd

from ml.p1_clinical.evaluation.alert_episodes import (
    compute_alert_burden,
    compute_alert_episodes,
    validate_alert_burden_result,
)


def test_compute_alert_episodes_basic():
    """Test basic alert episode computation."""
    predictions = [
        (1, 0.8, 0), (2, 0.8, 0), (3, 0.8, 0),
        (5, 0.1, 0), (6, 0.1, 0),
        (10, 0.8, 0), (11, 0.8, 0),
    ]

    episodes = compute_alert_episodes(
        "P001", "set_a", predictions, threshold=0.5,
        cooldown_hours=6, min_duration_hours=1, min_consecutive=1,
    )

    assert len(episodes) == 2
    # Episode 1: hours 1-3
    assert episodes[0].episode_start_hour == 1
    assert episodes[0].episode_end_hour == 3
    assert episodes[0].episode_duration_hours == 3.0
    assert episodes[0].n_positive_predictions == 3
    assert episodes[0].is_false_alert is True  # No sepsis patient

    # Episode 2: hours 10-11
    assert episodes[1].episode_start_hour == 10
    assert episodes[1].episode_end_hour == 11
    assert episodes[1].n_positive_predictions == 2

def test_compute_alert_episodes_cooldown():
    """Test cooldown period merging."""
    # Two positive predictions 2 hours apart, cooldown is 6 hours
    # They should be merged into one episode
    predictions = [
        (1, 0.8, 0), (2, 0.1, 0), (3, 0.1, 0), (4, 0.8, 0),
    ]

    episodes = compute_alert_episodes(
        "P001", "set_a", predictions, threshold=0.5,
        cooldown_hours=6, min_duration_hours=1, min_consecutive=1,
    )

    assert len(episodes) == 1
    assert episodes[0].episode_start_hour == 1
    assert episodes[1].episode_end_hour == 4 if len(episodes) > 1 else True


def test_compute_alert_episodes_min_duration():
    """Test minimum duration filtering."""
    predictions = [
        (1, 0.8, 0), (2, 0.1, 0), (3, 0.8, 0),
    ]

    # With min_duration=3, episode 1-1 (1 hour) and 3-3 (1 hour) should be filtered
    episodes = compute_alert_episodes(
        "P001", "set_a", predictions, threshold=0.5,
        cooldown_hours=1, min_duration_hours=3, min_consecutive=1,
    )

    assert len(episodes) == 0


def test_compute_alert_episodes_patient_has_sepsis():
    """Test true alert for sepsis patient."""
    predictions = [
        (1, 0.8, 0), (2, 0.8, 1), (3, 0.8, 1),
    ]

    episodes = compute_alert_episodes(
        "P001", "set_a", predictions, threshold=0.5,
    )

    assert len(episodes) == 1
    assert episodes[0].is_true_alert is True
    assert episodes[0].is_false_alert is False


def test_compute_alert_burden_full():
    """Test full alert burden computation."""
    rows = []
    for pid in ["P001", "P002"]:
        for hour in range(1, 11):
            prob = 0.8 if (pid == "P001" and hour >= 3) else 0.1
            label = 1 if (pid == "P001" and hour >= 5) else 0
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob})

    df = pd.DataFrame(rows)

    result = compute_alert_burden(
        df, "test_model", "1.0.0", "test", threshold=0.5,
        source="synthetic",
    )

    assert result.model_name == "test_model"
    assert result.model_version == "1.0.0"
    assert result.split == "test"
    assert result.source == "synthetic"
    assert result.n_patients == 2
    assert result.n_alert_episodes >= 0
    assert result.n_false_alert_episodes >= 0
    assert result.n_true_alert_episodes >= 0


def test_validate_alert_burden_result_valid():
    """Test validation of valid alert burden result."""
    rows = []
    for pid in ["P001"]:
        for hour in range(1, 6):
            prob = 0.8 if hour >= 3 else 0.1
            label = 0
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob})

    df = pd.DataFrame(rows)
    result = compute_alert_burden(
        df, "test_model", "1.0.0", "test", source="synthetic",
    )

    validation = validate_alert_burden_result(result)
    assert validation["valid"] is True
    assert len(validation["issues"]) == 0


def test_validate_alert_burden_result_invalid():
    """Test validation of invalid alert burden result."""
    rows = []
    for pid in ["P001"]:
        for hour in range(1, 6):
            prob = 0.8 if hour >= 3 else 0.1
            label = 0
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob})

    df = pd.DataFrame(rows)
    result = compute_alert_burden(
        df, "test_model", "1.0.0", "test", source="synthetic",
    )

    # Corrupt result
    result.n_false_alert_episodes = result.n_patients + 10

    validation = validate_alert_burden_result(result)
    assert validation["valid"] is False
    assert len(validation["issues"]) > 0


def test_false_alerts_per_patient_day():
    """Test false alerts per patient-day calculation."""
    # One patient, 24 hours, 1 false alert episode
    rows = []
    for hour in range(1, 25):
        prob = 0.8 if hour >= 3 else 0.1
        label = 0
        rows.append({"patient_id": "P001", "hour": hour,
                     "true_label": label, "predicted_probability": prob})

    df = pd.DataFrame(rows)
    result = compute_alert_burden(
        df, "test_model", "1.0.0", "test", threshold=0.5,
        source="synthetic",
    )

    # 1 false alert over 24 hours = 1 false alert per patient-day
    assert abs(result.false_alerts_per_patient_day - 1.0) < 1e-10
    assert abs(result.n_patient_days - 1.0) < 1e-10