"""Tests for alert burden evaluation in P1 research evaluation pipeline.

Tests alert episode aggregation, cooldown behavior, false alert calculation,
and false alerts per patient-day calculations.

All tests use synthetic data. No real patient data is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.evaluation.alert_burden import (
    compute_alert_burden,
)
from ml.p1_clinical.evaluation.alert_episodes import (
    compute_alert_episodes,
)


def _make_patient_timeline(
    patient_id: str = "p00001",
    hours: list[int] | None = None,
    predictions: list[float] | None = None,
    labels: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic patient timeline DataFrame for alert testing."""
    rng = np.random.default_rng(seed)
    if hours is None:
        hours = list(range(1, 25))
    n = len(hours)
    if predictions is None:
        predictions = rng.random(n).tolist()
    if labels is None:
        labels = [0] * n
    return pd.DataFrame(
        {
            "patient_id": [patient_id] * n,
            "hour": hours,
            "true_label": labels,
            "predicted_probability": predictions,
            "predicted_class": [int(p >= 0.5) for p in predictions],
        }
    )


# ------------------------------------------------------------------
# Alert episode construction tests
# ------------------------------------------------------------------


def test_compute_alert_episodes_basic():
    """Test basic alert episode detection."""
    predictions = [
        (1, 0.8, 0), (2, 0.8, 0), (3, 0.8, 0),
        (10, 0.1, 0), (11, 0.1, 0),
        (14, 0.8, 0), (15, 0.8, 0),
    ]
    episodes = compute_alert_episodes(
        "P001", "set_a", predictions, threshold=0.5,
        cooldown_hours=6, min_duration_hours=1, min_consecutive=1,
    )
    assert len(episodes) == 2
    # Episode 1: hours 1-3
    assert episodes[0].episode_start_hour == 1
    assert episodes[0].episode_end_hour == 3
    assert episodes[0].n_positive_predictions == 3
    assert episodes[0].is_false_alert is True  # No sepsis patient

    # Episode 2: hours 14-15
    assert episodes[1].episode_start_hour == 14
    assert episodes[1].episode_end_hour == 15
    assert episodes[1].n_positive_predictions == 2
    assert episodes[1].is_false_alert is True


def test_alert_episodes_cooldown_merges():
    """Test that observations within cooldown window merge into one episode."""
    predictions = [
        (1, 0.8, 0), (2, 0.8, 0), (3, 0.8, 0), (4, 0.8, 0),
    ]
    episodes = compute_alert_episodes(
        "P001", "set_a", predictions, threshold=0.5,
        cooldown_hours=6, min_duration_hours=1, min_consecutive=1,
    )
    assert len(episodes) == 1
    assert episodes[0].episode_start_hour == 1
    assert episodes[0].episode_end_hour == 4


def test_alert_episodes_separated_by_cooldown():
    """Test that gaps larger than cooldown create separate episodes."""
    predictions = [
        (1, 0.8, 0), (2, 0.8, 0),
        (10, 0.8, 0), (11, 0.8, 0),
    ]
    episodes = compute_alert_episodes(
        "P001", "set_a", predictions, threshold=0.5,
        cooldown_hours=6, min_duration_hours=1, min_consecutive=1,
    )
    assert len(episodes) == 2


def test_alert_episodes_min_duration():
    """Test minimum duration filtering."""
    predictions = [
        (1, 0.8, 0), (2, 0.1, 0), (3, 0.8, 0),
    ]
    episodes = compute_alert_episodes(
        "P001", "set_a", predictions, threshold=0.5,
        cooldown_hours=1, min_duration_hours=3, min_consecutive=1,
    )
    assert len(episodes) == 0  # Both episodes are 1 hour, below min duration


def test_compute_alert_burden_basic():
    """Test alert burden computation."""
    rows = []
    for pid in ["P001", "P002"]:
        for hour in range(1, 25):
            prob = 0.8 if (hour >= 3 and pid == "P001") else 0.1
            label = 1 if (pid == "P001" and hour >= 4) else 0
            rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label, "predicted_probability": prob,
                "predicted_class": int(prob >= 0.5),
            })
    df = pd.DataFrame(rows)

    result = compute_alert_burden(
        df, "test_model", "1.0.0", "test", threshold=0.5,
        source="synthetic",
    )

    assert result.model_name == "test_model"
    assert result.split == "test"
    assert result.n_patients == 2
    assert result.n_patient_days == 2.0
    assert result.n_alert_episodes >= 0
    assert result.false_alerts_per_patient_day >= 0
    assert result.alerts_per_patient >= 0


def test_alert_burden_validation():
    """Test validation of alert burden result."""
    from ml.p1_clinical.evaluation.alert_burden import validate_alert_burden_result

    rows = []
    for pid in ["P001"]:
        for hour in range(1, 25):
            prob = 0.8 if hour >= 3 else 0.1
            label = 0
            rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label, "predicted_probability": prob,
                "predicted_class": int(prob >= 0.5),
            })
    df = pd.DataFrame(rows)

    result = compute_alert_burden(
        df, "test_model", "1.0.0", "test", source="synthetic",
    )

    validation = validate_alert_burden_result(result)
    assert validation["valid"] is True
    assert len(validation["issues"]) == 0