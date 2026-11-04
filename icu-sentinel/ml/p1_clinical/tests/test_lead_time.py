"""Tests for lead_time.py.

Tests lead-time calculations based on official PhysioNet/CinC Challenge 2019
label convention. Does not shift labels or reconstruct sepsis onset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.evaluation.lead_time import (
    compute_lead_time,
    compute_lead_time_per_patient,
    validate_lead_time_result,
)


def test_compute_lead_time_per_patient_basic():
    """Test basic lead-time computation for a single patient."""
    # Patient with alert at hour 10, first positive label at hour 15
    # Lead time = 15 - 10 = 5 hours
    predictions = [
        (1, 0.1, 0), (2, 0.1, 0), (3, 0.1, 0), (4, 0.1, 0), (5, 0.1, 0),
        (6, 0.1, 0), (7, 0.1, 0), (8, 0.1, 0), (9, 0.1, 0),
        (10, 0.8, 0), (11, 0.8, 0), (12, 0.8, 0),
        (13, 0.8, 0), (14, 0.8, 0), (15, 0.9, 1), (16, 0.9, 1),
    ]

    result = compute_lead_time_per_patient(
        "P001", "set_a", predictions, threshold=0.5
    )

    assert result.patient_id == "P001"
    assert result.first_alert_hour == 10
    assert result.first_positive_label_hour == 15
    assert result.lead_time_hours == 5.0
    assert result.alert_before_onset is True
    assert result.alert_after_onset is False
    assert result.missed is False


def test_compute_lead_time_per_patient_no_alert():
    """Test patient with no alert (missed)."""
    predictions = [
        (1, 0.1, 0), (2, 0.1, 0), (3, 0.1, 1), (4, 0.1, 1),
    ]

    result = compute_lead_time_per_patient(
        "P002", "set_a", predictions, threshold=0.5
    )

    assert result.first_alert_hour is None
    assert result.first_positive_label_hour == 3
    assert result.lead_time_hours is None
    assert result.alert_before_onset is False
    assert result.alert_after_onset is False
    assert result.missed is True


def test_compute_lead_time_per_patient_no_sepsis():
    """Test non-sepsis patient with false alert."""
    predictions = [
        (1, 0.1, 0), (2, 0.8, 0), (3, 0.8, 0), (4, 0.1, 0),
    ]

    result = compute_lead_time_per_patient(
        "P003", "set_a", predictions, threshold=0.5
    )

    assert result.first_alert_hour == 2
    assert result.first_positive_label_hour is None
    assert result.lead_time_hours is None
    assert result.alert_before_onset is False
    assert result.alert_after_onset is True  # False alert
    assert result.missed is False  # Not a missed sepsis case


def test_compute_lead_time_per_patient_no_sepsis_no_alert():
    """Test non-sepsis patient with no alert."""
    predictions = [
        (1, 0.1, 0), (2, 0.1, 0), (3, 0.1, 0),
    ]

    result = compute_lead_time_per_patient(
        "P004", "set_a", predictions, threshold=0.5
    )

    assert result.first_alert_hour is None
    assert result.first_positive_label_hour is None
    assert result.lead_time_hours is None
    assert result.alert_before_onset is False
    assert result.alert_after_onset is False
    assert result.missed is False


def test_compute_lead_time_full():
    """Test full lead-time computation for multiple patients."""
    # Build predictions dataframe
    rows = []
    for pid in ["P001", "P002", "P003"]:
        for hour in range(1, 11):
            prob = 0.8 if (pid == "P001" and hour >= 5) else 0.1
            label = 1 if (pid == "P001" and hour >= 8) else 0
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob})

    df = pd.DataFrame(rows)

    result = compute_lead_time(
        df, "test_model", "1.0.0", "test", threshold=0.5, source="synthetic",
    )

    assert result.model_name == "test_model"
    assert result.model_version == "1.0.0"
    assert result.split == "test"
    assert result.source == "synthetic"
    assert result.n_patients == 3
    assert result.n_with_alert >= 1
    assert result.n_with_onset >= 1


def test_validate_lead_time_result_valid():
    """Test validation of valid lead-time result."""
    rng = np.random.default_rng(42)
    rows = []
    for pid in ["P001", "P002"]:
        for hour in range(1, 6):
            prob = rng.random()
            label = int(rng.random() < 0.3)
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob})

    df = pd.DataFrame(rows)
    result = compute_lead_time(df, "test_model", "1.0.0", "test",
                               source="synthetic")

    validation = validate_lead_time_result(result)
    assert validation["valid"] is True
    assert len(validation["issues"]) == 0


def test_validate_lead_time_result_invalid():
    """Test validation of invalid lead-time result."""
    rng = np.random.default_rng(42)
    rows = []
    for pid in ["P001", "P002"]:
        for hour in range(1, 6):
            prob = rng.random()
            label = int(rng.random() < 0.3)
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob})

    df = pd.DataFrame(rows)
    result = compute_lead_time(df, "test_model", "1.0.0", "test",
                               source="synthetic")

    # Corrupt result
    result.n_alert_before_onset = result.n_patients + 10

    validation = validate_lead_time_result(result)
    assert validation["valid"] is False
    assert len(validation["issues"]) > 0


def test_lead_time_label_shift_respected():
    """Test that SepsisLabel is not shifted again."""
    # SepsisLabel already incorporates 6-hour shift.
    # We use it as-is without additional shifting.
    predictions = [
        (1, 0.1, 0), (2, 0.1, 0), (3, 0.1, 0),
        (4, 0.9, 1), (5, 0.9, 1), (6, 0.9, 1),
    ]

    result = compute_lead_time_per_patient(
        "P001", "set_a", predictions, threshold=0.5
    )

    # First alert at hour 4, first positive label at hour 4
    # Lead time = 0 (alert at same hour as first positive label)
    assert result.first_alert_hour == 4
    assert result.first_positive_label_hour == 4
    assert result.lead_time_hours == 0.0
    assert result.alert_before_onset is True