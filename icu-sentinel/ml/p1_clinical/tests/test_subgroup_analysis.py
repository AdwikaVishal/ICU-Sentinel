"""Tests for subgroup_analysis.py.

Tests subgroup sample counts, small subgroup handling, and metric computation.
"""

from __future__ import annotations

import pandas as pd

from ml.p1_clinical.evaluation.subgroup_analysis import (
    compute_subgroup_analysis,
    validate_subgroup_analysis_result,
)


def test_compute_subgroup_analysis_basic():
    """Test basic subgroup analysis computation."""
    rows = []
    for pid in ["P001", "P002", "P003", "P004"]:
        for hour in range(1, 6):
            gender = "M" if pid in ["P001", "P003"] else "F"
            prob = 0.8 if (pid == "P001" and hour >= 3) else 0.1
            label = 1 if (pid == "P001" and hour >= 4) else 0
            rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label, "predicted_probability": prob,
                "predicted_class": int(prob >= 0.5),
                "gender": gender,
            })

    df = pd.DataFrame(rows)

    result = compute_subgroup_analysis(
        df, "test_model", "1.0.0", "test",
        subgroup_dimension="gender",
        subgroup_column="gender",
        threshold=0.5, source="synthetic",
    )

    assert result.model_name == "test_model"
    assert result.subgroup_dimension == "gender"
    assert len(result.subgroups) == 2  # M and F

    male_subgroup = next(s for s in result.subgroups if s.subgroup_value == "M")
    female_subgroup = next(s for s in result.subgroups if s.subgroup_value == "F")

    assert male_subgroup.n_patients == 2
    assert female_subgroup.n_patients == 2


def test_compute_subgroup_analysis_small_subgroup():
    """Test small subgroup handling."""
    rows = []
    # Only 1 patient in subgroup "Z"
    for hour in range(1, 6):
        rows.append({"patient_id": "P001", "hour": hour,
                     "true_label": 0, "predicted_probability": 0.1,
                     "predicted_class": 0, "gender": "Z"})
    # Many patients in subgroup "M"
    for pid in ["P002", "P003", "P004"]:
        for hour in range(1, 6):
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": 0, "predicted_probability": 0.1,
                         "predicted_class": 0, "gender": "M"})

    df = pd.DataFrame(rows)

    result = compute_subgroup_analysis(
        df, "test_model", "1.0.0", "test",
        subgroup_dimension="gender",
        subgroup_column="gender",
        threshold=0.5, source="synthetic",
        min_subgroup_size=3,
    )

    z_subgroup = next(s for s in result.subgroups if s.subgroup_value == "Z")
    assert z_subgroup.n_patients == 1
    assert z_subgroup.n_patients < 3  # Below min_subgroup_size


def test_validate_subgroup_analysis_result_valid():
    """Test validation of valid subgroup analysis result."""
    rows = []
    for pid in ["P001", "P002"]:
        for hour in range(1, 6):
            gender = "M" if pid == "P001" else "F"
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": 0, "predicted_probability": 0.1,
                         "predicted_class": 0, "gender": gender})

    df = pd.DataFrame(rows)
    result = compute_subgroup_analysis(
        df, "test_model", "1.0.0", "test",
        subgroup_dimension="gender",
        subgroup_column="gender",
        threshold=0.5, source="synthetic",
    )

    validation = validate_subgroup_analysis_result(result)
    assert validation["valid"] is True
    assert len(validation["issues"]) == 0


def test_validate_subgroup_analysis_result_invalid():
    """Test validation of invalid subgroup analysis result."""
    rows = []
    for pid in ["P001"]:
        for hour in range(1, 6):
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": 0, "predicted_probability": 0.1,
                         "predicted_class": 0, "gender": "M"})

    df = pd.DataFrame(rows)
    result = compute_subgroup_analysis(
        df, "test_model", "1.0.0", "test",
        subgroup_dimension="gender",
        subgroup_column="gender",
        threshold=0.5, source="synthetic",
    )

    # Corrupt result
    result.subgroups[0].n_positive_patients = result.subgroups[0].n_patients + 10

    validation = validate_subgroup_analysis_result(result)
    assert validation["valid"] is False
    assert len(validation["issues"]) > 0


def test_subgroup_metrics_computed():
    """Test that subgroup metrics are computed correctly."""
    rows = []
    for pid in ["P001"]:
        for hour in range(1, 6):
            prob = 0.8 if hour >= 3 else 0.1
            label = 1 if hour >= 4 else 0
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob,
                         "predicted_class": int(prob >= 0.5),
                         "gender": "M"})

    df = pd.DataFrame(rows)
    result = compute_subgroup_analysis(
        df, "test_model", "1.0.0", "test",
        subgroup_dimension="gender",
        subgroup_column="gender",
        threshold=0.5, source="synthetic",
    )

    subgroup = result.subgroups[0]
    assert subgroup.n_patients == 1
    assert subgroup.n_rows == 5
    assert "auroc" in subgroup.metrics
    assert "f1" in subgroup.metrics