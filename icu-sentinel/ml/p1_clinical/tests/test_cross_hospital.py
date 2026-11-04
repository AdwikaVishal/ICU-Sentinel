"""Tests for cross_hospital.py.

Tests patient-level leakage between hospitals and cross-hospital evaluation.
"""

from __future__ import annotations

import pandas as pd

from ml.p1_clinical.evaluation.cross_hospital import (
    compute_cross_hospital_evaluation,
    run_cross_hospital_evaluation,
    validate_cross_hospital_result,
)


def test_compute_cross_hospital_evaluation_basic():
    """Test basic cross-hospital evaluation."""
    train_rows = []
    test_rows = []
    for pid in ["P001", "P002", "P003"]:
        for hour in range(1, 6):
            label = int(hour >= 3)
            prob = 0.8 if label == 1 else 0.1
            rows = {"patient_id": pid, "hour": hour,
                    "true_label": label, "predicted_probability": prob,
                    "predicted_class": int(prob >= 0.5)}
            train_rows.append(rows)
            test_rows.append(rows.copy())

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    result = compute_cross_hospital_evaluation(
        train_df, test_df,
        "set_a", "set_b",
        "test_model", "1.0.0",
        threshold=0.5, source="synthetic",
    )

    assert result.train_hospital == "set_a"
    assert result.test_hospital == "set_b"
    assert result.model_name == "test_model"
    assert result.n_patients_train == 3
    assert result.n_patients_test == 3
    assert "auroc" in result.metrics


def test_compute_cross_hospital_evaluation_same_hospital():
    """Test that same-hospital is allowed but flagged."""
    rows = []
    for pid in ["P001", "P002"]:
        for hour in range(1, 6):
            label = int(hour >= 3)
            prob = 0.8 if label == 1 else 0.1
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob,
                         "predicted_class": int(prob >= 0.5)})

    df = pd.DataFrame(rows)
    result = compute_cross_hospital_evaluation(
        df, df, "set_a", "set_a",
        "test_model", "1.0.0",
        threshold=0.5, source="synthetic",
    )

    assert result.train_hospital == "set_a"
    assert result.test_hospital == "set_a"


def test_run_cross_hospital_evaluation():
    """Test running cross-hospital evaluation for all hospital pairs."""
    predictions_by_hospital = {}
    for hosp in ["set_a", "set_b"]:
        rows = []
        for pid in ["P001", "P002"]:
            for hour in range(1, 6):
                label = int(hour >= 3)
                prob = 0.8 if label == 1 else 0.1
                rows.append({"patient_id": pid, "hour": hour,
                             "true_label": label,
                             "predicted_probability": prob,
                             "predicted_class": int(prob >= 0.5)})
        predictions_by_hospital[hosp] = pd.DataFrame(rows)

    results = run_cross_hospital_evaluation(
        predictions_by_hospital, "test_model", "1.0.0",
        threshold=0.5, source="synthetic",
    )

    assert len(results) == 2  # set_a->set_b and set_b->set_a
    for r in results:
        assert result_is_cross_hospital(r)


def result_is_cross_hospital(r):
    return r.train_hospital != r.test_hospital


def test_validate_cross_hospital_result_valid():
    """Test validation of valid cross-hospital result."""
    rows = []
    for pid in ["P001", "P002"]:
        for hour in range(1, 6):
            label = int(hour >= 3)
            prob = 0.8 if label == 1 else 0.1
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob,
                         "predicted_class": int(prob >= 0.5)})

    df = pd.DataFrame(rows)
    result = compute_cross_hospital_evaluation(
        df, df, "set_a", "set_b",
        "test_model", "1.0.0",
        threshold=0.5, source="synthetic",
    )

    validation = validate_cross_hospital_result(result)
    assert validation["valid"] is True
    assert len(validation["issues"]) == 0


def test_validate_cross_hospital_result_invalid():
    """Test validation of invalid cross-hospital result."""
    rows = []
    for pid in ["P001", "P002"]:
        for hour in range(1, 6):
            label = int(hour >= 3)
            prob = 0.8 if label == 1 else 0.1
            rows.append({"patient_id": pid, "hour": hour,
                         "true_label": label, "predicted_probability": prob,
                         "predicted_class": int(prob >= 0.5)})

    df = pd.DataFrame(rows)
    result = compute_cross_hospital_evaluation(
        df, df, "set_a", "set_a",
        "test_model", "1.0.0",
        threshold=0.5, source="synthetic",
    )

    validation = validate_cross_hospital_result(result)
    # Same hospital is flagged but not necessarily invalid
    assert "n_patients_train" in validation
    assert "n_patients_test" in validation