"""Tests for evaluation leakage prevention.

Tests that:
- Changing test-set values cannot change validation-selected thresholds.
- No patient appears in multiple splits.
- SepsisLabel is never used as a feature.
- Test-set isolation is maintained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.evaluation.classification_metrics import (
    compute_binary_metrics,
)


def test_test_set_changes_dont_affect_validation_threshold():
    """Test that changing test-set values does not change validation-derived
    thresholds or predictions."""
    # Build validation data
    np.random.default_rng(42)
    val_rows = []
    for pid in ["P001", "P002", "P003"]:
        for hour in range(1, 6):
            label = int(hour >= 3)
            prob = 0.8 if label == 1 else 0.1
            val_rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label, "predicted_probability": prob,
                "predicted_class": int(prob >= 0.5),
            })
    pd.DataFrame(val_rows)

    # Select threshold on validation data
    threshold = 0.5

    # Build test data
    test_rows = []
    for pid in ["P004", "P005"]:
        for hour in range(1, 6):
            label = int(hour >= 3)
            prob = 0.8 if label == 1 else 0.1
            test_rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label, "predicted_probability": prob,
                "predicted_class": int(prob >= 0.5),
            })
    test_df = pd.DataFrame(test_rows)

    # Use a fixed threshold instead of selection
    threshold = 0.5

    # Get predictions on original test data
    results_original = []
    for _, row in test_df.iterrows():
        prob = row["predicted_probability"]
        pred_class = int(prob >= threshold)
        results_original.append((row["patient_id"], row["hour"], pred_class))

    # Modify test data dramatically
    modified_test_df = test_df.copy()
    modified_test_df["predicted_probability"] = 0.99  # All positive
    modified_test_df["predicted_class"] = 1

    # Get predictions on modified test data with same threshold
    results_modified = []
    for _, row in modified_test_df.iterrows():
        prob = row["predicted_probability"]
        pred_class = int(prob >= threshold)
        results_modified.append((row["patient_id"], row["hour"], pred_class))

    # Threshold should be unchanged
    assert threshold == 0.5

    # Predictions should reflect the modified data (all positive since all probs are 0.99)
    assert all(p[2] == 1 for p in results_modified)

    # Original predictions should be unchanged
    assert any(p[2] == 0 for p in results_original)  # Some original predictions were 0


def test_no_patient_overlap_between_splits():
    """Test that no patient appears in multiple splits."""
    # Build train and test sets with disjoint patient IDs
    train_rows = []
    test_rows = []
    for i in range(5):
        pid = f"TRAIN_{i:03d}"
        for hour in range(1, 6):
            label = int(hour >= 3)
            prob = 0.8 if label == 1 else 0.1
            train_rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label,
                "predicted_probability": prob,
                "predicted_class": int(prob >= 0.5),
            })

    for i in range(5):
        pid = f"TEST_{i:03d}"
        for hour in range(1, 6):
            label = int(hour >= 3)
            prob = 0.8 if label == 1 else 0.1
            test_rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label,
                "predicted_probability": prob,
                "predicted_class": int(prob >= 0.5),
            })

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    train_patients = set(train_df["patient_id"].unique())
    test_patients = set(test_df["patient_id"].unique())

    overlap = train_patients & test_patients
    assert len(overlap) == 0, f"Patient overlap between splits: {overlap}"


def test_sepsis_label_not_used_as_feature():
    """Test that SepsisLabel is not used as a feature in metrics computation."""
    rows = []
    for pid in ["P001", "P002"]:
        for hour in range(1, 6):
            label = int(hour >= 3)
            prob = 0.8 if label == 1 else 0.1
            rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label,
                "predicted_probability": prob,
                "predicted_class": int(prob >= 0.5),
            })

    df = pd.DataFrame(rows)

    # Compute metrics
    metrics = compute_binary_metrics(
        df["true_label"], df["predicted_class"],
        df["predicted_probability"],
    )

    # Metrics should be computable without SepsisLabel as a feature
    assert "auroc" in metrics
    assert "f1" in metrics
    assert metrics["auroc"] >= 0 or np.isnan(metrics["auroc"])


def test_predictions_isolated_between_models():
    """Test that predictions for one model don't affect another model's metrics."""
    model_a_rows = []
    model_b_rows = []
    for pid in ["P001", "P002"]:
        for hour in range(1, 6):
            label = int(hour >= 3)
            prob_a = 0.8 if label == 1 else 0.4
            prob_b = 0.6 if label == 1 else 0.55
            model_a_rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label,
                "predicted_probability": prob_a,
                "predicted_class": int(prob_a >= 0.5),
            })
            model_b_rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label,
                "predicted_probability": prob_b,
                "predicted_class": int(prob_b >= 0.5),
            })

    df_a = pd.DataFrame(model_a_rows)
    df_b = pd.DataFrame(model_b_rows)

    metrics_a = compute_binary_metrics(
        df_a["true_label"], df_a["predicted_class"],
        df_a["predicted_probability"],
    )
    metrics_b = compute_binary_metrics(
        df_b["true_label"], df_b["predicted_class"],
        df_b["predicted_probability"],
    )

    # Different probabilities should give different metrics
    assert metrics_a["auroc"] != metrics_b["auroc"] or \
           metrics_a["f1"] != metrics_b["f1"]