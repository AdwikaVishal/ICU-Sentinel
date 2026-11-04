"""Tests for evaluation_pipeline.py.

Tests end-to-end evaluation pipeline with synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.evaluation.evaluation_config import EvaluationConfig
from ml.p1_clinical.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
    run_evaluation_pipeline,
)


def _make_prediction_df(n_patients=5, n_hours=5, seed=42):
    """Create synthetic prediction DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in [f"P{i:03d}" for i in range(n_patients)]:
        for hour in range(1, n_hours + 1):
            label = int(hour >= 3)
            prob = 0.8 if (label == 1 and rng.random() < 0.8) else 0.1
            rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label,
                "predicted_probability": prob,
                "predicted_class": int(prob >= 0.5),
                "source_dataset": "set_a" if int(pid[1:]) % 2 == 0 else "set_b",
            })
    return pd.DataFrame(rows)


def test_evaluation_pipeline_init():
    """Test pipeline initialization."""
    config = EvaluationConfig()
    pipeline = EvaluationPipeline(config=config, source="synthetic")
    assert pipeline.config is config
    assert pipeline.source == "synthetic"


def test_run_evaluation_pipeline_basic():
    """Test basic end-to-end evaluation pipeline."""
    predictions = {}
    for name in ["logistic_regression", "random_forest", "sirs"]:
        predictions[name] = _make_prediction_df(n_patients=5, n_hours=5,
                                                seed=42)

    result = run_evaluation_pipeline(
        predictions, "test",
        config=EvaluationConfig(),
        source="synthetic",
    )

    assert result.source == "synthetic"
    assert len(result.classification_metrics) == 3
    assert len(result.calibration_results) == 3
    assert len(result.lead_time_results) == 3
    assert len(result.alert_burden_results) == 3
    assert len(result.model_comparison_results) == 1


def test_run_evaluation_pipeline_model_types():
    """Test that model types are correctly classified."""
    predictions = {
        "logistic_regression": _make_prediction_df(n_patients=3, n_hours=5, seed=1),
        "sirs": _make_prediction_df(n_patients=3, n_hours=5, seed=2),
        "gru": _make_prediction_df(n_patients=3, n_hours=5, seed=3),
    }

    result = run_evaluation_pipeline(
        predictions, "test",
        config=EvaluationConfig(),
        source="synthetic",
    )

    comp = result.model_comparison_results[0]
    model_types = {e.model_name: e.model_type for e in comp.models}
    assert model_types["logistic_regression"] == "classical_ml"
    assert model_types["sirs"] == "clinical_baseline"
    assert model_types["gru"] == "temporal"


def test_evaluation_pipeline_with_subgroup_column():
    """Test pipeline with subgroup column in predictions."""
    rows = []
    for pid in ["P001", "P002"]:
        for hour in range(1, 6):
            gender = "M" if pid == "P001" else "F"
            prob = 0.8 if hour >= 3 else 0.1
            label = 1 if hour >= 4 else 0
            rows.append({
                "patient_id": pid, "hour": hour,
                "true_label": label, "predicted_probability": prob,
                "predicted_class": int(prob >= 0.5),
                "gender": gender,
            })

    df = pd.DataFrame(rows)
    predictions = {"test_model": df}

    result = run_evaluation_pipeline(
        predictions, "test",
        config=EvaluationConfig(),
        source="synthetic",
    )

    # Should have subgroup analysis for gender
    assert len(result.subgroup_results) >= 1
    subgroup = result.subgroup_results[0]
    assert subgroup.subgroup_dimension == "gender"


def test_evaluation_pipeline_source_tracking():
    """Test that source is tracked in all results."""
    predictions = {
        "test_model": _make_prediction_df(n_patients=3, n_hours=5, seed=42),
    }

    result = run_evaluation_pipeline(
        predictions, "test",
        config=EvaluationConfig(),
        source="real",
    )

    assert result.source == "real"
    for cm in result.classification_metrics:
        assert cm.source == "real"
    for lt in result.lead_time_results:
        assert lt.source == "real"
    for ab in result.alert_burden_results:
        assert ab.source == "real"
    for mc in result.model_comparison_results:
        assert mc.source == "real"