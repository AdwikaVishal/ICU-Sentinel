# P2/P3 Evaluation Handoff — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 7 — Research Evaluation  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Overview

This document describes the evaluation interface that P1 provides for P2
(boosting models: Extra Trees, LightGBM, XGBoost, CatBoost) and P3
(temporal models: GRU, LSTM, Transformer) to integrate their models into
the ICU Sentinel evaluation framework.

---

## 2. Prediction Input Contract

P2 and P3 must provide predictions in the following format:

### Prediction Record Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `patient_id` | string | Yes | Patient identifier |
| `hour` | int | Yes | ICU hour since admission |
| `true_label` | int (0/1) | Yes | True SepsisLabel |
| `predicted_probability` | float | Yes | Predicted probability of sepsis |
| `predicted_class` | int (0/1) | Yes | Predicted class (probability >= threshold) |
| `model_name` | string | Yes | Model name |
| `model_version` | string | Yes | Model version |
| `split` | string | Yes | Split identity (train/validation/test) |
| `source_dataset` | string | No | Source dataset identity |
| `threshold` | float | No | Decision threshold used |
| `features_used` | list[str] | No | Features used for prediction |

### Prediction Batch Format

Predictions are grouped by model, version, and split:

```json
{
  "model_name": "lightgbm",
  "model_version": "1.0.0",
  "split": "test",
  "n_predictions": 10000,
  "n_positive_true": 500,
  "n_positive_pred": 600,
  "predictions": [...]
}
```

---

## 3. Evaluation API

P2 and P3 can use the `P1 evaluation pipeline` to evaluate their models:

```python
from ml.p1_clinical.evaluation.evaluation_pipeline import run_evaluation_pipeline

result = run_evaluation_pipeline(
    predictions_by_model={
        "lightgbm": lgb_predictions_df,
        "xgboost": xgb_predictions_df,
    },
    split="test",
    config=EvaluationConfig(),
    source="real",
)

# Access results
for split_result in result.model_comparison_results:
    for model_entry in split_result.models:
        print(f"{model_entry.model_name}: AUROC = {model_entry.metrics.get('auroc')}")
```

---

## 4. Output Format

All evaluation outputs are saved as JSON files under
`data/reports/research_evaluation/`:

| File | Description |
|------|-------------|
| `classification_metrics.json` | Per-model classification metrics |
| `calibration_report.json` | Calibration results (ECE, MCE) |
| `lead_time_report.json` | Lead-time evaluation results |
| `alert_burden_report.json` | Alert burden (false alerts per patient-day) |
| `feature_ablation_report.json` | Feature ablation results |
| `subgroup_analysis_report.json` | Subgroup analysis results |
| `bootstrap_confidence_intervals.json` | Bootstrap confidence intervals |
| `cross_hospital_report.json` | Cross-hospital evaluation results |
| `model_comparison.json` | Model comparison across all models |
| `final_evaluation_summary.json` | Final evaluation summary |

---

## 5. Evaluation Metrics

P2 and P3 models are evaluated using the same metrics as P1 classical ML models:

| Metric | Description |
|--------|-------------|
| AUROC | Area Under ROC Curve |
| AUPRC | Area Under Precision-Recall Curve |
| Sensitivity | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Precision | TP / (TP + FP) |
| F1-score | 2 × (Precision × Recall) / (Precision + Recall) |
| Accuracy | (TP + TN) / Total |
| Brier score | Mean squared probability error |

---

## 6. Data Integrity

- All evaluation reports generated from synthetic fixtures are marked
  `"source": "synthetic"`
- No clinical validation claims are made from synthetic experiments
- Real data evaluation requires PhysioNet/CinC Challenge 2019 access
- Patient-level bootstrap resampling preserves patient-level correlation
- Cross-hospital evaluation requires genuine hospital identifiers
