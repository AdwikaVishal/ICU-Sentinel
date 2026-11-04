# Feature Ablation Protocol — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 7 — Research Evaluation  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Overview

Feature ablation evaluates the contribution of individual feature groups to
model performance by removing one feature group at a time, retraining the
model, and comparing performance against the full feature model.

This helps identify which feature groups contribute most to model
performance and which features could potentially be removed without
significant performance degradation.

---

## 2. Feature Groups

Feature ablation tests the following feature groups defined in the Part 4
feature schema:

| Feature Group | Description | Example Features |
|---------------|-------------|------------------|
| **current** | Raw observation features | HR, Temp, SBP, etc. |
| **missingness** | Missingness indicators, time-since-last, availability | HR_is_missing, HR_hours_since_last, HR_availability_6h |
| **forward_fill** | Causally forward-filled features | HR_ffilled, Temp_ffilled, etc. |
| **rolling** | Rolling statistics (6/12/24h) | HR_rolling_mean_6h, Temp_rolling_max_24h |
| **trend** | Temporal slopes and changes | HR_abs_change_6h, Temp_rel_change_12h, HR_slope_24h |
| **clinical** | Derived clinical features | shock_index, sirs_count, sirs_components_available |

---

## 3. Ablation Methodology

For each feature group:

1. **Remove** the feature group from the feature set
2. **Retrain** the model using training data only
3. **Evaluate** the ablated model on the same frozen test population
4. **Compare** metrics against the full feature model
5. **Record** absolute and relative metric changes

### Important Constraints

- **No data leakage**: Feature ablation is performed on training data only. The ablation model is retrained using only training data, never validation or test data.
- **Same test population**: All ablated models are evaluated on the same frozen test population for fair comparison.
- **Same hyperparameters**: The same hyperparameters are used for the full model and all ablated models.
- **Feature ablation ≠ feature importance**: Feature ablation measures the impact of removing a feature group, which is different from model-based feature importance (e.g., Random Forest feature importance).
- **Feature ablation ≠ SHAP**: SHAP explanations are handled by P3 and are a different methodology.

---

## 6. Ablation Results Format

Feature ablation results are saved to
`data/reports/classical_ml/feature_ablation.json`:

```json
{
  "model_name": "random_forest",
  "model_version": "1.0.0",
  "split": "test",
  "n_features_full": 40,
  "ablations": [
    {
      "feature_group": "rolling",
      "n_features_removed": 120,
      "metrics_full": {
        "auroc": 0.85,
        "auprc": 0.52,
        "sensitivity": 0.75,
        "specificity": 0.82,
        "precision": 0.27,
        "f1": 0.39
      },
      "metrics_ablated": {
        "auroc": 0.78,
        "auprc": 0.42,
        "sensitivity": 0.68,
        "specificity": 0.75,
        "precision": 0.22,
        "f1": 0.33
      },
      "metric_changes": {
        "auroc": -0.07,
        "auprc": -0.10,
        "sensitivity": -0.07,
        "specificity": -0.07,
        "precision": -0.05,
        "f1": -0.06
      },
      "metric_changes_relative": {
        "auroc": -0.082,
        "auprc": -0.192,
        "sensitivity": -0.093,
        "specificity": -0.085,
        "precision": -0.227,
        "f1": -0.154
      },
      "source": "synthetic"
    }
  ],
  "source": "synthetic"
}
```

---

## 8. Interpretation

### Feature Importance from Ablation

Feature ablation provides a **model-agnostic** measure of feature importance:
- A large drop in performance when removing a feature group indicates that
  the group is important for model performance.
- A small drop indicates that the feature group is less important or
  redundant with other features.

### Limitations

- **Feature ablation is not causal**: Removing a feature group changes the
  feature space, which may affect model behavior in unexpected ways.
- **Feature ablation is not SHAP**: SHAP provides instance-level
  explanations, while feature ablation provides group-level importance.
- **Feature ablation is not feature selection**: Removing a feature group
  during ablation does not mean that feature group should be removed from
  the final model.

---

## 8. Data Integrity

- All ablation reports generated from synthetic fixtures are marked
  `"source": "synthetic"`
- No clinical validation claims are made from synthetic experiments
- Feature ablation is performed using training data only; test data is
  never used for ablation decisions
