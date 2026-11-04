# Classical ML Evaluation Protocol — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Classical ML  
**Part:** PART 6 — Classical ML models  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Evaluation Overview

P1 evaluates three classical ML models (Logistic Regression, Random Forest,
SVM) against clinical baselines (SIRS, qSOFA, NEWS2-style) for early sepsis
prediction.

All evaluation uses the Part 3 patient-level train/validation/test splits.
No hourly observation is treated as an independent patient.

---

## 2. Prediction Unit

The prediction unit is a **patient-hour**. Each row in the dataset represents
one hour of ICU stay for one patient, with:

- Features: Clinical variables at that hour (Part 4 engineered features)
- Target: SepsisLabel at that hour (Part 3 official label)
- Metadata: patient_id, hour, source_dataset

Hourly predictions are aggregated to patient-level outcomes:
- A patient is "detected" if any hourly prediction in a clinically relevant
  window is positive
- Alert sensitivity is measured at the patient level
- False alerts per patient-day requires an alert aggregation policy

---

## 3. Alert Definition

An **alert** is generated when `predicted_class == 1` (probability >= threshold).

Alert aggregation policy:
- Consecutive positive predictions within a time window are grouped into a
  single alert
- Alert onset = first positive prediction hour
- Alert duration = number of consecutive positive hours
- False alert = alert for a non-sepsis patient
- True alert = alert for a sepsis patient within the clinically relevant window

False alerts per patient-day = (total false alerts) / (total patient-days)

---

## 4. Train/Validation/Test Protocol

### Training Set

- Fit imputers, scalers, feature selectors, models
- Training-only cross-validation for hyperparameter selection

### Validation Set

- Compare model configurations
- Select hyperparameters (grid search)
- Select alert thresholds (optimize F1, precision, recall, or balanced accuracy)
- Assess calibration (calibration curves, Brier score)

### Test Set

- Evaluate final frozen models **once**
- **No** parameter tuning
- **No** model selection based on test results
- **No** threshold adjustment on test data

---

## 5. Metrics Calculation

### Threshold-Dependent Metrics

Calculated at the selected threshold:

| Metric | Formula |
|--------|---------|
| Sensitivity (Recall) | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Precision | TP / (TP + FP) |
| F1-score | 2 × (Precision × Recall) / (Precision + Recall) |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |

### Threshold-Independent Metrics

| Metric | Description |
|--------|-------------|
| AUROC | Area Under ROC Curve (all thresholds) |
| AUPRC | Area Under Precision-Recall Curve (all thresholds) |
| Brier score | Mean squared probability error |

### Clinical Utility Metrics

| Metric | Description |
|--------|-------------|
| False alerts per patient-day | False alerts / patient-days |
| Lead time | Hours between alert and sepsis onset |
| Patient-level detection rate | Patients with ≥1 alert / total sepsis patients |
| Alert sensitivity | True alerts / total alerts |

---

## 6. Clinical Baseline Comparison

Clinical baselines from Part 5 are compared against ML models:

| Baseline | Availability | Comparison Notes |
|----------|-------------|------------------|
| SIRS | Complete when all 4 components available | Compare on complete-case subset |
| qSOFA | Partial (AMS not available) | Compare on hours where AMS available |
| NEWS2-style | Modified (O2/consciousness not available) | Clearly labeled as modified |

**Important**: Do not compare a clinical score evaluated on a small
complete-case subset against an ML model evaluated on the entire dataset
without clearly explaining the difference. Report both complete-case and
full-dataset metrics where applicable.

---

## 7. Evaluation Population

Metrics are reported at two levels:

1. **Hourly level**: Each patient-hour is one prediction
2. **Patient-level**: Aggregated alerts per patient

Metrics are reported for both levels where applicable.

---

## 8. Reporting

Evaluation reports saved under `data/reports/classical_ml/`:

| File | Content |
|------|-----------|
| model_comparison.json | Side-by-side model comparison |
| model_metrics.json | Per-model metrics |
| feature_importance.csv | Feature importance rankings |
| training_summary.json | Training protocol summary |
| prediction_summary.json | Prediction output summary |

All synthetic reports marked `"source": "synthetic"`.

---

## 9. Synthetic vs Real-Data

**Real-data verification: Not available.**

Real PhysioNet/CinC Challenge 2019 data requires credentialed access
which has not been obtained. All evaluation metrics from synthetic data
are **not** clinical performance evidence.

No clinical validation claims are made from synthetic experiments.

---

## 10. Limitations

- No real data access for validation
- Class imbalance strategy limited to class weighting
- No temporal cross-validation (time-based splits not implemented)
- No external validation cohort
- Alert aggregation policy is configurable but not clinically validated
