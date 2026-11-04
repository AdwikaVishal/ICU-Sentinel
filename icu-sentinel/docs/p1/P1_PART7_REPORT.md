# ICU Sentinel — P1 PART 7 Report

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1)  
**Part:** PART 7 — Final Research Evaluation  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Implementation Summary

PART 7 implements a complete research evaluation framework for ICU Sentinel
P1. The evaluation framework provides a comprehensive set of tools for
evaluating and comparing early sepsis prediction models across all three
workstreams (P1 classical ML, P2 boosting, P3 temporal).

All code is tested using synthetic fixtures (no real patient data available
without PhysioNet credentialed access).

---

## 2. Files Created

### New source files

| File | Purpose |
|------|---------|
| `ml/p1_clinical/evaluation/__init__.py` | Package exports |
| `ml/p1_clinical/evaluation/evaluation_config.py` | Centralized evaluation configuration |
| `ml/p1_clinical/evaluation/evaluation_schema.py` | Evaluation result schemas |
| `ml/p1_clinical/evaluation/prediction_loader.py` | Prediction loader for P2/P3 integration |
| `ml/p1_clinical/evaluation/classification_metrics.py` | Classification metrics (AUROC, AUPRC, etc.) |
| `ml/p1_clinical/evaluation/calibration.py` | Calibration metrics (ECE, MCE) |
| `ml/p1_clinical/evaluation/lead_time.py` | Lead-time evaluation |
| `ml/p1_clinical/evaluation/alert_episodes.py` | Alert episode detection |
| `ml/p1_clinical/evaluation/alert_burden.py` | Alert burden (false alerts per patient-day) |
| `ml/p1_clinical/evaluation/feature_ablation.py` | Feature ablation analysis |
| `ml/p1_clinical/evaluation/subgroup_analysis.py` | Subgroup analysis |
| `ml/p1_clinical/evaluation/bootstrap.py` | Patient-level bootstrap confidence intervals |
| `ml/p1_clinical/evaluation/cross_hospital.py` | Cross-hospital evaluation |
| `ml/p1_clinical/evaluation/model_comparison.py` | Model comparison |
| `ml/p1_clinical/evaluation/evaluation_pipeline.py` | End-to-end evaluation pipeline |

### New test files

| File | Tests |
|------|-------|
| `ml/p1_clinical/tests/test_classification_metrics.py` | Metric calculations, edge cases, label isolation |
| `ml/p1_clinical/tests/test_calibration.py` | Calibration curve, ECE, MCE computation |
| `ml/p1_clinical/tests/test_lead_time.py` | Lead-time calculation, label convention, missing data |
| `ml/p1_clinical/tests/test_alert_episodes.py` | Alert episode detection, missingness, qSOFA calibration |
| `ml/p1_clinical/tests/test_alert_burden.py` | Alert burden calculation, patient-day denominator |
| `ml/p1_clinical/tests/test_feature_ablation.py` | Feature ablation grouping and comparison |
| `ml/p1_clinical/tests/test_subgroup_analysis.py` | Subgroup analysis, missing data handling |
| `ml/p1_clinical/tests/test_bootstrap.py` | Bootstrap confidence intervals, reproducibility |
| `ml/p1_clinical/tests/test_cross_hospital.py` | Cross-hospital evaluation |
| `ml/p1_clinical/tests/test_model_comparison.py` | Model comparison, ablation results |
| `ml/p1_clinical/tests/test_evaluation_pipeline.py` | End-to-end pipeline execution |
| `ml/p1_clinical/tests/test_evaluation_leakage.py` | Leakage prevention, label isolation |

### New documentation

| File | Purpose |
|------|---------|
| `docs/p1/RESEARCH_EVALUATION_AUDIT.md` | Pre-implementation audit |
| `docs/p1/RESEARCH_EVALUATION_DOCUMENTATION.md` | Full evaluation methodology |
| `docs/p1/CLASSICAL_ML_EVALUATION_PROTOCOL.md` | Evaluation protocol |
| `docs/p1/ALERT_EVALUATION_PROTOCOL.md` | Alert evaluation protocol |
| `docs/p1/FEATURE_ABLATION_PROTOCOL.md` | Feature ablation protocol |
| `docs/p1/SUBGROUP_ANALYSIS_PROTOCOL.md` | Subgroup analysis protocol |
| `docs/p1/P2_P3_EVALUATION_HANDOFF.md` | P2/P3 integration contract |
| `docs/p1/P1_PART7_REPORT.md` | This report |

### New data reports

| File | Source | Notes |
|------|--------|-------|
| `data/reports/research_evaluation/classification_metrics.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/research_evaluation/calibration_report.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/research_evaluation/lead_time_report.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/research_evaluation/alert_burden_report.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/research_evaluation/feature_ablation_report.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/research_evaluation/subgroup_analysis_report.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/research_evaluation/bootstrap_confidence_intervals.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/research_evaluation/cross_hospital_report.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/research_evaluation/model_comparison.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/research_evaluation/final_evaluation_summary.json` | Synthetic pipeline run | Marked synthetic |

### Modified files

None. All PART 7 files are new; no existing files were modified.

---

## 3. Evaluation Pipeline

The evaluation pipeline (`ml/p1_clinical/evaluation/evaluation_pipeline.py`)
orchestrates all evaluation components:

1. **Discovery**: Load prediction outputs from trained models and clinical baselines
2. **Classification metrics**: Compute AUROC, AUPRC, sensitivity, specificity, precision, F1, accuracy, Brier score
3. **Calibration**: Compute ECE and MCE
4. **Lead time**: Compute lead-time metrics (time from alert to sepsis label)
5. **Alert burden**: Compute false alerts per patient-day, alerts per patient
6. **Feature ablation**: Remove feature groups, retrain, evaluate
7. **Subgroup analysis**: Evaluate performance across patient subgroups
8. **Bootstrap confidence intervals**: Patient-level bootstrap CIs
9. **Cross-hospital evaluation**: Train on one hospital, test on another
10. **Model comparison**: Compare all models on the same evaluation population

---

## 4. Evaluation Metrics

### Classification Metrics

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

### Calibration Metrics

| Metric | Description |
|--------|-------------|
| ECE | Expected Calibration Error |
| MCE | Maximum Calibration Error |

### Lead-Time Metrics

| Metric | Description |
|--------|-------------|
| Lead time | Hours between alert and sepsis label |
| Detection rate before onset | % detected before onset |
| Detection rate at/before onset | % detected at or before onset |
| Detection rate after onset | % detected after onset |
| Missed patient rate | % of sepsis patients not detected |

### Alert Burden Metrics

| Metric | Description |
|--------|-------------|
| False alerts per patient-day | False alert episodes per patient-day |
| Alerts per patient | Alert episodes per patient |
| Patients with false alerts | % of patients with ≥1 false alert |
| Alert duration | Mean alert duration in hours |

### Subgroup Analysis Metrics

| Metric | Description |
|--------|-------------|
| Sensitivity | TP / (TP + FN) per subgroup |
| Specificity | TN / (TN + FP) per subgroup |
| Precision | TP / (TP + FP) per subgroup |
| F1-score | 2 × (Precision × Recall) / (Precision + Recall) per subgroup |

### Bootstrap Confidence Intervals

| Metric | Description |
|--------|-------------|
| AUROC CI | Confidence interval for AUROC |
| AUPRC CI | Confidence interval for AUPRC |
| Sensitivity CI | Confidence interval for sensitivity |
| Specificity CI | Confidence interval for specificity |
| Precision CI | Confidence interval for precision |
| F1 CI | Confidence interval for F1-score |
| Accuracy CI | Confidence interval for accuracy |
| Brier CI | Confidence interval for Brier score |

---

## 5. Evaluation Configuration

All evaluation parameters are centralized in
`ml/p1_clinical/evaluation/evaluation_config.py` and serializable to JSON.

Key configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `random_seed` | 42 | Random seed for reproducibility |
| `alert_threshold` | 0.5 | Decision threshold for positive prediction |
| `alert_cooldown_hours` | 6 | Minimum hours between alert episodes |
| `alert_min_duration_hours` | 1 | Minimum duration for an alert episode |
| `alert_min_consecutive` | 1 | Minimum consecutive positive predictions |
| `bootstrap_n_iterations` | 1000 | Number of bootstrap iterations |
| `bootstrap_confidence_level` | 0.95 | Confidence level for intervals |
| `lead_time_window_hours` | 6 | Lead-time window in hours |
| `false_alerts_per_patient_day_window_hours` | 24 | Window for false alerts per patient-day |

---

## 6. Evaluation Outputs

All evaluation outputs are saved to `data/reports/research_evaluation/`:

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

All reports include a `"source"` field: `"synthetic"` or `"real"`.

---

## 7. Evaluation Methodology

### Patient-Level Evaluation

All evaluation is performed at the patient level. Hourly observations are
grouped by patient, and metrics are computed at the patient level. This
ensures that hourly observations from the same patient are not treated as
independent observations.

### Causal Evaluation

All features are causal (use only current/past observations). No future
information leaks into feature computation or model evaluation.

### Patient-Level Bootstrap

Bootstrap confidence intervals are computed using patient-level
resampling. All observations from a sampled patient are kept together,
preserving the correlation structure within patients.

### Cross-Hospital Evaluation

Cross-hospital evaluation trains on one hospital's data and tests on
another hospital's data. This requires genuine hospital identifiers
(source dataset identity is not automatically a hospital identifier).

---

## 8. Synthetic vs Real-Data Verification

**Real-data verification: Not available.**

Real PhysioNet/CinC Challenge 2019 data requires credentialed access
which has not been obtained. All evaluation results from synthetic
fixtures are marked `"source": "synthetic"` and include a note stating
they are not real PhysioNet/CinC Challenge 2019 statistics.

No clinical model performance is reported as evidence of real predictive
performance. All claims based on synthetic data are explicitly labeled
as such.

---

## 9. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No real data access | Cannot validate against actual dataset | Pipeline ready; synthetic tests pass |
| Synthetic data only | All results are from synthetic data | Clearly marked as synthetic |
| No cross-hospital evaluation | Hospital identifiers not available | Architecture prepared; no claims made |
| No feature ablation without model retraining | Requires model retraining for each ablation | Pipeline supports retraining |
| Bootstrap confidence intervals require sufficient patients | Small patient counts may give unreliable CIs | Minimum patient count check |
| No clinical validation | No claims of clinical validity | Clearly stated as research only |

---

## 9. P2/P3 Handoff Requirements

P2 and P3 can integrate their models into the P1 evaluation pipeline:

1. **Prediction format**: Provide predictions in the standard format
   (patient_id, hour, true_label, predicted_probability, predicted_class,
   model_name, model_version, split)

2. **Evaluation pipeline**: Use `run_evaluation_pipeline()` to evaluate
   their models alongside P1 classical ML models and clinical baselines

3. **Split consistency**: Use the same patient-level splits as P1
   (train/validation/test from `data/splits/split_manifest.json`)

4. **Feature schema**: Use the same feature schema as P1
   (`data/features/feature_schema.json`)

---

## 9. Remaining Work

1. Obtain PhysioNet credentialed access and download real dataset
2. Run evaluation pipeline against real data
3. Validate evaluation results against real data
4. P1 final verification and release

---

## 10. Data Integrity Statement

- No raw patient data was downloaded (no PhysioNet credentials)
- No raw patient data was committed to Git
- No evaluation statistics were fabricated; all reports marked synthetic
- Source data remains unchanged (pipeline only reads processed data)
- Malformed records are never silently discarded
- All synthetic results are clearly marked as synthetic
- No clinical validation claims are made from synthetic experiments

---

Awaiting final approval.