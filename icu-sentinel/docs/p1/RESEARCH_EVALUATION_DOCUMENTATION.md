# Research Evaluation Documentation — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 7 — Final Research Evaluation  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Overview

The P1 research evaluation pipeline provides a comprehensive framework for
evaluating and comparing early sepsis prediction models. The pipeline
supports:

- **P1 classical ML models**: Logistic Regression, Random Forest, SVM
- **P1 clinical baselines**: SIRS, qSOFA, NEWS-style
- **P2 boosting models**: Extra Trees, LightGBM, XGBoost, CatBoost (future)
- **P3 temporal models**: GRU, LSTM, Transformer (future)

All evaluations are causal (use only current/past observations) and preserve
patient-level splits. No raw patient data is committed to Git.

---

## 2. Evaluation Pipeline Architecture

The evaluation pipeline is implemented in `ml/p1_clinical/evaluation/` and
consists of the following modules:

| Module | Purpose |
|--------|---------|
| `evaluation_config.py` | Centralized configuration |
| `prediction_loader.py` | Load prediction outputs from models |
| `evaluation_schema.py` | Shared evaluation output schemas |
| `classification_metrics.py` | AUROC, AUPRC, sensitivity, specificity, etc. |
| `calibration.py` | Expected Calibration Error, Maximum Calibration Error |
| `lead_time.py` | Lead-time evaluation |
| `alert_episodes.py` | Alert episode detection |
| `alert_burden.py` | Alert burden metrics (false alerts per patient-day) |
| `feature_ablation.py` | Feature ablation analysis |
| `subgroup_analysis.py` | Subgroup evaluation |
| `bootstrap.py` | Patient-level bootstrap confidence intervals |
| `cross_hospital.py` | Cross-hospital evaluation |
| `model_comparison.py` | Model comparison |
| `evaluation_pipeline.py` | End-to-end pipeline orchestration |

---

## 3. Evaluation Flow

The evaluation pipeline follows this sequence:

1. **Load predictions** from trained models and clinical baselines
2. **Compute classification metrics** for each model on each split
3. **Compute calibration metrics** (ECE, MCE)
4. **Compute lead-time metrics** (time from alert to sepsis label)
5. **Compute alert burden** (false alerts per patient-day)
6. **Compute feature ablation** (remove feature groups, retrain, evaluate)
7. **Compute subgroup analysis** (evaluate by demographic subgroups)
8. **Compute bootstrap confidence intervals** (patient-level resampling)
9. **Run cross-hospital evaluation** (train on one hospital, test on another)
10. **Generate model comparison report** (compare all models)

---

## 4. Evaluation Outputs

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

All reports include a `"source"` field: `"synthetic"` or `"real"`.

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

## 6. Evaluation Metrics

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
| Detection rate before onset | % of sepsis patients detected before onset |
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

## 7. Data Integrity

- All evaluation reports generated from synthetic fixtures are marked
  `"source": "synthetic"`
- No clinical validation claims are made from synthetic experiments
- Real data evaluation requires PhysioNet/CinC Challenge 2019 access
- Patient-level bootstrap resampling preserves patient-level correlation
- Cross-hospital evaluation requires genuine hospital identifiers
