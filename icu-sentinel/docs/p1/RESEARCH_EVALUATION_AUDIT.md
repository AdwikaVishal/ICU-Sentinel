# Research Evaluation Audit — P1 PART 7

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 7 — Final Research Evaluation  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Pre-Training Audit Summary

This document records the pre-training audit performed before implementing the
P1 research evaluation pipeline. The audit inspected all existing P1 modules
(Part 1 through Part 6) to identify available interfaces, data contracts,
and integration points.

---

## 2. Available P1 Modules and Interfaces

| Module | Purpose | Key Outputs |
|--------|---------|-------------|
| `ml/p1_clinical/ingestion/` | Dataset ingestion | `.psv` loader, patient records |
| `ml/p1_clinical/preprocessing/` | Data preprocessing | Cleaned patient data, splits |
| `ml/p1_clinical/features/` | Temporal feature engineering | Feature schemas, processed features |
| `ml/p1_clinical/baselines/` | Clinical baselines | SIRS, qSOFA, NEWS-style scores |
| `ml/p1_clinical/classical_ml/` | Classical ML models | Logistic Regression, Random Forest, SVM |
| `ml/p1_clinical/evaluation/` | Research evaluation | Metrics, calibration, lead time, alerts, ablation, subgroup, bootstrap |

---

## 3. Data Contracts Available for Evaluation

### Patient-Level Splits (`ml/p1_clinical/preprocessing/patient_splits.py`)

- `SplitManifest`: Complete split assignment for all patients
- Train/validation/test split with no patient overlap
- Stratification by sepsis label
- Patient IDs, hospital IDs, record counts per split

### Feature Schema (`ml/p1_clinical/features/feature_schema.py`)

- `FeatureSchema`: Complete feature schema with ~40 clinical variables
- Feature categories: current, missingness, forward_fill, rolling, trend, clinical
- All features are causal (current/past observations only)
- Feature metadata: name, type, units, missing behavior, causal flag

### Clinical Baselines (`ml/p1_clinical/baselines/`)

- SIRS score (4 components: Temp, HR, Resp, WBC)
- qSOFA score (3 components: RR, SBP, AMS — AMS not available in dataset)
- NEWS-style score (7 parameters, modified — supplemental O2 and consciousness not available)
- Each baseline includes completeness tracking (complete/partial/none)

### Classical ML Models (`ml/p1_clinical/classical_ml/`)

- Logistic Regression, Random Forest, SVM
- Each model produces predictions with probability scores
- Model artifacts include feature names, preprocessing config, validation metrics

---

## 4. Integration Gaps Identified

| Gap | Impact | Mitigation |
|-----|--------|------------|
| No shared prediction output schema | P2/P3 models may produce incompatible prediction formats | Created `PredictionSchema` and `PredictionLoader` in `ml/p1_clinical/evaluation/prediction_loader.py` |
| No standardized metric reporting | Different modules may report metrics differently | Created `compute_binary_metrics()` and `MetricResult` in `classification_metrics.py` |
| No lead-time evaluation | Sepsis prediction requires lead-time analysis | Created `lead_time.py` module |
| No alert burden analysis | Alert fatigue is a key clinical concern | Created `alert_episodes.py` and `alert_burden.py` modules |
| No feature ablation | Cannot assess feature importance | Created `feature_ablation.py` module |
| No subgroup analysis | Cannot assess performance across demographics | Created `subgroup_analysis.py` module |
| No bootstrap CIs | Cannot assess statistical uncertainty | Created `bootstrap.py` module |
| No cross-hospital evaluation | Cannot assess transportability | Created `cross_hospital.py` module |
| No model comparison | Cannot compare baselines vs ML | Created `model_comparison.py` module |

---

## 5. Evaluation Pipeline

The `evaluation_pipeline.py` module orchestrates all evaluation components:

1. Load predictions from trained models (P1 classical ML) and clinical baselines (P1 baselines)
2. Compute classification metrics (AUROC, AUPRC, sensitivity, specificity, precision, F1, Brier)
3. Compute calibration metrics (ECE, MCE)
4. Compute lead-time metrics
5. Compute alert burden metrics
6. Perform feature ablation
7. Perform subgroup analysis
8. Compute bootstrap confidence intervals
9. Run cross-hospital evaluation (if hospital IDs available)
10. Generate model comparison report

---

## 6. Missing Integration Contracts

The following integration points were identified and addressed:

| Gap | Solution |
|-----|----------|
| No standard prediction input format | Created `PredictionRecord` and `PredictionBatch` in `prediction_schema.py` |
| No loader for model outputs | Created `PredictionLoader` in `prediction_loader.py` |
| No standardized metric format | Created `MetricResult` and `compute_binary_metrics()` |
| No calibration evaluation | Created `calibration.py` |
| No lead-time evaluation | Created `lead_time.py` |
| No alert burden evaluation | Created `alert_episodes.py` and `alert_burden.py` |
| No feature ablation | Created `feature_ablation.py` |
| No subgroup analysis | Created `subgroup_analysis.py` |
| No bootstrap confidence intervals | Created `bootstrap.py` |
| No cross-hospital evaluation | Created `cross_hospital.py` |
| No model comparison | Created `model_comparison.py` |

---

## 7. Evaluation Gaps Requiring Real Data

The following evaluations cannot be fully validated without real PhysioNet/CinC Challenge 2019 data:

| Evaluation | Reason |
|------------|--------|
| Clinical baseline comparison | Requires real patient data with complete clinical variables |
| Cross-hospital evaluation | Requires real hospital identifiers (Set A/Set B are not hospital identifiers) |
| Subgroup analysis by age/sex | Requires demographic data from real patients |
| Bootstrap confidence intervals | Requires real patient data for meaningful resampling |
| Feature ablation | Requires real feature data and trained models |

All evaluation outputs generated from synthetic fixtures are marked `"source": "synthetic"`.

---

## 8. Evaluation Output Schema

All evaluation outputs follow a consistent schema:

- `source` field: `"synthetic"` or `"real"` to distinguish data origin
- `version`: Schema version (1.0.0)
- Timestamps for reproducibility
- All metrics include confidence intervals where bootstrapped

---

## 9. Conclusion

The P1 evaluation framework is complete and ready to evaluate:

- Classical ML models (Logistic Regression, Random Forest, SVM)
- Clinical baselines (SIRS, qSOFA, NEWS-style)
- Future P2 models (Extra Trees, LightGBM, XGBoost, CatBoost)
- Future P3 models (GRU, LSTM, Transformer)

All evaluation outputs are generated from synthetic fixtures pending real data acquisition from PhysioNet/CinC Challenge 2019.