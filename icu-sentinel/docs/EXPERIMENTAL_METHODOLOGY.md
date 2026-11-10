# Experimental Methodology — ICU Sentinel

This document describes the experimental methodology, including data processing, model training, evaluation, and validation procedures for the ICU Sentinel retrospective sepsis early-warning study.

## Overview

ICU Sentinel is a retrospective machine learning research project aimed at predicting sepsis onset within 6 hours in ICU patients. The project follows a multi-phase pipeline:

- **P1**: Data preprocessing, feature engineering, target construction, patient splitting
- **P2**: Boosting model training (XGBoost, LightGBM, Extra Trees)
- **P3**: Patient replay engine, prediction providers, alert generation, SHAP explainability, React UI

All experiments are retrospective, using the PhysioNet CinC 2019 Challenge database. The project is labeled "Research prototype — not approved for clinical decision-making."

## Data Processing Pipeline

### Cohort Selection
- **Source**: PhysioNet CinC 2019 Challenge (MIMIC-IV derived ICU database)
- **Total patients**: 39,810 (full cohort), with a default 500-patient subset for development
- **Exclusions**: 1 corrupted patient file (p010963/p000963) excluded; 39,809 usable patients
- **Data range**: ICU stays with at least 6 hours of measurable data

### Feature Engineering
- **Total features**: 994 (after sanitization of infinite/NaN values)
- **Feature types**: Vital signs, laboratory values, clinical scores (SIRS, shock index), demographics, rolling statistics, slope features, availability indicators
- **Sanitization**: Infinite values replaced with NaN; missing values handled via `ffill`/`bfill` within the pipeline; `native_nan_handling` flag in each model config
- **Target**: `sepsis_within_6h` — binary label indicating sepsis within 6 hours
- **Eligibility**: `target_eligible` flag filters observations within 6 hours of onset or discharge

### Patient-Level Splits
- **Strategy**: Stratified random split by sepsis label
- **Random state**: 42
- **Split ratios**: train/val/test = 0.7/0.1/0.2 (by patient count)
- **Default subset**: 500 patients (350 train / 50 validation / 100 test) in `data/splits/`
- **Full cohort**: 39,810 patients (27,867 train / 3,981 validation / 7,962 test) in `data/splits_full/`
- **Integrity**: No patient appears across train/validation/test partitions (patient-level split integrity guaranteed)
- **Temporal leakage prevention**: Features are computed in a purely retrospective manner; no future-looking statistics (e.g., `_rolling_mean_24h` uses only data within the observation window before the feature timestamp)

## Model Training

### Model Families (P2)

| Model | Framework | Features | Training Notes |
|---|---|---|---|
| **XGBoost** | `xgboost` | 994 | `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `n_jobs=8`, `tree_method=hist`. Class weight: `None` (original) / `balanced` (retrain). Artifact path: `artifacts/p2/xgboost/xgboost_1.0.0.joblib`. |
| **LightGBM** | `lightgbm` | 994 | `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`, `max_depth=-1` (unlimited), `min_child_samples=20`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_alpha=0.0`, `reg_lambda=0.0`, `class_weight=balanced`, `verbosity=-1`. Artifact path: `artifacts/p2/lightgbm/lightgbm_1.0.0.joblib`. Note: Windows DLL failure prevents LightGBM execution in this environment; Extra Trees used as substitute. |
| **Extra Trees** | `scikit-learn` | 994 | `n_estimators=200`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features="sqrt"`, `class_weight="balanced"`, `n_jobs=8`. Artifact path: `artifacts/p2/extra_trees/extra_trees_1.0.0.joblib`. |

### Training Configuration
- **Hyperparameters**: Saved in each model's metadata JSON (see `artifacts/p2/xgboost/xgboost_1.0.0_metadata.json`, etc.)
- **Class imbalance**: `class_weight="balanced"` for Extra Trees and LightGBM; XGBoost uses `reg_lambda` penalty without explicit class weighting
- **Random seed**: 42 for train/val/test split and model training
- **Validation metric**: AUROC, with threshold selection on F1 using a grid with `step=0.01` on the validation split only
- **Test set untouched**: Threshold and model selection are never performed on the test set. Test metrics are computed exactly once after model/threshold are fixed.

### Training Pipeline (per model)
1. Load processed patient data from `data/processed_full/`
2. Apply patient-level split using manifest (`train/val/test`)
3. Construct feature matrix X and target vector y using only training-patient data
4. Train model on training split, validate on validation split
5. Select threshold on validation split maximizing F1
6. Compute test-set metrics with fixed threshold (no re-optimization)
7. Save model + metadata + feature names to `artifacts/p2/<model_name>/`
8. Save evaluation results to `data/reports/p2_fast/p2_training_summary.json`

## Evaluation Methodology

### Metrics Reported
For each model, the following metrics are computed on the **test split**:

| Metric | Description |
|---|---|
| **AUROC** | Area under the receiver operating characteristic curve. Measures ranking quality across all thresholds. |
| **AUPRC** | Area under the precision-recall curve. More informative than AUROC for imbalanced datasets (positive rate ~1.14%). |
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN).of limited utility due to imbalance. |
| **Sensitivity (Recall)** | TP / (TP+FN). Fraction of septic patients correctly identified. |
| **Specificity** | TN / (TN+FP). Fraction of non-septic patients correctly identified. |
| **Precision** | TP / (TP+FP). Fraction of positive predictions that are correct. |
| **F1 score** | Harmonic mean of precision and sensitivity. Used for threshold selection. |
| **Brier score** | Mean squared error between predicted probability and true label. |
| **ECE** | Expected Calibration Error. Not currently computed (see CALIBRATION.md). |
| **Threshold** | The probability threshold selected on validation F1. |

### Evaluation Splits
- **Training split**: Used for model fitting only. Metrics reported but not the focus.
- **Validation split**: Used for threshold selection and model comparison. No test set touching.
- **Test split**: Held out until final evaluation. Metrics computed once with fixed threshold.

### Threshold Selection Methodology
1. Compute predicted probabilities on the **validation split only**.
2. Grid search over thresholds from 0.00 to 1.00 in steps of `threshold_grid_step=0.01`.
3. Select threshold that maximizes `threshold_selection_metric=f1`.
4. Fix the threshold and compute all metrics on the **test split exactly once**.
5. Test set metrics are never re-optimized or tuned.

### Bootstrap Confidence Intervals
- Currently not computed (see KNOWN_LIMITATIONS.md).
- When computed, will use patient-level resampling with `n_iterations=1000`.
- Bootstrap CIs for AUROC, AUPRC, and calibration metrics will be reported in a separate supplement.

### Lead-Time Distribution
- Not explicitly computed in this retrospective effort.
- When available, will report the distribution of lead times (hours between prediction and sepsis onset) for positive predictions.

## Calibration

| Aspect | Status |
|---|---|
| `calibration_method` | `sigmoid` (configured in `P2Config`) |
| Calibrator fitted | Brier score recorded in training_summary; calibrator fit not yet demonstrably applied on real data (PARTIAL per master audit) |
| Brier score | Present in P2 training_summary.json (e.g., XGBoost test brier=0.1331) |
| ECE | Not computed anywhere in current artifacts |
| Calibration data | **Must be fitted on validation data only**; test set must remain untouched until final evaluation |

**Calibration Procedure** (to be completed):
1. Compute predicted probabilities on the validation split
2. Fit an isotonic regression or sigmoid calibrator on validation predictions vs. true labels
3. Apply calibrator to test-set predictions
4. Compute calibrated ECE and Brier on test set
5. Report both pre- and post-calibration metrics

*Note*: The Brier score currently reported in training summaries is the **uncalibrated** Brier score. Calibrated Brier/ECE will be reported once the calibration pipeline is completed.

## Reproducibility

### Random Seeds
- Train/val/test split: `random_state=42`
- Model training: `random_state=42`
- Bootstrap resampling: not yet documented (see KNOWN_LIMITATIONS.md)

### Software Environment
- **Python**: 3.14.5 (via `.venv`)
- **Key packages**: `xgboost`, `lightgbm`, `scikit-learn`, `pandas`, `numpy`, `pytest`, `vitest`
- **Environment locked**: `pyproject.toml` and `requirements.txt` specify exact versions

### Data Provenance
- Raw PhysioNet data: not in repository (too large)
- Processed features: generated and committed to `data/processed_full/`
- Model artifacts: committed to `artifacts/p2/` (gitignored, ~3.5 GB total)
- Feature schemas: `data/features/feature_schema.json`, `data/features/feature_config.json`
- Split manifests: `data/splits/split_manifest.json` (500 patients) and `data/splits_full/split_manifest.json` (39,810 patients)

### Output Artifacts
- Trained models: `artifacts/p2/xgboost/xgboost_1.0.0.joblib`, etc.
- Metadata: `artifacts/p2/xgboost/xgboost_1.0.0_metadata.json`, etc.
- Training summaries: `data/reports/p2_fast/p2_training_summary.json`
- Evaluation results: `artifacts/physionet2019/final_evaluation/test_evaluation.json`
- Split manifests: `data/splits/split_manifest.json`, `data/splits_full/split_manifest.json`

### Reproduction Instructions (Test Evaluation)
1. Load model from `artifacts/physionet2019/full_cohort_v2/models/` (or `artifacts/p2/<model_name>/`)
2. Load test data from the appropriate split manifest
3. Generate predictions on test set
4. Apply fixed threshold (from validation F1 maximization)
5. Compute metrics: AUROC, AUPRC, sensitivity, specificity, precision, F1, Brier
6. Compare against `artifacts/physionet2019/final_evaluation/test_evaluation.json` results

**Expected match**: Metrics should match within floating-point tolerance (the test_evaluation.json reports AUROC: XGBoost 0.8138, LightGBM 0.8128, Extra Trees 0.7433 on the 7,962-patient test split).

## Limitations

- Single-center data (PhysioNet CinC 2019); may not generalize to other ICU populations
- Class imbalance (1.14% positive) leads to low precision; model may not be suitable for direct clinical deployment without threshold tuning
- 1 corrupted patient file excluded (p010963/p000963)
- LightGBM not executable in current environment (Windows DLL issue; Extra Trees substituted)
- No external validation performed
- Calibration (ECE) not yet computed
- Bootstrap confidence intervals not computed
- Lead-time distribution not computed