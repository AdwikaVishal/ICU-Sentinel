# ICU SENTINEL: Artifacts Manifest

**Generated:** November 10, 2026  
**Project Status:** 70% Complete (Core Evaluation COMPLETE)

---

## 1. TRAINED MODELS

### Location: `/Users/shantanu/ICU-Sentinel/artifacts/physionet2019/full_dataset_retrain/`

#### Extra Trees
```
extra_trees/
├── extra_trees_1.0.0.joblib            (1.0 GB) - Trained model
├── extra_trees_1.0.0_metadata.json     (57 KB)  - Training metadata
└── extra_trees_1.0.0_features.json     (28 KB)  - Feature names
```

**Model Specs:**
- Training patients: 27,867
- Training observations: 1,065,260
- Features: 994
- n_estimators: 200
- Validation AUROC: 0.7340
- **Test AUROC: 0.7433**
- **Test ECE: 0.0008** (excellent calibration)

#### LightGBM
```
lightgbm/
├── lightgbm_1.0.0.joblib              (788 KB) - Trained model
├── lightgbm_1.0.0_metadata.json       (57 KB)  - Training metadata
└── lightgbm_1.0.0_features.json       (28 KB)  - Feature names
```

**Model Specs:**
- Training patients: 27,867
- Training observations: 1,065,260
- Features: 994
- n_estimators: 200, learning_rate: 0.05
- Validation AUROC: 0.8051
- **Test AUROC: 0.8128**
- **Test ECE: 0.2542** (needs calibration)

#### XGBoost
```
xgboost/
├── xgboost_1.0.0.joblib               (415 KB) - Trained model
├── xgboost_1.0.0_metadata.json        (57 KB)  - Training metadata
└── xgboost_1.0.0_features.json        (28 KB)  - Feature names
```

**Model Specs:**
- Training patients: 27,867
- Training observations: 1,065,260
- Features: 994
- n_estimators: 200, learning_rate: 0.05, max_depth: 4
- Validation AUROC: 0.7986
- **Test AUROC: 0.8138** ⭐ BEST
- **Test ECE: 0.3173** (needs calibration)

**Model Configuration:**
```
p2_config.json - Shared configuration for all three models
```

---

## 2. EVALUATION RESULTS

### Location: `/Users/shantanu/ICU-Sentinel/artifacts/physionet2019/final_evaluation/`

#### Test Evaluation (COMPLETE)
```
test_evaluation.json - Comprehensive test set metrics for all three models
```

**Contents:**
- Model: extra_trees, lightgbm, xgboost
- Patients: 7,962 (untouched during training)
- Observations: 303,709
- Positive class: 3,464 (1.14%)
- Metrics per model:
  - AUROC, AUPRC
  - Brier score, ECE
  - Sensitivity, Specificity, Precision, F1
  - Confusion matrix (TP, FP, FN, TN)
  - Threshold used

#### Bootstrap Confidence Intervals (SKIPPED)
```
bootstrap_confidence_intervals.json - Status: skipped (infrastructure issue)
```

**Contents:**
```json
{
  "status": "skipped",
  "reason": "Patient-level bootstrap requires patient_id column which needs dataset reload"
}
```

**To Complete:**
1. Optimize data structure to include patient_id
2. Run 1000-iteration patient-level bootstrap
3. Expected runtime: 10-30 minutes per model

#### Clinical Baselines (PARTIAL)
```
Status: In progress (blocked by dataset loading time)
```

**Preliminary Results:**
- SIRS: AUROC ~0.563 (computable on 86.7% of observations)
- qSOFA (partial): AUROC ~0.514 (computable on 80.4% of observations)
- NEWS2 (partial): Calculation in progress

---

## 3. DATASET AND SPLITS

### Dataset Summary
**Location:** `/Users/shantanu/ICU-Sentinel/artifacts/physionet2019/dataset_summary.json`

**Contents:**
- Total patients: 39,810
- Usable patients: 39,809 (1 corrupted file: p010963)
- Total observations: 1,530,732
- Sepsis patients: 2,896 (7.27%)
- Training SetA: 20,336 patients
- Training SetB: 19,474 patients
- Missingness statistics per feature

### Split Manifest
**Location:** `/Users/shantanu/ICU-Sentinel/icu-sentinel/data/splits_full/split_manifest.json`

**Contents:**
- Strategy: stratified_random
- Random seed: 42
- Training: 27,867 patients, 1,072,418 observations
- Validation: 3,981 patients, 152,554 observations
- Test: 7,962 patients, 305,760 observations
- Per-patient split assignment
- Sepsis label per patient
- Source dataset per patient

### Feature Schema
**Location:** `/Users/shantanu/ICU-Sentinel/icu-sentinel/data/features_full/feature_schema.json`

**Contents:**
- 994 features (validated)
- Feature types and descriptions
- Missing value handling strategy
- Temporal window definitions

### Feature Files
**Location:** `/Users/shantanu/ICU-Sentinel/icu-sentinel/data/features_full/patients/`

**Contents:**
- 39,809 patient feature files (1 excluded)
- Format: patient_id.parquet
- Each file contains temporal feature vectors

---

## 4. HISTORICAL EVALUATION REPORTS

### Phase Reports
```
/Users/shantanu/ICU-Sentinel/
├── PHASE0_BASELINE_AUDIT.md                    - Initial data audit
├── PHASE1_BOOTSTRAP_FIX_VERIFICATION.md        - Bootstrap correction
├── PHASE1_DATA_AUDIT_REPORT.md                 - Data quality audit
├── PHASE1_TRAINING_VERIFICATION.md             - Training verification
├── PHASE2_SPLIT_LEAKAGE_VERIFICATION.md        - Split integrity check
├── PHASE3_VALIDATION_EVALUATION.md             - Validation results
├── PHASE4_CALIBRATION_ANALYSIS.md              - Calibration analysis
├── PHASE5_TEST_EVALUATION.md                   - Test evaluation
└── PHASES_6_7_8_SUMMARY.md                     - Final phases summary
```

### Comprehensive Reports
```
├── COMPREHENSIVE_STATUS_REPORT.md              - Mid-project status
├── EXECUTION_SUMMARY_INTERIM.md                - Interim execution log
├── FINAL_PROJECT_READINESS.md                  - Pre-final status
├── FINAL_COMPREHENSIVE_EVALUATION_REPORT.md    - ⭐ FINAL REPORT
└── EXECUTIVE_SUMMARY.md                        - ⭐ EXECUTIVE SUMMARY
```

---

## 5. EVALUATION SCRIPTS

### Completed Evaluations
```
/Users/shantanu/ICU-Sentinel/
├── complete_final_evaluation.py          - Main evaluation script
├── evaluate_clinical_baselines.py        - Clinical baseline summary
└── train_full_cohort.py                  - Full cohort training script
```

### Test Scripts
```
├── test_bootstrap_fix.py                 - Bootstrap unit test
└── icu-sentinel/ml/p1_clinical/tests/
    ├── test_bootstrap.py                 - Bootstrap functionality tests
    └── test_bootstrap_indexed_equivalence.py - Bootstrap index tests
```

### Utility Scripts
```
├── reconcile_p2_experiments.py           - Experiment reconciliation
└── [Various data processing scripts]
```

---

## 6. EXECUTION LOGS

### Training Logs
```
/Users/shantanu/ICU-Sentinel/artifacts/physionet2019/full_dataset_retrain/
└── [Training stdout/stderr would be here if captured]
```

### Evaluation Logs
```
/Users/shantanu/ICU-Sentinel/
├── evaluation_run.log                    - Partial evaluation run
└── evaluation_run_final.log              - Final evaluation run
```

---

## 7. BACKUP ARTIFACTS

### Pre-Full-Run Backup
**Location:** `/Users/shantanu/ICU-Sentinel/artifacts/_backup_pre_full_run/`

```
artifacts_physionet2019_20261109_142915.tar.gz - Full artifact backup before final run
```

---

## 8. LEGACY ARTIFACTS (Historical)

### P1 - Matched Cohort (6,000 patients)
**Location:** `/Users/shantanu/ICU-Sentinel/artifacts/p1/`

**Contents:**
- Training patients: 351
- Validation patients: 51
- Test patients: 101
- Clinical baseline results (SIRS, qSOFA, NEWS)
- Data quality reports
- Temporal leakage audits

**Status:** Superseded by full cohort evaluation

### P2 - Multiple Experimental Runs
**Locations:**
- `/Users/shantanu/ICU-Sentinel/artifacts/p2/` - Matched cohort (6,000)
- `/Users/shantanu/ICU-Sentinel/artifacts/physionet2019/p2_debug*/` - Debug runs
- `/Users/shantanu/ICU-Sentinel/artifacts/physionet2019/p2_974/` - 974-feature experiment
- `/Users/shantanu/ICU-Sentinel/artifacts/physionet2019/full_dataset_experiment/` - Experimental
- `/Users/shantanu/ICU-Sentinel/artifacts/physionet2019/full_dataset_retrain_test/` - Test run

**Status:** Historical experiments, superseded by full_dataset_retrain

---

## 9. KEY DELIVERABLES FOR REPRODUCTION

### Minimum Required Files

To reproduce test evaluation results:

1. **Models:**
   ```
   artifacts/physionet2019/full_dataset_retrain/
   ├── extra_trees/extra_trees_1.0.0.joblib
   ├── lightgbm/lightgbm_1.0.0.joblib
   └── xgboost/xgboost_1.0.0.joblib
   ```

2. **Dataset:**
   ```
   icu-sentinel/data/splits_full/split_manifest.json
   icu-sentinel/data/features_full/patients/ (39,809 files)
   icu-sentinel/data/features_full/feature_schema.json
   ```

3. **Evaluation Script:**
   ```
   complete_final_evaluation.py
   ```

4. **Results:**
   ```
   artifacts/physionet2019/final_evaluation/test_evaluation.json
   ```

### To Reproduce Training

1. **Data:**
   - Raw PhysioNet 2019 files (not included, available from PhysioNet)
   - Feature engineering pipeline (icu-sentinel/ml/p1_clinical/)

2. **Training Script:**
   ```
   train_full_cohort.py
   ```

3. **Configuration:**
   ```
   Random seed: 42
   Models: Extra Trees, LightGBM, XGBoost
   Split: 70% train, 10% validation, 20% test
   ```

---

## 10. FILE SIZE SUMMARY

### Large Files (>100 MB)
```
extra_trees_1.0.0.joblib                      1.0 GB
artifacts_physionet2019_20261109_142915.tar.gz ~500 MB (estimated)
```

### Medium Files (1-100 MB)
```
Feature files (39,809 × ~50KB avg)            ~2 GB total
Split manifest                                ~20 MB
```

### Small Files (<1 MB)
```
Metadata, configs, reports                    <50 MB total
```

**Total Artifact Size:** ~3.5 GB

---

## 11. VERIFICATION CHECKSUMS

### Key Result Files

```bash
# Test evaluation results
sha256sum artifacts/physionet2019/final_evaluation/test_evaluation.json
# Output: [would include actual checksum]

# Model files
sha256sum artifacts/physionet2019/full_dataset_retrain/*/
*_1.0.0.joblib
# Output: [would include actual checksums]
```

---

## 12. USAGE EXAMPLES

### Load and Use Trained Model

```python
import joblib
import pandas as pd

# Load model
model_data = joblib.load(
    "artifacts/physionet2019/full_dataset_retrain/xgboost/xgboost_1.0.0.joblib"
)
model = model_data['model']
pipeline = model_data['preprocessing_pipeline']

# Load test data
# [Code to load test features]

# Make predictions
if pipeline:
    X_transformed = pipeline.transform(X_test)
else:
    X_transformed = X_test

y_pred_proba = model.predict_proba(X_transformed)[:, 1]

# Apply threshold (from validation)
threshold = 0.78  # XGBoost optimal threshold
y_pred_class = (y_pred_proba >= threshold).astype(int)
```

### Evaluate Model

```python
from sklearn.metrics import roc_auc_score, average_precision_score

auroc = roc_auc_score(y_true, y_pred_proba)
auprc = average_precision_score(y_true, y_pred_proba)

print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
```

---

## 13. NEXT STEPS TO COMPLETE EVALUATION

### High Priority (1-2 days)

1. **Clinical Baselines:**
   - Complete SIRS, qSOFA, NEWS2 evaluation
   - Runtime: 30 minutes (once dataset loads)
   - Output: `clinical_baselines.json`

2. **Probability Calibration:**
   - Fit isotonic calibrator on validation set
   - Re-evaluate LightGBM and XGBoost
   - Runtime: 30 minutes
   - Output: `calibrated_test_evaluation.json`

### Medium Priority (1 week)

3. **Bootstrap Confidence Intervals:**
   - Optimize data structure for patient_id
   - 1000-iteration bootstrap × 3 models
   - Runtime: 10-30 minutes per model
   - Output: `bootstrap_confidence_intervals.json`

4. **Early Warning Analysis:**
   - Lead time calculations
   - False alerts per patient-day
   - Event detection rates
   - Runtime: 8 hours
   - Output: `early_warning_analysis.json`

---

## 14. CONTACT AND DOCUMENTATION

**Primary Documentation:**
- `FINAL_COMPREHENSIVE_EVALUATION_REPORT.md` - Full technical report
- `EXECUTIVE_SUMMARY.md` - Executive overview
- `ARTIFACTS_MANIFEST.md` - This file

**Code Repository:**
- `/Users/shantanu/ICU-Sentinel/`

**Dataset:**
- PhysioNet/CinC Challenge 2019 v1.0.0
- https://physionet.org/content/challenge-2019/

---

**Manifest Generated:** November 10, 2026  
**Status:** COMPLETE (core artifacts), PARTIAL (supplementary analyses)  
**Primary Evaluator:** Kiro AI System  
