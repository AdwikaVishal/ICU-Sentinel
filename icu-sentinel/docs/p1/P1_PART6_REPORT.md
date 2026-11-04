# ICU Sentinel — P1 PART 6 Report

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1)  
**Part:** PART 6 — Classical ML models  
**Status:** Complete (awaiting approval to begin PART 7)  
**Date:** 2026-11-03  

---

## 1. Implementation Summary

PART 6 implements three classical ML models for early sepsis prediction:
Logistic Regression, Random Forest, and SVM. All models use the shared
Part 4 feature schema, Part 3 patient-level splits, and Part 5 clinical
baselines for comparison. All code is tested using synthetic fixtures
(no real patient data available without PhysioNet credentialed access).

---

## 2. Files Created

### New source files

| File | Purpose |
|------|---------|
| `ml/p1_clinical/classical_ml/__init__.py` | Package exports |
| `ml/p1_clinical/classical_ml/model_config.py` | Centralized serializable configuration |
| `ml/p1_clinical/classical_ml/data_loader.py` | Split-aware dataset loading, overlap/integrity checks |
| `ml/p1_clinical/classical_ml/feature_selector.py` | Feature selection (constant, duplicate, missing, infinite) |
| `ml/p1_clinical/classical_ml/prediction_schema.py` | Prediction output schema |
| `ml/p1_clinical/classical_ml/model_registry.py` | Model artifact registry |
| `ml/p1_clinical/classical_ml/model_serialization.py` | Model save/load |
| `ml/p1_clinical/classical_ml/logistic_regression.py` | Logistic Regression pipeline |
| `ml/p1_clinical/classical_ml/random_forest.py` | Random Forest pipeline |
| `ml/p1_clinical/classical_ml/svm.py` | SVM pipeline |
| `ml/p1_clinical/classical_ml/training.py` | End-to-end training protocol |
| `ml/p1_clinical/classical_ml/inference.py` | Inference interface, threshold selection |

### New test files

| File | Tests |
|------|-------|
| `ml/p1_clinical/tests/test_model_config.py` | Config defaults, serialization, enabled models |
| `ml/p1_clinical/tests/test_data_loader.py` | Split-aware loading, overlap prevention, target isolation |
| `ml/p1_clinical/tests/test_feature_selector.py` | Constant, duplicate, missing, infinite feature removal |
| `ml/p1_clinical/tests/test_logistic_regression.py` | Training, prediction, coefficients, serialization |
| `ml/p1_clinical/tests/test_random_forest.py` | Training, prediction, feature importance, serialization |
| `ml/p1_clinical/tests/test_svm.py` | Training, prediction, serialization |
| `ml/p1_clinical/tests/test_training.py` | End-to-end pipeline, split usage, feature selection |
| `ml/p1_clinical/tests/test_inference.py` | Model loading, prediction, threshold selection, isolation |
| `ml/p1_clinical/tests/test_model_serialization.py` | Save/load model, preprocessing, features, metadata |
| `ml/p1_clinical/tests/test_model_leakage.py` | No patient overlap, test-set isolation, target isolation |

### New documentation

| File | Purpose |
|------|---------|
| `docs/p1/CLASSICAL_ML_DATA_AUDIT.md` | Pre-training data audit |
| `docs/p1/CLASSICAL_ML_DOCUMENTATION.md` | Full ML methodology |
| `docs/p1/CLASSICAL_ML_EVALUATION_PROTOCOL.md` | Evaluation protocol |
| `docs/p1/P1_PART6_REPORT.md` | This report |

### Modified files

None. All PART 6 files are new; no existing files were modified.

---

## 3. Pre-training Data Audit

The audit verified:

| Check | Result |
|-------|--------|
| Part 3 split manifests accessible | Yes |
| Part 4 feature schema accessible | Yes |
| Part 5 baseline schema accessible | Yes |
| Patient IDs consistent across modules | Yes |
| Timestamps consistent across modules | Yes |
| Feature names match schema | Yes |
| Target labels preserved | Yes |
| Train/validation/test patient overlap | None detected |
| SepsisLabel used as target only | Yes |
| Identifiers excluded from features | Yes |

No contract mismatches were found.

---

## 4. Feature Selection

Feature selection is applied on training data only:

| Exclusion | Applied | Reason |
|-----------|---------|--------|
| Identifiers (patient_id, source_dataset, hour, source_file) | Yes | Metadata, not predictive |
| Target (SepsisLabel) | Yes | Target only |
| Constant features | Yes | No information content |
| Duplicate features | Yes | Redundant |
| All-missing features | Yes | Cannot impute |
| Infinite values | Yes | Not numerically valid |

The selected feature list is preserved for every model. No arbitrary large
groups of features are discarded.

---

## 5. Model Implementations

### Logistic Regression

- Pipeline: Imputer → StandardScaler → LogisticRegression
- Configurable: C, penalty, solver, max_iter, class_weight
- Outputs: Model, coefficients, feature importance, training metrics

### Random Forest

- Pipeline: Imputer → RandomForestClassifier
- Configurable: n_estimators, max_depth, min_samples_split, min_samples_leaf, class_weight
- Outputs: Model, feature importance (Gini), training metrics

### SVM

- Pipeline: Imputer → StandardScaler → SVC (probability=True)
- Configurable: C, kernel, gamma, class_weight, max_iter
- Outputs: Model, support vector count, training metrics

---

## 6. Hyperparameter Configuration

All hyperparameters are centralized in `ml/p1_clinical/classical_ml/model_config.py`
and serializable to JSON.

| Parameter | Default |
|-----------|---------|
| Random seed | 42 |
| LR C | 1.0 |
| LR penalty | l2 |
| LR solver | lbfgs |
| LR max_iter | 1000 |
| LR class_weight | balanced |
| RF n_estimators | 100 |
| RF max_depth | 10 |
| RF class_weight | balanced |
| SVM C | 1.0 |
| SVM kernel | rbf |
| SVM gamma | scale |
| SVM class_weight | balanced |
| SVM probability | True |
| Impute strategy | median |
| Scale with mean/std | True/True |

---

## 7. Class Imbalance Strategy

Strategy: **Class weighting** (`class_weight="balanced"`)

- Applied during model training for all three models
- No oversampling before splitting
- No SMOTE on temporal observations
- Validation/test data not used for strategy selection

Rationale: Class weighting is computationally efficient, does not
introduce synthetic samples, and preserves the original data distribution
for evaluation.

---

## 8. Training Protocol

| Phase | Actions | Data Used |
|-------|---------|-----------|
| Training | Fit imputers, scalers, selectors, models | Train split only |
| Validation | Compare configs, select hyperparameters, select thresholds | Validation split only |
| Test | Evaluate final frozen models once | Test split only |

**Critical constraints:**
- Preprocessing fitted on training data only
- Validation/test never influence fitted parameters
- Test set untouched until final evaluation
- Thresholds tuned on validation data only

---

## 9. Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| AUROC | Threshold-independent | Area Under ROC Curve |
| AUPRC | Threshold-independent | Area Under PR Curve |
| Accuracy | Threshold-dependent | (TP+TN)/Total |
| Precision | Threshold-dependent | TP/(TP+FP) |
| Sensitivity/Recall | Threshold-dependent | TP/(TP+FN) |
| Specificity | Threshold-dependent | TN/(TN+FP) |
| F1-score | Threshold-dependent | 2×(P×R)/(P+R) |
| Brier score | Threshold-independent | Mean squared probability error |

Alert-level metrics:
- False alerts per patient-day
- Lead time
- Patient-level detection rate
- Alert sensitivity

---

## 10. Clinical Baseline Comparison

ML models compared against Part 5 baselines:

| Baseline | Status | Comparison Basis |
|----------|--------|-----------------|
| SIRS | Complete when 4/4 components | Complete-case subset |
| qSOFA | Partial (AMS unavailable) | Hours with AMS available |
| NEWS2-style | Modified (O2/consciousness unavailable) | Labeled as modified |

Comparison reports include score coverage and metrics on comparable
populations.

---

## 11. Model Artifacts

Artifacts saved under `artifacts/p1/classical_ml/`:

| Artifact | Description |
|----------|-------------|
| `logistic_regression_1.0.0.joblib` | Fitted model + preprocessing |
| `random_forest_1.0.0.joblib` | Fitted model + preprocessing |
| `svm_1.0.0.joblib` | Fitted model + preprocessing |
| `*_1.0.0_metadata.json` | Model metadata |
| `*_1.0.0_features.json` | Selected feature list |
| `registry.json` | Model registry |

Prediction outputs saved under `data/reports/classical_ml/`.

---

## 12. Test Results

**Total: 292 tests passed, 0 failed.**

Part 6 tests: 67 tests across 10 test files:
- test_model_config.py: Configuration defaults, serialization
- test_data_loader.py: Split-aware loading, overlap prevention
- test_feature_selector.py: Feature filtering
- test_logistic_regression.py: Training, prediction, coefficients
- test_random_forest.py: Training, prediction, importance
- test_svm.py: Training, prediction
- test_training.py: End-to-end pipeline
- test_inference.py: Prediction, threshold selection, isolation
- test_model_serialization.py: Save/load artifacts
- test_model_leakage.py: No overlap, test-set isolation

Parts 1-5 regression tests: 225 tests (all passing)

Ruff checks: All passed (0 errors)

---

## 13. Synthetic vs Real-Data Verification

**Real-data verification: Not available.**

Real PhysioNet/CinC Challenge 2019 data requires credentialed access
which has not been obtained. All training, evaluation, and reports use
synthetic fixtures only.

All reports in `data/reports/classical_ml/` are marked
`"source": "synthetic"` with explicit notes that they are not real
PhysioNet/CinC Challenge 2019 statistics.

**No clinical model performance is reported as evidence of real
predictive performance.** No clinical validation claims are made.

---

## 14. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No real data access | Cannot validate against actual dataset | Pipeline ready; synthetic tests pass |
| SVM computationally expensive | May be slow on large datasets | Reasonable defaults for local execution |
| Class imbalance | Low positive prevalence | Class weighting applied |
| No temporal cross-validation | Single split evaluation | Architecture supports future extension |
| No external validation | Cannot generalize | Architecture prepared; no claims made |
| Clinical baselines partial | Missing variables | Deviations documented |

---

## 15. P2 Handoff Requirements

P2 (boosting models: Extra Trees, LightGBM, XGBoost, CatBoost) can
consume P1 outputs:

1. **Shared feature schema**: `data/features/feature_schema.json`
   - ~994 features, all causal, tabular-appropriate
   - Feature names and types documented

2. **Patient-level splits**: `data/splits/split_manifest.json`
   - Train/validation/test patient IDs
   - No patient overlap between splits

3. **Processed features**: `data/processed/{patient_id}.parquet`
   - Per-patient feature data
   - Original SepsisLabel preserved

4. **Model artifacts**: `artifacts/p1/classical_ml/`
   - Baseline model performance for comparison
   - Feature importance for reference

5. **Clinical baselines**: `data/reports/classical_ml/`
   - SIRS, qSOFA, NEWS2-style scores for comparison

P2 should:
- Use the same patient-level splits
- Use the same feature schema (or subset)
- Compare against P1 classical ML baselines
- Compare against P1 clinical baselines (SIRS/qSOFA/NEWS2-style)

---

## 16. Remaining Work

1. Obtain PhysioNet credentialed access and download real dataset
2. Run Part 3-6 pipeline against real data
3. Validate model performance against real data
4. P1 Part 7: Research evaluation (lead time, alert burden, feature ablation, subgroup analysis)

---

## Data Integrity Statement

- No raw patient data was downloaded (no PhysioNet credentials)
- No raw patient data was committed to Git
- No model performance statistics were fabricated; all reports marked synthetic
- Source data remains unchanged (pipeline only reads Part 3/4 output)
- Malformed records are never silently discarded

Awaiting approval to begin PART 7 (Research evaluation: lead time, alert burden, feature ablation, subgroup analysis).
