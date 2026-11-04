# Classical ML Documentation — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Classical ML  
**Part:** PART 6 — Classical ML models  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Overview

P1 implements three classical ML models for early sepsis prediction:

1. **Logistic Regression** — Regularized linear model with probability output
2. **Random Forest** — Ensemble of decision trees with feature importance
3. **Support Vector Machine (SVM)** — Kernel-based classifier with probability estimates

All models use the shared Part 4 feature schema, Part 3 patient-level splits,
and Part 5 clinical baselines for comparison.

---

## 2. Model Implementations

### 2.1 Logistic Regression

**File:** `ml/p1_clinical/classical_ml/logistic_regression.py`

Pipeline:
1. SimpleImputer (median strategy, fitted on train only)
2. StandardScaler (fitted on train only)
3. LogisticRegression (configurable C, penalty, solver, class_weight)

**Configuration defaults:**
- C: 1.0
- Penalty: l2
- Solver: lbfgs
- Max iterations: 1000
- Class weight: balanced
- Add missingness indicator: False

**Outputs:** Model artifact, coefficients, feature importance (absolute coefficients).

### 2.2 Random Forest

**File:** `ml/p1_clinical/classical_ml/random_forest.py`

Pipeline:
1. SimpleImputer (median strategy, fitted on train only)
2. RandomForestClassifier (configurable n_estimators, max_depth, etc.)

**Configuration defaults:**
- n_estimators: 100
- Max depth: 10
- Min samples split: 2
- Min samples leaf: 1
- Class weight: balanced
- n_jobs: 1

**Outputs:** Model artifact, feature importance (Gini importance).

### 2.3 SVM

**File:** `ml/p1_clinical/classical_ml/svm.py`

Pipeline:
1. SimpleImputer (median strategy, fitted on train only)
2. StandardScaler (fitted on train only)
3. SVC (configurable C, kernel, gamma, class_weight, probability)

**Configuration defaults:**
- C: 1.0
- Kernel: rbf
- Gamma: scale
- Class weight: balanced
- Probability: True
- Max iterations: -1 (no limit)

**Outputs:** Model artifact, support vector count.

---

## 3. Preprocessing

All preprocessing is fitted on **training data only**:

| Step | Fitted On | Applied To |
|------|-----------|------------|
| Imputation | Training X | Train, Val, Test |
| Scaling | Training X | Train, Val, Test |
| Feature selection | Training X | Train, Val, Test |

No information from validation or test data influences fitted parameters.

---

## 4. Training Protocol

### 4.1 Training Set

- Fit imputers, scalers, feature selectors
- Fit models
- Training-only cross-validation where needed

### 4.2 Validation Set

- Compare model configurations
- Select hyperparameters
- Select alert thresholds
- Assess calibration if implemented

### 4.3 Test Set

- Evaluate final frozen models once
- No parameter or threshold tuning
- No model selection based on test results

---

## 5. Configuration

All parameters are centralized in `ml/p1_clinical/classical_ml/model_config.py`
and serializable to/from JSON.

Key configuration groups:
- Model hyperparameters (LR, RF, SVM)
- Preprocessing (imputation, scaling)
- Feature selection (constant, duplicate, all-missing, infinite)
- Class imbalance (class_weight strategy)
- Threshold selection (metric, grid step)

Configuration is saved alongside model artifacts for reproducibility.

---

## 6. Missing-Value Handling

- Missing values preserved as NaN from Part 3/4
- No clinical imputation by P1
- Imputers fitted on training data only
- Missingness indicators available from Part 4 features
- Infinite values detected and removed during feature selection

---

## 7. Class Imbalance

Strategy: Class weighting (`class_weight="balanced"`)

- Applied during model training
- No oversampling before splitting
- No SMOTE on temporal observations
- Validation/test not used for strategy selection

---

## 8. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AUROC | Area Under ROC Curve |
| AUPRC | Area Under Precision-Recall Curve |
| Accuracy | (TP+TN)/(TP+TN+FP+FN) |
| Precision | TP/(TP+FP) |
| Recall/Sensitivity | TP/(TP+FN) |
| Specificity | TN/(TN+FP) |
| F1-score | 2×(Precision×Recall)/(Precision+Recall) |
| Brier score | Mean squared probability error |
| Confusion matrix | TP, TN, FP, FN counts |

---

## 9. Clinical Baseline Comparison

ML models are compared against Part 5 clinical baselines:

- **SIRS**: Complete when all 4 components available
- **qSOFA**: Partial (AMS not available)
- **NEWS2-style**: Modified (O2/consciousness not available)

Comparison reports:
- Score coverage (complete/partial/none)
- Metrics on comparable evaluation populations
- Clear distinction between complete-case and full-dataset evaluation

---

## 10. Explainability Preparation

For each model, the following are saved:

- Model artifact (.joblib)
- Feature names
- Selected feature names
- Training configuration
- Preprocessing configuration
- Validation metrics
- Model version
- Training timestamp
- Dataset/split identity

**Logistic Regression:** Coefficients saved  
**Random Forest:** Feature importance saved  
**SVM:** Model metadata preserved for future explanation

SHAP implementation belongs to P3.

---

## 11. Model Artifacts

Artifacts saved under `artifacts/p1/classical_ml/`:

```
artifacts/p1/classical_ml/
├── logistic_regression/
│   ├── logistic_regression_1.0.0.joblib
│   ├── logistic_regression_1.0.0_metadata.json
│   └── logistic_regression_1.0.0_features.json
├── random_forest/
│   ├── random_forest_1.0.0.joblib
│   ├── random_forest_1.0.0_metadata.json
│   └── random_forest_1.0.0_features.json
├── svm/
│   ├── svm_1.0.0.joblib
│   ├── svm_1.0.0_metadata.json
│   └── svm_1.0.0_features.json
└── metadata/
    └── registry.json
```

Prediction outputs saved under `data/reports/classical_ml/`.

Patient identifiers are **not** used as model features.

---

## 12. Reproducibility

- Random seed: 42 (configurable)
- Configuration saved with each experiment
- Model artifacts versioned
- Training timestamp recorded
- Split identity recorded

Reproduction:
```python
from ml.p1_clinical.classical_ml.training import run_training_protocol
result = run_training_protocol(
    raw_dir="data/raw",
    processed_dir="data/processed",
    splits_dir="data/splits",
    reports_dir="data/reports",
    config=ModelConfig.load("config.json"),
)
```

---

## 13. Limitations

- Real data not accessed; synthetic fixtures only
- No clinical performance claims
- SVM may be computationally expensive on large datasets
- Class imbalance strategy is class weighting only (no oversampling)
- No clinical validation claimed
