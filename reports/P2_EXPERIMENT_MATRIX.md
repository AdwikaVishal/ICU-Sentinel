# P2 Experiment Matrix

## Core Comparison Models

| Model | Implementation | Enabled | Library |
|-------|---------------|---------|---------|
| Logistic Regression | P1 Classical ML | ✅ (P1) | scikit-learn |
| Random Forest | P1 Classical ML | ✅ (P1) | scikit-learn |
| SVM | P1 Classical ML | ✅ (P1) | scikit-learn |
| Extra Trees | P2 Boosting | ✅ | scikit-learn |
| XGBoost | P2 Boosting | ✅ | xgboost |
| LightGBM | P2 Boosting | ✅ | lightgbm |

## Experimental Design

### Data Splits (from P1 - Fixed)
- **Train**: 350 patients, 10,501 eligible rows
- **Validation**: 50 patients, 1,476 eligible rows
- **Test**: 100 patients, 3,001 eligible rows
- **Target**: `sepsis_within_6h` (strict 6h pre-onset)
- **Features**: 659 selected features (P1 feature selection on train only)
- **No patient overlap**: Verified

### Shared Protocol (All Models)
1. **Preprocessing** (fitted on train only):
   - Median imputation + missing indicator (Extra Trees)
   - No imputation (LightGBM, XGBoost handle NaN natively)
   - Standard scaling (Logistic Regression, SVM)
   
2. **Training**:
   - Fit on train split only
   - Class weighting: `balanced` for all models
   - Random seed: 42
   
3. **Threshold Selection** (on validation only):
   - Metric: F1-score
   - Grid search: 0.00 to 1.00 step 0.01
   - Test set NEVER used for threshold selection
   
4. **Evaluation** (on test, one-shot):
   - Metrics: AUROC, AUPRC, Sensitivity, Specificity, Precision, F1, Brier
   - Confusion matrix at validation-selected threshold
   - ROC curve data (FPR, TPR, thresholds)
   - PR curve data (Precision, Recall, thresholds)
   - Calibration: reliability diagrams, ECE

### Hyperparameter Strategy

| Model | Tuning Approach |
|-------|-----------------|
| Extra Trees | Fixed defaults (n_estimators=200, max_depth=None) |
| XGBoost | Fixed defaults (n_estimators=200, max_depth=4, lr=0.05) |
| LightGBM | Fixed defaults (n_estimators=200, num_leaves=31, lr=0.05) |
| Logistic Regression | Fixed (C=1.0, l2, max_iter=1000) |
| Random Forest | Fixed (n_estimators=100, max_depth=10) |
| SVM | Fixed (C=1.0, rbf, gamma=scale) |

**Rationale**: Prioritize reproducibility and fair comparison over extensive tuning. All models use sensible defaults appropriate for imbalanced clinical data.

### Evaluation Metrics (All Models)

| Metric | Purpose |
|--------|---------|
| AUROC | Discrimination (ranking) |
| AUPRC | Discrimination (imbalanced) |
| Sensitivity | Clinical: catch sepsis cases |
| Specificity | Clinical: avoid false alerts |
| Precision | Clinical: PPV of alerts |
| F1 | Balance precision/recall |
| Brier Score | Overall probabilistic accuracy |
| ECE | Calibration quality |

### Statistical Validation

- **Bootstrap**: Patient-level resampling (n=1000 iterations)
- **Unit**: Patient (not hourly rows)
- **Metrics with CIs**: AUROC, AUPRC, Sensitivity, Specificity, Precision, F1, Brier
- **Random seed**: 42

## Expected Outputs

### Per-Model Artifacts
```
artifacts/p2/
├── extra_trees/
│   ├── extra_trees_1.0.0.joblib
│   └── metadata.json
├── lightgbm/
│   ├── lightgbm_1.0.0.joblib
│   └── metadata.json
├── xgboost/
│   ├── xgboost_1.0.0.joblib
│   └── metadata.json
├── model_metrics.csv
├── model_predictions/
│   ├── extra_trees_test.parquet
│   ├── lightgbm_test.parquet
│   └── xgboost_test.parquet
├── confusion_matrices/
│   ├── extra_trees.json
│   ├── lightgbm.json
│   └── xgboost.json
├── roc_data/
│   ├── extra_trees.json
│   ├── lightgbm.json
│   └── xgboost.json
├── pr_data/
│   ├── extra_trees.json
│   ├── lightgbm.json
│   └── xgboost.json
├── calibration_results.json
├── bootstrap_results.json
└── bootstrap_confidence_intervals.csv
```

### Reports
- `reports/P2_MODEL_EVALUATION.md` - Full model comparison
- `reports/P2_CALIBRATION.md` - Calibration analysis
- `reports/P2_STATISTICAL_VALIDATION.md` - Bootstrap CIs
- `reports/P2_RESEARCH_INTERPRETATION.md` - Clinical interpretation
- `reports/P2_LIGHTGBM_BLOCKER.md` - LightGBM status (working on this platform)

## LightGBM Status

On this macOS (Apple Silicon) environment with Python 3.14:
- LightGBM 4.7.0 installed successfully
- All LightGBM tests pass (7/7)
- No DLL compatibility issues
- Training completes without errors

**Conclusion**: LightGBM blocker is **RESOLVED** for this platform. Include in full comparison.

## Execution Order

1. Run P2 training on P1 artifacts → `artifacts/p2/`
2. Generate predictions on test set
3. Compute metrics, ROC/PR curves
4. Run calibration (validation-fitted, test-evaluated)
5. Run patient-level bootstrap (1000 iterations)
6. Generate comparison tables and reports
7. Run P2 tests to verify

## Reproducibility

All experiments use:
- Fixed random seed: 42
- Same P1 splits (patient-level, stratified)
- Same feature set (659 features, selected on train only)
- Same target definition (`sepsis_within_6h`)
- Thresholds selected on validation only
- Test set evaluated once per model