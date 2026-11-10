# Calibration — ICU Sentinel

This document covers the calibration pipeline, Brier scores, Expected Calibration Error (ECE), and the verified status of probability calibration for the ICU Sentinel sepsis early-warning models.

## Calibration Status

| Aspect | Status |
|---|---|
| `calibration_method` | `sigmoid` (configured in `P2Config`) |
| Calibrator fitted on validation data | **Complete** — sigmoid calibrators fitted on validation split for all three models (XGBoost, LightGBM, Extra Trees) |
| Brier score | Present in P2 training summaries (uncalibrated) and in artifact metadata (calibrated) |
| ECE | **Computed on test data**: XGBoost=0.0112, LightGBM=0.0124, Extra Trees=0.0102 |
| Calibrator type | Sigmoid regression (Platt scaling via `LogisticRegression(C=1e6, solver="lbfgs")` |
| Calibrated Brier scores | **Improve over uncalibrated for all three models** |
| Calibrator wired into P3 prediction pipeline | **Yes** — `ModelPredictionProvider.predict()` applies calibrator when available in bundle metadata |

## Current State

### What Is Present

- **Fitted calibrator joblib files** for all three models:
  - `artifacts/p2/xgboost/xgboost_1.0.0_calibrator.joblib`
  - `artifacts/p2/lightgbm/lightgbm_1.0.0_calibrator.joblib`
  - `artifacts/p2/extra_trees/extra_trees_1.0.0_calibrator.joblib`

- **Calibrator metadata** in artifact JSON files, containing:
  - `method`: `"sigmoid"`
  - `partition`: `"validation"` (fitted on validation split only, test split untouched)
  - `n_samples`: 50 (validation set size)
  - `ece_test`: XGBoost=0.01117, LightGBM=0.01245, Extra Trees=0.01020
  - `brier_original`: XGBoost=0.1416, LightGBM=0.1494, Extra Trees=0.1406
  - `brier_calibrated`: XGBoost=0.1277, LightGBM=0.1285, Extra Trees=0.1277

- **`P2Config.calibration_method`** is set to `"sigmoid"` in the training configuration.

- **Calibration pipeline is wired into the P3 prediction pathway**: When real model artifacts are loaded via `ICU_SENTINEL_P1_MODEL_DIR` and `ICU_SENTINEL_P1_MODEL_NAME` env vars, the P3 container (_resolve_bundle) attaches the calibrator to the bundle metadata, and `ModelPredictionProvider.predict()` applies the sigmoid calibrator to predicted probabilities before returning risk scores.

- **Calibrated Brier scores improve over uncalibrated** for all three models (documented in artifact metadata).

### Calibration Pipeline Summary

The complete calibration pipeline consists of:

1. **Step 1**: Fit sigmoid calibrator on validation split only (50 patients) for each model
   - Uses `sklearn.linear_model.LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=42)`
   - Fits on logit-transformed probabilities and true labels
   - Critical: test split is never used for fitting

2. **Step 2**: Calibrator persisted as `*_1.0.0_calibrator.joblib` and metadata JSON
   - Method: `"sigmoid"`, Partition: `"validation"`, N samples: 50
   - ECE and Brier scores computed on test split and stored in metadata

3. **Step 3**: P3 prediction pipeline applies calibrator automatically
   - `_resolve_bundle()` loads calibrator from `artifacts/p2/<model_name>/*_1.0.0_calibrator.joblib`
   - Calibrator attached to `bundle.metadata["calibrator"]`, `bundle.metadata["calibration_method"] = "sigmoid"`, `bundle.metadata["calibration_partition"] = "validation"`
   - `ModelPredictionProvider.predict()` applies `apply_probability_calibrator(calibrator, probabilities)` when calibrator is present

4. **Step 4**: ECE computed on test data (independent from validation)
   - XGBoost: ECE = 0.0112
   - LightGBM: ECE = 0.0124
   - Extra Trees: ECE = 0.0102

### Historical vs Actual Calibration Status

| Metric | Earlier Reports | **Actual Status** |
|---|---|---|
| Brier score present | Yes (in training_summary.json) | Yes (verified in metadata JSON) |
| ECE computed | No (reported as "NOT DONE") | **Computed on test data**: XGBoost=0.0112, LightGBM=0.0124, Extra Trees=0.0102 |
| Calibrator fitted | No (marked as blocker) | **Fitted on validation data for all three models** |
| Calibrated Brier scores | N/A | **Improve over uncalibrated for all three models** |
| Calibrator wired into P3 pipeline | No | **Wired into `ModelPredictionProvider.predict()`** |

## Calibration in the P3 Container

The P3 container's `_resolve_bundle()` method loads model bundles with calibration attached:

- **Priority 1** (explicit config): When `ICU_SENTINEL_P1_MODEL_DIR` and `ICU_SENTINEL_P1_MODEL_NAME` are set, the model bundle is loaded and the calibrator is attached from `artifacts/p2/<model_name>/*_1.0.0_calibrator.joblib`
- **Priority 2** (auto-detect): When the data directory points to the full PhysioNet processed dataset, auto-detected model artifacts also have calibrators attached
- **Priority 3** (demo fallback): When no real-data configuration is present, the demo model is used without calibration (opt-in via `ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT=false`)

## Calibration and Alert Thresholds

The ICU Sentinel alert engine uses probability thresholds to generate sepsis alerts.

- **Threshold selection**: Fixed on validation F1 maximization (see MODEL_EVALUATION.md)
- **Calibrated thresholds**: Available — alert thresholds can be re-selected on calibrated validation probabilities
- **Impact of calibration**: If a sigmoid calibrator is fitted and applied, the probability values will shift, potentially requiring threshold adjustment

**Recommendation**: Once calibration is implemented, re-select alert thresholds on the calibrated validation probabilities using the same F1-maximization procedure.

## Next Steps for Calibration (Post-Publication, Not Blocking)

1. [x] Fit an isotonic regression or sigmoid calibrator on the **validation split only** — completed for all three models
2. [x] Apply the calibrator to **test set predictions** and compute calibrated ECE and Brier — completed, ECE: XGBoost=0.0112, LightGBM=0.0124, Extra Trees=0.0102
3. [x] Update `artifacts/p2/<model_name>/<model_name>_1.0.0_metadata.json` with calibrated metrics — completed
4. [x] Wire calibration into the P3 prediction pipeline (`backend/p3/container.py`) — completed
5. [ ] Re-select alert thresholds on calibrated probabilities
6. [ ] Add bootstrap confidence intervals for calibration metrics
7. [ ] Add SHAP explanations on real trained models
8. [ ] Per-source-dataset (Set A vs Set B) model evaluation with "not hospitals" disclaimer

## References

- `P2Config.calibration_method = "sigmoid"` — training configuration
- `artifacts/p2/xgboost/xgboost_1.0.0_metadata.json` — XGBoost calibration metadata
- `artifacts/p2/lightgbm/lightgbm_1.0.0_metadata.json` — LightGBM calibration metadata
- `artifacts/p2/extra_trees/extra_trees_1.0.0_metadata.json` — Extra Trees calibration metadata
- `ml/p2_boosting/calibration.py` — calibration pipeline (fit_probability_calibrator, apply_probability_calibrator)
- `ml.p3_temporal.replay.prediction_provider` — P3 prediction provider with calibration applied
- `backend/p3/container.py` — P3 container with calibrator loading via `_load_calibrator` and `_resolve_bundle`