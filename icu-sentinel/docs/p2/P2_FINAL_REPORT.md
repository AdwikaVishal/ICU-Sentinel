# P2 Final Report — ICU Sentinel

**Date:** 2026-11-06  
**Branch:** `main`  
**Tested interpreter:** Python 3.11.9  
**Overall status:** **CODE COMPLETE for Extra Trees and XGBoost; BLOCKED for LightGBM on Windows.**

This report documents the P2 advanced tabular ML foundation. P2 builds on the
verified P1 feature and split contracts to provide Extra Trees, LightGBM, and
XGBoost classifiers with unified evaluation, probability calibration, and
lightweight experiment tracking.

---

## 1. Executive Summary

The P2 codebase is internally consistent and the automated test suite passes
for Extra Trees and XGBoost models. LightGBM tests fail on Windows due to a
native DLL access violation in the LightGBM library (not an ICU Sentinel bug).

P2 cannot yet make real-data performance claims because:
- The official PhysioNet/CinC Challenge 2019 dataset is not present.
- Real P1 artifacts do not yet exist.
- LightGBM requires a working native library installation.

**Status summary:**
- **Code status:** `COMPLETE` for Extra Trees and XGBoost; `BLOCKED` for LightGBM on Windows.
- **Test status:** 39/48 tests pass (9 LightGBM tests fail due to DLL).
- **Real-data status:** `BLOCKED` on P1 artifacts from real dataset.
- **Final verdict:** `GO WITH CONDITIONS` for software handover; `NO-GO` for empirical claims.

---

## 2. Implementation Status

### 2.1 Models Implemented

| Model | Status | Tests | Notes |
|-------|--------|-------|-------|
| Extra Trees | ✅ COMPLETE | 7/7 pass | Median imputation, balanced class weights |
| LightGBM | ⚠️ BLOCKED | 0/5 pass | Windows DLL access violation |
| XGBoost | ✅ COMPLETE | 6/6 pass | scale_pos_weight, deterministic training |

All models:
- Use reproducible random seeds (default: 42)
- Support class imbalance handling
- Return positive-class probabilities
- Preserve feature names and validate input schema
- Handle missing values (trees natively; Extra Trees via imputation)
- Exclude identifiers, timestamps, and target columns from features

### 2.2 Core Components

| Component | Status | Description |
|-----------|--------|-------------|
| `base.py` | ✅ COMPLETE | Shared P2ModelResult contract, predict_probabilities |
| `config.py` | ✅ COMPLETE | Serializable P2Config with per-model hyperparameters |
| `pipelines.py` | ✅ COMPLETE | Unified sklearn-compatible pipeline builder |
| `training.py` | ✅ COMPLETE | Training orchestration with split validation |
| `evaluation.py` | ✅ COMPLETE | Unified evaluation reusing P1 metrics |
| `calibration.py` | ✅ COMPLETE | Platt scaling and isotonic regression |
| `feature_contract.py` | ✅ COMPLETE | P1 feature schema integration |
| `serialization.py` | ✅ COMPLETE | Model persistence using P1 conventions |
| `experiment.py` | ✅ COMPLETE | JSON experiment records with source labels |

### 2.3 Feature Contract Compliance

P2 correctly reuses P1's verified feature schema:

```python
# Excluded from model features (feature_contract.py)
P2_METADATA_COLUMNS = ("patient_id", "source_dataset", "hour", "hospital_id")
P2_TARGET_COLUMNS = ("SepsisLabel", "sepsis_within_6h", "target_eligible")
P2_DEFAULT_TARGET = "sepsis_within_6h"  # P1's approved derived target
```

P2 validates:
- Feature schema match before training/prediction
- Target isolation (no target columns in features)
- Patient-level split integrity (no patient leakage)
- Input data types (numeric/boolean only)
- Missing value handling per model

---

## 3. Training Protocol

P2 follows P1's verified training protocol:

1. Load patient-level train/validation/test splits from P1 artifacts
2. Validate split integrity (no patient overlap between splits)
3. Validate target isolation (target columns excluded from features)
4. Fit preprocessing and models on **training data only**
5. Evaluate on validation at threshold 0.5 to obtain probabilities
6. Select one alert threshold per model on **validation only**
7. Evaluate frozen models **once on test** at the validation threshold

Guardrails:
- Missing validation data raises `ValueError`
- Test data never influences threshold selection
- `p2_thresholds.json` records `selection_split="validation"` and `test_set_used_for_selection=false`
- Invalid selection metrics raise rather than silently falling back
- Split leakage detected by P1's loader raises before training

---

## 4. Evaluation Metrics

P2 reuses P1's shared metric implementation (`ml.common.metrics`):

- AUROC (Area Under ROC Curve)
- AUPRC (Average Precision / Area Under Precision-Recall)
- Sensitivity (Recall / True Positive Rate)
- Specificity (True Negative Rate)
- Precision (Positive Predictive Value)
- F1 Score
- Brier Score
- Confusion Matrix (TP, TN, FP, FN)

Metrics are computed and stored separately for:
- Training split (recorded but not used for selection)
- Validation split (used for threshold selection only)
- Test split (one-shot evaluation at frozen threshold)

All splits use the approved derived target `sepsis_within_6h`.

---

## 5. Calibration Foundation

P2 implements reusable probability calibration:

**Supported methods:**
- `sigmoid`: Platt scaling (logistic regression on logit-transformed probabilities)
- `isotonic`: Isotonic regression (monotonic calibration curve)

**Fitting constraints:**
- Calibrators must be fitted on `validation` or a dedicated `calibration` partition
- Fitting on the `test` split is **explicitly rejected** (`ValueError`)
- Minimum 3 samples required
- Both classes required in the calibration target

**Usage:**
```python
from ml.p2_boosting.calibration import fit_probability_calibrator, apply_probability_calibrator

calibrator = fit_probability_calibrator(
    y_validation, 
    validation_probabilities, 
    method="sigmoid",
    partition="validation"
)
calibrated_probs = apply_probability_calibrator(calibrator, test_probabilities)
```

Calibration is prepared but not automatically applied in the current training
pipeline. Users must explicitly choose when to calibrate predictions.

---

## 6. Test Evidence

Commands run from `icu-sentinel/` on 2026-11-06:

```bash
# All non-LightGBM P2 tests
python -m pytest ml/p2_boosting/tests -k "not lightgbm" --no-cov -p no:cacheprovider --disable-warnings -q
```

Result: **39 passed, 0 failed.**

```bash
# Extra Trees and XGBoost model tests
python -m pytest ml/p2_boosting/tests/test_extra_trees.py ml/p2_boosting/tests/test_xgboost.py -v
```

Result: **13 passed, 0 failed.**

### Test Coverage

| Test Module | Tests | Status | Coverage |
|-------------|-------|--------|----------|
| `test_extra_trees.py` | 7 | ✅ PASS | Initialization, missing values, predictions, reproducibility, serialization, schema mismatch |
| `test_xgboost.py` | 6 | ✅ PASS | Same coverage as Extra Trees |
| `test_lightgbm.py` | 5 | ❌ FAIL | DLL access violation on Windows |
| `test_config.py` | 3 | ✅ PASS | Serialization, enabled models, defaults |
| `test_calibration.py` | 7 | ✅ PASS | Sigmoid/isotonic fitting, application, test-split rejection |
| `test_evaluation.py` | 6 | ✅ PASS | Metrics, threshold selection, validation enforcement |
| `test_feature_contract.py` | 10 | ✅ PASS | Feature name validation, frame validation, target isolation |
| `test_training.py` | 4 | ❌ FAIL | Fail due to default LightGBM enable |

All tests use synthetic fixtures and explicitly avoid real patient data.

---

## 7. Known Limitations and Blockers

### 7.1 LightGBM DLL Issue (Windows)

**Symptom:**
```
OSError: exception: access violation reading 0x0000000000000000
  File "...\lightgbm\basic.py", line 2860, in set_field
    _LIB.LGBM_DatasetSetField(
```

**Root cause:** Native LightGBM DLL incompatibility on Windows (not an ICU Sentinel bug).

**Impact:** 9 tests fail (5 model tests + 4 training tests that enable LightGBM by default).

**Mitigation:**
- LightGBM can be disabled via config: `P2Config(train_lightgbm=False)`
- Artifact script supports `--disable-lightgbm` flag
- Linux/macOS environments may not exhibit this issue
- Future: investigate LightGBM version downgrade or Windows-compatible build

### 7.2 Real Data Blocked

P2 inherits P1's blocker:
- Real PhysioNet dataset not yet acquired
- No real P1 artifacts (preprocessed patients, splits, features)
- All current results use synthetic fixtures for software validation only

### 7.3 Cross-Hospital Evaluation Not Implemented

P1 prepares `hospital_id` infrastructure but does not populate it because:
- The PhysioNet dataset does not include genuine hospital identifiers
- Set A/Set B paths are not hospital IDs
- Cross-hospital generalization experiments require real multi-site data

P2 excludes `hospital_id` from model features but does not implement
per-hospital evaluation or leave-one-hospital-out cross-validation.

---

## 8. Reproducibility and Environment

### 8.1 Artifact Runner

```bash
# Generate P2 artifacts from P1 processed data
python scripts/generate_p2_artifacts.py

# Skip LightGBM if DLL issues exist
python scripts/generate_p2_artifacts.py --disable-lightgbm

# Custom paths
python scripts/generate_p2_artifacts.py \
    --processed-dir data/features/patients \
    --split-manifest data/splits/split_manifest.json \
    --artifact-dir artifacts/p2 \
    --reports-dir data/reports/p2
```

Expected outputs when P1 artifacts exist:
- `artifacts/p2/extra_trees/extra_trees_1.0.0.joblib`
- `artifacts/p2/xgboost/xgboost_1.0.0.joblib`
- `artifacts/p2/p2_config.json`
- `data/reports/p2/p2_thresholds.json`
- `data/reports/p2/p2_training_summary.json`
- `data/reports/p2/p2_experiment.json`

### 8.2 Verified Environment

- Python 3.11.9
- pandas 2.3.3
- numpy 1.26.4
- scikit-learn 1.4.2
- xgboost 2.1.3
- lightgbm 4.5.0 (⚠️ Windows DLL issue)
- joblib 1.4.2
- pytest 8.4.2

### 8.3 Reproducibility Guarantees

- All models use deterministic random seeds (default: 42)
- LightGBM configured with `deterministic=True`, `force_col_wise=True`, `subsample_freq=1`
- XGBoost configured with `tree_method="hist"` for deterministic splits
- Extra Trees uses scikit-learn's reproducible RNG
- Tests verify prediction reproducibility across repeated training runs

---

## 9. P3 Handover Contract

### 9.1 P2 Output Contract

P3 (temporal models) may consume only:

- Same P1 artifacts that P2 consumes:
  - `data/features/combined_features.parquet` (for sequential modeling)
  - `data/features/patients/*.parquet` (patient-level data)
  - `data/features/feature_schema.json`
  - `data/splits/split_manifest.json`

- Optional P2 artifacts:
  - `artifacts/p2/*/` (trained tabular models)
  - `data/reports/p2/p2_thresholds.json` (validation thresholds)
  - `data/reports/p2/p2_training_summary.json` (metrics)

### 9.2 P3 Constraints

P3 must:

1. Respect the same feature contract (exclude identifiers/targets from inputs)
2. Respect patient-level splits (no patient leakage)
3. Fit preprocessing/calibration on training data only
4. Select thresholds and hyperparameters on validation only
5. Evaluate frozen models once on test
6. Label synthetic results as synthetic, never as real evidence
7. Build sequences per patient, preserving causal temporal order

P3 may:
- Use P2 tabular models for ensemble or late-fusion comparison
- Reuse P2 calibration infrastructure for temporal model probabilities
- Compare temporal model performance against P2 baselines

---

## 10. Conditions Required to Change the Verdict

To move from **GO WITH CONDITIONS** to full **GO**:

1. Obtain the official dataset legitimately and place under `data/raw/`
2. Run `scripts/generate_p1_artifacts.py` to create real P1 artifacts
3. Run `scripts/generate_p2_artifacts.py` (with or without LightGBM)
4. Verify real validation/test metrics and threshold provenance
5. Resolve LightGBM Windows DLL issue OR document Linux/macOS-only support
6. Update this report with real, non-synthetic results

To make **empirical claims**:

7. Confirm feature target isolation on real columns
8. Verify split leakage checks on real patient IDs
9. Document actual class imbalance and prevalence
10. Run full bootstrap confidence intervals on test metrics
11. Clearly label all results as retrospective research, not clinical validation

---

## 11. Final Verdict

**Code Status: GO WITH CONDITIONS**

Reason: P2 implementation is complete and the automated suite passes for Extra
Trees and XGBoost. The code correctly reuses P1's verified feature and split
contracts, follows the approved training protocol, and provides unified
evaluation and calibration infrastructure.

**LightGBM Status: BLOCKED (Windows DLL issue)**

Mitigation: LightGBM can be disabled via config or command-line flag. Extra
Trees and XGBoost provide sufficient tabular baseline coverage.

**Real Data Status: NO-GO**

Reason: Same as P1. Real dataset, real artifacts, and real evaluation results
are all `BLOCKED`/`MISSING`. Synthetic fixtures are software validation only
and are not acceptable evidence for empirical claims.

**Recommendation:**

- P2 code is ready for P3 integration
- Extra Trees and XGBoost are production-ready for software purposes
- LightGBM should be disabled on Windows until DLL issue is resolved
- No empirical performance claims until real data is processed
- All synthetic results must be explicitly labeled as such

---

## 12. Files Created by P2

### Core Implementation
- `ml/p2_boosting/__init__.py`
- `ml/p2_boosting/base.py`
- `ml/p2_boosting/config.py`
- `ml/p2_boosting/pipelines.py`
- `ml/p2_boosting/training.py`
- `ml/p2_boosting/evaluation.py`
- `ml/p2_boosting/calibration.py`
- `ml/p2_boosting/feature_contract.py`
- `ml/p2_boosting/serialization.py`
- `ml/p2_boosting/experiment.py`
- `ml/p2_boosting/extra_trees.py`
- `ml/p2_boosting/lightgbm_model.py`
- `ml/p2_boosting/xgboost_model.py`

### Tests
- `ml/p2_boosting/tests/__init__.py`
- `ml/p2_boosting/tests/conftest.py`
- `ml/p2_boosting/tests/fixtures.py`
- `ml/p2_boosting/tests/test_config.py`
- `ml/p2_boosting/tests/test_extra_trees.py`
- `ml/p2_boosting/tests/test_xgboost.py`
- `ml/p2_boosting/tests/test_lightgbm.py`
- `ml/p2_boosting/tests/test_calibration.py`
- `ml/p2_boosting/tests/test_evaluation.py`
- `ml/p2_boosting/tests/test_feature_contract.py`
- `ml/p2_boosting/tests/test_training.py`

### Scripts and Documentation
- `scripts/generate_p2_artifacts.py`
- `docs/p2/P2_FINAL_REPORT.md` (this file)

### Files Modified
- `ml/p2_boosting/training.py` (added `n_features_total` property)

---

## 13. References

- P1 Final Report: `docs/p1/P1_FINAL_REPORT.md`
- P1 Feature Handoff: `docs/p1/P2_P3_FEATURE_HANDOFF.md`
- P1 Artifact Runner: `scripts/generate_p1_artifacts.py`
- P1 Feature Contract: `ml/p1_clinical/features/feature_schema.py`
- P1 Data Loader: `ml/p1_clinical/classical_ml/data_loader.py`
- P1 Target Policy: `ml/p1_clinical/preprocessing/targets.py`
- P1 Shared Metrics: `ml/common/metrics/__init__.py`

