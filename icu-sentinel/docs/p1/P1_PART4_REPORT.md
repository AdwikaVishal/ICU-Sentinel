# ICU Sentinel — P1 PART 4 Report

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1)  
**Part:** PART 4 — Temporal feature engineering  
**Status:** Complete (awaiting approval to begin PART 5)  
**Date:** 2026-11-03  

---

## 1. Implementation Summary

PART 4 implements a complete temporal feature-engineering pipeline for
the PhysioNet/CinC Challenge 2019 dataset. The pipeline transforms
cleaned hourly patient observations (Part 3 output) into a feature-rich
dataset suitable for P1 classical ML, P2 boosting models, and P3
temporal deep-learning models. All code is tested using synthetic
fixtures (no real patient data available without PhysioNet credentialed
access).

## 2. Files Created

### New source files

| File | Purpose |
|------|---------|
| `ml/p1_clinical/features/__init__.py` | Package exports |
| `ml/p1_clinical/features/feature_config.py` | Centralized serializable configuration |
| `ml/p1_clinical/features/feature_schema.py` | Stable feature schema (P2/P3 contract) |
| `ml/p1_clinical/features/current_features.py` | Raw observation features |
| `ml/p1_clinical/features/missingness_features.py` | Missingness indicators, time-since-last, availability |
| `ml/p1_clinical/features/rolling_features.py` | Causal rolling 6/12/24h statistics |
| `ml/p1_clinical/features/trend_features.py` | Absolute/relative change, slopes |
| `ml/p1_clinical/features/clinical_features.py` | Shock index, SIRS component count |
| `ml/p1_clinical/features/feature_pipeline.py` | End-to-end orchestration |

### New test files

| File | Tests |
|------|-------|
| `ml/p1_clinical/tests/test_current_features.py` | Patient ID/hour/label preservation, column order, no imputation, NaN preservation |
| `ml/p1_clinical/tests/test_missingness_features.py` | Missingness indicators, time-since-last, availability fractions, never-observed vs previously-observed |
| `ml/p1_clinical/tests/test_rolling_features.py` | Causal rolling mean/min/max/std/latest, window boundaries, insufficient obs, no future leakage |
| `ml/p1_clinical/tests/test_trend_features.py` | Abs/rel change, slope, zero denominator, no future leakage |
| `ml/p1_clinical/tests/test_clinical_features.py` | Shock index (zero/missing denominator), SIRS count (missing components not normal), label preservation |
| `ml/p1_clinical/tests/test_feature_pipeline.py` | End-to-end pipeline, label integrity, output schema, reproducibility, no patient contamination, config save/load |
| `ml/p1_clinical/tests/test_temporal_leakage.py` | Future observation doesn't affect hour t, no cross-patient contamination, sorting, missing hours, sparse observations, duplicate timestamps |
| `ml/p1_clinical/tests/test_feature_schema.py` | Feature entry/schema creation, category organization, serialization roundtrip, target not appropriate, all features causal |

### New data outputs

| File | Source | Notes |
|------|--------|-------|
| `data/features/feature_schema.json` | Feature schema | Stable contract for P2/P3 |
| `data/features/feature_config.json` | Feature config | Serialized configuration |
| `data/features/combined_features.parquet` | Synthetic pipeline output | Combined features (marked synthetic) |
| `data/reports/feature_summary.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/feature_missingness_report.csv` | Synthetic pipeline run | Per-feature missingness |
| `data/reports/feature_validation_report.json` | Synthetic pipeline run | Validation results (0 issues, causal passed) |

### New documentation

| File | Purpose |
|------|---------|
| `docs/p1/FEATURE_ENGINEERING_DOCUMENTATION.md` | Full feature methodology, categories, formulas |
| `docs/p1/P2_P3_FEATURE_HANDOFF.md` | Integration contract for P2 and P3 |
| `docs/p1/P1_PART4_REPORT.md` | This report |

### Modified files

None. All PART 4 files are new; no existing files were modified.

## 3. Feature Categories and Count

Generated from a synthetic run with 5 patients (108 rows):

| Category | Count | Tabular | Sequential |
|----------|-------|---------|------------|
| Current (raw observations) | 39 | Yes | Yes |
| Missingness (indicators, time-since-last, availability) | 102 | Yes | Yes |
| Forward-filled | 34 | Yes | Yes |
| Rolling (6/12/24h mean/min/max/std/latest) | 510 | Yes | Yes |
| Trend (abs/rel change, slope over 6/12/24h) | 306 | Yes | Yes |
| Clinical (shock index, SIRS count, components available) | 3 | Yes | Yes |
| **Total** | **~994** | | |

Exact count depends on configuration (which feature groups are enabled).

## 4. Configuration Defaults

| Parameter | Default |
|-----------|---------|
| Rolling windows | [6, 12, 24] hours |
| Rolling aggregations | mean, min, max, std, latest |
| Rolling min observations | 1 |
| Trend windows | [6, 12, 24] hours |
| Trend min observations | 2 |
| Forward-fill max window | default: 4h, vital_signs: 4h, laboratory: 6h |
| Feature groups enabled | All (missingness, forward_fill, rolling, trend, clinical) |
| Random seed | 42 |

Configuration is serialized to `data/features/feature_config.json` and
saved with each experiment.

## 5. Leakage Tests Performed

Automated leakage tests verify:

1. **No future observation affects hour t:** Modifying observations
   after hour t does not change features at hour t (for rolling,
   trend, and forward-fill features)

2. **No cross-patient contamination:** Patient A's data never appears
   in patient B's features

3. **Correct sorting:** Output sorted by patient_id and hour

4. **Missing hours:** Gaps in hour sequence handled correctly

5. **Sparse observations:** Patients with many missing values handled
   without crash

6. **Duplicate timestamps:** Duplicate hour values handled without crash

7. **Time-since-last-measurement:** `inf` for never-observed, correct
   values for previously-observed; no future leakage

8. **All schema features causal:** Every feature entry in the schema
   has `causal: true`

All leakage tests pass.

## 6. Target-Label Integrity

The official `SepsisLabel` is preserved unchanged through the feature
pipeline:

- Not shifted a second time
- Not reconstructed
- Not used as a model input feature (tabular_appropriate: false,
  sequential_appropriate: false)
- Tests verify labels match input after feature generation

## 7. Test Results

**Total: 169 tests passed, 0 failed.**

Breakdown:
- Part 1 foundation (`test_foundation.py`): 5 passed
- Part 2 ingestion (`test_ingestion.py`): 25 passed
- Part 3 preprocessing (`test_cleaning.py`, `test_patient_splits.py`,
  `test_preprocessing_pipeline.py`, `test_leakage.py`): 42 passed
- Part 4 features (8 test files): 97 passed

All tests use synthetic fixtures. No real patient data is used.

Ruff checks: All passed (0 errors).

## 8. Real-Data Verification Status

**Not available.** Real PhysioNet/CinC Challenge 2019 data requires
credentialed access which has not been obtained. All validation,
feature generation, and reports use synthetic fixtures only. The
pipeline is ready to run against Part 3 output once credentials and
data are obtained.

## 9. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No real data access | Cannot validate against actual dataset | Pipeline ready; synthetic tests pass |
| Forward-fill windows experimental | Not clinically established | Documented as requiring validation |
| No imputation | Downstream models must handle NaN | Documented; missingness indicators provided |
| Hospital ID inferred from source | May not match actual hospital IDs | Preserved when genuinely available |
| No cross-hospital validation | Cannot claim generalization | Architecture prepared; no claims made |
| PerformanceWarning on DataFrame fragmentation | Minor performance | Cosmetic only; results correct |

## 10. Handoff Instructions for P2 and P3

### Consuming processed features

```python
import pandas as pd

df = pd.read_parquet("data/features/combined_features.parquet")

# Feature columns (exclude identifiers and target)
feature_cols = [c for c in df.columns
                if c not in ("patient_id", "source_dataset", "hour", "SepsisLabel")]

# Split by patient using P1 split manifests
import json
with open("data/splits/train_patient_ids.json") as f:
    train_ids = json.load(f)["patient_ids"]

train_df = df[df["patient_id"].isin(train_ids)]
X_train = train_df[feature_cols]
y_train = train_df["SepsisLabel"]
```

### Key contract points

- All features are causal (no future leakage)
- `SepsisLabel` is target-only, never an input
- Missing values preserved as NaN (no imputation by P1)
- `hospital_id` preserved when available from source
- Configuration and schema saved with experiment outputs

Full details: `docs/p1/P2_P3_FEATURE_HANDOFF.md`

## 11. Remaining Work

1. Obtain PhysioNet credentialed access and download real dataset to
   `data/raw/`
2. Run Part 3 preprocessing pipeline against real data
3. Run Part 4 feature pipeline against real processed data
4. Validate feature reports against real data
5. P1 Part 5: Clinical baselines (SIRS, qSOFA, NEWS-style scoring)
6. P1 Part 6: Classical ML models (Logistic Regression, Random Forest, SVM)

---

## Data Integrity Statement

- No raw patient data was downloaded (no PhysioNet credentials)
- No raw patient data was committed to Git
- No dataset statistics were fabricated; all reports marked synthetic
- Source data would remain unchanged (pipeline only reads Part 3 output)
- Malformed records are never silently discarded

Awaiting approval to begin PART 5 (Clinical baselines: SIRS, qSOFA, NEWS-style scoring).
