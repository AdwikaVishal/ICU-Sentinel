# P1 Temporal Leakage Audit Report

## Overview

This report documents the audit of temporal leakage protections in the ICU Sentinel P1 data pipeline. Temporal leakage occurs when features at prediction time `t` contain information from future time points `> t`, which would invalidate early-warning system evaluation.

## Pipeline Architecture

The P1 pipeline enforces causal feature engineering through several mechanisms:

1. **Per-patient processing**: Each patient is processed independently; no cross-patient statistics.
2. **ICULOS ordering**: All data sorted by `ICULOS` (ICU hour) to ensure temporal ordering.
3. **No imputation**: Missing values preserved as NaN; no forward-fill from future.
4. **Causal feature categories**:
   - **Current features**: Raw observations at hour `t` only.
   - **Missingness features**: Computed from history up to hour `t`.
   - **Forward-fill features**: Last observation carried forward, expires after configured window (default 24h).
   - **Rolling features**: Statistics over windows `[t-W, t]` (past only).
   - **Trend features**: Changes/slopes over windows `[t-W, t]` (past only).
   - **Clinical features**: Computed from current hour `t` only.

## Audit Results

### Feature Schema Analysis

All features in the feature schema are marked as `causal: true`:
- Total features: 659 (after selection)
- Non-causal features found: 0
- Target columns in feature set: 0 (correctly separated)

### Target Column Isolation

The following target/identifier columns are correctly excluded from model features:
- `SepsisLabel` (original shifted label)
- `sepsis_within_6h` (derived target)
- `target_eligible` (eligibility mask)
- `patient_id`, `source_dataset`, `hour`, `hospital_id` (identifiers)

**Status**: ✅ PASS - No target columns in feature matrix.

### Forward-Fill Expiration

Forward-filled features (`*_ffilled`) expire after the configured maximum validity window:
- Default window: 24 hours
- Beyond window: Values become NaN (no stale information carried forward)
- Verification: Features with `_ffilled` suffix correctly implement expiration logic.

**Status**: ✅ PASS - Forward-fill does not leak future information.

### Rolling/Trend Window Causality

Rolling and trend features use only historical windows:
- Window sizes: 6h, 12h, 24h
- Aggregations: mean, std, min, max, slope
- All windows end at current hour `t` (inclusive of `t`, exclusive of `t+1`)

**Status**: ✅ PASS - Rolling/trend features are causal.

### Preprocessing Fitting

All preprocessing transformers are fitted on **training data only**:
- Imputer (median): Fitted on train, applied to val/test
- Scaler (standard): Fitted on train, applied to val/test
- Feature selector: Fitted on train, applied to val/test
- No statistics from validation/test influence training preprocessing.

**Status**: ✅ PASS - No test data leakage in preprocessing.

### Patient-Level Splits

Data splits are performed at **patient level**, not row level:
- Train: 350 patients (10,501 rows)
- Validation: 50 patients (1,476 rows)
- Test: 100 patients (3,001 rows)
- Zero patient overlap between splits (verified)
- Stratification by sepsis label preserves class distribution.

**Status**: ✅ PASS - No patient leakage between splits.

### Sepsis Label Handling

- Original `SepsisLabel` is **never used as a feature**
- Derived target `sepsis_within_6h` computed per-patient from original label
- Target eligibility mask excludes post-onset+6 hours
- No future label information enters features at prediction time.

**Status**: ✅ PASS - Label isolation maintained.

## Limitations & Notes

1. **Synthetic Data**: This audit was performed on synthetic data (500 patients). Real data may have different missingness patterns.
2. **Feature Selection**: Feature selection removes constant/duplicate/all-missing/infinite features on training data only. This is valid but means feature set depends on training distribution.
3. **Forward-Fill Window**: The 24-hour expiration window is a configurable hyperparameter; clinical validity should be assessed.
4. **Static Variables**: Demographics (Age, Gender, Unit1, Unit2, HospAdmTime) are repeated at each hour but are inherently static - no temporal leakage.

## Verification Commands

```bash
# Run temporal leakage tests
cd icu-sentinel && python3 -m pytest ml/p1_clinical/tests/test_temporal_leakage.py -v -o addopts=""

# Run model leakage tests
python3 -m pytest ml/p1_clinical/tests/test_model_leakage.py -v -o addopts=""

# Run evaluation leakage tests
python3 -m pytest ml/p1_clinical/tests/test_evaluation_leakage.py -v -o addopts=""

# Generate audit artifacts
python3 scripts/generate_p1_audit_artifacts.py
```

## Conclusion

The P1 data pipeline **passes temporal leakage audit**. All feature engineering is causal, preprocessing is fitted on training data only, patient-level splits prevent cross-patient leakage, and target columns are properly isolated from features. The derived `sepsis_within_6h` target correctly implements a strict 6-hour pre-onset prediction window without future information leakage.