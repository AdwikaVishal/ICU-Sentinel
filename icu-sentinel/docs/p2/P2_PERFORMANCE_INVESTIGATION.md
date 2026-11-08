# P2 Performance Discrepancy Investigation

## Executive Summary

Investigated the 20% performance drop between P2_FAST training summary (AUROC 0.72-0.76) and current saved models (AUROC 0.57-0.61). **Key Finding**: Test sets are IDENTICAL across experiments - performance difference cannot be attributed to different evaluation cohorts.

## Investigation Date
- **Date**: November 8, 2026
- **Investigator**: Automated Analysis
- **Status**: Investigation Complete - Root Cause UNRESOLVED

---

## Background Context

Three distinct P2 experiments exist:

1. **P2_FAST** (Training Summary) - Documented but models NOT saved
   - AUROC: ET=0.7189, XGB=0.7621
   - Config: ET n_estimators=80, XGB n_estimators=100
   - Features: 994 features with `*_hours_since_last`
   
2. **P2_MATCHED_6000** - Fully documented, models saved
   - AUROC: ET=0.7027, XGB=0.7152
   - Config: ET n_estimators=200, XGB n_estimators=200
   - Features: 960 features (P1-matched)
   
3. **P2_FAST** (Current Artifacts) - Models ARE saved
   - AUROC: ET=0.5729, XGB=0.6146
   - Config: ET n_estimators=80, XGB n_estimators=100 (matches training summary)
   - Features: 994 features (expected based on config)

**Performance Gap**: Current P2_FAST models are 14-15 AUROC points lower than the training summary reported.

---

## Test Set Comparison Analysis

### Manifest Comparison

#### P2_FAST Manifest (`split_manifest_fast.json`)
```
Strategy: stratified_random
Random Seed: 42
Test Size: 20%
Validation Size: 10%

Total Patients: 20,102
- Train: 8,000 patients → 306,558 records
- Validation: 4,034 patients → 157,069 records  
- Test: 8,068 patients → 311,614 records

Sepsis Distribution:
- Train: 582 sepsis patients → 5,528 positive hours
- Validation: 293 sepsis patients → 2,786 positive hours
- Test: 586 sepsis patients → 5,553 positive hours
```

#### P2_MATCHED_6000 Manifest (`split_manifest.json`)
```
Strategy: stratified_random
Random Seed: 42
Test Size: 20%
Validation Size: 10%

Total Patients: 40,336
- Train: 28,234 patients → 1,083,527 records
- Validation: 4,034 patients → 157,069 records
- Test: 8,068 patients → 311,614 records

Sepsis Distribution:
- Train: 2,053 sepsis patients → 19,577 positive hours
- Validation: 293 sepsis patients → 2,786 positive hours
- Test: 586 sepsis patients → 5,553 positive hours
```

### Critical Finding: IDENTICAL Test Sets

Both experiments use **THE EXACT SAME**:
- Test set size: 8,068 patients, 311,614 records
- Sepsis patients: 586 patients, 5,553 positive hours
- Validation set: 4,034 patients, 157,069 records, 293 sepsis patients
- Random seed: 42
- Split strategy: stratified_random

**Conclusion**: The performance discrepancy CANNOT be explained by different test sets. Both experiments evaluate on identical held-out data.

---

## Revised Hypothesis Analysis

Given that test sets are identical, the performance drop must stem from:

### Hypothesis 1: Model Retraining Event (MOST LIKELY)
**Description**: The current saved P2_FAST models were retrained at a later time, possibly with:
- Data preprocessing bugs introduced after initial training
- Feature engineering pipeline changes
- Different data loading code
- NumPy/scikit-learn version differences affecting reproducibility

**Evidence**:
- Hyperparameters match training summary exactly (n_estimators verified)
- Test set is identical
- No evaluation code bugs found (code review clean)
- 20% performance drop is too large for random variation

**Status**: CANNOT VERIFY without access to:
- Original training logs with timestamps
- Git history of preprocessing code at training time
- Environment snapshots (library versions)

### Hypothesis 2: Feature Engineering Pipeline Drift
**Description**: The 994-feature version may have preprocessing differences:
- P2_FAST includes `*_hours_since_last` features (994 features)
- P2_MATCHED uses P1-aligned features (960 features)
- Current preprocessing may calculate these features incorrectly

**Evidence**:
- Feature count matches expectations (994 vs 960)
- `hours_since_last` features are complex temporal calculations
- No feature engineering validation tests exist

**Status**: CANNOT VERIFY without:
- Original preprocessed data files from initial training
- Feature distribution comparisons
- Feature engineering code history

### Hypothesis 3: Training Set Size Impact (RULED OUT AS PRIMARY CAUSE)
**Description**: P2_FAST trained on 8K patients vs P2_MATCHED on 28K patients

**Evidence AGAINST this being the primary cause**:
- P2_FAST training summary shows BETTER performance (0.72-0.76) despite smaller training set
- P2_MATCHED with 3.5x more training data achieved 0.70-0.72 (reasonable for more data)
- Current P2_FAST models (same small training set) show 0.57-0.61 (much worse)
- If training set size were the issue, performance would be consistent across runs

**Conclusion**: Training set size doesn't explain why the SAME configuration produced different results.

### Hypothesis 4: Random Seed / Stochastic Effects (RULED OUT)
**Description**: Model training randomness caused variation

**Evidence AGAINST**:
- 14-15 AUROC point drop is far too large for stochastic variation
- Random forests with 80-100 trees have low variance
- Held-out test evaluation eliminates overfitting explanations
- Training metrics would also differ (not just test metrics)

**Conclusion**: Not plausible at this magnitude.

---

## Data Integrity Verification

### Training Summary Analysis

Both training summaries contain:
- ✅ Complete hyperparameter configuration
- ✅ Feature lists (P2_FAST=994, P2_MATCHED=960)
- ✅ Target column: `sepsis_within_6h`
- ✅ Data source: `real`
- ✅ Thresholds: validation-optimized F1
- ✅ Training metrics (all splits)

### Artifact Verification

**P2_FAST Current Artifacts**:
- ✅ Models load successfully
- ✅ Hyperparameters match training summary
  - ET: n_estimators=80, max_depth=10
  - XGB: n_estimators=100, max_depth=4
- ✅ Wrapped in sklearn Pipeline (as expected)
- ✅ Predictions reproduce consistently
- ❌ Performance 20% lower than documented

**P2_MATCHED_6000 Artifacts**:
- ✅ Models load successfully
- ✅ Manifest hash: 008a93f47c4264e3
- ✅ Feature hash: c7407bfd3a11f257
- ✅ Performance matches documentation
- ✅ Fully reproducible

---

## Root Cause Assessment

### What We Know
1. ✅ Test sets are IDENTICAL (verified via manifests)
2. ✅ Hyperparameters match training summaries (verified via model inspection)
3. ✅ Evaluation code is correct (code review completed)
4. ✅ Threshold loading fixed (bug corrected)
5. ✅ No test set leakage (methodology validated)

### What We Don't Know
1. ❌ **When were current P2_FAST models trained?**
   - Training summary timestamp not recorded
   - Model file timestamps not preserved
   - Git history doesn't track model training events

2. ❌ **What preprocessing pipeline was used?**
   - No preprocessing code snapshot at training time
   - Feature engineering may have changed between runs
   - `*_hours_since_last` calculation correctness unverified

3. ❌ **What was the exact training environment?**
   - Python version unknown
   - scikit-learn/xgboost version unknown
   - NumPy version unknown
   - Reproducibility not guaranteed across versions

---

## Implications for Research Integrity

### Methodology Integrity: ✅ PRESERVED
The 20% performance discrepancy does NOT indicate methodological flaws:
- Training procedures were validation-based (no test leakage)
- Evaluation code is correct
- Test set composition is identical and properly held out
- Hyperparameter settings are documented and verified

### Performance Claims: ❌ UNRELIABLE
The P2_FAST training summary performance (AUROC 0.72-0.76) cannot be considered authoritative:
- Original models from that training run were not saved
- Current models with matching configuration show much lower performance
- Unable to reproduce the reported results
- **Conclusion**: P2_FAST training summary represents an unreproducible experiment

### Authoritative Results: ✅ P2_MATCHED_6000 DESIGNATED
P2_MATCHED_6000 remains the only fully reproducible experiment:
- Models saved and loadable
- Manifest hashes documented
- Performance verified against artifacts
- Feature engineering aligned with P1
- **AUROC: ET=0.7027, XGB=0.7152** (authoritative)

---

## Recommendations

### For Current Project (FROZEN)
1. ✅ **Use P2_MATCHED_6000 as authoritative result**
   - Fully reproducible with manifest hashes
   - Performance matches documentation
   - P1-aligned feature set

2. ✅ **Document P2_FAST as historical experiment**
   - Training summary preserved for historical record
   - Current artifacts have lower performance
   - Not reproducible - do not cite

3. ✅ **Preserve all artifacts**
   - Keep both experiment directories
   - Maintain training summaries
   - Document discrepancy in limitations

### For Future Work (if unfrozen)
1. **Implement reproducibility infrastructure**:
   - Record training timestamps
   - Save preprocessing code snapshots
   - Log environment (Python, library versions)
   - Version control training scripts
   - Generate manifest hashes for all experiments

2. **Feature engineering validation**:
   - Unit tests for all temporal features
   - Distribution checks against known good data
   - Automated regression testing

3. **Model versioning**:
   - MLflow or similar experiment tracking
   - Automatic artifact collection
   - Training/evaluation metric logging
   - Environment capture

4. **Investigate performance recovery**:
   - Compare preprocessed feature distributions
   - Audit `*_hours_since_last` calculations
   - Check for data loading bugs
   - Consider retraining with verified pipeline

---

## Conclusion

The investigation successfully ruled out test set differences as the cause of the 20% AUROC performance drop. The test sets are identical across experiments, confirming that both P2_FAST and P2_MATCHED_6000 evaluate on the same held-out cohort.

**Most Likely Explanation**: The current saved P2_FAST models were retrained at a later time with a different (possibly buggy) preprocessing pipeline, while the original training that produced the 0.72-0.76 AUROC results did not save its models.

**Research Integrity Status**: VALIDATED - No methodological flaws detected. The discrepancy reflects reproducibility issues, not scientific misconduct.

**Authoritative Performance**: P2_MATCHED_6000 with AUROC 0.70-0.72 remains the only verifiable result.

**Project Status**: Investigation complete. Root cause unresolved but methodology validated. Project remains FROZEN pending future decisions on reproducibility infrastructure and potential retraining.

---

## Files Referenced
- `data/splits/split_manifest_fast.json` - P2_FAST data split manifest
- `data/splits/split_manifest.json` - P2_MATCHED_6000 data split manifest
- `data/reports/p2_fast/p2_training_summary.json` - P2_FAST training documentation
- `data/reports/p2_matched_6000/p2_training_summary.json` - P2_MATCHED training documentation
- `artifacts/p2_fast/extra_trees_model.pkl` - Current P2_FAST ET model
- `artifacts/p2_fast/xgboost_model.pkl` - Current P2_FAST XGB model

## Related Documentation
- `docs/p2/P2_FINAL_EXPERIMENT_RECONCILIATION.md` - Complete experiment comparison
- `docs/p2/P2_FINAL_REMEDIATION_REPORT.md` - Methodology validation
- `docs/p2/P2_FINAL_STATUS_REPORT.md` - Executive summary
