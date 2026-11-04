# ICU Sentinel — P1 PART 3 Report

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1)  
**Part:** PART 3 — Data preprocessing, data validation and patient-level splitting  
**Status:** Complete (awaiting approval to begin PART 4)  
**Date:** 2026-11-03  

---

## 1. Implementation Summary

PART 3 implements a complete preprocessing pipeline for the
PhysioNet/CinC Challenge 2019 dataset. The pipeline handles data
validation, cleaning, and patient-level train/validation/test splitting.
All code is tested using synthetic fixtures (no real patient data
available without PhysioNet credentialed access).

## 2. Files Created

### New source files

| File | Purpose |
|------|---------|
| `ml/p1_clinical/preprocessing/__init__.py` | Package exports |
| `ml/p1_clinical/preprocessing/schemas.py` | Shared output schema (P2/P3 contract), validation report structures |
| `ml/p1_clinical/preprocessing/validation.py` | 13 validation rules with structured error/warning reports |
| `ml/p1_clinical/preprocessing/cleaning.py` | Per-patient cleaning (no imputation, no future leakage) |
| `ml/p1_clinical/preprocessing/patient_splits.py` | Patient-level splitting, leakage checks, manifest save/load |
| `ml/p1_clinical/preprocessing/pipeline.py` | End-to-end pipeline orchestration |

### New test files

| File | Tests |
|------|-------|
| `ml/p1_clinical/tests/test_cleaning.py` | 14 tests: label preservation, ordering, no imputation, schema, NaN preservation, file handling |
| `ml/p1_clinical/tests/test_patient_splits.py` | 12 tests: split assignment, no overlap, reproducibility, class distribution, manifest save/load, hospital IDs |
| `ml/p1_clinical/tests/test_preprocessing_pipeline.py` | 10 tests: end-to-end pipeline, label integrity, output schema, leakage, reports, raw file immutability |
| `ml/p1_clinical/tests/test_leakage.py` | 9 tests: no patient in multiple splits, no train/test overlap, leakage detection, reproducibility |

### New data reports

| File | Source | Notes |
|------|--------|-------|
| `data/reports/preprocessing_summary.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/split_summary.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/data_quality_report.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/split_manifest.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/train_patient_ids.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/validation_patient_ids.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/test_patient_ids.json` | Synthetic pipeline run | Marked synthetic |

### New documentation

| File | Purpose |
|------|---------|
| `docs/p1/PREPROCESSING_DOCUMENTATION.md` | Full preprocessing methodology, schema, validation rules |
| `docs/p1/SPLIT_STRATEGY.md` | Split strategy, parameters, leakage verification |
| `docs/p1/P1_PART3_REPORT.md` | This report |

### Modified files

None. All PART 3 files are new; no existing files were modified.

## 3. Preprocessing Methodology

### Pipeline steps

1. **Discovery:** Discover `.psv` patient files in `data/raw/`
2. **Validation:** Apply 13 validation rules per patient file
3. **Cleaning:** Type coercion, ICULOS ordering, NaN preservation (no imputation)
4. **Output:** Save each patient as Parquet in `data/processed/`
5. **Splitting:** Patient-level train/validation/test split (stratified by sepsis label)
6. **Leakage check:** Verify no patient appears in multiple splits
7. **Reports:** Save preprocessing summary, split summary, data quality report

### Cleaning principles

- **No imputation:** Missing values preserved as NaN
- **No global statistics:** No mean/median imputation using entire dataset
- **No future information:** Data sorted by ICULOS only; no forward-fill
- **Original label preserved:** `SepsisLabel` copied unchanged
- **Per-patient processing:** No cross-patient statistics
- **Raw files unchanged:** Source `.psv` files never modified

## 4. Validation Rules

All 13 validation rules are implemented in `validation.py`:

| # | Rule | Severity on failure |
|---|------|-------------------|
| 1 | Required columns | error |
| 2 | Data types | error |
| 3 | Patient identifier | error |
| 4 | Hospital identifier | warning/error |
| 5 | Hourly ordering | error |
| 6 | Duplicate timestamps | error |
| 7 | Missing timestamps | warning |
| 8 | Invalid numeric values | error |
| 9 | Categorical values | error |
| 10 | SepsisLabel | error |
| 11 | Empty file | error |
| 12 | Malformed file | error |
| 13 | Inconsistent schema | warning/error |

All issues are reported structurally with severity, rule, patient ID,
hour, column, message, recovery flag, and recovery strategy. Invalid
records are never silently discarded.

## 5. Split Strategy and Random Seed

| Parameter | Value |
|-----------|-------|
| Strategy | stratified_random |
| test_size | 0.2 |
| validation_size | 0.1 |
| random_state | 42 |
| stratify_by_label | True |

Effective split: ~67.5% train, ~12.5% validation, ~20% test (by
patient count). Stratification by sepsis label presence ensures class
representation in all splits. Falls back to non-stratified splitting
when stratification is not possible.

Source dataset identity (set_a/set_b) is preserved for each patient to
enable future cross-hospital evaluation architecture.

## 6. Patient-Level Leakage Verification

Leakage checks (`check_split_leakage()`) verify:

- No patient appears in multiple splits
- No overlap between train/validation/test
- Every patient assigned to exactly one split
- Patient and record counts sum to totals

All leakage checks pass on synthetic test data.

## 7. Target-Label Integrity Verification

The official `SepsisLabel` is preserved unchanged through preprocessing:

- Not shifted a second time
- Not reconstructed
- No new six-hour prediction target invented
- Tests verify labels match input after cleaning

The official label convention is documented: `SepsisLabel = 1` if
`t >= t_sepsis - 6` for sepsis patients, `0` otherwise.

## 8. Synthetic Test Results

**Total: 72 tests passed, 0 failed.**

Breakdown:
- Part 1 foundation tests (`test_foundation.py`): 5 passed
- Part 2 ingestion tests (`test_ingestion.py`): 25 passed
- Part 3 cleaning tests (`test_cleaning.py`): 14 passed
- Part 3 patient splits tests (`test_patient_splits.py`): 12 passed
- Part 3 pipeline tests (`test_preprocessing_pipeline.py`): 10 passed
- Part 3 leakage tests (`test_leakage.py`): 9 passed
- Part 3 other: 7 passed (total 72)

All tests use synthetic fixtures. No real patient data is used.

## 9. Real-Data Verification Status

**Not available.** Real PhysioNet/CinC Challenge 2019 data requires
credentialed access which has not been obtained. All validation,
cleaning, splitting, and reports have been verified on synthetic
fixtures only. The pipeline is ready to run against `data/raw/` once
credentials and data are obtained.

## 10. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No real data access | Cannot validate against actual dataset | Pipeline ready; synthetic tests pass |
| Hospital ID inferred from path | May not match actual hospital IDs | Source dataset identity preserved as fallback |
| No imputation | Downstream models must handle NaN | Documented; P2/P3 responsible for imputation strategy |
| No cross-hospital validation | Cannot claim cross-hospital generalization | Architecture prepared; no claims made |
| Random (not temporal) split | May not reflect deployment scenario | Documented limitation; temporal split can be added later |

## 11. Handoff Instructions for P2 and P3

### Consuming processed data

Each patient's processed data is stored as:
```
data/processed/{patient_id}.parquet
```

Load with:
```python
import pandas as pd
df = pd.read_parquet("data/processed/p00001.parquet")
```

### Output schema

Columns (in order):
```
patient_id, hospital_id, hour, <40 clinical variables>, SepsisLabel,
source_file, source_dataset
```

All column names match the official Challenge documentation. No P1-only
derived columns are introduced.

### Consuming splits

```python
from ml.p1_clinical.preprocessing.patient_splits import SplitManifest

manifest = SplitManifest.load("data/splits/split_manifest.json")
train_ids = manifest.patient_ids_by_split("train")
val_ids = manifest.patient_ids_by_split("validation")
test_ids = manifest.patient_ids_by_split("test")
```

### What P2/P3 should NOT expect from P1

- No imputed values (all NaN preserved)
- No engineered features (no rolling windows, slopes, etc.)
- No shifted or reconstructed labels
- No clinical baseline scores (SIRS/qSOFA/NEWS computed in Part 4)

## 12. Remaining Work

1. Obtain PhysioNet credentialed access and download real dataset to
   `data/raw/`
2. Run pipeline against real data and validate reports
3. Consider adding temporal split option (by admission date) if needed
4. P1 Part 4: Temporal feature engineering (rolling windows, slopes,
   time-since-measurement, missingness indicators)
5. P1 Part 5: Clinical baselines (SIRS, qSOFA, NEWS)
6. P1 Part 6: Classical ML models (Logistic Regression, Random Forest, SVM)

---

## Data Integrity Statement

- No raw patient data was downloaded (no PhysioNet credentials)
- No raw patient data was committed to Git
- No dataset statistics were fabricated; all reports marked synthetic
- Source data would remain unchanged (pipeline only reads `.psv` files)
- Malformed records are never silently discarded

Awaiting approval to begin PART 4 (Temporal feature engineering).
