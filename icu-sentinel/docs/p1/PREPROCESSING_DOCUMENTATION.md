# Preprocessing Documentation — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 3 — Data preprocessing, data validation and patient-level splitting  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Overview

The P1 preprocessing pipeline prepares clean, validated, patient-indexed
data from the PhysioNet/CinC Challenge 2019 dataset for consumption by:

- **P1:** Classical ML models (Logistic Regression, Random Forest, SVM)
  and clinical baselines (SIRS, qSOFA, NEWS).
- **P2:** Boosting models (Extra Trees, LightGBM, XGBoost, optional
  CatBoost), ensembles, calibration.
- **P3:** Temporal models (GRU, LSTM, optional Transformer).

The pipeline is implemented in `ml/p1_clinical/preprocessing/` and
consists of five modules:

| Module | Responsibility |
|--------|---------------|
| `schemas.py` | Shared output schema consumed by P2/P3 |
| `validation.py` | 13 validation rules with structured error/warning reports |
| `cleaning.py` | Per-patient cleaning without imputation or future leakage |
| `patient_splits.py` | Patient-level train/validation/test splitting |
| `pipeline.py` | End-to-end orchestration |

---

## 2. Shared Output Schema (P2/P3 Integration Contract)

The pipeline produces a stable output schema defined in
`ml/p1_clinical/preprocessing/schemas.py`. The output columns are:

```
patient_id, hospital_id, hour, <40 clinical variables>, SepsisLabel,
source_file, source_dataset
```

All 40 clinical variables and `SepsisLabel` use the exact column names
from the official Challenge documentation (Table 1). No P1-only derived
or engineered column names are introduced.

### Output column reference

| Column | Type | Description |
|--------|------|-------------|
| patient_id | string | Original patient identifier (e.g. "p00101") |
| hospital_id | string | Source dataset identity ("set_a" / "set_b") when available |
| hour | int | ICU hour since admission (from ICULOS) |
| HR … Platelets | float | 40 clinical variables; NaN = missing |
| Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS | float/int | Demographics and administrative fields |
| SepsisLabel | int (0/1) | Original target label, unchanged |
| source_file | string | Source .psv filename |
| source_dataset | string | Source dataset identity |

### Data contract for P2/P3

- Each patient is stored as a separate Parquet file: `data/processed/{patient_id}.parquet`
- All observations for a patient remain in a single file
- Original `SepsisLabel` is preserved (not shifted, not reconstructed)
- No imputation is applied; missing values remain as NaN
- No future information is used (data sorted by ICULOS only)
- Raw source files are never modified

---

## 3. Validation Rules (13 rules)

Implemented in `ml/p1_clinical/preprocessing/validation.py`. Every rule
produces a structured `ValidationIssue` with severity ("error" or
"warning"), rule name, patient ID, hour, column, message, recovery
flag, and recovery strategy. Invalid records are **never silently
discarded**.

| # | Rule | Severity | Description |
|---|------|----------|-------------|
| 1 | Required columns | error/warning | All 41 Challenge columns present |
| 2 | Data types | error | Numeric columns are numeric; integer columns are numeric |
| 3 | Patient identifier | error | Patient ID matches pattern `p[5-6 digits]`; consistent within file |
| 4 | Hospital identifier | warning/error | Hospital ID present and valid string |
| 5 | Hourly ordering | error | ICULOS monotonically non-decreasing |
| 6 | Duplicate timestamps | error | No duplicate ICULOS values |
| 7 | Missing timestamps | warning | No gaps in ICULOS sequence |
| 8 | Invalid numeric values | error | No infinite values in numeric columns |
| 9 | Categorical values | error | Gender in {0,1}; SepsisLabel in {0,1} |
| 10 | SepsisLabel | error | No missing or invalid (non-0/1) labels |
| 11 | Empty file | error | File contains at least one row |
| 12 | Malformed file | error | File parses successfully as pipe-delimited |
| 13 | Inconsistent schema | warning/error | No unexpected extra/missing columns |

### Validation report structure

```json
{
  "patient_id": "p00001",
  "n_issues": 3,
  "n_errors": 1,
  "n_warnings": 2,
  "passed": false,
  "issues": [
    {
      "severity": "error",
      "rule": "sepsis_label",
      "patient_id": "p00001",
      "hour": null,
      "column": "SepsisLabel",
      "message": "2 row(s) with invalid SepsisLabel values (not 0 or 1)",
      "recovered": false,
      "recovery_strategy": null
    }
  ]
}
```

---

## 4. Cleaning Pipeline

Implemented in `ml/p1_clinical/preprocessing/cleaning.py`.

### Cleaning principles

1. **No imputation:** Missing values are preserved as NaN. No arbitrary
   clinical imputation is performed.
2. **No global statistics:** No mean/median imputation using the entire
   dataset or any subset.
3. **No future information:** Data is sorted by ICULOS only; no
   forward-fill, rolling windows, or time-since-measurement features
   are introduced (those are Part 4 responsibilities).
4. **Original label preserved:** `SepsisLabel` is copied unchanged.
5. **Per-patient processing:** Each patient is cleaned independently;
   no cross-patient statistics are computed.
6. **Raw files unchanged:** Source `.psv` files are never modified.

### Cleaning steps

1. Validate the patient file (all 13 rules).
2. Coerce all columns to appropriate numeric types (failures → NaN).
3. Sort rows by ICULOS (stable sort to preserve causal ordering).
4. Derive `hour` from ICULOS.
5. Count and preserve missing values (no imputation).
6. Produce a `ProcessedPatient` matching the shared output schema.

---

## 5. Target-Label Integrity

The official PhysioNet/CinC Challenge 2019 `SepsisLabel` already
incorporates the six-hour shift:

- For sepsis patients: `SepsisLabel = 1` if `t >= t_sepsis - 6`, else `0`
- For non-sepsis patients: `SepsisLabel = 0` for all hours

**The preprocessing pipeline:**

- Preserves the original `SepsisLabel` unchanged
- Does **not** shift it a second time
- Does **not** reconstruct sepsis onset
- Does **not** invent a new six-hour prediction target
- Adds tests to verify labels remain unchanged through preprocessing

Any future target transformation must be documented separately and
approved before implementation.

---

## 6. How to Run

```bash
# Activate environment
source .venv/bin/activate

# Run the full pipeline on synthetic data (for testing)
python - <<'EOF'
from ml.p1_clinical.preprocessing.pipeline import run_pipeline
result = run_pipeline(
    raw_dir="data/raw",
    processed_dir="data/processed",
    splits_dir="data/splits",
    reports_dir="data/reports",
    test_size=0.2,
    validation_size=0.1,
    random_state=42,
)
print(f"Processed {result.n_patients_processed} patients")
print(f"Leakage detected: {result.leakage_report['leakage_detected']}")
EOF
```

### Pipeline steps

1. Discover `.psv` patient files in `data/raw/`
2. Validate each file (13 validation rules)
3. Clean each file (type coercion, ordering, no imputation)
4. Save processed patients as Parquet (one file per patient)
5. Split patients into train/validation/test (patient-level)
6. Check for split-level leakage
7. Save split manifests and reports

---

## 7. Outputs

| Path | Description |
|------|-------------|
| `data/processed/{patient_id}.parquet` | Per-patient processed data |
| `data/splits/split_manifest.json` | Complete split manifest |
| `data/splits/{train,validation,test}_patient_ids.json` | Per-split patient ID lists |
| `data/reports/preprocessing_summary.json` | Pipeline execution summary |
| `data/reports/split_summary.json` | Split distribution summary |
| `data/reports/data_quality_report.json` | Validation and cleaning quality report |

> **Note:** Reports currently in `data/reports/` were generated from
> synthetic fixture data because real PhysioNet data is not accessible
> without credentialed access. They are marked with `"source": "synthetic"`.

---

## 8. Limitations

- Real PhysioNet/CinC Challenge 2019 data has not been accessed (no
  credentialed account available). All validation and reports use
  synthetic fixtures.
- Hospital identifier is inferred from file path naming convention;
  actual hospital IDs may require additional metadata.
- No imputation is performed; downstream models must handle missing
  values themselves.
- Cross-hospital evaluation architecture is prepared (source dataset
  identity preserved) but no cross-hospital validation has been
  performed.
