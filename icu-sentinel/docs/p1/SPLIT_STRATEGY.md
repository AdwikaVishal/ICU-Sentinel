# Split Strategy — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 3 — Patient-level splitting  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Strategy Overview

The dataset is split **by patient**, never by individual hourly
observations. All observations belonging to a single patient remain in
exactly one split (train, validation, or test). There is zero patient
overlap between splits.

This is implemented in `ml/p1_clinical/preprocessing/patient_splits.py`
via the `PatientSplitter` class.

---

## 2. Split Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| strategy | `stratified_random` | Split strategy (currently only stratified random supported) |
| test_size | 0.2 | Fraction of patients assigned to test |
| validation_size | 0.1 | Fraction of patients assigned to validation (of remaining) |
| random_state | 42 | Random seed for reproducibility |
| stratify_by_label | True | Stratify by sepsis label presence |

The effective split proportions are:
- **Test:** `test_size` of all patients
- **Validation:** `validation_size / (1 - test_size)` of train+val patients
- **Train:** Remaining patients

With defaults (test=0.2, val=0.1):
- Test: 20% of patients
- Validation: 12.5% of all patients (0.125 × 0.8)
- Train: 67.5% of all patients

---

## 3. Stratification

Splits are stratified by whether a patient has any sepsis-positive
hours (`has_sepsis` = any `SepsisLabel == 1`). This ensures the sepsis
class is represented in all splits.

If stratification is not possible (e.g., all patients have the same
label, or too few patients in a class), the splitter falls back to
non-stratified random splitting with the same random seed.

---

## 4. Reproducibility

- All splits use a configurable `random_state` (default: 42).
- Patients are sorted by `patient_id` before splitting for
  deterministic ordering.
- `sklearn.model_selection.train_test_split` is used with the
  specified random state.
- Running the splitter twice with the same seed produces identical
  split assignments.

---

## 5. Patient-Level Constraints

| Constraint | Implementation |
|------------|---------------|
| Split by patient, not rows | All observations of a patient assigned together |
| No patient overlap | Verified by `check_split_leakage()` |
| Preserve hospital IDs | `hospital_id` carried through to split entries |
| Record class distributions | `n_sepsis_patients_*` tracked per split |
| Record patient counts | `n_patients_*` tracked per split |
| Record hourly counts | `n_records_*` tracked per split |
| Save split manifest | `split_manifest.json` with all patient assignments |
| Leakage checks | `check_split_leakage()` validates no overlap |

---

## 6. Source Dataset Identity (Set A vs Set B)

The official Challenge training database consists of two parts:
- **Set A:** 20,336 subjects
- **Set B:** 20,000 subjects

The pipeline preserves source dataset identity as `source_dataset`
("set_a" or "set_b") for each patient. This prepares the architecture
for future cross-hospital evaluation, but **no cross-hospital
validation has been performed** and no claims are made about
cross-hospital generalizability.

---

## 7. Split Manifest

The split manifest (`data/splits/split_manifest.json`) contains:

```json
{
  "strategy": "stratified_random",
  "random_state": 42,
  "test_size": 0.2,
  "validation_size": 0.1,
  "stratify_by_label": true,
  "n_patients_total": N,
  "n_patients_train": N_train,
  "n_patients_validation": N_val,
  "n_patients_test": N_test,
  "n_records_train": R_train,
  "n_records_validation": R_val,
  "n_records_test": R_test,
  "n_sepsis_patients_train": S_train,
  "n_sepsis_patients_validation": S_val,
  "n_sepsis_patients_test": S_test,
  "n_sepsis_positive_hours_train": H_train,
  "n_sepsis_positive_hours_validation": H_val,
  "n_sepsis_positive_hours_test": H_test,
  "patients": [
    {
      "patient_id": "p00001",
      "hospital_id": "set_a",
      "split": "train",
      "n_records": 24,
      "n_sepsis_positive_hours": 5,
      "n_sepsis_negative_hours": 19,
      "has_sepsis": true,
      "source_file": "p00001.psv",
      "source_dataset": "set_a"
    }
  ]
}
```

Per-split patient ID lists are also saved:
- `data/splits/train_patient_ids.json`
- `data/splits/validation_patient_ids.json`
- `data/splits/test_patient_ids.json`

---

## 8. Leakage Verification

`check_split_leakage()` verifies:

1. **No patient appears in multiple splits:** Each `patient_id` is
   assigned to exactly one split.
2. **No overlap between any pair of splits:** Train ∩ validation =
   ∅, train ∩ test = ∅, validation ∩ test = ∅.
3. **All patients assigned:** Every patient in the manifest has a
   valid split assignment.
4. **Counts sum correctly:** Patient counts and record counts sum to
   totals.

---

## 9. Limitations

- Real PhysioNet/CinC Challenge 2019 data is not accessible without
  credentialed access. Split strategy has been validated on synthetic
  fixtures only.
- The default 80/10/10 split may need adjustment based on actual class
  imbalance in the real dataset.
- Cross-hospital evaluation (Set A vs Set B) is architecturally
  prepared but not yet performed.
- No temporal split (e.g., by admission date) is implemented; all
  splitting is random at the patient level.

---

## 10. Handoff to P2 and P3

P2 and P3 can consume the split outputs as follows:

```python
from ml.p1_clinical.preprocessing.patient_splits import SplitManifest

manifest = SplitManifest.load("data/splits/split_manifest.json")

train_ids = manifest.patient_ids_by_split("train")
val_ids = manifest.patient_ids_by_split("validation")
test_ids = manifest.patient_ids_by_split("test")
```

Each patient's processed data is available as:
`data/processed/{patient_id}.parquet`

All splits share the same output schema defined in
`ml/p1_clinical/preprocessing/schemas.py`.
