# P2/P3 Feature Handoff — P1

**Project:** ICU Sentinel  
**From:** Person 1 (P1) — Data engineering  
**To:** Person 2 (P2) — Boosting/ensembles; Person 3 (P3) — Temporal models  
**Part:** PART 4 — Temporal feature engineering  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Purpose

This document describes the stable feature output contract that P1
provides for consumption by P2 (boosting models) and P3 (temporal deep
learning models). The same underlying feature definitions are usable
across all three workstreams.

---

## 2. Output Location

Processed patient features are stored as Parquet files:

```
data/features/combined_features.parquet    (all patients, one row per patient-hour)
data/features/{patient_id}.parquet         (optional, per-patient files)
```

The pipeline can also output per-patient Parquet files. Configuration
and schema are saved alongside:

```
data/features/feature_config.json
data/features/feature_schema.json
```

---

## 3. Output Schema

The output DataFrame has one row per patient-hour. Columns are
documented in `data/features/feature_schema.json` and
`ml/p1_clinical/features/feature_schema.py`.

### 3.1 Preserved Columns (Not Model Inputs)

| Column | Type | Notes |
|--------|------|-------|
| patient_id | str | Original patient identifier |
| source_dataset | str | Source dataset identity ("set_a"/"set_b") |
| hour | int | ICU hour since admission |
| SepsisLabel | int (0/1) | Original target label, unchanged (target only) |

**SepsisLabel is a target field only.** It must never be used as a model
input feature. It has not been shifted or reconstructed.

`hospital_id` is preserved when genuinely available from the source
data. It is not inferred from patient IDs.

### 3.2 Feature Categories

| Category | Count | Tabular | Sequential | Description |
|----------|-------|---------|------------|-------------|
| current | 39 | Yes | Yes | Raw clinical variables |
| missingness | 102 | Yes | Yes | Is-missing, hours-since-last, availability |
| forward_fill | 34 | Yes | Yes | Causally forward-filled variables |
| rolling | 510 | Yes | Yes | 6/12/24h mean/min/max/std/latest |
| trend | 306 | Yes | Yes | Abs/rel change, slope over 6/12/24h |
| clinical | 3 | Yes | Yes | Shock index, SIRS count, components available |

Total: ~994 features (exact count depends on configuration).

All features are **causal** (use only current/past observations). All
are appropriate for both tabular and sequential models.

---

## 4. Consumption Instructions

### 4.1 For P2 (Tabular Models: Extra Trees, LightGBM, XGBoost, CatBoost)

```python
import pandas as pd

# Load combined features
df = pd.read_parquet("data/features/combined_features.parquet")

# Split by patient (never by row)
train_ids = [...]  # from data/splits/train_patient_ids.json
val_ids = [...]    # from data/splits/validation_patient_ids.json
test_ids = [...]   # from data/splits/test_patient_ids.json

train_df = df[df["patient_id"].isin(train_ids)]
val_df = df[df["patient_id"].isin(val_ids)]
test_df = df[df["patient_id"].isin(test_ids)]

# Features: all columns except identifiers and target
feature_cols = [c for c in df.columns
                if c not in ("patient_id", "source_dataset", "hour", "SepsisLabel")]
X_train = train_df[feature_cols]
y_train = train_df["SepsisLabel"]

# Handle missing values: NaN preserved; impute as needed by your model
```

### 4.2 For P3 (Sequential Models: GRU, LSTM, Transformer)

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("data/features/combined_features.parquet")

# Build sequences per patient
feature_cols = [c for c in df.columns
                if c not in ("patient_id", "source_dataset", "hour", "SepsisLabel")]

sequences = []
targets = []
patient_ids = []

for pid, group in df.groupby("patient_id"):
    group = group.sort_values("hour")
    seq = group[feature_cols].values  # (n_hours, n_features)
    target = group["SepsisLabel"].values  # (n_hours,)
    sequences.append(seq)
    targets.append(target)
    patient_ids.append(pid)

# Pad sequences to equal length for batching
# Use masking for variable-length sequences
```

---

## 5. Missing Data Handling

P1 does **not** perform imputation. Missing values are preserved as
`NaN`. Downstream models must handle missingness:

- Tree-based models (P2): Most handle NaN natively or require imputation
- Sequential models (P3): Use masking or impute before batching

Missingness indicator features (`{var}_is_missing`) are provided and
can be used as explicit missingness signals.

---

## 6. Causality Guarantee

All features are causal. For any patient-hour `t`, no feature uses
information from hours after `t`. This is verified by automated
leakage tests (`test_temporal_leakage.py`).

P2 and P3 must maintain this causality constraint: no future
information leakage in training or inference.

---

## 7. Reproducibility

- Feature configuration is saved in `data/features/feature_config.json`
- Feature schema is saved in `data/features/feature_schema.json`
- Splits are saved in `data/splits/split_manifest.json`
- All use a configurable random seed (default: 42)

Use the saved configuration to reproduce features exactly.

---

## 8. Limitations

- Real PhysioNet data not yet accessed; features generated from
  synthetic fixtures
- Forward-fill validity windows are experimental defaults
- No imputation performed; NaN values preserved
- Cross-hospital evaluation architecture prepared but not performed

---

## 9. Contact

P1 maintains the feature pipeline. Questions about feature definitions,
causality, or schema should be directed to P1 before modifying feature
generation logic.
