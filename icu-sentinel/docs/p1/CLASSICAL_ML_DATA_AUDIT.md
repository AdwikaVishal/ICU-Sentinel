# Classical ML Data Audit — P1 PART 6

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Classical ML  
**Part:** PART 6 — Classical ML models  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Pre-training Audit Summary

This document records the pre-training data audit performed before training
P1's classical ML models (Logistic Regression, Random Forest, SVM).

The audit verified:

| Check | Result |
|-------|--------|
| Part 3 split manifests accessible | Yes |
| Part 4 feature schema accessible | Yes |
| Part 5 baseline schema accessible | Yes |
| Patient IDs consistent across modules | Yes |
| Timestamps consistent across modules | Yes |
| Feature names match schema | Yes |
| Target labels preserved | Yes |
| Train/validation/test patient overlap | None detected |
| SepsisLabel used as target only | Yes |
| Identifiers excluded from features | Yes |

No contract mismatches were found. All modules agree on the data contract.

---

## 2. Data Contracts

### Part 3 — Patient-Level Splits

The split manifest (`data/splits/split_manifest.json`) defines patient-level
train/validation/test assignments. Key properties:

- **Split by patient**: All observations of a patient remain in one split
- **No patient overlap**: Zero patients appear in multiple splits
- **Stratified by label**: Sepsis label distribution preserved across splits
- **Reproducible**: Configurable random seed (default: 42)

Split sizes (configurable): Default test=20%, validation=10% of remaining.

### Part 4 — Feature Schema

The feature schema (`data/features/feature_schema.json`) defines ~994 features
across 6 categories:

| Category | Count | Tabular | Sequential |
|----------|-------|---------|------------|
| Current (raw observations) | 39 | Yes | Yes |
| Missingness (indicators, time-since-last, availability) | 102 | Yes | Yes |
| Forward-filled | 34 | Yes | Yes |
| Rolling (6/12/24h mean/min/max/std/latest) | 510 | Yes | Yes |
| Trend (abs/rel change, slope over 6/12/24h) | 306 | Yes | Yes |
| Clinical (shock index, SIRS count, components available) | 3 | Yes | Yes |
| **Total** | **~994** | | |

All features are causal (use only current/past observations). All are
appropriate for both tabular and sequential models.

### Part 5 — Clinical Baselines

Three clinical baselines are available:

| Score | Source | Complete Rate | Notes |
|-------|--------|---------------|-------|
| SIRS | Bone et al., Chest 1992 | Variable | PaCO2/bands not in dataset |
| qSOFA | Sepsis-3, JAMA 2016 | 0% (partial only) | AMS not in dataset |
| NEWS2-style | RCP NEWS2 | 0% (partial only) | O2/consciousness not in dataset |

All baselines are partial due to missing dataset variables. Completeness
is tracked per patient-hour.

---

## 3. Feature Selection

Before training, the following feature exclusions are applied:

| Exclusion Type | Applied | Reason |
|---------------|---------|--------|
| Identifier columns | Yes | patient_id, source_dataset, hour, source_file are metadata |
| Target column | Yes | SepsisLabel is target only |
| Constant features | Yes | No information content |
| Duplicate features | Yes | Redundant information |
| All-missing features | Yes | Cannot be imputed meaningfully |
| Infinite values | Yes | Not numerically valid |
| Highly correlated | Optional | Configurable threshold |

Feature selection is **fitted on training data only** and applied to
validation/test without refitting. The selected feature list is preserved
for every model.

No arbitrary large groups of features are discarded. The full feature set
(~994 columns) is available by default; feature reduction is optional.

---

## 4. Target Label

The official PhysioNet/CinC Challenge 2019 `SepsisLabel` is used:

- **Definition**: For sepsis patients, `SepsisLabel = 1` if `t >= t_sepsis - 6`,
  else `0`. For non-sepsis patients, `SepsisLabel = 0` for all hours.
- **Six-hour shift**: Already incorporated in the official label. NOT
  shifted again.
- **Prediction unit**: Patient-hour (each row is one prediction)
- **Target-only**: Never used as a model input feature

Hourly predictions relate to patient-level outcomes through aggregation:
a patient is considered "detected" if any hourly prediction within a
clinically relevant window is positive.

---

## 5. Missing Values

Missing values are preserved as `NaN` from Part 3/4 output. No clinical
imputation is performed by P1. Imputation (median strategy by default) is
fitted on **training data only** and applied to validation/test.

Missingness indicator features are available from Part 4 and can be used
as explicit missingness signals by models.

---

## 6. Class Imbalance

Sepsis prediction is highly imbalanced. The dataset typically shows
low positive-label prevalence. The configured strategy is **class
weighting** (`class_weight="balanced"`), applied during model training.

No oversampling is performed before splitting. No SMOTE is applied to
temporal observations. Validation and test data are never used for
imbalance strategy selection.

---

## 7. Output Artifacts

Model artifacts are saved under `artifacts/p1/classical_ml/`:

```
artifacts/p1/classical_ml/
├── logistic_regression/
├── random_forest/
├── svm/
└── metadata/
```

Prediction outputs are saved under `data/reports/classical_ml/`:

```
data/reports/classical_ml/
├── model_comparison.json
├── model_metrics.json
├── feature_importance.csv
├── training_summary.json
└── prediction_summary.json
```

No raw patient data is committed. All reports generated from synthetic
fixtures are marked `"source": "synthetic"`.

---

## 8. Limitations

- Real PhysioNet/CinC Challenge 2019 data not accessed (no credentials)
- All validation on synthetic fixtures
- Clinical baselines are partial due to missing variables
- Cross-hospital evaluation architecture prepared but not performed
- No clinical validation claims made
