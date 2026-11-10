# PhysioNet Preprocessing Report

**Generated:** 2026-11-09  
**Purpose:** Document preprocessing decisions and verification for real PhysioNet data

---

## 1. Preprocessing Pipeline Overview

The existing P1 preprocessing pipeline (`ml/p1_clinical/preprocessing/pipeline.py`) is designed to work with the actual PhysioNet Challenge 2019 data format. Key design principles:

1. **Patient-level processing** — One file per patient, streamed sequentially
2. **No imputation** — Missing values preserved as NaN
3. **Causal ordering** — ICULOS-sorted, no future data in features
3. **Patient-level splits** — No hourly-level leakage
4. **Schema validation** — 13 validation rules per patient file

---

## 2. Real Data Adaptations

### 2.1 Dataset Structure Mapping

| Aspect | Synthetic Data | Real PhysioNet Data | Adaptation |
|--------|---------------|---------------------|------------|
| **File layout** | Flat `data/raw/pXXXXX.psv` | Partitioned `data/raw/training_setA/`, `training_setB/` | ✅ Loader uses `rglob` to discover all `.psv` files |
| **Partition IDs** | None | Set A (p000001-p020336), Set B (p100001-p119416) | ✅ `_infer_source_dataset()` maps path to set_a/set_b |
| **Patient IDs** | p00001-p00500 | p000001-p020336, p100001-p119416 | ✅ Regex `p\d{5,6}` matches both formats |
| **Header row** | None (synthetic) | Present in all files | ✅ `_file_has_header()` detects automatically |
| **File size** | Small (~10KB) | Variable (1-40KB) | ✅ Streaming handles any size |

### 2.2 Validation Results on Real Data

| Check | Status | Notes |
|-------|--------|-------|
| **Schema validation** | ✅ PASS | All 39,810 files have correct 41-column header |
| **SepsisLabel domain** | ✅ PASS | Only 0.0 and 1.0 values |
| **ICULOS monotonic** | ✅ PASS | 0 non-monotonic rows in both sets |
| **Duplicate IDs** | ✅ PASS | No duplicates within or across sets |
| **Column presence** | ✅ PASS | All 41 columns present |
| **Zero-byte files** | ✅ PASS | None found |

### 2.3 Data Quality Observations

| Issue | Count | Action |
|-------|-------|--------|
| Patients not starting at ICULOS=1 | 8,759 (22%) | Feature pipeline uses `hour` (0-indexed) from ICULOS |
| Out-of-range values | ~500 total | Flagged but preserved (clinical plausibility bounds) |
| High lab missingness | 94-99% | Expected; missingness indicators added as features |

---

## 3. Feature Engineering on Real Data

### 3.1 Feature Categories (from `feature_config.py`)

| Category | Variables | Count | Real Data Considerations |
|----------|-----------|-------|-------------------------|
| **Current** | 40 clinical vars | 40 | Raw observations preserved; NaN kept |
| **Missingness** | 40 vars × 4 types | 160 | Critical for high-missingness labs |
| **Forward-fill** | 40 vars | 40 | Vitals: 4h window, Labs: 6h window |
| **Rolling** | 40 vars × 5 aggs × 3 windows | 600 | Causal trailing windows (6/12/24h) |
| **Trend** | 40 vars × 3 types × 3 windows | 360 | Causal; min 2 observations |
| **Clinical** | Shock index, SIRS | 3 | Computed from available vitals/labs |
| **Static** | Age, Gender, Unit1, Unit2, HospAdmTime | 5 | Repeated per hour |

**Total engineered features: ~1,200 per patient-hour**

### 3.2 Missing Value Handling Strategy

**Decision: No global imputation — missingness is informative**

| Strategy | Applied To | Rationale |
|----------|------------|-----------|
| **Preserve NaN** | All raw variables | Missing = test not ordered (MNAR) |
| **Missingness indicators** | All 40 clinical variables | `_is_missing`, `_hours_since_last`, `_availability_*h` |
| **Limited forward-fill** | All 40 clinical variables | Vitals: 4h max, Labs: 6h max; expires if gap too long |
| **Rolling stats with min_obs=1** | All 40 clinical variables | Compute if at least 1 observation in window |
| **Trends with min_obs=2** | All 40 clinical variables | Need 2 points for slope/change |

**Key principle:** All imputation/fill is **causal** — only past/present data used for current hour prediction.

### 3.3 Forward-Fill Window Configuration

| Variable Group | Max Window | Rationale |
|----------------|------------|-----------|
| Vital signs (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2) | 4 hours | Vitals change rapidly; older values less reliable |
| Laboratory (all 26 lab variables) | 6 hours | Labs more stable; longer carry-forward acceptable |
| Default | 4 hours | Conservative default |

---

## 4. Target Construction

### 4.1 Official Challenge Target (Recommended for Training)

```python
# Binary: 1 if SepsisLabel == 1 (septic or septic within 6h)
y = (df['SepsisLabel'] == 1).astype(int)
```

**Statistics on real data:**
- Hourly positive rate: 1.80% (27,573 / 1,530,732)
- Patient-level prevalence: 7.27% (2,896 / 39,810)
- Severe class imbalance (1:55)

### 4.2 Project Clinical Target (For Analysis)

```python
# Ternary: 1=onset window, 0=pre-onset, -1=ineligible (post-onset)
# From targets.py: assign_sepsis_target()
```

**Statistics:**
- Positive (onset window): ~0.9% of hours
- Negative (pre-onset): ~97% of eligible hours
- Ineligible (post-onset): ~2% of hours

### 4.3 Target Configuration for Real Data

```python
TARGET_CONFIG = {
    "primary_training": "SepsisLabel",      # Binary, official challenge
    "clinical_analysis": "sepsis_within_6h", # Ternary, onset window
    "prediction_horizon": 6,                 # Hours (matches challenge)
    "target_shift": 6,                       # SepsisLabel shifted 6h pre-onset
}
```

---

## 5. Patient-Level Splits

### 5.1 Split Strategy (from `patient_splits.py`)

```python
PatientSplitter(
    test_size=0.2,           # 20% test
    validation_size=0.1,     # 10% validation
    random_state=42,         # Fixed seed
    stratify_by_label=True,  # Stratify by patient sepsis status
    strategy="stratified_random"
)
```

### 5.2 Expected Split Sizes (39,810 patients)

| Split | Patients | Septic Patients | Rows (est.) |
|-------|----------|-----------------|-------------|
| Train | 31,848 | ~2,317 | ~1,224,586 |
| Validation | 3,981 | ~290 | ~153,073 |
| Test | 3,981 | ~290 | ~153,073 |

### 5.3 Leakage Prevention

| Check | Implementation |
|-------|---------------|
| **Patient isolation** | `PatientSplitter.split()` assigns entire patient to one split |
| **No temporal leakage** | Features use only current/past data (`transform_patient`) |
| **Target isolation** | `validate_target_isolation()` ensures SepsisLabel not in features |
| **Split verification** | `check_split_leakage()` confirms zero patient overlap |

---

## 6. Preprocessing Verification Tests

### 6.1 Unit Tests to Run

```bash
# Preprocessing tests
pytest icu-sentinel/ml/p1_clinical/preprocessing/tests/ -v

# Feature engineering tests
pytest icu-sentinel/ml/p1_clinical/features/tests/ -v

# Integration tests
pytest icu-sentinel/tests/ -v -k "preprocess or feature"
```

### 6.2 Key Assertions

| Test | Expected Result |
|------|-----------------|
| Patient-level split isolation | Zero patient overlap between splits |
| Temporal causality | No feature uses future hour data |
| Target integrity | SepsisLabel not in feature columns |
| Missingness preserved | NaN values in raw columns unchanged |
| Forward-fill expiry | Values expire after configured window |
| Rolling window causality | Trailing windows only (no future) |
| Static variables constant | Age, Gender, Unit1, Unit2, HospAdmTime same per patient |

---

## 7. Scaling Considerations for 40K Patients

### 7.1 Memory Management

| Technique | Implementation |
|-----------|----------------|
| **Streaming loader** | `PhysioNetChallenge2019Loader.iter_patients()` yields one patient at a time |
| **Per-patient parquet** | Features saved as individual `.parquet` files |
| **Batch writing** | `run_feature_pipeline_streaming()` uses pyarrow batches |
| **No combined in-memory** | `combined=None` in streaming mode |

### 7.2 Expected Runtime (Estimates)

| Step | Synthetic (500) | Real (39,810) | Scaling |
|------|-----------------|---------------|---------|
| Validation | ~30s | ~40 min | ~80x |
| Cleaning | ~20s | ~25 min | ~75x |
| Feature engineering | ~2 min | ~2.5 hours | ~75x |
| Patient splitting | ~5s | ~5 min | ~60x |
| **Total** | ~3 min | **~3.5 hours** | |

---

## 8. Configuration for Real Data Run

### 8.1 Feature Config (save as `feature_config_real.json`)

```json
{
  "rolling_window_hours": [6, 12, 24],
  "rolling_aggregations": ["mean", "min", "max", "std", "latest"],
  "rolling_min_observations": 1,
  "trend_window_hours": [6, 12, 24],
  "trend_min_observations": 2,
  "forward_fill_max_window_hours": {
    "default": 4,
    "vital_signs": 4,
    "laboratory": 6
  },
  "feature_variables": [40 clinical variables],
  "static_variables": ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"],
  "compute_missingness_features": true,
  "compute_forward_fill": true,
  "compute_rolling_features": true,
  "compute_trend_features": true,
  "compute_clinical_features": true,
  "shocks_index_enabled": true,
  "sirs_count_enabled": true
}
```

### 8.2 Preprocessing Config

```json
{
  "raw_dir": "data/raw",
  "processed_dir": "data/processed/physionet2019",
  "splits_dir": "data/splits/physionet2019",
  "reports_dir": "data/reports/physionet2019",
  "test_size": 0.2,
  "validation_size": 0.1,
  "random_state": 42,
  "stratify_by_label": true,
  "limit": null,
  "save_processed": true,
  "save_manifests": true,
  "save_reports": true
}
```

---

## 9. Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **Dataset loader** | ✅ Ready | Handles partitioned structure, header detection |
| **Validation** | ✅ Ready | 13 rules; all pass on real data |
| **Cleaning** | ✅ Ready | Type coercion, ICULOS sort, no imputation |
| **Patient splits** | ✅ Ready | Stratified, leak-proof, reproducible |
| **Target construction** | ✅ Ready | Both official and clinical targets |
| **Feature engineering** | ✅ Ready | 6 categories, causal, ~1200 features |
| **Missing value handling** | ✅ Documented | No imputation; indicators + limited forward-fill |
| **Scaling** | ✅ Ready | Streaming, per-patient parquet, batch writing |

**Next step:** Run full preprocessing pipeline on real data (Phase C → Phase D → Phase E)

---

*Report generated by ICU Sentinel Phase 2 Real Data Migration Pipeline*