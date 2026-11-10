# Real Data Migration Audit — ICU Sentinel

**Generated:** 2026-11-09
**Purpose:** Document the transition from synthetic data to PhysioNet/CinC Challenge 2019 real data.

---

## 1. Existing Implementation Summary

### 1.1 Repository Structure

```
icu-sentinel/
├── ml/
│   ├── p1_clinical/          # P1: Data Engineering, Clinical Baselines, Classical ML
│   │   ├── ingestion/        # Dataset loading (PhysioNetChallenge2019Loader)
│   │   ├── preprocessing/    # Validation, cleaning, patient splits, target construction
│   │   ├── features/         # Feature engineering (current, missingness, rolling, trend, clinical)
│   │   ├── baselines/        # SIRS, qSOFA, NEWS-style clinical scores
│   │   ├── classical_ml/     # Logistic Regression, Random Forest, SVM
│   │   └── evaluation/       # Metrics, calibration, bootstrap, lead time, alert burden
│   ├── p2_boosting/          # P2: Advanced Tabular Models
│   │   ├── training.py       # Training orchestration with checkpointing
│   │   ├── config.py         # Model configurations (Extra Trees, LightGBM, XGBoost)
│   │   ├── pipelines.py      # Model training pipelines
│   │   ├── evaluation.py     # Model evaluation and threshold selection
│   │   └── calibration.py    # Probability calibration
│   └── p3_temporal/          # P3: Temporal Models (GRU/LSTM), SHAP, Alert Engine
├── backend/                  # FastAPI services
├── scripts/                  # Generation and training scripts
└── tests/                    # Integration tests
```

### 1.2 P1 Pipeline (Validated on Synthetic Data)

**Components Verified (408/408 tests passing):**
- Dataset ingestion with catalog discovery and streaming
- 13 validation rules for data integrity
- Cleaning: type coercion, ICULOS sorting, no imputation
- Patient-level stratified splits (train/val/test) with leakage checks
- Target construction: `sepsis_within_6h` (6-hour pre-onset window)
- Feature engineering: 6 categories, 994 features total
- Clinical baselines: SIRS (complete), qSOFA (partial - AMS missing), NEWS (partial - O2/consciousness missing)
- Classical ML: LR, RF, SVM with feature selection and threshold optimization

**Synthetic Data Statistics (500 patients):**
- Patients: 500 total (350 train, 50 val, 100 test)
- Hourly records: 14,978 total
- Sepsis patients: 75 (15.0%)
- Sepsis hours: 1,519 (10.1%)
- Missingness: ~20% per clinical variable

### 1.3 P2 Pipeline (Validated on Synthetic Data)

**Models Implemented (58/60 tests passing):**
- Extra Trees (n_estimators=200, class_weight=balanced)
- LightGBM (n_estimators=200, learning_rate=0.05, class_weight=balanced)
- XGBoost (n_estimators=200, learning_rate=0.05, max_depth=4)

**Synthetic Data Results:**
| Model | AUROC | AUPRC | Sensitivity | Specificity | F1 |
|-------|-------|-------|-------------|-------------|-----|
| Extra Trees | 0.490 | 0.031 | 0.522 | 0.516 | 0.065 |
| LightGBM | 0.503 | 0.035 | 0.067 | 0.942 | 0.048 |
| XGBoost | 0.501 | 0.033 | 0.033 | 0.945 | 0.025 |

**Note:** All AUROC CIs include 0.5 → No signal in synthetic data (expected).

---

## 2. Required Changes for Real Data Migration

### 2.1 Data Acquisition (BLOCKER)

| Task | Status | Notes |
|------|--------|-------|
| Verify official dataset source | PENDING | PhysioNet/CinC Challenge 2019 |
| Confirm access requirements | PENDING | Requires credentialed PhysioNet account |
| Download training_setA (20,336 patients) | BLOCKED | S3: s3://physionet-open/challenge-2019/1.0.0/training/training_setA/ |
| Download training_setB (20,000 patients) | BLOCKED | S3: s3://physionet-open/challenge-2019/1.0.0/training/training_setB/ |
| Verify file integrity (checksums) | PENDING | Post-download |

### 2.2 Data Organization

Current structure expects:
```
data/
  raw/
    training_setA/
      pXXXXX.psv
    training_setB/
      pXXXXX.psv
```

**Change required:** The synthetic data is currently flat in `data/raw/` (p00001.psv–p00500.psv). Real data must be organized into `training_setA/` and `training_setB/` subdirectories per the official partitioning.

### 2.3 Pipeline Adaptations

| Module | Change Required | Risk |
|--------|-----------------|------|
| `dataset_loader.py` | Already supports real data schema | LOW |
| `preprocessing/pipeline.py` | Already handles Set A/Set B inference | LOW |
| `preprocessing/targets.py` | Verify `sepsis_within_6h` matches official 6h-shifted label | MEDIUM |
| `features/` | Feature engineering expects same 40 variables | LOW |
| `baselines/` | qSOFA/NEWS still partial (AMS, O2, consciousness missing) | LOW |
| `classical_ml/` | Feature selector must handle real missingness patterns | MEDIUM |
| `p2_boosting/` | Config defaults may need tuning for real data scale | MEDIUM |

### 2.4 Configuration Changes

**New configuration mechanism needed:**
```python
# Environment variable or config file
DATASET_MODE = "physionet2019"  # vs "synthetic"
```

**Separate artifact directories:**
```
artifacts/
  physionet2019/
    p1/
    p2/
  synthetic/
    p1/
    p2/
```

---

## 3. Reusable Modules (No Changes Needed)

| Module | Status | Notes |
|--------|--------|-------|
| `PhysioNetChallenge2019Loader` | ✅ Ready | Handles header detection, NA values, cataloging |
| `validate_patient_file` | ✅ Ready | 13 validation rules |
| `clean_patient_dataframe` | ✅ Ready | Type coercion, ICULOS sort, no imputation |
| `PatientSplitter` | ✅ Ready | Stratified patient-level splits, leakage checks |
| `sepsis_within_6h` target | ⚠️ Verify | Must match official SepsisLabel semantics |
| Feature engineering pipeline | ✅ Ready | 6 categories, causal windows |
| Clinical baseline calculators | ✅ Ready | SIRS complete; qSOFA/NEWS partial |
| Classical ML training protocol | ✅ Ready | Train/val/test separation, feature selection |
| P2 training orchestration | ✅ Ready | Checkpointing, threshold selection on val |
| Calibration (sigmoid/isotonic) | ✅ Ready | Validation-only fitting |
| Patient-level bootstrap | ✅ Ready | Resamples patients, not hours |

---

## 4. Potential Incompatibilities

| Issue | Impact | Resolution |
|-------|--------|------------|
| **Dataset partitioning** | Synthetic: flat 500 files; Real: Set A (20,336) + Set B (20,000) | Update `_infer_source_dataset` to use directory structure |
| **Sepsis label semantics** | Synthetic: random onset; Real: 6h-shifted official label | Verify `targets.py` logic matches challenge definition |
| **Missingness patterns** | Synthetic: 20% uniform; Real: clinical missingness (MNAR) | Feature selector must handle all-infinite columns |
| **Class imbalance** | Synthetic: ~10% positive hours; Real: ~3-5% (challenge docs) | Class weights and threshold selection already implemented |
| **Scale** | Synthetic: 500 patients; Real: 40,336 patients | Memory-efficient streaming already implemented |
| **Variable units** | Synthetic: normalized; Real: clinical units (mmHg, °C, etc.) | No scaling in pipeline — models handle raw values |

---

## 5. Data Acquisition Requirements

### 5.1 Official Dataset Information

| Attribute | Value |
|-----------|-------|
| **Title** | PhysioNet/CinC Challenge 2019: Early Prediction of Sepsis |
| **Source** | PhysioNet (physionet.org) |
| **Version** | 1.0.0 |
| **Access** | Credentialed account required; must accept data use agreement |
| **License** | PhysioNet Credentialed Health Data License 1.5.0 |
| **Citation** | Reyna MA, et al. "Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019" |

### 5.2 Dataset Structure

| Partition | Patients | Files | Source |
|-----------|----------|-------|--------|
| training_setA | 20,336 | 20,336 .psv | S3: training_setA/ |
| training_setB | 20,000 | 20,000 .psv | S3: training_setB/ |
| **Total** | **40,336** | **40,336** | |

### 5.3 File Format

- One pipe-delimited (.psv) file per patient
- 41 columns: 40 time-dependent variables + SepsisLabel
- Separator: "|" (no spaces)
- Missing values: "NaN"
- Header row: Present in official files
- SepsisLabel: 1 if t ≥ t_sepsis - 6, else 0 (6-hour early window)

### 5.4 Variables (40 Clinical + 1 Label)

**Vital Signs (8):** HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2  
**Laboratory (26):** BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets  
**Demographics (6):** Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS  
**Outcome (1):** SepsisLabel

---

## 6. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cannot obtain PhysioNet credentials | HIGH | BLOCKER | Document exact steps; request owner action |
| Dataset schema differs from expectation | LOW | MEDIUM | Validation step will catch; loader is flexible |
| Memory/compute for 40K patients | MEDIUM | HIGH | Streaming pipeline; parquet per patient; batch processing |
| Sepsis label definition mismatch | MEDIUM | HIGH | Explicit verification in Phase B3 |
| qSOFA/NEWS still incomplete | HIGH | LOW | Document limitation; SIRS is complete |
| Platform variance in golden tests | LOW | LOW | Accept functional correctness; document variance |

---

## 7. Execution Plan

### Phase A: Data Acquisition (Weeks 1-2)
1. **A1** Verify official dataset source and access requirements
2. **A2** Obtain credentials / download via authorized S3 access
3. **A3** Organize into `data/raw/training_setA/` and `data/raw/training_setB/`
4. **A4** Run `validate_dataset.py` to verify integrity
5. **A5** Generate `PHYSIONET_DATA_ACQUISITION.md` report

### Phase B: Dataset Understanding (Week 2)
1. **B1** Run exploratory analysis on real data
2. **B2** Generate `dataset_summary.json`, `feature_inventory.csv`, `missingness_report.csv`
3. **B3** Verify sepsis label definition vs `sepsis_within_6h` target
4. **B4** Generate `PHYSIONET_DATA_QUALITY.md` and `PHYSIONET_TARGET_DEFINITION.md`

### Phase C: Preprocessing Adaptation (Week 3)
1. **C1** Create feature mapping document
2. **C2** Validate missing-value handling for real patterns
3. **C3** Run feature engineering, verify no temporal leakage
4. **C4** Generate `PHYSIONET_PREPROCESSING.md`

### Phase D: Patient Splitting (Week 3)
1. **D1** Run patient-level splits on 40,336 patients
2. **D2** Leakage audit
3. **D3** Generate split manifests and summary

### Phase E: P1 on Real Data (Week 4)
1. **E1** Clinical baselines (SIRS complete; qSOFA/NEWS partial)
2. **E2** Classical ML (LR, RF, SVM) if computationally feasible
3. **E3** Generate P1 artifacts and reports

### Phase F: P2 on Real Data (Weeks 4-5)
1. **F1** Train Extra Trees, LightGBM, XGBoost
2. **F2** Calibration (sigmoid/isotonic)
3. **F3** Patient-level bootstrap (1000 iterations)
4. **F4** Model comparison and early-warning evaluation

### Phase G: Early-Warning Evaluation (Week 5)
1. **G1** Lead time analysis
2. **G2** Alert burden calculation
3. **G3** Threshold selection on validation

### Phase H: Final Validation (Week 5)
1. **H1** Full pipeline reproducibility test
2. **H2** All tests pass
3. **H3** Final reports

### Phase I: P3 Handoff (Week 5)
1. **I1** Document real-data feature schema, model I/O
2. **I2** Generate `REAL_DATA_P3_HANDOFF.md`

---

## 8. Immediate Next Steps

1. **Create `data/raw/training_setA/` and `data/raw/training_setB/` directories**
2. **Attempt S3 download using `audit_dataset.py` (requires AWS CLI and credentials)**
3. **If credentials unavailable, document exact steps for project owner**
4. **Begin writing `PHYSIONET_DATA_ACQUISITION.md` with current status**

---

## 9. Synthetic Data Preservation

**Strategy:** Keep synthetic pipeline fully functional.

- Synthetic data generation: `scripts/generate_synthetic_data.py`
- Synthetic artifacts: `artifacts/synthetic/` (new)
- Real data artifacts: `artifacts/physionet2019/` (new)
- Configuration: `DATASET_MODE` environment variable
- Tests: Continue to use synthetic fixtures for unit tests

---

**Prepared by:** ICU Sentinel Real Data Migration Audit  
**Date:** 2026-11-09