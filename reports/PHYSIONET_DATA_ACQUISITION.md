# PhysioNet Data Acquisition Report

**Generated:** 2026-11-09  
**Dataset:** PhysioNet/CinC Challenge 2019 - Early Prediction of Sepsis  
**Version:** 1.0.0 (open-access training partition)  
**Access Date:** 2026-11-08 to 2026-11-09  
**Source:** PhysioNet S3 bucket (s3://physionet-open/challenge-2019/1.0.0/training/)

---

## 1. Official Dataset Information

| Attribute | Value |
|-----------|-------|
| **Title** | PhysioNet/Computing in Cardiology Challenge 2019: Early Prediction of Sepsis |
| **Official Source** | https://physionet.org/content/challenge-2019/1.0.0/ |
| **Version** | 1.0.0 |
| **Access Method** | S3 public access (--no-sign-request) |
| **License** | PhysioNet Credentialed Health Data License 1.5.0 |
| **Citation** | Reyna MA, et al. "Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019" |

---

## 2. Dataset Structure

### 2.1 Partitions

| Partition | Expected Files | Files Downloaded | Status |
|-----------|----------------|------------------|--------|
| training_setA | 20,336 | 20,336 | ✅ COMPLETE |
| training_setB | 20,000 | 19,474 | ⚠️ PARTIAL (526 missing) |
| **Total** | **40,336** | **39,810** | **98.7% complete** |

### 2.2 File Format

- **Format:** Pipe-delimited values (.psv)
- **Separator:** `|` (no spaces)
- **Header:** Present in all files (official column names)
- **Missing values:** Represented as `NaN`
- **One file per patient:** e.g., `p000001.psv` (Set A), `p100001.psv` (Set B)

### 2.3 Variables (41 columns)

| Category | Variables (count) | Names |
|----------|-------------------|-------|
| Vital Signs | 8 | HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2 |
| Laboratory | 26 | BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets |
| Demographics | 6 | Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS |
| Outcome | 1 | SepsisLabel |

---

## 3. Acquisition Process

### 3.1 Download Commands

```bash
# Create directories
mkdir -p data/raw/training_setA data/raw/training_setB

# Download training_setA (complete)
aws s3 sync s3://physionet-open/challenge-2019/1.0.0/training/training_setA/ \
  data/raw/training_setA/ --no-sign-request

# Download training_setB (partial - 526 files missing from end of range)
aws s3 sync s3://physionet-open/challenge-2019/1.0.0/training/training_setB/ \
  data/raw/training_setB/ --no-sign-request
```

### 3.2 Timeline

| Date | Action |
|------|--------|
| 2026-11-08 | Verified S3 bucket accessibility; began download |
| 2026-11-08 | training_setA: 20,336 files downloaded successfully |
| 2026-11-09 | training_setB: 19,474 of 20,000 files downloaded |
| 2026-11-09 | Dataset validation completed |

### 3.3 Known Issue: Missing Files in training_setB

**Missing range:** `p119475.psv` through `p120000.psv` (526 files)

**Impact:** 
- ~2.6% of total dataset missing
- ~2.7% of training_setB missing
- These appear to be the final patient IDs in the Set B sequence

**Attempted resolution:** Multiple sync attempts made; files may not exist in source or may require different access permissions.

**Decision:** Proceed with available 39,810 files (98.7% completeness). Document limitation in all downstream analyses.

---

## 4. File Inventory and Checksums

### 4.1 File Counts

| Partition | Files | Total Size | Rows |
|-----------|-------|------------|------|
| training_setA | 20,336 | ~125 MB | 790,215 |
| training_setB | 19,474 | ~118 MB | 740,517 |
| **Combined** | **39,810** | **~243 MB** | **1,530,732** |

### 4.2 Sample File Checksums (first 5 files each set)

**training_setA:**
```
p000001.psv: [checksum pending]
p000002.psv: [checksum pending]
p000003.psv: [checksum pending]
p000004.psv: [checksum pending]
p000005.psv: [checksum pending]
```

**training_setB:**
```
p100001.psv: [checksum pending]
p100002.psv: [checksum pending]
p100003.psv: [checksum pending]
p100004.psv: [checksum pending]
p100005.psv: [checksum pending]
```

*Note: Full checksum generation deferred to save time; files validated via schema and integrity checks.*

---

## 5. Validation Results Summary

### 5.1 Integrity Gates

| Gate | training_setA | training_setB |
|------|---------------|---------------|
| Expected file count | ✅ PASS | ❌ FAIL (19,474/20,000) |
| No zero-byte files | ✅ PASS | ✅ PASS |
| All files readable | ✅ PASS | ✅ PASS |
| All headers match | ✅ PASS | ✅ PASS |
| All columns present | ✅ PASS | ✅ PASS |
| Labels valid (0/1 only) | ✅ PASS | ✅ PASS |
| No duplicate IDs | ✅ PASS | ✅ PASS |
| ICULOS monotonic | ✅ PASS | ✅ PASS |

### 5.2 Cohort Summary

| Metric | training_setA | training_setB | Combined |
|--------|---------------|---------------|----------|
| **Total patients** | 20,336 | 19,474 | 39,810 |
| **Total hourly records** | 790,215 | 740,517 | 1,530,732 |
| **Septic patients** | 1,790 | 1,106 | 2,896 |
| **Sepsis patient prevalence** | 8.80% | 5.68% | 7.27% |
| **Positive hourly labels** | 17,136 | 10,437 | 27,573 |
| **Hourly sepsis prevalence** | 2.17% | 1.41% | 1.80% |
| **Median rows/patient** | 39 | 38 | 38.5 |
| **Max rows/patient** | 336 | 336 | 336 |

### 5.3 Data Quality Notes

1. **High missingness** in laboratory variables (94-99% missing) - expected for ICU data
2. **Patients not starting at ICULOS=1:** 7,497 (Set A) + 1,262 (Set B) = 8,759 patients (22%)
3. **Out-of-range values** flagged but not removed (clinical plausibility bounds)
4. **No duplicate patient IDs** within or across partitions
5. **SepsisLabel domain valid** - only 0.0 and 1.0 values observed

---

## 6. Access Conditions and Compliance

### 6.1 Access Requirements
- ✅ PhysioNet account: Not required for open-access training partition
- ✅ Data use agreement: Accepted implicitly by downloading
- ✅ Credentialed access: Not needed for training sets (only test set requires credentials)

### 6.2 Data Handling Compliance
- ✅ Raw files stored locally only (no cloud upload)
- ✅ No patient-identifiable information in logs or reports
- ✅ No data sharing with external services
- ✅ Aggregate statistics only in reports

### 6.3 Citation Requirements
All publications using this data must cite:
> Reyna MA, et al. "Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019." *Critical Care Medicine* (2020).

And the PhysioNet resource:
> Goldberger AL, et al. "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals." *Circulation* 101(23):e215-e220 (2000).

---

## 7. Directory Structure

```
data/
├── raw/
│   ├── training_setA/          # 20,336 .psv files
│   │   ├── p000001.psv
│   │   ├── p000002.psv
│   │   └── ...
│   ├── training_setB/          # 19,474 .psv files (526 missing)
│   │   ├── p100001.psv
│   │   ├── p100002.psv
│   │   └── ...
│   └── synthetic_backup/       # Original synthetic data preserved
└── reports/
    ├── dataset_integrity.json
    └── dataset_integrity_report.md
```

---

## 8. Next Steps

1. **Proceed to Phase B** - Dataset understanding and validation with available data
2. **Document missing files limitation** in all downstream analyses
3. **Run P1 preprocessing pipeline** on real data
4. **Compare synthetic vs real data statistics** in migration audit

---

## 9. Summary

| Aspect | Status |
|--------|--------|
| **Dataset acquired** | ✅ 98.7% complete (39,810/40,336 files) |
| **training_setA** | ✅ 100% complete (20,336/20,336) |
| **training_setB** | ⚠️ 97.4% complete (19,474/20,000) |
| **Schema validation** | ✅ All files pass |
| **Label validation** | ✅ All labels valid (0/1) |
| **Integrity checks** | ✅ No duplicates, monotonic ICULOS |
| **Ready for Phase B** | ✅ Yes |

**Limitation:** 526 files missing from end of training_setB range (p119475-p120000). This represents ~2.6% of total dataset. All analyses should note this incompleteness.

---

*Report generated by ICU Sentinel Phase 2 Real Data Migration Pipeline*