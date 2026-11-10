# PhysioNet Data Quality Report

**Generated:** 2026-11-09  
**Dataset:** PhysioNet/CinC Challenge 2019 v1.0.0 (training partitions)  
**Source:** `data/reports/dataset_integrity.json` (validated 2026-11-09)

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Total patients** | 39,810 (98.7% of expected 40,336) |
| **Total hourly records** | 1,530,732 |
| **Septic patients** | 2,896 (7.27%) |
| **Positive hourly labels** | 27,573 (1.80%) |
| **Partitions complete** | Set A: 100%, Set B: 97.4% |
| **Schema validation** | ✅ All files pass |
| **Label validity** | ✅ 100% valid (0/1 only) |
| **Temporal ordering** | ✅ ICULOS monotonic in all files |

---

## 2. File-Level Quality

### 2.1 Inventory

| Partition | Expected | Found | Missing | Zero-byte | Unreadable |
|-----------|----------|-------|---------|-----------|------------|
| training_setA | 20,336 | 20,336 | 0 | 0 | 0 |
| training_setB | 20,000 | 19,474 | 526 | 0 | 0 |
| **Total** | **40,336** | **39,810** | **526** | **0** | **0** |

**Missing files:** `p119475.psv` through `p120000.psv` (526 files at end of Set B range)

### 2.2 Schema Compliance
- ✅ All files have correct 41-column header
- ✅ Separator is `|` (pipe) as specified
- ✅ Missing values encoded as `NaN`
- ✅ SepsisLabel domain: {0.0, 1.0} only
- ✅ No duplicate patient IDs within or across partitions

---

## 3. Cohort Characteristics

### 3.1 Patient-Level Statistics

| Metric | training_setA | training_setB | Combined |
|--------|---------------|---------------|----------|
| **Patients** | 20,336 | 19,474 | 39,810 |
| **Septic patients** | 1,790 | 1,106 | 2,896 |
| **Sepsis prevalence** | 8.80% | 5.68% | 7.27% |
| **Median LOS (hours)** | 39 | 38 | 38.5 |
| **Mean LOS (hours)** | 38.9 | 38.0 | 38.5 |
| **Max LOS (hours)** | 336 | 336 | 336 |
| **Min LOS (hours)** | 8 | 8 | 8 |

### 3.2 Hourly-Level Statistics

| Metric | training_setA | training_setB | Combined |
|--------|---------------|---------------|----------|
| **Total hours** | 790,215 | 740,517 | 1,530,732 |
| **Positive hours** | 17,136 | 10,437 | 27,573 |
| **Hourly sepsis rate** | 2.17% | 1.41% | 1.80% |

---

## 4. Missingness Analysis

### 4.1 Missingness by Category

| Category | Variables | Set A Avg Missing | Set B Avg Missing | Combined Avg |
|----------|-----------|-------------------|-------------------|--------------|
| **Vital Signs** | 8 | 33.6% | 29.3% | 31.5% |
| **Laboratory** | 26 | 94.2% | 96.1% | 95.2% |
| **Demographics** | 6 | 16.3% | 9.9% | 13.1% |
| **Outcome** | 1 | 0.0% | 0.0% | 0.0% |

### 4.2 Top 15 Most Missing Variables (Combined)

| Rank | Variable | Category | Missing % |
|------|----------|----------|-----------|
| 1 | EtCO2 | Vital Signs | 96.34% |
| 2 | TroponinI | Laboratory | 99.06% |
| 3 | Bilirubin_direct | Laboratory | 99.81% |
| 4 | Fibrinogen | Laboratory | 99.34% |
| 5 | HCO3 | Laboratory | 95.76% |
| 6 | BaseExcess | Laboratory | 94.51% |
| 7 | PaCO2 | Laboratory | 94.40% |
| 8 | SaO2 | Laboratory | 96.53% |
| 9 | Lactate | Laboratory | 97.33% |
| 10 | PTT | Laboratory | 97.02% |
| 11 | Phosphate | Laboratory | 95.98% |
| 12 | Platelets | Laboratory | 94.05% |
| 13 | Alkalinephos | Laboratory | 98.39% |
| 14 | AST | Laboratory | 98.37% |
| 15 | Bilirubin_total | Laboratory | 98.50% |

### 4.3 Least Missing Variables (Combined)

| Rank | Variable | Category | Missing % |
|------|----------|----------|-----------|
| 1 | Age | Demographics | 0.00% |
| 2 | Gender | Demographics | 0.00% |
| 3 | HospAdmTime | Demographics | 0.00% |
| 4 | ICULOS | Demographics | 0.00% |
| 5 | SepsisLabel | Outcome | 0.00% |
| 6 | HR | Vital Signs | 9.85% |
| 7 | MAP | Vital Signs | 12.42% |
| 8 | SBP | Vital Signs | 14.59% |
| 9 | Resp | Vital Signs | 15.29% |
| 10 | O2Sat | Vital Signs | 13.04% |

---

## 5. Out-of-Range Values

Values exceeding clinical plausibility bounds (flagged, not removed):

| Variable | Set A Count | Set B Count | Combined | Notes |
|----------|-------------|-------------|----------|-------|
| MAP | 94 | 106 | 200 | Mean arterial pressure |
| DBP | 13 | 90 | 103 | Diastolic BP |
| Temp | 4 | 2 | 6 | Temperature |
| TroponinI | 0 | 189 | 189 | Cardiac biomarker |
| WBC | 11 | 7 | 18 | White blood cells |
| FiO2 | 2 | 19 | 21 | Inspired O2 fraction |
| Resp | 0 | 48 | 48 | Respiratory rate |
| Calcium | 2 | 14 | 16 | Calcium |
| Hct | 1 | 1 | 2 | Hematocrit |
| Hgb | 1 | 2 | 3 | Hemoglobin |
| Creatinine | 1 | 1 | 2 | Creatinine |
| Potassium | 1 | 1 | 2 | Potassium |
| Platelets | 0 | 1 | 1 | Platelets |
| Lactate | 1 | 0 | 1 | Lactate |
| BaseExcess | 1 | 0 | 1 | Base excess |
| Fibrinogen | 1 | 0 | 1 | Fibrinogen |
| Chloride | 2 | 0 | 2 | Chloride |

**Note:** These are extreme but physiologically possible values in critical care. No automatic removal applied.

---

## 6. Temporal Consistency

| Check | training_setA | training_setB | Status |
|-------|---------------|---------------|--------|
| ICULOS monotonic (no decreasing) | ✅ 0 violations | ✅ 0 violations | PASS |
| Starts at ICULOS=1 | 12,839 (63.1%) | 18,212 (93.5%) | ⚠️ See below |

### 6.1 Patients Not Starting at ICULOS=1

| Partition | Count | Percentage | Implication |
|-----------|-------|------------|-------------|
| training_setA | 7,497 | 36.9% | May indicate transfer from another ICU/ward |
| training_setB | 1,262 | 6.5% | Lower rate, possible different data collection |
| **Combined** | **8,759** | **22.0%** | Must handle in feature engineering |

**Action:** Feature engineering must not assume hour 1 = ICU admission for these patients. Use `HospAdmTime` for hospital-relative timing.

---

## 7. Label Quality

### 7.1 SepsisLabel Validation
- ✅ Domain: Only 0.0 and 1.0 values present
- ✅ No missing labels (0 missing in both sets)
- ✅ No invalid labels (0 invalid in both sets)

### 7.2 Label Distribution
- **Patient-level:** 7.27% of patients have at least one positive hour
- **Hourly-level:** 1.80% of all hours are SepsisLabel=1
- **Interpretation:** SepsisLabel is shifted 6 hours before clinical onset (per challenge definition)

### 7.3 Class Imbalance
- **Severe imbalance:** 1.8% positive hourly rate
- **Implication:** Requires class weighting, appropriate metrics (AUPRC > AUROC), careful threshold selection

---

## 8. Partition Differences

| Aspect | training_setA | training_setB | Difference |
|--------|---------------|---------------|------------|
| Patients | 20,336 | 19,474 | -862 |
| Sepsis prevalence | 8.80% | 5.68% | -3.12 pp |
| Hourly sepsis rate | 2.17% | 1.41% | -0.76 pp |
| Median LOS | 39h | 38h | -1h |
| Not starting at ICULOS=1 | 36.9% | 6.5% | -30.4 pp |
| Avg lab missingness | 94.2% | 96.1% | +1.9 pp |
| Vital signs missingness | 33.6% | 29.3% | -4.3 pp |

**Key finding:** Set A has higher sepsis prevalence and more patients with ICU transfer (not starting at ICULOS=1). Set B has slightly more missing lab data. These are **dataset partitions, not hospitals** — do not interpret as site differences.

---

## 9. Data Quality Recommendations

### 9.1 For Preprocessing
1. **Do not impute** laboratory values globally — missingness is informative (test not ordered)
2. **Use missingness indicators** as features (already in feature pipeline)
3. **Forward-fill with limits** for vital signs (max 6 hours per challenge protocol)
4. **Preserve out-of-range values** — they may be clinically significant

### 9.2 For Feature Engineering
1. **Time since last measurement** is critical given high missingness
2. **Rolling windows** must be causal (no future data)
3. **Demographics** (Age, Gender, Unit1, Unit2) are constant per patient — use once
4. **HospAdmTime** provides hospital-relative timeline for transfer patients

### 9.3 For Modeling
1. **Class weights** essential (1:55 positive:negative ratio hourly)
2. **Patient-level splits** mandatory (no hourly-level leakage)
3. **AUPRC preferred** over AUROC for evaluation
4. **Threshold selection** on validation set only

---

## 10. Limitations

1. **Missing 526 files** from training_setB (p119475-p120000) — ~2.6% of data
2. **No test set** available (requires credentialed PhysioNet access)
3. **Set A/B are partitions, not hospitals** — no cross-hospital validation possible
4. **No outcome adjudication** beyond SepsisLabel — cannot validate clinical onset
5. **Static demographics only** — no comorbidities, medications, or interventions recorded

---

*Report generated from validated dataset_integrity.json using ICU Sentinel Phase 2 pipeline*