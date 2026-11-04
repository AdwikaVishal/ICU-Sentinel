# Clinical Variable Mapping — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 5 — Clinical baseline scoring systems  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Purpose

This document maps each clinical score component to the available dataset
variables in the PhysioNet/CinC Challenge 2019. It identifies which
components can be computed reliably and which cannot.

---

## 2. Dataset schema

The Challenge 2019 dataset contains 41 columns per patient-hour:

- **8 vital signs:** HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2
- **26 laboratory values:** BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST,
  BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct,
  Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total,
  TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets
- **6 demographics:** Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS
- **1 outcome:** SepsisLabel

---

## 3. Variable mapping table

### SIRS (4 components)

| Component | Official threshold | Required variable | Dataset variable | Unit | Available | Missingness behavior | Computable reliably |
|-----------|-------------------|-------------------|-----------------|------|-----------|---------------------|---------------------|
| Temperature | > 38.3 or < 36.0 C | Temperature | Temp | deg_c | Yes | NaN preserved | Yes |
| Heart rate | > 90 bpm | Heart rate | HR | beats_per_minute | Yes | NaN preserved | Yes |
| Respiration | > 20 breaths/min | Respiration | Resp | breaths_per_minute | Yes | NaN preserved | Yes (PaCO2 not available) |
| WBC | > 12 or < 4 (10^3/uL) | WBC count | WBC | count_times_10_3_per_ul | Yes | NaN preserved | Yes (bands not available) |

**SIRS completeness:** Complete = all 4 components available. Partial scores
exposed but not labeled as complete SIRS.

**Deviations from original SIRS:** PaCO2 criterion and WBC bands criterion
not available; only Resp and WBC count used.

---

### qSOFA (3 components)

| Component | Official threshold | Required variable | Dataset variable | Unit | Available | Missingness behavior | Computable reliably |
|-----------|-------------------|-------------------|-----------------|------|-----------|---------------------|---------------------|
| Respiratory rate | >= 22 breaths/min | Respiration | Resp | breaths_per_minute | Yes | NaN preserved | Yes |
| Systolic BP | <= 100 mmHg | Systolic BP | SBP | mm_hg | Yes | NaN preserved | Yes |
| Altered mental status | GCS < 15 | GCS/consciousness | NOT available | — | No | N/A | No |

**qSOFA completeness:** Complete = all 3 components available. Since altered
mental status is not available, qSOFA will be partial unless AMS data is
provided externally.

**Deviations from original qSOFA:** Altered mental status component not
available; no proxy used.

---

### NEWS2-style (7 components)

| Component | Official thresholds | Required variable | Dataset variable | Unit | Available | Missingness behavior | Computable reliably |
|-----------|-------------------|-------------------|-----------------|------|-----------|---------------------|---------------------|
| Respiratory rate | <=8→3, 9-11→1, 12-20→0, 21-24→1, >=25→3 | Respiration | Resp | breaths_per_minute | Yes | NaN preserved | Yes |
| SpO2 | <=91→3, 92-93→2, 94-95→1, 96-97→0, >=98→0 | SpO2 | O2Sat | percent | Yes | NaN preserved | Yes |
| Supplemental oxygen | No→0, Yes→+2 | O2 therapy | NOT available | — | No | N/A | No |
| Temperature | <=35.0→3, 35.1-36.0→1, 36.1-38.0→0, 38.1-39.0→0, >=39.1→2 | Temperature | Temp | deg_c | Yes | NaN preserved | Yes |
| Systolic BP | <=90→3, 91-100→2, 101-119→1, 120-219→0, >=220→3 | Systolic BP | SBP | mm_hg | Yes | NaN preserved | Yes |
| Heart rate | <=40→3, 41-50→1, 51-90→0, 91-110→0, 111-130→1, 131-140→2, >=141→3 | Heart rate | HR | beats_per_minute | Yes | NaN preserved | Yes |
| Consciousness | A→0, C/V/P/U→3 | Consciousness level | NOT available | — | No | N/A | No |

**NEWS2 completeness:** Complete = all 7 components available. Since
supplemental oxygen and consciousness are not available, NEWS2-style will
be partial unless those data are provided externally.

**Deviations from original NEWS2:** Supplemental oxygen and consciousness
components not available. FiO2 is NOT used as a proxy for supplemental
oxygen. This is a modified NEWS-style score, not full NEWS2.

---

## 4. Summary of unavailable components

| Score | Unavailable components | Reason |
|-------|----------------------|--------|
| SIRS | PaCO2 criterion, WBC bands | Not in Challenge dataset |
| qSOFA | Altered mental status (GCS) | Not in Challenge dataset |
| NEWS2-style | Supplemental oxygen, Consciousness | Not in Challenge dataset |

---

## 5. Missingness policy

- Missing values are **never** treated as normal.
- Missing components are **never** counted as negative criteria.
- Missing components reduce the completeness level (COMPLETE → PARTIAL → NONE).
- Partial scores are exposed but not labeled as complete validated scores.
- `components_available` and `components_required` are always reported.

---

## 6. Data integrity

- No clinical thresholds were invented; all thresholds are from the
  authoritative sources cited in CLINICAL_BASELINE_DEFINITIONS.md.
- No unavailable variables are inferred from other variables.
- FiO2 is not used as proof of supplemental oxygen administration.
- No altered mental status proxy is invented from unrelated variables.
- All reports are generated from synthetic fixture data and marked as such.
