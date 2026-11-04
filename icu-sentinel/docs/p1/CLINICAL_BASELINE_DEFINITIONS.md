# Clinical Baseline Definitions — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 5 — Clinical baseline scoring systems  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Overview

This document defines the three clinical baseline scoring systems implemented
in P1:

1. **SIRS** (Systemic Inflammatory Response Syndrome)
2. **qSOFA** (quick Sequential Organ Failure Assessment)
3. **NEWS2-style** (National Early Warning Score 2, modified)

These baselines provide clinically interpretable reference points for
comparison against P1's classical ML models and P2's boosting models.

All scores are computed **per patient-hour** using only current/past
observations (causal). The original `SepsisLabel` is preserved separately
for later evaluation and is never used as a score input.

---

## 2. SIRS

### Source

Bone RC, et al. "Definitions for sepsis and organ failure and guidelines
for the use of innovative therapies in sepsis." *Chest*.
1992;101(6):1644-1655.

### Criteria (>= 2 of 4 for positive)

| Component | Threshold | Dataset Variable | Unit |
|-----------|-----------|-----------------|------|
| Temperature | > 38.3 C or < 36.0 C | Temp | deg_c |
| Heart rate | > 90 bpm | HR | beats_per_minute |
| Respiration | > 20 breaths/min | Resp | breaths_per_minute |
| WBC | > 12 or < 4 (10^3/uL) | WBC | count_times_10_3_per_ul |

### Implementation notes

- **PaCO2 criterion** (PaCO2 < 32 mmHg) is NOT available in the Challenge
  dataset. The respiration criterion uses only Resp (> 20 breaths/min).
- **WBC bands criterion** (> 10% bands) is NOT available. The WBC criterion
  uses only WBC count (> 12 or < 4 x 10^3/uL).
- Missing components are never counted as negative.
- A **complete** SIRS score requires all 4 components.
- Partial scores are exposed but NOT labeled as complete SIRS.
- Score range: 0-4. Positive: >= 2 (complete only).

---

## 3. qSOFA

### Source

Singer M, et al. "The Third International Consensus Definitions for Sepsis
and Septic Shock (Sepsis-3)." *JAMA*. 2016;315(8):801-810.

### Criteria (>= 2 of 3 for positive)

| Component | Threshold | Dataset Variable | Unit |
|-----------|-----------|-----------------|------|
| Respiratory rate | >= 22 breaths/min | Resp | breaths_per_minute |
| Systolic BP | <= 100 mmHg | SBP | mm_hg |
| Altered mental status | GCS < 15 | NOT available | — |

### Implementation notes

- **Altered mental status** is NOT available in the Challenge dataset and
  cannot be inferred from any other variable. No proxy is used.
- qSOFA will be **partial** unless altered mental status data is provided
  externally.
- Missing components are never counted as negative.
- A **complete** qSOFA score requires all 3 components.
- Partial scores are exposed but NOT labeled as complete qSOFA.
- Score range: 0-3. Positive: >= 2 (complete only).

---

## 4. NEWS2-style

### Source

Royal College of Physicians. National Early Warning Score (NEWS) 2.

### Parameters (scores 0, 1, 2, or 3 per parameter)

| Parameter | Thresholds → Score | Dataset Variable | Unit |
|-----------|-------------------|-----------------|------|
| Respiratory rate | <=8→3, 9-11→1, 12-20→0, 21-24→1, >=25→3 | Resp | breaths_per_minute |
| SpO2 | <=91→3, 92-93→2, 94-95→1, 96-97→0, >=98→0 | O2Sat | percent |
| Supplemental oxygen | No→0, Yes→+2 | NOT available | — |
| Temperature | <=35.0→3, 35.1-36.0→1, 36.1-38.0→0, 38.1-39.0→0, >=39.1→2 | Temp | deg_c |
| Systolic BP | <=90→3, 91-100→2, 101-119→1, 120-219→0, >=220→3 | SBP | mm_hg |
| Heart rate | <=40→3, 41-50→1, 51-90→0, 91-110→0, 111-130→1, 131-140→2, >=141→3 | HR | beats_per_minute |
| Consciousness | A→0, C/V/P/U→3 | NOT available | — |

### Clinical concern threshold

Score >= 5 indicates clinical concern (complete score only).

### Implementation notes

- **Supplemental oxygen** is NOT available in the dataset. FiO2 is NOT
  used as a proxy for supplemental oxygen administration.
- **Consciousness level** (A/C/V/P/U) is NOT available in the dataset.
- This is a **modified NEWS-style score**, not full NEWS2.
- Missing components are never counted as normal (score 0).
- Score range: 0-21 (7 parameters x max 3 points). Clinical concern: >= 5
  (complete only).

---

## 5. Completeness model

Every score result includes a completeness assessment:

| Level | Meaning | Score computed? | is_positive |
|-------|---------|-----------------|-------------|
| COMPLETE | All required components available | Yes | Yes/No |
| PARTIAL | Some components available | Yes | None (not labeled) |
| NONE | No components available | No | None |

- Missing components are never treated as normal values.
- Partial scores are useful for research exploration but must not be
  labeled as complete validated scores.
- `components_available` and `components_required` are always reported.

---

## 6. Temporal correctness

All scores are computed at each patient-hour using only observations
available at or before that hour:

- No future observations are used.
- No future imputation.
- No centered rolling windows.
- `SepsisLabel` is never used as a score input.
- Patient ID, hospital ID, and source cohort are never score components.
- The original `SepsisLabel` is preserved unchanged for later evaluation.
- `SepsisLabel` is never shifted or reconstructed.

---

## 7. Configuration

All thresholds, variable mappings, and scoring parameters are centralized
in `ml/p1_clinical/baselines/clinical_score_config.py` and are
serializable to/from JSON.

Default values are taken from the authoritative sources cited above and
are not adjusted from memory.

---

## 8. Scientific safeguards

- No clinical validation is claimed.
- Modified scores are not called original validated scores.
- NEWS2-style is explicitly labeled as modified, not full NEWS2.
- No clinical thresholds are invented.
- No unavailable variables are silently treated as normal.
- Missing components are explicitly tracked and reported.
- Every assumption and limitation is documented.
