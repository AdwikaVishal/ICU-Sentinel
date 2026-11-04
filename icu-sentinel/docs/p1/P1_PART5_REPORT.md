# ICU Sentinel — P1 PART 5 Report

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1)  
**Part:** PART 5 — Clinical baseline scoring systems  
**Status:** Complete (awaiting approval to begin PART 6)  
**Date:** 2026-11-03  

---

## 1. Implementation Summary

PART 5 implements three clinical baseline scoring systems for the
PhysioNet/CinC Challenge 2019 dataset: SIRS, qSOFA, and NEWS2-style.
All scores are computed per patient-hour using only current/past
observations (causal). The original `SepsisLabel` is preserved unchanged
for later evaluation.

All code is tested using synthetic fixtures (no real patient data
available without PhysioNet credentialed access).

---

## 2. Files Created

### New source files

| File | Purpose |
|------|---------|
| `ml/p1_clinical/baselines/__init__.py` | Package exports |
| `ml/p1_clinical/baselines/baseline_schema.py` | Shared score schema (P2/P3 contract) |
| `ml/p1_clinical/baselines/clinical_score_config.py` | Centralized thresholds and variable mappings |
| `ml/p1_clinical/baselines/score_completeness.py` | Completeness tracking (COMPLETE/PARTIAL/NONE) |
| `ml/p1_clinical/baselines/sirs.py` | SIRS implementation (Bone et al., Chest 1992) |
| `ml/p1_clinical/baselines/qsofa.py` | qSOFA implementation (Sepsis-3, JAMA 2016) |
| `ml/p1_clinical/baselines/news_style.py` | NEWS2-style implementation (RCP NEWS2) |
| `ml/p1_clinical/baselines/baseline_pipeline.py` | End-to-end pipeline orchestration |

### New test files

| File | Tests |
|------|-------|
| `ml/p1_clinical/tests/test_sirs.py` | Threshold boundaries, complete/partial, missing components, sepsis label preservation |
| `ml/p1_clinical/tests/test_qsofa.py` | Threshold boundaries, AMS not available, partial scores, sepsis label preservation |
| `ml/p1_clinical/tests/test_news_style.py` | Parameter scoring, supplemental O2/consciousness not available, partial scores |
| `ml/p1_clinical/tests/test_score_completeness.py` | Completeness levels, availability ratio, reason messages |
| `ml/p1_clinical/tests/test_baseline_pipeline.py` | Multi-patient pipeline, combined DataFrame, config/schema serialization, multiple hospitals |
| `ml/p1_clinical/tests/test_baseline_temporal_integrity.py` | No future leakage, cross-patient contamination, sorting, target-label isolation, sparse observations, duplicate timestamps |

### New data reports

| File | Source | Notes |
|------|--------|-------|
| `data/reports/clinical_variable_availability.json` | Variable mapping | Marked synthetic |
| `data/reports/baseline_completeness_report.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/baseline_feature_summary.json` | Synthetic pipeline run | Marked synthetic |
| `data/reports/clinical_score_config.json` | Configuration | Marked synthetic |
| `data/reports/baseline_schema.json` | Schema | Marked synthetic |

### New documentation

| File | Purpose |
|------|---------|
| `docs/p1/CLINICAL_BASELINE_DEFINITIONS.md` | Full definitions, thresholds, sources, safeguards |
| `docs/p1/CLINICAL_VARIABLE_MAPPING.md` | Variable-to-component mapping table |
| `docs/p1/P1_PART5_REPORT.md` | This report |

### Modified files

None. All PART 5 files are new; no existing files were modified.

---

## 3. Scoring Systems Implemented

### SIRS

- **Source:** Bone et al., Chest 1992;101(6):1644-1655
- **Components:** Temperature, Heart rate, Respiration, WBC
- **Thresholds:** Temp >38.3 or <36.0 C; HR >90 bpm; Resp >20/min; WBC >12 or <4 (10^3/uL)
- **Positive:** >= 2 criteria (complete score only)
- **Deviations:** PaCO2 criterion and WBC bands not available in dataset

### qSOFA

- **Source:** Singer et al., JAMA 2016;315(8):801-810 (Sepsis-3)
- **Components:** Respiratory rate, Systolic BP, Altered mental status
- **Thresholds:** RR >= 22/min; SBP <= 100 mmHg; AMS (GCS < 15)
- **Positive:** >= 2 criteria (complete score only)
- **Deviations:** Altered mental status NOT available in dataset; no proxy used

### NEWS2-style

- **Source:** Royal College of Physicians, NEWS2
- **Components:** RR, SpO2, Supplemental O2, Temperature, SBP, HR, Consciousness
- **Thresholds:** Per NEWS2 parameter scoring tables
- **Clinical concern:** >= 5 points (complete score only)
- **Deviations:** Supplemental oxygen and consciousness NOT available; modified NEWS-style, not full NEWS2

---

## 4. Completeness Model

Every score result includes:

| Field | Description |
|-------|-------------|
| `complete` | True if all required components available |
| `components_available` | Count of components with valid measurements |
| `components_required` | Total required components |
| `is_positive` | True/False (complete only), None (partial/none) |
| `score_value` | Raw score (complete/partial only), None (none) |

Completeness levels:
- **COMPLETE:** All components available → score and is_positive computed
- **PARTIAL:** Some components available → score computed, is_positive = None
- **NONE:** No components available → score = None, is_positive = None

Missing components are never counted as negative.

---

## 5. Configuration

All thresholds, variable mappings, and scoring parameters are centralized
in `ml/p1_clinical/baselines/clinical_score_config.py` and serializable
to/from JSON (`data/reports/clinical_score_config.json`).

Default thresholds are from authoritative sources and are not adjusted
from memory.

---

## 6. Test Results

**Total: 234 tests passed, 0 failed.**

Breakdown:
- Part 1 foundation: 5 passed
- Part 2 ingestion: 25 passed
- Part 3 preprocessing: 42 passed
- Part 4 features: 97 passed
- Part 5 baselines: 65 passed (6 test files)

All tests use synthetic fixtures. No real patient data is used.

Ruff checks: All passed (0 errors).

---

## 7. Synthetic vs Real-Data Verification

**Real-data verification: Not available.**

Real PhysioNet/CinC Challenge 2019 data requires credentialed access
which has not been obtained. All validation, scoring, and reports use
synthetic fixtures only.

All reports in `data/reports/` are marked `"source": "synthetic"` and
include a note stating they are not real PhysioNet/CinC Challenge 2019
statistics.

No clinical performance is reported from synthetic data as evidence of
real predictive performance.

---

## 8. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No real data access | Cannot validate against actual dataset | Pipeline ready; synthetic tests pass |
| qSOFA partial (no AMS) | qSOFA always partial without external AMS data | Partial scores exposed; documented |
| NEWS2 partial (no O2/consciousness) | NEWS2 always partial without external data | Partial scores exposed; documented |
| SIRS partial (no PaCO2/bands) | SIRS respiration/WBC criteria approximated | Deviations documented |
| Cross-hospital evaluation | Not yet performed | Architecture prepared; no claims made |

---

## 9. Handoff Requirements for Part 6

Part 6 (Classical ML: Logistic Regression, Random Forest, SVM) can
consume baseline scores as:

1. **Baseline features:** SIRS score, qSOFA score, NEWS2-style score can
   be used as features alongside engineered features from Part 4.
2. **Baseline comparison:** AUROC/AUPRC of baselines vs. ML models.
3. **Score outputs:** `data/reports/combined_baselines.parquet` contains
   one row per patient-hour with score value, completeness, and positivity.

Key contract points for Part 6:
- All scores are causal (no future leakage)
- `SepsisLabel` is target-only, never a score input
- Missing components are tracked via completeness flags
- Partial scores have `is_positive = None`; handle appropriately in evaluation

Full details: `docs/p1/CLINICAL_BASELINE_DEFINITIONS.md` and
`docs/p1/CLINICAL_VARIABLE_MAPPING.md`.

---

## 10. Remaining Work

1. Obtain PhysioNet credentialed access and download real dataset to
   `data/raw/`
2. Run Part 3 preprocessing pipeline against real data
3. Run Part 4 feature pipeline against real processed data
4. Run Part 5 baseline pipeline against real data
5. Validate baseline reports against real data
6. P1 Part 6: Classical ML models (Logistic Regression, Random Forest, SVM)

---

## Data Integrity Statement

- No raw patient data was downloaded (no PhysioNet credentials)
- No raw patient data was committed to Git
- No dataset statistics were fabricated; all reports marked synthetic
- Source data would remain unchanged (pipeline only reads Part 3/4 output)
- Malformed records are never silently discarded

Awaiting approval to begin PART 6 (Classical ML: Logistic Regression, Random Forest, SVM).
