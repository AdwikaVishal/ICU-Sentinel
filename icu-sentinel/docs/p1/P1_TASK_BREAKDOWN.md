# P1 Task Breakdown

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1)  
**Scope:** All P1-assigned responsibilities across the 7-part implementation
sequence.

> **Status update (2026-11-06).** The per-part tables below are the original
> PART 1 planning baseline and are retained for history. They are no longer
> the current status. Current verified state:
>
> - Parts 2–7 code is implemented (ingestion, preprocessing, features,
>   baselines, classical ML, evaluation) and the P1 test suite passes.
> - The strict 6h pre-onset target (`sepsis_within_6h` + `target_eligible`)
>   is implemented and `SepsisLabel` is preserved unchanged.
> - Training now evaluates on validation, selects thresholds on validation
>   only, and evaluates the frozen models once on test.
> - P1 patient API is implemented as read-only over processed artefacts.
> - Real-data artefacts remain **BLOCKED**: the PhysioNet/CinC 2019 dataset
>   is not present in `data/raw/`. Synthetic fixtures are test-only and are
>   not reported as real results.
> - P2 is still unimplemented (`ml/p2_boosting/`), so the P2 handover is a
>   contract/readiness report rather than completed P2 code.

## Overview

P1 owns: data engineering, clinical baselines (SIRS/qSOFA/NEWS), classical ML
(Logistic Regression / Random Forest / SVM), evaluation, and patient data APIs.
P1 creates shared interfaces/schemas so P2 (boosting/ensembles/calibration) and
P3 (temporal models/SHAP/React) can integrate later.

P2/P3 modules are intentionally NOT implemented by P1.

---

## PART 1 — Project foundation (COMPLETED)

| # | Task | Status |
|---|------|--------|
| 1.1 | Initialize repository | Done |
| 1.2 | Configure Python environment (venv) | Done |
| 1.3 | Install dependencies (`requirements.txt`) | Done |
| 1.4 | Create folder structure + `__init__.py` files | Done |
| 1.5 | Configure pytest (`pyproject.toml`) | Done |
| 1.6 | Configure logging (`logging.yaml`) | Done |
| 1.7 | Create common schemas (`ml/common/schemas`) | Done |
| 1.8 | Write README | Done |
| 1.9 | Verify project starts (tests + lint + import) | Done |

**Tests:** `ml/p1_clinical/tests/test_foundation.py` — 5 passed.

---

## PART 2 — Dataset acquisition and exploration (PENDING)

| # | Task | Status |
|---|------|--------|
| 2.1 | Inspect official PhysioNet/CinC Challenge 2019 documentation (access, license, schema) | Pending |
| 2.2 | Implement dataset loader (`ml/p1_clinical/ingestion/`) | Pending |
| 2.3 | Validate downloaded files | Pending |
| 2.4 | Generate dataset summary | Pending |
| 2.5 | Generate data dictionary | Pending |
| 2.6 | Analyse missingness | Pending |
| 2.7 | Document target labels | Pending |

**Note:** No patient data downloaded. Do not commit patient data to Git.

---

## PART 3 — Preprocessing (PENDING)

| # | Task | Status |
|---|------|--------|
| 3.1 | Validate patient and hospital IDs | Pending |
| 3.2 | Detect duplicates | Pending |
| 3.3 | Validate hourly ordering | Pending |
| 3.4 | Handle missing observations | Pending |
| 3.5 | Generate patient-level splits | Pending |
| 3.6 | Add automated tests | Pending |

---

## PART 4 — Feature engineering (PENDING)

| # | Task | Status |
|---|------|--------|
| 4.1 | Current observation features | Pending |
| 4.2 | Causal forward filling | Pending |
| 4.3 | Missingness indicators | Pending |
| 4.4 | Time-since-last-measurement | Pending |
| 4.5 | Rolling 6/12/24-hour features | Pending |
| 4.6 | Temporal slopes | Pending |
| 4.7 | Shock index | Pending |
| 4.8 | SIRS count | Pending |
| 4.9 | Feature schema | Pending |
| 4.10 | Leakage tests | Pending |

---

## PART 5 — Clinical baselines (PENDING)

| # | Task | Status |
|---|------|--------|
| 5.1 | SIRS | Pending |
| 5.2 | qSOFA | Pending |
| 5.3 | NEWS-style scoring | Pending |
| 5.4 | Unit tests | Pending |
| 5.5 | Baseline evaluation | Pending |

---

## PART 6 — Classical ML (PENDING)

| # | Task | Status |
|---|------|--------|
| 6.1 | Logistic Regression | Pending |
| 6.2 | Random Forest | Pending |
| 6.3 | SVM | Pending |
| 6.4 | Training and inference pipelines | Pending |
| 6.5 | Model serialization | Pending |
| 6.6 | Comparative evaluation | Pending |

---

## PART 7 — Research evaluation (PENDING)

| # | Task | Status |
|---|------|--------|
| 7.1 | AUROC | Pending |
| 7.2 | AUPRC | Pending |
| 7.3 | Sensitivity | Pending |
| 7.4 | Specificity | Pending |
| 7.5 | Precision | Pending |
| 7.6 | F1-score | Pending |
| 7.7 | Brier score | Pending |
| 7.8 | Lead time | Pending |
| 7.9 | False alerts per patient-day | Pending |
| 7.10 | Feature ablation | Pending |
| 7.11 | Subgroup analysis | Pending |
| 7.12 | Bootstrap confidence intervals | Pending |

---

## PART 8 — Patient APIs (PENDING)

| # | Endpoint | Status |
|---|----------|--------|
| 8.1 | `GET /patients` | Pending (stub app exists) |
| 8.2 | `GET /patients/{id}` | Pending |
| 8.3 | `GET /patients/{id}/timeline` | Pending |
| 8.4 | `GET /patients/{id}/features` | Pending |

---

## PART 9 — Documentation and release (PENDING)

| # | Task | Status |
|---|------|--------|
| 9.1 | Technical documentation | Pending |
| 9.2 | Experiment reports | Pending |
| 9.3 | Dataset citation | Pending |
| 9.4 | Reproducibility instructions | Pending |
| 9.5 | Automated tests | Pending |
| 9.6 | Final P1 verification report | Pending |

---

## Shared interfaces (created in PART 1, for P2/P3 integration)

- `ml/common/schemas/` — `PatientRecord`, `Observation`, `SplitMetadata`,
  `FeatureSet`, `ClinicalBaselineResult`, `MetricResult`
- `ml/common/metrics/` — `compute_binary_metrics()`
- `ml/common/fixtures/` — synthetic patient generators for tests

## Scientific constraints (P1, all parts)

- Follow official PhysioNet/CinC Challenge 2019 label convention.
- Split by patient, never by hourly observation.
- Never use future observations in features (causal only).
- Fit preprocessing on training data only.
- Tune thresholds on validation data only.
- Keep test set untouched until final evaluation.
- Report missingness honestly; never fabricate results.
- No clinical validation claims from retrospective experiments.

## Current state

Only PART 1 is complete. Awaiting approval to begin PART 2.
