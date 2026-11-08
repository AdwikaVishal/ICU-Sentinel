# P1 Final Report — ICU Sentinel

**Date:** 2026-11-06  
**Branch:** `main`  
**Tested interpreter:** Python 3.11.9  
**Overall status:** **NO-GO for empirical claims** (real dataset absent);
**GO WITH CONDITIONS for code handover.**

This report deliberately separates code completion from empirical
verification. No synthetic fixture is presented as a real result.

---

## 1. Executive summary

The P1 codebase is now internally consistent and its full automated suite
passes. Audit findings in bootstrap evaluation, duplicate feature detection,
target construction, training/evaluation protocol, patient API, dependency
documentation, and the SHAP/torch import path have been remediated.

P1 cannot yet make a real-data performance claim. The official
PhysioNet/CinC Challenge 2019 dataset is not present under `data/raw/`, and
the repository correctly refuses to fabricate a substitute. Therefore:

- **Code status:** `COMPLETE` for the P1 implementation scope.
- **Real-data status:** `BLOCKED` on dataset access.
- **Artifact status:** `MISSING` for real trained models and real metrics.
- **P2 status:** `MISSING` (`ml/p2_boosting/` remains unimplemented).
- **Final verdict:** `NO-GO` until real artifacts and real evaluation exist.

---

## 2. Audit remediation

| Finding | Resolution | Status |
|---|---|---|
| Bootstrap metric dispatch used arity-broken callables and broad exception suppression | Direct metric functions, explicit validation, warnings instead of silent suppression | `COMPLETE` |
| Bootstrap accepted unsupported metrics/iterations/confidence | `SUPPORTED_METRICS` plus range/column validation | `COMPLETE` |
| Undefined AUPRC on single-class resamples | Returns `NaN` explicitly | `COMPLETE` |
| Feature selector missed duplicate columns whose first values matched | Full-column `blake2b` hash plus `Series.equals` verification | `COMPLETE` |
| Target was only the noisy original `SepsisLabel` | Added strict 6h pre-onset derived target while preserving original label | `COMPLETE` |
| Training reported training metrics as validation/test | Validation evaluation, validation-only threshold selection, one-shot test evaluation | `COMPLETE` |
| No saved threshold provenance | `thresholds.json` records validation selection and `test_set_used_for_selection=False` | `COMPLETE` |
| Patient API was a `/health` stub | Read-only patients/timeline/features API with explicit 400/404/503 behaviour | `COMPLETE` |
| API could imply invented hospital IDs or synthetic fallback | Missing fields return `null`; absent data returns 503; no fallback data | `COMPLETE` |
| Environment versions in docs were unverified | Corrected to the installed Python 3.11.9 environment | `COMPLETE` |
| Dependency declaration gaps | Added `scipy`, `lightgbm`, `xgboost`; retained `shap` and documented torch | `COMPLETE` |
| SHAP import crashed on a broken optional torch DLL | Guarded lazy SHAP import; P3 suite now collects and passes | `COMPLETE` |
| No runnable end-to-end artifact path | Added `scripts/generate_p1_artifacts.py` | `COMPLETE` |

---

## 3. Dataset acquisition status — `BLOCKED`

- Source: <https://physionet.org/content/challenge-2019/1.0.0/> (DOI
  `10.13026/v64v-d857`).
- Access requires a credentialed PhysioNet account and completion of the
  required training. Access controls must not be bypassed.
- Expected layout: one pipe-delimited `.psv` file per patient under
  `data/raw/`, flat or nested, named `p#####.psv`.
- Dataset scale: 20,336 Set A + 20,000 Set B subjects (40,336 total).
- Set A/Set B path heuristics are **not** hospital identifiers.

The artifact runner exits with code 2 and an acquisition message when no
raw files are present. This is intentional.

---

## 4. Approved target — `COMPLETE`

Policy (`ml/p1_clinical/preprocessing/targets.py`):

- `onset_first_positive` = first hour where `SepsisLabel == 1`, per patient.
- `sepsis_within_6h = 1` for hours `[onset, onset + 5]`.
- `sepsis_within_6h = 0` for hours before onset and for non-septic patients.
- `target_eligible = False` for hours `>= onset + 6`.
- Original `SepsisLabel` is preserved unchanged.
- Missing labels make a row ineligible with `NaN` target.
- Non-binary, non-missing labels raise `ValueError`.
- The horizon is configurable through `TargetPolicy`; default is 6 hours.

Constants: `SepsisLabel`, `sepsis_within_6h`, `target_eligible`.

---

## 5. Preprocessing and splits — `PARTIAL` (code complete; real data blocked)

- Patient files are validated for identifiers, duplicates, hourly order, and
  missing observations.
- Cleaning enforces causal handling: no future values, bounded forward fill
  (4h vitals / 6h labs).
- Splits are patient-level, stratified by `has_sepsis`, with
  `test_size=0.2`, `validation_size=0.1`, seed 42.
- Split manifests and per-split patient ID lists are written by the pipeline.
- Leakage checks verify that no patient appears in more than one split.
- `hospital_id` is `null` when absent; it is never invented.
- Real split integrity remains `BLOCKED` until real data is processed.

---

## 6. Feature engineering — `PARTIAL` (code complete; real data blocked)

- Causal current-value, missingness, time-since-measurement, 6/12/24h
  rolling/trend, shock-index, and SIRS features.
- Approximately 994 engineered features in the current configured set.
- All target columns are excluded from feature summaries, invalid-value
  checks, and model feature lists.
- The pipeline validates that derived target values are internally
  consistent after assignment.
- Per-patient engineered features are written to `data/features/patients/`
  so the split-aware training loader can consume them.
- Real feature generation and distribution checks are `BLOCKED`.

---

## 7. Clinical baselines — `COMPLETE` (code)

- SIRS is available from the supplied variables.
- qSOFA is **never complete** because altered mental status is unavailable.
- NEWS2 is **never complete** because supplemental oxygen and consciousness
  are unavailable.
- Missing components must remain explicit; baselines are not silently
  imputed into a complete score.

---

## 8. Classical ML — `COMPLETE` (code)

- Logistic Regression, Random Forest, and SVM implementations exist with
  preprocessing and serialization.
- Preprocessing, imputers, scalers, and feature selection are fit on the
  training split only.
- Feature selection uses deterministic duplicate detection and retains the
  first occurrence by column order.
- Models, registry metadata, selected features, and feature-selection reports
  are persisted by the training protocol.
- P2 boosting/ensembles/calibration are **not implemented**.

---

## 9. Training and evaluation protocol — `COMPLETE`

Protocol implemented in `ml/p1_clinical/classical_ml/training.py`:

1. Load patient-level train/validation/test splits.
2. Fit preprocessing and feature selection on training data only.
3. Train enabled models on the training split.
4. Evaluate on validation at 0.5 to obtain probabilities.
5. Select one alert threshold per model on **validation only**, using
   `f1`, `precision`, `recall`, or `balanced_accuracy`.
6. Evaluate each frozen model **once** on the test split at the selected
   validation threshold.
7. Persist `thresholds.json`, validation/test metrics, selected features,
   feature-selection report, and model artifacts.

Guardrails:

- Missing validation data raises `ValueError`.
- Test data never influences threshold selection.
- `thresholds.json` explicitly records `selection_split="validation"` and
  `test_set_used_for_selection=false`.
- Invalid selection metrics raise rather than silently falling back.

---

## 10. Evaluation and bootstrap — `COMPLETE` (code)

- Bootstrap supports the declared binary metrics, validates required columns,
  iterations, confidence level, and metric names.
- Undefined metrics are warned about and omitted rather than silently
  returning fabricated values.
- AUPRC with no positive class returns `NaN`.
- Unexpected runtime errors propagate instead of being swallowed.
- Real bootstrap results are `BLOCKED` until real predictions exist.

---

## 11. Patient API — `COMPLETE` (code); `BLOCKED` (real data)

Read-only FastAPI application in `backend/p1_patients/app.py`:

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Service health only |
| `GET /patients` | List patients from processed artefacts |
| `GET /patients/{id}` | Patient summary |
| `GET /patients/{id}/timeline` | Hourly timeline with offset/limit |
| `GET /patients/{id}/features` | Feature rows with offset/limit |

Configuration:

- `ICU_SENTINEL_PROCESSED_DIR` (default `data/processed`)
- `ICU_SENTINEL_FEATURES_FILE` (default `data/features/combined_features.parquet`)

Safety behaviour:

- Missing/unconfigured data returns HTTP 503.
- Unknown patient returns HTTP 404.
- Invalid patient ID returns HTTP 400 (regex `^[A-Za-z0-9_-]{1,64}$`).
- Path traversal is rejected.
- NaN/inf/numpy values are converted to JSON-safe values.
- Missing `hospital_id` is returned as `null`, never invented.
- No synthetic demo patients are served.

Nine dedicated API tests pass.

---

## 12. Reproducibility and environment

Artifact runner:

```bash
python scripts/generate_p1_artifacts.py
python scripts/generate_p1_artifacts.py --skip-training
python scripts/generate_p1_artifacts.py --limit 500   # smoke only, never final results
```

Expected outputs when real data is available:

- `data/processed/*.parquet`
- `data/splits/split_manifest.json` and per-split patient ID lists
- `data/features/combined_features.parquet`
- `data/features/patients/*.parquet`
- `data/features/feature_config.json`
- `data/features/feature_schema.json`
- `data/features/target_policy.json`
- `data/reports/classical_ml/` metrics and `thresholds.json`
- `artifacts/p1/classical_ml/` fitted models and registry

Verified environment: Python 3.11.9, pandas 2.3.3, numpy 1.26.4,
scikit-learn 1.4.2, fastapi 0.162.0, uvicorn 0.54.0, pyarrow 24.0.0,
pytest 8.4.2, pytest-cov 6.0.0, ruff 0.7.4, mypy 1.13.0.

`torch` is not a repository dependency. SHAP normally probes it during
import. A broken local CPU torch (`WinError 1114`, `c10.dll`) can crash the
interpreter, so the explainability factory blocks that optional import by
default. Set `ICU_SENTINEL_ENABLE_TORCH=1` only in an environment with a
working torch install.

---

## 13. Test evidence

Commands run from `icu-sentinel/` on 2026-11-06:

```bash
python -m pytest --no-cov -p no:cacheprovider --disable-warnings --tb=short -q
```

Result: **485 passed, 0 failed.**

| Suite | Tests | Result |
|---|---:|---|
| `ml/p1_clinical/tests` | 395 | Passed |
| `ml/p3_temporal/tests` | 66 | Passed |
| `tests` (API and integration) | 24 | Passed |

Additional checks:

- `ruff check` on all touched P1/API/script files: **passed**.
- `mypy` on target, training, ingestion, pipeline, API, and artifact runner:
  **passed** (only an untyped-body note remains).
- `git diff --check`: no whitespace errors.
- Artifact runner without raw data: exits 2 with acquisition instructions.

Limitations: these are unit/integration tests on fixtures. They do not prove
real-data performance, calibration, fairness, or clinical utility.

---

## 14. P2 handover contract, conditions, and verdict

### P2 input contract

P2 may consume only:

- `data/features/combined_features.parquet` for shared schema/row checks.
- `data/features/patients/*.parquet` for split-aware model training.
- `data/features/feature_schema.json` for the feature contract.
- `data/features/feature_config.json` for window/config provenance.
- `data/features/target_policy.json` for target provenance.
- `data/splits/split_manifest.json` for patient-level split membership.
- `data/reports/classical_ml/thresholds.json` for the validation-selected
  threshold policy.

P2 must:

1. Keep all three target columns out of model features:
   `SepsisLabel`, `sepsis_within_6h`, `target_eligible`.
2. Respect patient-level splits; no patient may cross splits.
3. Fit preprocessing/calibration on training data only.
4. Select thresholds and hyperparameters on validation only.
5. Evaluate the final frozen ensemble once on test.
6. Never report training metrics as validation/test metrics.
7. Label any synthetic smoke result as synthetic, never as real evidence.

### Conditions required to change the verdict

1. Obtain the official dataset legitimately and place `.psv` files under
   `data/raw/`.
2. Run `scripts/generate_p1_artifacts.py` without `--limit` for final data.
3. Verify split leakage checks on the real manifest.
4. Verify real validation/test metrics and threshold provenance.
5. Confirm feature target isolation on real columns.
6. Update this report with real, non-synthetic results.
7. Implement P2 before making P2 performance claims.

### Final verdict

**NO-GO** for empirical/clinical claims at this time.

Reason: the target and protocol are now correct in code, and all automated
tests pass, but real data, real artifacts, real split verification, and real
evaluation results are all `BLOCKED`/`MISSING`. Synthetic fixtures are not
acceptable evidence for a GO decision.
