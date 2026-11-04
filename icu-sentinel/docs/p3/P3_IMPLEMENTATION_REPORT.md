# P3 Implementation Report

**Project:** ICU Sentinel  
**Owner:** Person 3 (P3)  
**Date:** 2026-11-04  
**Status:** Implemented, tested and documented

---

## 1. Executive summary

P3 adds three connected research modules on top of the existing P1 codebase:

1. **Patient Replay Engine** with prefix-only causal feature construction.
2. **Explainable AI** with genuine SHAP explanations and explicit
   unavailable/unsupported states.
3. **Alert Engine** with configurable thresholds, SQLite persistence,
   deduplication and an ACTIVE/ACKNOWLEDGED/CLEARED lifecycle.

A React/TypeScript application provides Patient Replay, Explanation Centre and
Alert Centre pages with typed clients, hooks, explicit loading/error/empty
states and non-alarmist wording.

No P1 source was modified. P3 code is additive; the only shared-file changes
are additive test configuration and dependency declarations.

---

## 2. Status by deliverable

| Deliverable | Status | Evidence |
|---|---|---|
| Repository inspection and baseline | Complete | Baseline 363 passed, 2 pre-existing failures |
| Shared prediction contracts and provider abstraction | Complete | `ml/p3_temporal/common/` |
| Synthetic demo cohort and patient repository | Complete | `replay/demo_data.py`, 6 demo patients |
| Timeline with missing-hour handling | Complete | `replay/timeline.py`, explicit unavailable states |
| Prefix-only causal feature builder | Complete | `replay/model_input.py` |
| Replay navigation and prediction API | Complete | `replay/replay_engine.py`, 200s via TestClient |
| Leakage tests (future data/labels/rows) | Complete | `tests/test_causality.py`, 5 tests |
| Genuine SHAP patient explanations | Complete | `explainability/`, additivity error ≈ 4.4e-16 |
| Per-model explainer factory | Complete | Linear/tree/kernel plans + unsupported reasons |
| Global importance and feature groups | Complete | `global_explainer.py` |
| Model catalogue and feature metadata | Complete | API catalogue + P1 units |
| Alert threshold configuration | Complete | `alert_rules.py`, env overrides, not validated |
| Alert deduplication | Complete | Update/out-of-order/gap rules, 6 h default |
| Alert lifecycle and persistence | Complete | `AlertStore` SQLite, ACTIVE/ACKNOWLEDGED/CLEARED |
| FastAPI integration | Complete | `backend/p3/`, error handlers, CORS, lifespan |
| API tests | Complete | `tests/test_p3_api.py`, 15 tests |
| React/TypeScript frontend | Complete | `frontend/`, 3 pages |
| Typed API clients and hooks | Complete | `frontend/src/api/`, `frontend/src/hooks/` |
| Loading/error/empty states | Complete | Shared state panels, page-level handling |
| Frontend tests, typecheck, build | Complete | 15 tests, `tsc --noEmit`, Vite build |
| Documentation | Complete | `docs/p3/` (this report + 5 documents) |
| P1 preserved | Complete | No P1 source edits |

---

## 3. Test results

### Python

Command:

```bash
python3 -m pytest -q -o addopts=''
```

Result:

```text
2 failed, 444 passed, 219733 warnings in 23.73s
```

The two failures are the pre-existing P1 baseline failures:

```text
ml/p1_clinical/tests/test_bootstrap.py::test_validate_bootstrap_result_invalid
ml/p1_clinical/tests/test_bootstrap.py::test_bootstrap_reproducibility
```

Both fail because `_safe_roc_auc` / `_safe_auprc` are called with an argument
the current implementations do not accept. They are unrelated to P3 and were
present before this work.

Baseline reconciliation:

```text
363 (P1 baseline passing) + 66 (P3 module tests) + 15 (P3 API tests) = 444 passed
```

P3-only command:

```bash
python3 -m pytest ml/p3_temporal/tests tests/test_p3_api.py -q -o addopts=''
```

### Frontend

From `frontend/`:

```bash
npm test
```

```text
Test Files  3 passed (3)
Tests       15 passed (15)
```

```bash
npm run typecheck
npm run build
```

Both pass. The production build emits a chunk-size advisory (678 kB
JavaScript, 203 kB gzip) due to Recharts; this is a performance note, not a
build failure.

### Verification scope

- The FastAPI app was exercised through `TestClient` (all 22 route/error
  cases used during smoke testing returned the expected status).
- The live backend and the Vite `/api` proxy were checked with `curl` against
  a running uvicorn process.
- The React app was verified with jsdom tests, `tsc` and a production build.
  A live desktop-browser walkthrough was not available in this environment.

---

## 4. Requirements traceability

| Requirement | Implementation |
|---|---|
| Prediction at `t` uses only data `<= t` | Prefix `ProcessedPatient` built before P1 feature pipeline; leakage tests |
| Automated leakage test at hour 24 | `test_every_future_mutation_leaves_hour_24_unchanged` and related |
| Label never a model input | `NON_FEATURE_COLUMNS`, separate label field, hash excludes label |
| Label never alert trigger/threshold input | `AlertEvaluationRequest` has no label field; threshold config |
| Synthetic data/predictions clearly labelled | `is_synthetic`, `PredictionSource`, disclaimers, UI badges |
| No fake SHAP | Factory plans/statuses, heuristic returns `unsupported` |
| Unsupported model explicit | `unsupported` / `unavailable` with reason; never placeholder values |
| Threshold configurable, not clinically validated | Env/config, `clinically_validated: false`, disclosure |
| Dedup and clearing documented | `P3_ALERTS.md`, UI notes, code docstrings |
| Alert lifecycle + persistence | SQLite store, ACTIVE/ACKNOWLEDGED/CLEARED, history |
| No duplicate active alerts | One episode per patient/rule, engine dedup |
| Replaceable prediction provider | `PredictionProvider` Protocol, `ModelPredictionProvider` |
| Explainer factory per model type | `ExplainerFactory` linear/tree/kernel plans |
| P1 untouched | No P1 source edits |
| React + TS + 3 pages | `frontend/src/pages/` |
| Typed API clients and hooks | `api/client.ts`, `api/types.ts`, `hooks/` |
| Loading/error/empty states | Shared panels and per-page handling |
| No medical alarmism | Calm labels, “model score”, persistent research disclaimer |

---

## 5. Integration defect found and fixed

During backend verification, `GET /api/v3/replay/{patient_id}` failed with:

```text
TypeError: ReplayEngine.start() got an unexpected keyword argument 'evaluate_alerts'
```

The route correctly forwarded `evaluate_alerts`, but `ReplayEngine.start()`
did not accept it. The engine method was updated to accept the keyword and
pass it through. This was a P3 integration defect, not a P1 defect.
`tests/test_p3_api.py::test_replay_start_uses_first_hour_without_alerts`
guards it.

---

## 6. Changes outside P3

| File | Change | Reason |
|---|---|---|
| `pyproject.toml` | Added `ml/p3_temporal/tests` to `testpaths` | Run P3 tests in the normal suite. |
| `requirements.txt` | Added `shap>=0.44.0`, `httpx>=0.24.0` | SHAP runtime; FastAPI `TestClient`. |
| `.gitignore` | Ignored `frontend/node_modules/`, `frontend/dist/`, coverage and local env | Do not commit build artifacts. |
| `README.md` | Added P3 quickstart/status section | Discoverability. |

No P1 Python source, schema or API was changed. The two pre-existing P1 test
failures were not “fixed” because that would be unrelated scope.

---

## 7. Key design decisions

1. **Replay builds features from a prefix, not a sliced full-stay matrix.**
   This is the only way to keep rolling/trend/forward-fill features causal.
2. **A real fitted demo model, not a fake SHAP mock.** When no artifact is
   configured, a logistic regression is fitted on synthetic demo patients.
   Explainability is genuine for that model and explicitly labelled synthetic.
3. **Explicit availability everywhere.** No prediction is represented by 0;
   missing hours, unsupported models and missing backgrounds have reasons.
4. **Input-hash binding.** Explanations must reference the same causal input
   as the prediction, or they are withheld.
5. **Output-space honesty.** Linear SHAP is reported in log-odds; tree SHAP in
   probability-like space. The UI displays the space.
6. **Alert evaluation is opt-in in replay.** Opening or navigating the page
   does not create alerts unless the caller enables evaluation.
7. **SQLite rather than an in-memory list.** Alert history survives restarts
   and tests use isolated temporary databases.

---

## 8. Known limitations

- Demo data and the demo model are synthetic; no clinical performance is
  claimed or implied.
- No P1/P2/P3 research artifact is bundled. A real deployment must configure
  one; a real data directory without an artifact fails fast.
- Linear/kernel SHAP require a background dataset. Without one, explanations
  are `unavailable`.
- Kernel SHAP is sampled and approximate; failed additivity is surfaced.
- SHAP describes model behaviour, not causality or treatment effect.
- Alert deduplication is process-level. Multi-process deployments sharing one
  SQLite file need a database-level guard.
- SQLite is appropriate for a demo/research prototype, not a high-write
  clinical deployment.
- The frontend bundle is not code-split; Recharts dominates the chunk.
- The threshold is a demonstration configuration and must not be presented as
  clinically validated.

---

## 9. Exact command reference

All Python commands are from the repository root; frontend commands from
`frontend/`.

```bash
# Full suite
python3 -m pytest -q -o addopts=''

# P3 module + API tests
python3 -m pytest ml/p3_temporal/tests tests/test_p3_api.py -q -o addopts=''

# Leakage tests only
python3 -m pytest ml/p3_temporal/tests/test_causality.py -q -o addopts=''

# Explanation tests only
python3 -m pytest ml/p3_temporal/tests/test_explainability.py -q -o addopts=''

# Alert tests only
python3 -m pytest ml/p3_temporal/tests/test_alerts.py \
  ml/p3_temporal/tests/test_integration.py tests/test_p3_api.py -q -o addopts=''

# Backend
PYTHONPATH=. python3 -m uvicorn backend.p3.app:app --host 127.0.0.1 --port 8000

# Frontend
cd frontend
npm install
npm test
npm run typecheck
npm run build
npm run dev
```

---

## 10. Documentation index

- `P3_ARCHITECTURE.md` — modules, dependency direction, provider/explainer
  abstractions, configuration.
- `P3_CAUSALITY_AND_LEAKAGE.md` — prefix construction, label exclusion,
  leakage test matrix.
- `P3_EXPLAINABILITY.md` — SHAP methods, output spaces, validation, limits.
- `P3_ALERTS.md` — threshold, lifecycle, deduplication, clearing, persistence.
- `P3_API_REFERENCE.md` — endpoints, payloads, statuses, errors.
- `P3_DEMO_GUIDE.md` — local run, guided walkthrough, troubleshooting.
