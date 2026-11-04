# P3 Architecture

**Project:** ICU Sentinel  
**Owner:** Person 3 (P3) — Temporal models, replay, explanations, alerts, UI  
**Status:** Implemented research prototype  
**Date:** 2026-11-04

---

## 1. Purpose

P3 provides three connected capabilities on top of the P1 processed-patient
contract:

1. **Patient Replay Engine** — reconstruct a patient-hour prediction using
   only data recorded at or before that hour.
2. **Explainable AI (SHAP)** — explain a fitted model's output with genuine
   SHAP contributions, or return an explicit `unavailable` / `unsupported`
   state.
3. **Alert Engine** — evaluate a configurable threshold against predictions,
   persist alert episodes, deduplicate them and manage their lifecycle.

A React/TypeScript single-page application exposes all three as **Patient
Replay**, **Explanation Centre** and **Alert Centre**.

Nothing in P3 is a medical device, a diagnosis, or a clinically validated
early-warning system. Synthetic demonstration data and predictions are
labelled as such in every API response and every page.

---

## 2. Repository layout

```text
ml/p3_temporal/
  common/
    prediction.py          # PredictionResult, PredictionContext, hashing, provider Protocol
    model_bundle.py        # ModelBundle, ModelRegistry, P1 artifact loader
    clinical_catalog.py    # display metadata for P1 observation columns
  replay/
    demo_data.py           # synthetic demo cohort + directory repository
    timeline.py            # recorded/missing hour index
    model_input.py         # CausalFeatureBuilder: prefix-only feature construction
    prediction_provider.py # model-backed PredictionProvider
    replay_engine.py       # navigation + prediction + alert integration
    replay_schemas.py      # replay API schemas
  explainability/
    demo_model.py          # fitted synthetic demonstration logistic regression
    explainer_factory.py   # per-model SHAP planning/building/validation
    patient_explainer.py   # per-patient and service-level explanations
    global_explainer.py    # mean |SHAP| over a background sample
    feature_mapping.py     # P1 feature name -> readable name/unit/category
  alerts/
    alert_rules.py         # threshold config + pure trigger/episode rules
    alert_engine.py        # evaluate/update/clear lifecycle
    alert_store.py         # SQLite persistence
    alert_schemas.py       # alert API schemas

backend/p3/
  container.py             # P3Settings + dependency container
  dependencies.py          # request-scoped container access
  routes/replay.py         # /api/v3 patients + replay
  routes/explanations.py   # /api/v3/explanations
  routes/alerts.py         # /api/v3/alerts
  schemas.py               # backend-only response helpers
  app.py                   # FastAPI app, CORS, error handlers, health

frontend/
  src/api/                 # typed client + schema mirrors
  src/hooks/               # resource + replay/explanation/alert controllers
  src/components/          # shared UI, charts, tables, state panels
  src/pages/               # Patient Replay, Explanation Centre, Alert Centre
```

P3 imports P1 (`ml.p1_clinical`) but P1 does not import P3. The only P1 change
made for P3 is an additive `testpaths` entry in `pyproject.toml`.

---

## 3. Dependency direction

```text
React UI
   │  typed fetch client
   ▼
FastAPI routes  ──►  P3Container
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   ReplayEngine   ExplanationService  AlertEngine
        │               │                │
        ▼               ▼                ▼
 PredictionProvider  ModelRegistry   AlertStore (SQLite)
        │               │
        └──────► ModelBundle ◄──────────┘
                        │
                        ▼
              P1 FeaturePipeline (prefix-only)
```

Routes depend on the container and the P3 services, never on concrete model
artifacts. The prediction provider, model bundles and alert store can be
replaced without changing route code.

---

## 4. Prediction provider abstraction

`ml/p3_temporal/common/prediction.py` defines:

- `PredictionContext` — the exact causal input (feature values, observations,
  input hash, data provenance, retrospective label kept separate).
- `PredictionResult` — probability, availability, model identity, source,
  input hash and disclaimer metadata.
- `PredictionProvider` — a runtime-checkable Protocol with `predict()` and
  `status()`.

`PredictionSource` is deliberately explicit:

| Source | Meaning |
|---|---|
| `synthetic_heuristic` | Hand-written demonstration score; never explained with SHAP. |
| `synthetic_demo_model` | A real fitted model trained on synthetic demo patients. |
| `p1_classical_model` | A loaded P1 research artifact. |
| `p2_boosting_model` | Reserved for P2 artifacts. |
| `p3_temporal_model` | Reserved for a P3 sequence model. |
| `unavailable` | No prediction is available; a reason is always supplied. |

`ModelPredictionProvider` wraps a `ModelBundle`. P1 artifacts are loaded by
`load_p1_model_bundle()`; if no artifact is configured, the backend builds a
logistic-regression demo model **only** for the synthetic demo repository.
For a real processed-data repository, startup fails instead of silently
serving a synthetic model.

---

## 5. Explanation abstraction

`ExplainerFactory` plans an explainer from the model bundle:

| Model family | Method | Output space | Background required |
|---|---|---|---|
| Linear classifier with `decision_function` | `linear_shap` | log-odds (`decision_function_log_odds`) | Yes |
| scikit-learn tree ensemble | `tree_shap` | probability-like tree output | No |
| XGBoost / LightGBM / CatBoost booster | `tree_shap` | probability when the library supports it, else raw | No |
| Unknown family | `kernel_shap` | probability | Yes |

Planning is separate from building. A model with no usable background data
returns `unavailable`; a model family that cannot be explained returns
`unsupported`. Both include a reason. SHAP values are never substituted with
zeros, random values or feature importances.

The explainer validates shape and finiteness, and `PatientExplainer` verifies
the prediction input hash matches the explanation input hash before exposing
contributions. `validate_additivity()` checks
`base_value + sum(shap_values) ≈ model_output` in the declared output space.

---

## 6. Alert integration

Replay calls `AlertEngine.evaluate_prediction()` only when the caller asks for
alert evaluation. The engine receives a `PredictionResult`; it has no access
to the retrospective label. The alert store is the only mutable state and is
persisted in SQLite.

The same engine is exposed directly at `/api/v3/alerts/evaluate`, so the
Alert Centre and replay share one rule and one persistence layer.

---

## 7. Frontend

The React app uses:

- typed service modules (`src/api/client.ts`) with a shared `ApiError`;
- resource hooks with `AbortController` so stale patient/hour requests cannot
  commit;
- explicit loading, error and empty states on every page;
- Recharts for SHAP contribution and global-importance charts, plus an
  accessible values table;
- calm wording and persistent provenance/disclaimer banners. No diagnosis,
  no “sepsis detected”, no red alarm styling.

Vite proxies `/api` and `/health` to `http://127.0.0.1:8000` in development.
Set `VITE_P3_API_BASE` for a different deployment.

---

## 8. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ICU_SENTINEL_DATA_DIR` | unset | Processed-patient directory; when unset the synthetic demo repository is used. |
| `ICU_SENTINEL_ALERT_DB` | `<repo>/data/p3_alerts.db` | SQLite alert database path. |
| `ICU_SENTINEL_P1_MODEL_DIR` | unset | Directory containing P1 model artifacts. |
| `ICU_SENTINEL_P1_MODEL_NAME` | unset | P1 model name (defaults to `logistic_regression` when a dir is set). |
| `ICU_SENTINEL_P1_MODEL_VERSION` | `1.0.0` | P1 artifact version. |
| `ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT` | `true` | Enable the synthetic demo model only for the synthetic repository. |
| `ICU_SENTINEL_ALERT_THRESHOLD` | `0.60` | Configurable demonstration threshold. |
| `ICU_SENTINEL_ALERT_DEDUP_WINDOW_HOURS` | `6` | Episode merge window. |
| `ICU_SENTINEL_ALERT_CLEAR_ON_BELOW` | `true` | Clear an active episode when the score falls below threshold. |
| `ICU_SENTINEL_ALERT_THRESHOLD_VALIDATED` | `false` | Must only be set after a genuine clinical validation process. |
| `ICU_SENTINEL_CORS_ORIGINS` | localhost dev origins | Comma-separated allowed origins. |
| `VITE_P3_API_BASE` | same origin | Frontend API base URL. |

---

## 9. Related documents

- `P3_CAUSALITY_AND_LEAKAGE.md`
- `P3_EXPLAINABILITY.md`
- `P3_ALERTS.md`
- `P3_API_REFERENCE.md`
- `P3_DEMO_GUIDE.md`
- `P3_IMPLEMENTATION_REPORT.md`
