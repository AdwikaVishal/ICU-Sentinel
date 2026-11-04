# P3 Demo Guide

This guide runs the P3 replay, explanation and alert stack locally with the
synthetic demonstration cohort. No PhysioNet data, P1 artifact or clinical
system is required.

---

## 1. Prerequisites

- Python 3.11+ (tested with 3.14.5), with the repository requirements.
- Node.js 20+ (tested with 26.7.0) and npm.
- No network access is needed at runtime; the first `npm install` and
  `pip install` do need package access.

```bash
cd /path/to/icu-sentinel
python3 -m pip install -r requirements.txt
```

---

## 2. Run the API

From the repository root:

```bash
PYTHONPATH=. python3 -m uvicorn backend.p3.app:app --host 127.0.0.1 --port 8000
```

Optional: isolate the alert database for a demo run.

```bash
ICU_SENTINEL_ALERT_DB=/tmp/p3_demo_alerts.db \
PYTHONPATH=. python3 -m uvicorn backend.p3.app:app --host 127.0.0.1 --port 8000
```

Check it:

```bash
curl -s http://127.0.0.1:8000/api/v3/provider/status | python3 -m json.tool
curl -s http://127.0.0.1:8000/api/v3/alerts/threshold | python3 -m json.tool
```

Expected provider source: `synthetic_demo_model` with `is_synthetic: true`.
If a P1 artifact is configured instead, the source is `p1_classical_model`.

Use the interactive docs at <http://127.0.0.1:8000/docs>.

---

## 3. Run the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:5173>. Vite proxies `/api` to
`http://127.0.0.1:8000`.

For a different API host:

```bash
VITE_P3_API_BASE=http://127.0.0.1:9000 npm run dev
```

Production build:

```bash
cd frontend
npm run build
npm run preview
```

The UI is a static SPA; serve `frontend/dist/` from any static host and point
`VITE_P3_API_BASE` at the API at build time.

---

## 4. Guided walkthrough

### Patient Replay

1. Pick `DEMO-0004` (24 recorded hours).
2. Move the hour slider or use **Previous/Next**. The score, observations and
   causality panel update.
3. Switch to `DEMO-0003` and move to hour 12 or 13. The page shows an explicit
   missing-hour state; no observation is shown as normal and no alert is
   evaluated.
4. Enable **Evaluate the configured alert rule when navigating**, then click
   **Evaluate this hour** at hour 20. The prediction panel shows the resulting
   episode status.
5. Check that the causality panel reports `Future data used: No`,
   `Retrospective label as input: No`, and an input hash.

### Explanation Centre

1. Pick a patient and an hour with observations.
2. Read the method and output space. The demo model uses `linear_shap` in
   `decision_function_log_odds`.
3. Inspect the contribution chart. Positive values push the model output up;
   negative values push it down.
4. Confirm the additivity check badge.
5. Try an hour that has no observations, or use a heuristic provider, to see an
   explicit `unavailable` / `unsupported` state with no fabricated values.
6. Inspect global importance and the model catalogue.

### Alert Centre

1. Review the configured threshold and the “Not clinically validated” badge.
2. Look at active episodes created from replay.
3. Acknowledge an episode; it leaves the active list and appears in history.
4. Clear it; it remains in history with a reason.
5. Navigating backwards in replay never creates a duplicate episode.

---

## 5. Useful API calls

```bash
# Patients and timeline
curl -s http://127.0.0.1:8000/api/v3/patients | python3 -m json.tool
curl -s http://127.0.0.1:8000/api/v3/patients/DEMO-0004/timeline | python3 -m json.tool

# Replay and causal input
curl -s "http://127.0.0.1:8000/api/v3/replay/DEMO-0004?hour=20" | python3 -m json.tool
curl -s "http://127.0.0.1:8000/api/v3/replay/DEMO-0004/model-input?hour=20" | python3 -m json.tool

# Explanation
curl -s http://127.0.0.1:8000/api/v3/explanations/DEMO-0004/20 | python3 -m json.tool
curl -s http://127.0.0.1:8000/api/v3/explanations/global | python3 -m json.tool

# Alerts
curl -s http://127.0.0.1:8000/api/v3/alerts/threshold | python3 -m json.tool
curl -s http://127.0.0.1:8000/api/v3/alerts/active | python3 -m json.tool

# Evaluate a high probability directly
curl -s -X POST http://127.0.0.1:8000/api/v3/alerts/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "patient_id": "DEMO-0004",
    "hour": 20,
    "probability": 0.99,
    "model_name": "demo_logistic_regression",
    "model_version": "demo-1.0.0",
    "prediction_source": "synthetic_demo_model",
    "is_synthetic": true
  }' | python3 -m json.tool
```

---

## 6. Using a real processed-data directory

```bash
ICU_SENTINEL_DATA_DIR=/path/to/processed/patients \
ICU_SENTINEL_ALERT_DB=/tmp/p3_alerts.db \
PYTHONPATH=. python3 -m uvicorn backend.p3.app:app --port 8000
```

Without a P1 artifact this fails by design: the demo model is only for the
synthetic demo cohort. Supply a P1 artifact:

```bash
ICU_SENTINEL_DATA_DIR=/path/to/processed/patients \
ICU_SENTINEL_P1_MODEL_DIR=/path/to/artifacts/p1 \
ICU_SENTINEL_P1_MODEL_NAME=logistic_regression \
ICU_SENTINEL_P1_MODEL_VERSION=1.0.0 \
PYTHONPATH=. python3 -m uvicorn backend.p3.app:app --port 8000
```

The P1 loader expects
`<dir>/<name>/<name>_<version>.joblib` and
`<dir>/<name>/<name>_<version>_features.json`. An optional background file in
that model directory is required for linear/kernel SHAP; without it,
explanations are explicitly `unavailable`.

---

## 7. Tests

Python (from the repository root):

```bash
python3 -m pytest -q -o addopts=''
```

P3 only:

```bash
python3 -m pytest ml/p3_temporal/tests tests/test_p3_api.py -q -o addopts=''
```

Frontend (from `frontend/`):

```bash
npm test
npm run typecheck
npm run build
```

Two pre-existing P1 failures in `ml/p1_clinical/tests/test_bootstrap.py` are
unrelated to P3 (`_safe_roc_auc` / `_safe_auprc` argument count). See
`P3_IMPLEMENTATION_REPORT.md`.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Could not reach the P3 API` | Backend not running, wrong port, or CORS origin not allowed. Start uvicorn on `127.0.0.1:8000` or set `VITE_P3_API_BASE`. |
| `No patients are available` | Synthetic repository disabled and no `ICU_SENTINEL_DATA_DIR`; or the directory has no processed patients. |
| Startup error about a missing P1 artifact | A real data directory is configured without `ICU_SENTINEL_P1_MODEL_DIR`/`NAME`. This is intentional. |
| Explanation `unavailable` for a linear model | No background dataset was found for SHAP. Add one or accept the explicit unavailable state. |
| Explanation `unsupported` | Model family has no exact explainer and no background for Kernel SHAP, or preprocessing changes the feature count. |
| Alerts keep updating one episode | This is deduplication inside the configured window. Increase the hour gap or inspect the active episode. |
| Alert DB locked | Use a single backend process; SQLite WAL supports the demo but is not a multi-worker production database. |

---

## 9. Shutdown and cleanup

Stop uvicorn and Vite with `Ctrl+C`. Delete a temporary alert database if used:

```bash
rm -f /tmp/p3_demo_alerts.db
```
