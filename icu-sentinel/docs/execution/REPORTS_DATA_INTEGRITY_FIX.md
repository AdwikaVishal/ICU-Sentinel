# ICU-Sentinel — Reports Data Integrity Diagnosis and Fix

**Project:** ICU-Sentinel  
**Report path:** `icu-sentinel/docs/execution/REPORTS_DATA_INTEGRITY_FIX.md`  
**Branch:** `main`  
**Date:** 2026-11-10  
**Verdict:** PASS

---

## 1. Root Cause of Zero-Valued Metrics

The Reports page at `http://localhost:5173/reports` displayed zeros for all patients
(`total_hours: 0`, `recorded_hours: 0`, `missing_hours: 0`, `observation_count: 0`)
because of a defect in `DirectoryPatientRepository.list_patients()` in
`icu-sentinel/ml/p3_temporal/replay/demo_data.py`:

1. **`hour_min`/`hour_max` were never populated** in `list_patients()`. The method
   called `entry.get("hour_min")` and `entry.get("hour_max")`, but these fields
   were only set by `_get_hour_range()`, which was never invoked from
   `list_patients()`. With both `None`, the code took the branch
   `total_hours = 0; recorded_hours = 0`, ignoring the `n_records` value from
   the manifest entirely.

2. **`observation_count` was hardcoded to 0** with the comment
   "counted lazily per patient on demand" — but it was never actually computed
   on demand either.

3. **`n_records` from the split manifest was available** (each patient's parquet
   row count) but was only used as `recorded_hours` in the `else` branch that
   required `hour_max` to be non-None — which never happened.

As a result, every patient in the Reports data-quality table showed zeros
regardless of the actual data.

---

## 2. Files Modified

| File | Change |
|------|--------|
| `icu-sentinel/ml/p3_temporal/replay/demo_data.py` | Added `import json`; changed `__init__` to call `_load_metrics_cache()` instead of `_compute_patient_metrics()`; added `_load_metrics_cache()` method; rewrote `list_patients()` to read pre-computed metrics from the patient index |
| `compute_patient_metrics.py` | New utility script that reads all patient parquet files and writes a JSON cache of per-patient data-quality metrics |
| `.gitignore` | Added `/icu-sentinel/data/reports/patient_metrics_cache.json` (generated cache artifact) |

### Files NOT modified (frozen)
- All P1/P2 model artifacts (`artifacts/p2/matched_6000/*`)
- All P1/P2 evaluation results (`data/reports/p2_matched_6000/*`)
- All training targets and clinical definitions
- All P3 replay, prediction, and SHAP explanation code paths

---

## 3. Backend Fix Explanation

### Data-quality metric definitions

| Metric | Definition |
|--------|------------|
| **Total hours** | Number of hourly records in the patient's source time series (row count of the parquet file) |
| **Recorded hours** | Number of hourly records containing at least one valid clinical observation (ICULOS excluded) |
| **Missing hours** | Total hours minus recorded hours |
| **Observations** | Number of non-null clinical measurements across all supported clinical variables (ICULOS excluded per task requirements) |
| **Coverage** | Recorded hours / Total hours, expressed as a percentage |

### Implementation

The fix uses a **pre-computation + cache** approach to avoid loading the entire
40,336-patient dataset into memory at backend startup:

1. **`compute_patient_metrics.py`** — A standalone script that reads each
   patient's parquet file (only the `hour` and clinical columns, not the full
   row) and computes all four metrics. Uses `ProcessPoolExecutor` with 8 workers
   for parallelism. Writes results to
   `icu-sentinel/data/reports/patient_metrics_cache.json` (40,336 entries,
   ~4MB JSON).

2. **`_load_metrics_cache()`** — Called in `DirectoryPatientRepository.__init__()`.
   Loads the JSON cache and merges the per-patient metrics into `self._index`.
   If the cache is absent or corrupt, metrics remain at their defaults and
   individual patients can still be computed on demand.

3. **`list_patients()`** — Reads pre-computed metrics from `self._index[pid]`
   instead of computing them on the fly. No parquet reads occur during listing.

Backend startup time is dominated by reading the 4MB JSON cache (fast), not by
reading 40,336 parquet files.

---

## 4. Frontend Fix Explanation

No frontend code changes were required. The `ReportsPage.tsx` component already
read the correct fields from the API response (`total_hours`, `recorded_hours`,
`missing_hours`, `observation_count`, `data_source`, `is_synthetic`). The bug
was entirely in the backend returning zeros. Once the backend returns accurate
values, the frontend displays them correctly.

The Vite proxy configuration (`vite.config.ts`) already proxies `/api` and
`/health` to `http://127.0.0.1:8000`, so frontend requests reach the backend
without additional changes.

---

## 5. Verified Real-Patient Results

Three patients were independently verified against their source parquet files:

| Patient | Source | Total Hours | Recorded Hours | Missing Hours | Observations | Coverage |
|---------|--------|-------------|----------------|---------------|--------------|----------|
| p000001 | set_a | 54 | 54 | 0 | 456 | 100% |
| p000002 | set_a | 23 | 23 | 0 | 249 | 100% |
| p000003 | set_a | 48 | 48 | 0 | 579 | 100% |

### Independent source-versus-API comparison

For each patient, the source parquet was read independently and metrics were
computed using the same definitions. The API response matched the source
computation exactly for all three patients:

```
p000001: source=(54,54,0,456)  API=(54,54,0,456)  MATCH=True
p000002: source=(23,23,0,249)  API=(23,23,0,249)  MATCH=True
p000003: source=(48,48,0,579)  API=(48,48,0,579)  MATCH=True
```

All 40,336 patients in the cache were also verified internally: every patient
has `total_hours == recorded_hours` (no fully-empty rows in this dataset), and
observation counts vary based on missing clinical measurements.

---

## 6. Test Results

### Python backend tests

| Suite | Passed | Failed | Notes |
|-------|--------|--------|-------|
| `ml/p1_clinical/tests` | Included in total | — | |
| `ml/p2_boosting/tests` | Included in total | 5 | All LightGBM tests — pre-existing C-library crash (`OSError: access violation`), unrelated to this change; LightGBM is not part of the frozen P2_MATCHED_6000 results |
| `ml/p3_temporal/tests` | Included in total | — | |
| `tests/` | Included in total | — | |
| **Total** | **553** | **5** | |

### Frontend tests

| Suite | Passed | Failed | Notes |
|-------|--------|--------|-------|
| `npm run typecheck` | Passed | 0 | No TypeScript errors |
| `npm test` (vitest) | 17 | 1 | 1 pre-existing jsdom heading lookup issue (`App.test.tsx:112`); unrelated to this change |

No new test failures were introduced by this fix.

---

## 7. Replay, Prediction, and SHAP — Still Working

All existing P3 functionality continues to work with real data:

| Endpoint | Result |
|----------|--------|
| `GET /health` | 200 OK |
| `GET /api/v3/patients` | 200, 40,336 real patients |
| `GET /api/v3/patients/{id}` | 200, real patient summary |
| `GET /api/v3/patients/{id}/timeline` | 200, hourly observations |
| `GET /api/v3/replay/{id}?hour=N` | 200, 2 predictions (Extra Trees + XGBoost) |
| `GET /api/v3/replay/{id}/model-input?hour=N` | 200, 960 causal features |
| `GET /api/v3/explanations/{id}/{hour}` | 200, Tree SHAP, status `available` |
| `GET /api/v3/provider/status` | 200, `p2_boosting_model`, available |
| `GET /api/v3/provider/threshold` | 200, ET=0.04, XGB=0.69 |

Sample patient `p000001` at hour 1:
- Extra Trees: prob=0.0000, threshold=0.04, class=0
- XGBoost: prob=0.5502, threshold=0.69, class=0
- SHAP explanation: available, Tree SHAP method, contributions sum to model output (additivity validated)

---

## 8. Scientific Integrity

- **No clinical validity claims:** AUROC ~0.70–0.72 on frozen test set; well below clinical deployment thresholds. The project explicitly states "DO NOT DEPLOY."
- **No six-hour warning claims:** No held-out lead-time analysis performed.
- **No synthetic data presented as real:** All 40,336 patients are labeled `is_synthetic: false`, `data_source: processed_parquet`.
- **No train/test leakage:** Split integrity verified (0 overlapping patients, every patient in exactly one split).
- **No target leakage:** Target isolation valid; retrospective labels never used as model inputs.
- **No frozen metric modification:** P2_MATCHED_6000 thresholds (ET=0.04, XGB=0.69) and AUROC values unchanged.
- **ICULOS excluded from observations** per task requirements; ICULOS is not counted as a clinical observation.

---

## 9. Remaining Limitations

1. **Cache staleness:** If the underlying parquet files change, the cache becomes stale. The cache must be regenerated by running `compute_patient_metrics.py`.
2. **All patients have complete data in this dataset:** Every patient has `recorded_hours == total_hours` (no fully-empty hourly rows). The `missing_hours` metric is 0 for all patients, which is correct for this dataset but means the "missing hours" code path is not exercised by current data.
3. **Observation count excludes ICULOS** per task requirements; if ICULOS should be counted, the `OBS_COLUMNS` filter in `compute_patient_metrics.py` would need adjustment.
4. **Backend startup requires the cache file** to be present for accurate metrics. If the cache is missing, metrics default to zeros until regenerated.

---

## 10. Exact Commands to Run the Application

### Backend (real PhysioNet mode)

From `C:\Users\Saatvika Reddy\ICU-Sentinel\icu-sentinel`:

```powershell
$env:PYTHONPATH="."
$env:ICU_SENTINEL_MODE="real"
$env:ICU_SENTINEL_DATA_DIR="C:\Users\Saatvika Reddy\ICU-Sentinel\icu-sentinel\data\processed"
$env:ICU_SENTINEL_P2_MODEL_DIR="C:\Users\Saatvika Reddy\ICU-Sentinel\icu-sentinel\artifacts\p2\matched_6000"
python -m uvicorn backend.p3.app:app --host 127.0.0.1 --port 8000
```

**Backend URL:** `http://127.0.0.1:8000`  
**Health:** `http://127.0.0.1:8000/health` → 200

### Frontend

From `C:\Users\Saatvika Reddy\ICU-Sentinel\icu-sentinel\frontend`:

```powershell
npm install   # first time only
npm run dev
```

**Frontend URL:** `http://localhost:5173`  
**Reports page:** `http://localhost:5173/reports` → 200  
**API proxy:** `/api` and `/health` are proxied to `http://127.0.0.1:8000`

### Regenerating the metrics cache (if source data changes)

```powershell
python compute_patient_metrics.py
```

Run from the repository root. This reads all 40,336 patient parquet files and
writes `icu-sentinel/data/reports/patient_metrics_cache.json`.

---

## 10. Git Status

- **Commit:** `c3ef722` on `main`
- **Remote:** `origin/main` at `c3ef722` (pushed successfully)
- **Working tree:** Clean (except untracked backup file `demo_data.py.reports_backup`)
