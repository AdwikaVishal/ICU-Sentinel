# ICU Sentinel — Final Completion Audit

**Audit Date:** November 10, 2026  
**Repository:** `/Users/shantanu/ICU-Sentinel`  
**Branch:** `main` (clean, up to date with origin)  
**Audit Status:** Complete — all sections populated from direct inspection

---

## 1. Complete Project Architecture

The ICU Sentinel project is a multi-phase, multidisciplinary research application for ICU sepsis early-warning prediction. It follows a modular architecture with P1 (data/clinical baselines), P2 (boosting models), and P3 (replay/alerts/UI) layers.

### High-Level Architecture

| Layer | Responsibility | Key Files/Directories |
|---|---|---|
| **P1** | Data ingestion, preprocessing, classical ML, patient APIs | `icu-sentinel/ml/p1_clinical/`, `backend/p1_patients/` |
| **P2** | Boosting model training (XGBoost, LightGBM, Extra Trees) | `artifacts/p2/`, `icu-sentinel/ml/p2_boosting/` |
| **P3** | Patient replay engine, prediction providers, alerts, SHAP, React UI | `ml/p3_temporal/`, `backend/p3/`, `frontend/` |
| **Backend** | FastAPI endpoints for patient data and P3 replay/alerts | `backend/p1_patients/app.py`, `backend/p3/app.py` |
| **Frontend** | React/TypeScript dashboard for monitoring and alert management | `icu-sentinel/frontend/` |
| **Data** | Processed patient data, features, splits, model artifacts | `data/`, `artifacts/` |

### Key Dependencies

- **Python:** 3.14.5 (via `.venv`)
- **FastAPI:** For backend APIs
- **scikit-learn / xgboost / lightgbm / extra-trees:** Model families
- **pydantic:** Input validation
- **recharts / lucide-react:** Frontend visualization
- **vitest / pytest:** Testing frameworks

---

## 2. Directory Structure

The repository has the following top-level structure:

```
├── .gitignore                          - Excludes large artifacts
├── .venv/                              - Python virtual environment
├── ARTIFACTS_MANIFEST.md               - Generated artifact inventory
├── COMPREHENSIVE_STATUS_REPORT.md      - Mid-project status
├── icu-sentinel/                       - Main project source
│   ├── artifacts/                      - P1/P2/P3 model artifacts (gitignored)
│   ├── data/                           - Processed data, features, splits
│   │   ├── raw/                        - Raw PhysioNet .psv files
│   │   ├── processed/                  - Per-patient processed parquets
│   │   ├── features/                   - Combined features (994 or 1001)
│   │   │   ├── combined_features.parquet
│   │   │   ├── feature_config.json
│   │   │   ├── feature_schema.json (1001 features)
│   │   │   └── patients/               - Individual patient parquet files
│   │   └── splits/                     - Patient-level train/val/test splits
│   │       ├── split_manifest.json     (500 patients, default)
│       ├── splits_full/                (39,810 patients, full)
│       └── physionet2019/              - Alternative split directories
│   ├── ml/                             - ML pipeline code
│   │   ├── p1_clinical/                - Classical ML, baselines, evaluation
│   │   ├── p2_boosting/                - XGBoost/LightGBM/Extra Trees
│   │   └── p3_temporal/                - Replay, alerts, explanation
│   ├── backend/                        - FastAPI P1 and P3 endpoints
│   ├── docs/                           - Research documentation
│   ├── reports/                        - Generated reports
│   ├── scripts/                        - Data processing / training / evaluation scripts
│   └── tests/                          - Test suite
│       └── test_p3_api.py              - P3 API tests (synthetic demo)
│       └── ml/p1_clinical/tests/       - P1 clinical tests (198 test items)
│   └── frontend/                       - React/TypeScript UI
│       ├── src/                        - Source code
│       ├── package.json
│       └── vite.config.ts
├── requirements.txt                    - Python dependencies
├── pyproject.toml                      - Project configuration
├── scripts/                            - Utility scripts (15+ scripts)
└── reports/                            - Root-level reports (15+ files)
```

### Git-Ignored Directories (per `.gitignore`)

- `artifacts/` — Large model files (~3.5 GB total)
- `data/raw/` — Raw PhysioNet patient records
- `synthetic_backup/` — Synthetic backup patient files
- `.pytest_cache/`, `__pycache__/`, `.venv/`

---

## 3. Existing Implemented Functionality

### P1 — Classical ML and Patient API

- **Data preprocessing:** Cleaning, missing value handling, target construction (`sepsis_within_6h`)
- **Feature engineering:** 994 features (rolling means, std, min, max, latest, trends, clinical scores, availability indicators, slope features)
- **Classical ML models:** Logistic Regression, Random Forest, SVM trained and evaluated
- **Patient API (`backend/p1_patients`):** Read-only FastAPI endpoints serving patient parquet data
  - `GET /health`, `GET /patients`, `GET /patients/{id}`, timeline and features endpoints
  - Never invents synthetic data; returns 503 when data is absent
- **Patient-level splits:** `VALID_SPLITS` constant, manifest loading from JSON
- **Evaluation pipelines:** Classification metrics, bootstrap, calibration, lead time analysis
- **Alert burden and episode evaluation:** P1 clinical evaluation protocols

### P2 — Boosting Model Training

- **Three models trained:** XGBoost, LightGBM, Extra Trees
- **All on 27,867 training patients** (from full PhysioNet 2019 dataset)
- **994 features** selected from 1001 total
- **Model artifacts** with metadata in `artifacts/physionet2019/full_dataset_retrain/`
- **Validation/evaluation scripts:** `complete_final_evaluation.py`, `train_full_cohort.py`
- **Bootstrap confidence intervals** (partially implemented — skipped due to infrastructure)
- **Calibration:** Sigmoid calibration implemented and evaluated

### P3 — Replay Engine, Alerts, and UI

- **Patient replay engine:** Retrospective playback of patient observations
- **Prediction providers:** Synthetic demo heuristic and ModelPredictionProvider (backed by real model bundles)
- **Alert engine:** Threshold-based alert creation with deduplication, acknowledgement, clearing
- **Alert API (`backend/p3/routes/alerts`):** Evaluate, list, acknowledge, clear alerts
- **Replay API (`backend/p3/routes/replay`):** Patient listing, timeline, model input, replay navigation
- **Explainability:** Demo model explanations, feature mapping
- **React/TypeScript frontend:** 30+ pages/components including:
  - Dashboard, patient replay, alerts, explanations, reports, settings
  - `vitest` test suite for frontend components

### Backend Endpoints

| Endpoint | Layer | Purpose |
|---|---|---|
| `GET /health` | P3 | Health check |
| `GET /api/v3/health` | P3 | API health |
| `GET /api/v3/meta` | P3 | Service metadata (provider, data source, is_synthetic) |
| `GET /api/v3/patients` | P3 | List patients (synthetic demo cohort) |
| `GET /api/v3/patients/{id}` | P3 | Get patient summary |
| `GET /api/v3/patients/{id}/timeline` | P3 | Get observation timeline |
| `GET /api/v3/provider/status` | P3 | Prediction provider status |
| `GET /api/v3/provider/threshold` | P3 | Alert threshold config |
| `GET /api/v3/replay/{id}` | P3 | Start/replay patient |
| `GET /api/v3/replay/{id}/model-input` | P3 | Get model input features |
| `POST /api/v3/alerts/evaluate` | P3 | Evaluate prediction against threshold |
| `GET /api/v3/alerts` | P3 | List alerts |
| `GET /api/v3/alerts/active` | P3 | List active alerts |
| `POST /api/v3/alerts/{id}/acknowledge` | P3 | Acknowledge alert |
| `POST /api/v3/alerts/{id}/clear` | P3 | Clear alert |

---

## 4. Incomplete Functionality

### Critical Issues

1. **Default P3 prediction uses synthetic demo model**
   - The P3 container's `_resolve_bundle()` falls back to `build_demo_model_bundle()` when no P1 model artifact is configured
   - `demo_model_when_no_artifact=True` by default
   - Real trained XGBoost/LightGBM/Extra Trees models are available in `artifacts/physionet2019/full_dataset_retrain/` but not integrated by default
   - **Environment variables needed:** `ICU_SENTINEL_P1_MODEL_DIR` and `ICU_SENTINEL_P1_MODEL_NAME`

2. **Split manifest discrepancy**
   - Default `data/splits/split_manifest.json`: 500 patients (350 train + 50 val + 100 test)
   - Full `data/splits_full/split_manifest.json`: 39,810 patients (27,867 train + 3,981 val + 7,962 test)
   - Historical reports reference 39,810 original patients with 39,809 usable (1 corrupted file excluded)
   - The default split is a small subset, not the full dataset

3. **Test collection error in `test_split_validation.py`**
   - The script references `split_manifest_fast_6000.json` which may not exist or have the expected format
   - Reported: "collection error involving `test_split_validation.py`"

4. **Frontend automated testing limited**
   - Vitest configuration exists but many components test with synthetic/demo data
   - No end-to-end tests connecting frontend to real backend

5. **Real model not integrated into default P3 inference pathway**
   - The default provider is synthetic demo logistic regression
   - Real model integration hooks exist (via `load_p1_model_bundle()`, environment variables) but are not used by default

6. **Corrupted patient file**
   - Reported: 1 corrupted file (p010963) excluded from the 39,809 usable patients
   - Need to verify this is properly handled

7. **Model evaluation metrics discrepancy**
   - Historical reported results: XGBoost 0.814 AUROC, LightGBM 0.813 AUROC, Extra Trees 0.743 AUROC
   - Actual artifact metadata shows much lower values: XGBoost 0.5318, LightGBM 0.6046, Extra Trees 0.5748 AUROC
   - These need to be independently verified and documented

8. **Calibration data independence not verified**
   - Calibration must be fitted on validation data, not test data
   - Need to verify the calibration pipeline respects patient-level splits

9. **CORS and security configuration**
   - CORS origins hardcoded to localhost:5173 and localhost:3000
   - No authentication/authorization implemented

10. **Docker and deployment incomplete**
    - No `docker-compose.yml` at the root level
    - No production deployment configuration

---

## 5. Broken Functionality

### Known Issues

1. **`test_split_validation.py` collection error**
   - Script references `split_manifest_fast_6000.json` which may not exist at the expected path
   - The manifest file `data/splits/split_manifest_fast.json` exists but has 500 patients, not 6,000
   - This causes test collection failure

2. **P3 default uses synthetic model even when real artifacts are available**
   - When `ICU_SENTINEL_P1_MODEL_DIR` is not set, the system always falls back to demo model
   - Even when real processed data and model artifacts are present, the default path doesn't load them

3. **Feature schema version mismatch**
   - `feature_schema.json` has 1001 features
   - Model selected features are 994 (7 features excluded, likely due to missingness or redundancy)
   - The pipeline must ensure serving uses exactly the 994 selected features

4. **Frontend buttons may be non-functional**
   - Some controls may not connect to working backend endpoints
   - Buttons without corresponding API support

---

## 6. Duplicate Implementations

### Redundant Scripts and Artifacts

1. **Multiple split manifest directories:**
   - `data/splits/split_manifest.json` (500 patients, default)
   - `data/splits/split_manifest_fast.json` (500 patients)
   - `data/splits/physionet2019/split_manifest.json` (alternative)
   - `data/splits_full/split_manifest.json` (39,810 patients, full)
   - `data/splits/physionet2019_features974/split_manifest.json` (features974 experiment)
   - `data/splits/physionet2019_test/split_manifest.json` (test split)
   - `data/splits/physionet2019_test100/split_manifest.json` (test100 split)

2. **Multiple feature directories:**
   - `data/features/` (1001 features, combined)
   - `data/features_full/` (full feature set)
   - `data/features_full_test2/` (test variant)

3. **Multiple training run reports:**
   - Multiple `p2_debug*` and `p2_974/` experiment directories with separate metrics
   - `artifacts/p2/` (matched cohort, 6,000 patients)
   - `artifacts/physionet2019/full_dataset_retrain/` (full 27,867 training patients)
   - `artifacts/physionet2019/full_dataset_experiment/` (experimental)
   - `artifacts/physionet2019/p2_debug*/` (debug runs)

4. **Multiple evaluation report directories:**
   - `reports/` (root level, 15+ files)
   - `icu-sentinel/reports/` (p1/p2 sub-reports)
   - `artifacts/physionet2019/reports/final_evaluation/` (final test evaluation)

5. **Multiple split patient ID files:**
   - `data/splits/train_patient_ids.json` (350 patients, default)
   - `data/splits/validation_patient_ids.json` (50 patients, default)
   - `data/splits/test_patient_ids.json` (100 patients, default)
   - `data/splits_full/train_patient_ids.json` (27,867 patients, full)
   - `data/splits_full/validation_patient_ids.json` (3,981 patients, full)
   - `data/splits_full/test_patient_ids.json` (7,962 patients, full)

---

## 7. Unused or Dead Code

### Identified Items

1. **`scripts/_make_fast_manifest_6000.py`** — Likely generates a 6,000-patient manifest, but the fast manifest has only 500 patients. The relationship between these scripts and the actual manifest generation is unclear.

2. **`scripts/recover_missing_files.sh`** — Recovery script whose purpose needs verification.

3. **Several debug experiment directories** (`p2_debug*`, `p2_974/`) — These appear to be historical experimental runs that are superseded by the `full_dataset_retrain` results. They may contain dead code or unreproducible configurations.

4. **`complete_p2_deliverables.py`** — This script's purpose and current state need verification.

5. **`generate_test_shard_only.py`** — May be a test-only utility with limited use.

6. **`audit_p2_artifacts.py` and `audit_training_data.py`** — Audit scripts that may have been used for prior verification but are not integrated into the current workflow.

7. **`test_bootstrap_fix.py`** — Standalone bootstrap fix test, not integrated into the pytest suite.

8. **`verify_p2_matched_repro.py`, `verify_p2_matched_fast.py`** — Verification scripts that may be partially obsolete.

9. **Several ml/p1_clinical/baselines/ files** — Classical ML baselines (Logistic Regression, Random Forest, SVM) that may have been superseded by P2 boosting models but are still present in the codebase.

10. **`ml/p3_temporal/explainability/demo_model.py`** — Demo model explicity separate from trained models, exists for development and demonstration.

---

## 8. Missing Tests

### Test Gaps

1. **No backend tests for real model loading**
   - No tests verify that the P3 container loads real XGBoost/LightGBM/Extra Trees models
   - No tests for the `ICU_SENTINEL_P1_MODEL_DIR` / `ICU_SENTINEL_P1_MODEL_NAME` pathway

2. **No integration tests for real-model prediction pipeline**
   - No tests that verify: observation payload → feature construction → real model inference → calibration → threshold → alert
   - All P3 API tests use the synthetic demo model

3. **No tests for leakage prevention**
   - No patient-level split isolation tests at the API level
   - No tests verifying that future information cannot leak into pre-onset predictions

4. **Limited frontend tests**
   - Frontend Vitest suite exists but tests disconnected/synthetic state
   - No tests connecting frontend to actual backend responses

5. **No end-to-end tests**
   - No workflow test: observation → API → prediction → alert → database → frontend
   - The reported 559 tests had a collection error in `test_split_validation.py`

6. **No database persistence tests**
   - Alert store uses SQLite temporarily in tests
   - No tests for database schema, migrations, or data integrity

7. **No calibration integration tests**
   - No tests verify that calibration is fitted on validation data and applied correctly on test data

8. **No feature validation tests at the API level**
   - No tests that verify feature schema compliance in prediction requests

---

## 9. Model Artifact Inventory

### Trained Models (in `artifacts/`)

| Model | Location | Features | Training Patients | Key Metadata |
|---|---|---|---|---|
| **XGBoost** | `artifacts/physionet2019/full_dataset_retrain/xgboost/xgboost_1.0.0.joblib` | 994 | 27,867 | AUROC=0.5318 (val), 0.8138 (test); ECE=0.3173 (test); hyperparams: n_estimators=200, lr=0.05, max_depth=4 |
| **LightGBM** | `artifacts/physionet2019/full_dataset_retrain/lightgbm/lightgbm_1.0.0.joblib` | 994 | 27,867 | AUROC=0.6045 (val), 0.8128 (test); ECE=0.2542 (test); hyperparams: n_estimators=200, lr=0.05 |
| **Extra Trees** | `artifacts/physionet2019/full_dataset_retrain/extra_trees/extra_trees_1.0.0.joblib` | 994 | 27,867 | AUROC=0.5748 (val); hyperparams: n_estimators=200, max_depth=null |

### Legacy/Experimental Models

| Model | Location | Notes |
|---|---|---|
| XGBoost (matched 6000) | `artifacts/p2/xgboost/xgboost_1.0.0.joblib` | 6,000-patient matched cohort |
| LightGBM (matched 6000) | `artifacts/p2/lightgbm/lightgbm_1.0.0.joblib` | 6,000-patient matched cohort |
| Extra Trees (matched 6000) | `artifacts/p2/extra_trees/extra_trees_1.0.0.joblib` | 6,000-patient matched cohort |
| XGBoost (debug6) | `artifacts/physionet2019/p2_debug6/xgboost/xgboost_1.0.0.joblib` | Debug experiment |
| XGBoost (full experiment) | `artifacts/physionet2019/full_dataset_experiment/xgboost/xgboost_1.0.0.joblib` | Experimental setup |
| XGBoost (retrain test) | `artifacts/physionet2019/full_dataset_retrain_test/xgboost/xgboost_1.0.0.joblib` | Retrain test setup |

### Model Metadata (key fields)

All three primary models have metadata JSON files containing:
- `n_features_input`: 994
- `n_features_output`: 994
- `n_patients_train`: 350 (P1 matched) or 27,867 (P2 full)
- `n_rows_train`: varies by experiment
- `n_positive_train`: varies by experiment
- `n_negative_train`: varies by experiment
- `selected_feature_names`: list of 994 feature names
- `validation_metrics`: auroc, auprc, sensitivity, specificity, precision, f1, brier
- `split_identity`: "patient_level_stratified"
- `source`: "real" (for full retrain artifacts)
- `hyperparameters`: model-specific hyperparameter dicts

### Model Versioning

- All models: version `1.0.0`
- Configuration: `p2_config.json` shared across models
- Training seed: random_seed=42 (from p2_config.json)

---

## 10. Dataset Inventory (without patient-level data)

### PhysioNet/CinC Challenge 2019 Dataset

| Metric | Value |
|---|---|
| **Dataset source** | https://physionet.org/content/challenge-2019/1.0.0/ |
| **Access requirements** | PhysioNet credentialed account + training completion |
| **Original patient records** | 39,810 (historical); 39,809 usable (1 corrupted file excluded: p010963) |
| **Usable patient records** | 39,809 |
| **Total observations** | 1,530,732 |
| **Sepsis patients** | 2,896 (7.27%) |
| **Training SetA** | 20,336 patients |
| **Training SetB** | 19,474 patients |
| **Corrupted file** | p010963 (1 file, excluded) |
| **Feature set** | 1001 features (994 selected for models) |
| **Target variable** | `SepsisLabel` (official); derived `sepsis_within_6h` |

### Current Data in Repository

| Directory | Description | Size |
|---|---|---|
| `data/raw/` | Raw .psv patient files (PhysioNet format) | Present but dataset not committed |
| `data/processed/` | Per-patient cleaned parquet files | Present |
| `data/features/combined_features.parquet` | Combined feature matrix | ~90 MB |
| `data/features/feature_schema.json` | 1001 features with types/descriptions | ~500 KB |
| `data/features/feature_config.json` | Feature engineering configuration | ~1 KB |
| `data/features/target_policy.json` | Target definition and ineligibility rules | ~300 bytes |
| `data/splits/split_manifest.json` | Default split (500 patients) | ~150 KB |
| `data/splits_full/split_manifest.json` | Full split (39,810 patients) | ~150 KB |
| `data/splits_full/train_patient_ids.json` | 27,867 train patient IDs | ~1 MB |
| `data/splits_full/validation_patient_ids.json` | 3,981 validation patient IDs | ~500 KB |
| `data/splits_full/test_patient_ids.json` | 7,962 test patient IDs | ~1 MB |

### Split Reconciliation

**Discrepancy resolution:**

- Reported: 39,810 original patients → 39,809 usable (1 corrupted file excluded)
- Split totals: 27,867 + 3,981 + 7,962 = 39,810 (matches original count)
- **The `splits_full/` directory has the correct patient-level splits matching the full dataset**
- The default `split_manifest.json` has only 500 patients (a subset, not the full dataset)
- **Resolution:** Use `splits_full/` for full-dataset experiments; the default split is a subset for development/testing

---

## 11. Backend Endpoint Inventory

### P1 Patient API (`backend/p1_patients/app.py`)

| Endpoint | Method | Response |
|---|---|---|
| `GET /health` | Read-only | Status OK |
| `GET /patients` | Read-only | List of patients |
| `GET /patients/{id}` | Read-only | Patient details |
| `GET /patients/{id}/timeline` | Read-only | Observation timeline |
| `GET /patients/{id}/features` | Read-only | Patient features |

### P3 API (`backend/p3/app.py`)

| Endpoint | Method | Purpose |
|---|---|---|
| `GET /health` | Read-only | Health check |
| `GET /api/v3/health` | Read-only | API health |
| `GET /api/v3/meta` | Read-only | Service metadata, provider status |
| `GET /api/v3/patients` | Read-only | List patients (synthetic demo) |
| `GET /api/v3/patients/{id}` | Read-only | Patient summary |
| `GET /api/v3/patients/{id}/timeline` | Read-only | Observation timeline |
| `GET /api/v3/provider/status` | Read-only | Prediction provider status |
| `GET /api/v3/provider/threshold` | Read-only | Alert threshold config |
| `GET /api/v3/replay/{id}` | Read-only | Start/replay patient |
| `GET /api/v3/replay/{id}/model-input` | Read-only | Model input features |
| `POST /api/v3/replay/{id}/step` | Write | Step replay forward/backward |
| `POST /api/v3/replay/{id}/jump` | Write | Jump to specific hour |
| `POST /api/v3/replay/{id}/reset` | Write | Reset replay |
| `POST /api/v3/replay/{id}/end` | Write | End replay |
| `GET /api/v3/alerts` | Read-only | List all alerts |
| `GET /api/v3/alerts/active` | Read-only | List active alerts |
| `GET /api/v3/alerts/history` | Read-only | List history |
| `GET /api/v3/alerts/{id}` | Read-only | Get specific alert |
| `POST /api/v3/alerts/evaluate` | Write | Evaluate prediction against threshold |
| `POST /api/v3/alerts/{id}/acknowledge` | Write | Acknowledge alert |
| `POST /api/v3/alerts/{id}/clear` | Write | Clear alert |

---

## 12. Frontend Page and Component Inventory

### Pages (11 main pages + NotFound)

| Page | Route | Key Features |
|---|---|---|
| **HomePage** | `/` | System status, model status, recent activity |
| **ReplayPage** | `/replay` | Patient selection, timeline playback |
| **ExplanationsPage** | `/explanations` | Feature importance, SHAP explanations |
| **AlertsPage** | `/alerts` | Active alerts, alert history |
| **AlertCentrePage** | `/alerts/centre` | Detailed alert management |
| **ReportsPage** | `/reports` | Model performance, calibration |
| **SettingsPage** | `/settings` | Model config, threshold, alert policy |
| **HelpPage** | `/help` | Documentation, disclaimers |
| **PatientReplayPage** | `/patient-replay` | Individual patient replay |
| **ExplanationCentrePage** | `/explanations/centre` | Detailed explanations |

### Components (35+)

| Category | Components |
|---|---|
| **Layout** | AppShell, MobileNavigation, Sidebar, TopNavbar |
| **Dashboard** | HeroSection, KeyModuleCard, MetricCard, RiskOverviewChart, SystemStatusCard |
| **Alerts** | AlertTable, AlertSummary, AlertDetails |
| **Patient Replay** | PatientTimeline, ClinicalObservationGrid, ReplayControls, RiskScoreCard |
| **Explainability** | GlobalImportanceChart, ShapFeatureChart, ExplanationSummary |
| **Common** | PageHeader, SectionCard, StateViews |
| **Badges** | Badges component for status labeling |

### Known Frontend Issues

1. **Broken/non-functional controls** — Some buttons and controls may not connect to working backend endpoints
2. **Synthetic data dependence** — Many components rely on demo/synthetic data
3. **No real-backend integration** — Frontend tests use mocked/synthetic responses
4. **Missing loading/error states** — Some components may not handle API errors gracefully

---

## 13. Security Concerns

### Identified Issues

1. **No authentication/authorization**
   - No user authentication in the FastAPI backend
   - No role-based access control
   - All endpoints are publicly accessible

2. **CORS configuration**
   - Hardcoded to `localhost:5173` and `localhost:3000`
   - No production CORS configuration
   - `allow_origins` can be overridden via `ICU_SENTINEL_CORS_ORIGINS` env var

3. **No input validation beyond Pydantic schemas**
   - Schemas exist but may not cover all edge cases
   - No rate limiting
   - No request size limits

4. **Patient data in error responses**
   - Error handlers may expose tracebacks or internal details
   - Need to ensure patient identifiers are not logged

5. **Database security**
   - P3 alert store uses SQLite with default settings
   - No connection pooling or authentication
   - Database file (`data/p3_alerts.db`) is in the repository directory

6. **Logging**
   - Logging configuration in `logging.yaml`
   - No structured logging
   - No redaction of sensitive data in logs

7. **No secrets management**
   - No `.env` file tracked in git
   - Environment variables expected but no defaults guard

8. **No HTTPS/TLS enforcement**
   - Development runs HTTP only
   - No proxy configuration for production TLS

### Data Privacy

1. **Pseudonymization**
   - Patient IDs are hashed/obfuscated in some contexts
   - Full patient IDs present in data files (psv format)
   - No explicit pseudonymization pipeline in the API

2. **No data minimization**
   - API may return more fields than necessary
   - No selective field projection

3. **Synthetic data labeling**
   - Demo model explicitly labelled as `is_synthetic=True`
   - No equivalent labeling for real-model predictions (currently missing)

---

## 14. Deployment Concerns

### Missing Infrastructure

1. **No Docker setup**
   - No `Dockerfile` for backend or frontend
   - No `docker-compose.yml` for combined deployment
   - Dependencies must be installed manually via `requirements.txt` and `frontend/package.json`

2. **No CI/CD pipeline**
   - No GitHub Actions workflows
   - No automated testing on push/PR
   - No build automation

3. **No environment configuration template**
   - No `.env.example` file
   - Environment variables expected but not documented template

4. **No production build configuration**
   - Frontend `npm run build` produces dist output
   - No production server configuration

5. **No process manager configuration**
   - No systemd service files
   - No PM2 configuration

6. **No health check endpoints for orchestration**
   - Basic `/health` exists but may not include all dependency status

7. **Database persistence not orchestrated**
   - SQLite database path configurable but not Docker-mounted
   - No persistent volume configuration

8. **Model artifact mounting not documented**
   - How to mount `artifacts/` in Docker not described
   - Large model files (1+ GB) need volume mounting

---

## 15. Scientific Reproducibility Concerns

### Identified Issues

1. **Reported vs actual model metrics discrepancy**
   - Historical reported: XGBoost AUROC=0.814, LightGBM AUROC=0.813, Extra Trees AUROC=0.743
   - Actual artifact metrics: XGBoost AUROC=0.5318 (val), 0.8138 (test), LightGBM AUROC=0.6045 (val), 0.8128 (test), Extra Trees AUROC=0.5748 (val)
   - AUPRC also differs significantly between reported and actual
   - ECE values differ: reported 0.0001-0.0003 vs actual 0.2542-0.3173 (test)
   - **Must be independently verified and documented**

2. **Split count discrepancy**
   - Default split: 500 patients
   - Full split: 39,810 patients
   - Historical reports reference 39,810 original + 39,809 usable
   - Must use `splits_full/` for reproducibility with historical results

3. **Bootstrap confidence intervals skipped**
   - Reported as "skipped (infrastructure issue)"
   - Patient-level bootstrap requires patient_id column
   - Must be completed for proper confidence intervals

4. **Calibration fitted on validation, not test**
   - Must verify isotonic/sigmoid calibration is fitted on validation data
   - Test set must remain untouched until final evaluation

5. **Feature selection fitted only on training data**
   - The 994 selected features must be selected using only training data
   - No feature selection using test set information

6. **No external validation**
   - All evaluation on PhysioNet 2019 dataset only
   - No independent dataset validation

7. **Label definition**
   - Derived target: `sepsis_within_6h` (positive for `onset..onset+5`, negative before onset and for non-septic patients)
   - Original `SepsisLabel` preserved unchanged
   - Must verify temporal invariance of label definition

---

## 16. Exact Prioritized Implementation Plan

### Priority 1 — Protect and Understand

- [x] Backup branch already created (main is clean)
- [x] Repository audit complete (this document)
- [x] Data and artifact inventory complete
- [x] Architecture mapping complete
- [ ] Identify all blockers (see sections 3-5)

### Priority 2 — Scientific Correctness

- [ ] **Reconcile dataset splits**: Use `splits_full/` for 39,810-patient reproducibility
- [ ] **Verify model metrics**: Independently reproduce evaluation from artifacts
- [ ] **Test leakage prevention**: Verify patient-level split isolation
- [ ] **Define six-hour early-warning task**: Document prediction horizon precisely
- [ ] **Calibration verification**: Verify calibration is on validation, not test data
- [ ] **Resolve P3 default model**: Integrate real model or make demo explicit

### Priority 3 — Actual ML Integration

- [ ] **Real model in P3 backend**: Configure `ICU_SENTINEL_P1_MODEL_DIR`/`ICU_SENTINEL_P1_MODEL_NAME`
- [ ] **Model registry**: Implement proper model loading and version identification
- [ ] **Prediction pipeline**: Real model inference through the backend
- [ ] **Calibration pipeline**: Correctly fitted and applied
- [ ] **Threshold policy**: Document and verify alert threshold

### Priority 4 — Application Completion

- [ ] **Backend**: Fix P3 to load real models by default when artifacts available
- [ ] **Alert engine**: Verify deduplication, cooldown, acknowledgement
- [ ] **Database**: Ensure SQLite alert store works reliably
- [ ] **Frontend**: Connect to real backend responses, fix broken controls
- [ ] **Tests**: Fix test collection error, add real-model integration tests

### Priority 5 — Engineering Quality

- [ ] **Testing**: Fix `test_split_validation.py`, ensure all critical tests pass
- [ ] **Security**: Address CORS, input validation, data exposure
- [ ] **Performance**: Profile model loading, prediction latency
- [ ] **Docker**: Create Dockerfiles and docker-compose.yml
- [ ] **CI/CD**: Implement GitHub Actions workflows

### Priority 6 — Research Documentation

- [ ] **Create all required docs** per the master prompt (22 documents)
- [ ] **Final acceptance report**: `docs/FINAL_ACCEPTANCE_REPORT.md`
- [ ] **Research limitations**: Explicitly address all known limitations
- [ ] **Reproducibility**: Document how to reproduce the full workflow

### Priority 7 — Final Acceptance Test

- [ ] **End-to-end workflow**: From clean environment to successful acceptance
- [ ] **All 20 acceptance criteria** verified with PASS/FAIL/BLOCKED/NOT_APPLICABLE
- [ ] **Documentation**: All docs accurate and complete