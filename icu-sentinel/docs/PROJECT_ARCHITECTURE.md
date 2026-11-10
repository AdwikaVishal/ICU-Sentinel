# ICU Sentinel — Project Architecture

**Last Updated:** November 10, 2026  
**Repository:** `/Users/shantanu/ICU-Sentinel`  
**Branch:** `main`

## 1. Overview

ICU Sentinel is a multi-phase, multidisciplinary research application for ICU sepsis early-warning prediction. It implements a modular architecture dividing responsibilities across P1 (data/clinical baselines), P2 (boosting models), and P3 (replay/alerts/UI) layers.

The project uses the PhysioNet Computing in Cardiology Challenge 2019 dataset, with the central research objective being early prediction of sepsis onset (up to six hours before onset).

## 2. Architecture Layers

### P1 — Data Engineering and Classical ML

**Responsibility:** Data ingestion, preprocessing, classical ML baselines (SIRS/qSOFA/NEWS), evaluation, and patient data APIs.

**Key Components:**
- **Ingestion:** `icu-sentinel/ml/p1_clinical/ingestion/` — Dataset loading, exploration, validation
- **Preprocessing:** `icu-sentinel/ml/p1_clinical/preprocessing/` — Cleaning, target construction (`sepsis_within_6h`), patient splits
- **Features:** `icu-sentinel/ml/p1_clinical/features/` — 994 feature engineering (rolling stats, trends, clinical scores, missingness indicators)
- **Classical ML:** `icu-sentinel/ml/p1_clinical/classical_ml/` — Logistic Regression, Random Forest, SVM
- **Evaluation:** `icu-sentinel/ml/p1_clinical/evaluation/` — Classification metrics, bootstrap, calibration, lead time analysis
- **Patient API:** `backend/p1_patients/` — FastAPI read-only endpoints serving patient parquet data

**Data Flow:**
1. Raw PhysioNet `.psv` files → preprocessing → cleaned per-patient parquet
2. Cleaned data → feature engineering → combined features parquet (994 features + targets)
3. Features + splits → classical ML training → model artifacts
4. Model artifacts → P1 patient API → read-only patient data endpoints

### P2 — Boosting Model Training

**Responsibility:** XGBoost, LightGBM, and Extra Trees model training, evaluation, calibration, and comparison.

**Key Components:**
- **Training:** `icu-sentinel/ml/p2_boosting/` — Training scripts, configuration, model artifacts
- **Evaluation:** `icu-sentinel/ml/p2_boosting/tests/` — Model comparison, bootstrap, performance characterization
- **Calibration:** Isotonic/sigmoid calibration on validation data
- **Model artifacts:** `artifacts/physionet2019/full_dataset_retrain/` — Trained models with metadata

**Model Artifacts (in `artifacts/physionet2019/full_dataset_retrain/`):**
| Model | Features | Training Patients | Validation AUROC | Test AUROC | Test ECE |
|---|---|---|---|---|---|
| XGBoost | 994 | 27,867 | 0.7986 | 0.8138 | 0.3173 |
| LightGBM | 994 | 27,867 | 0.8051 | 0.8128 | 0.2542 |
| Extra Trees | 994 | 27,867 | N/A | 0.7433 | 0.0008 |

**Data Flow:**
1. Combined features + patient-level splits → train/val/test split at patient level
2. Training data (patient-level stratified) → model training with fixed random seed (42)
3. Validation predictions → calibration (isotonic/sigmoid)
4. Test predictions → final evaluation metrics

### P3 — Replay Engine, Alerts, and UI

**Responsibility:** Patient retrospective playback, genuine SHAP explanations, persisted alert workflow, and React/TypeScript UI.

**Key Components:**
- **Replay engine:** `ml/p3_temporal/replay/` — Patient listing, timeline, model input, replay navigation
- **Prediction providers:** `ml/p3_temporal/replay/prediction_provider.py` — Synthetic demo heuristic and ModelPredictionProvider (backed by real model bundles)
- **Alert engine:** `ml/p3_temporal/alerts/` — Threshold-based alert creation with deduplication, acknowledgement, clearing
- **Explainability:** `ml/p3_temporal/explainability/` — Demo model explanations, feature mapping
- **React/TypeScript UI:** `icu-sentinel/frontend/` — 30+ pages/components including dashboard, patient replay, alerts, explanations, reports, settings

**Backend API:**
- `backend/p3/app.py` — FastAPI application with lifespan-based container wiring
- `backend/p3/container.py` — Dependency container choosing repositories, prediction providers, and persistence
- `backend/p3/routes/` — Replay, explanations, and alert routes
- `backend/p3/schemas.py` — Pydantic schemas for requests and responses

**Data Flow:**
1. Patient repository → observation timeline → model input construction → prediction provider → model inference → calibration → threshold → alert decision → database persistence → frontend display

### Backend (FastAPI)

**P1 Patient API** (`backend/p1_patients/app.py`):
- Read-only endpoints: `GET /health`, `GET /patients`, `GET /patients/{id}`, timeline and features
- Configuration via `ICU_SENTINEL_PROCESSED_DIR` and `ICU_SENTINEL_FEATURES_FILE` env vars
- Returns 503 when required data is absent

**P3 API** (`backend/p3/app.py`):
- Meta endpoints: `GET /health`, `GET /api/v3/health`, `GET /api/v3/meta`
- Patient replay: `GET /api/v3/patients`, `GET /api/v3/patients/{id}/timeline`
- Provider status: `GET /api/v3/provider/status`
- Prediction provider: `GET /api/v3/replay/{id}`, `GET /api/v3/replay/{id}/model-input`
- Alert evaluation: `POST /api/v3/alerts/evaluate`
- Alert management: `GET /api/v3/alerts`, `GET /api/v3/alerts/active`, `POST /api/v3/alerts/{id}/acknowledge`, `POST /api/v3/alerts/{id}/clear`

### Frontend (React/TypeScript)

**11 main pages:**
- `HomePage` — System status, model status, current prediction status, active alerts summary
- `ReplayPage` — Patient selection, timeline playback
- `ExplanationsPage` — Feature importance, global explanations
- `AlertsPage` — Active alerts, alert history
- `AlertCentrePage` — Detailed alert management
- `ReportsPage` — Model performance, calibration information
- `SettingsPage` — Model information, threshold configuration, alert policy
- `HelpPage` — Documentation, disclaimers
- `PatientReplayPage` — Individual patient replay
- `ExplanationCentrePage` — Detailed explanations
- `NotFoundPage` — Page not found handler

**35+ components:**
- Layout: AppShell, MobileNavigation, Sidebar, TopNavbar
- Dashboard: HeroSection, KeyModuleCard, MetricCard, RiskOverviewChart, SystemStatusCard
- Alerts: AlertTable, AlertSummary, AlertDetails
- Patient Replay: PatientTimeline, ClinicalObservationGrid, ReplayControls, RiskScoreCard
- Explainability: GlobalImportanceChart, ShapFeatureChart, ExplanationSummary
- Common: PageHeader, SectionCard, StateViews
- Badges: Status badges for prediction source, model status, alert status

## 3. Data Management

### Dataset: PhysioNet/CinC Challenge 2019

- **Source:** https://physionet.org/content/challenge-2019/1.0.0/
- **Access:** Requires PhysioNet credentialed account + training completion
- **Original patients:** 39,810
- **Usable patients:** 39,809 (1 corrupted file excluded: p010963)
- **Total observations:** 1,530,732
- **Sepsis patients:** 2,896 (7.27%)
- **Feature set:** 1001 features (994 selected for models)
- **Target:** `sepsis_within_6h` (derived from official `SepsisLabel`)

### Patient-Level Splits

Three split configurations exist:

| Split Directory | Total Patients | Train | Validation | Test | Sepsis Train | Sepsis Val | Sepsis Test |
|---|---|---|---|---|---|---|---|
| `data/splits/` (default) | 500 | 350 | 50 | 100 | 53 | 7 | 15 |
| `data/splits_full/` (full) | 39,810 | 27,867 | 3,981 | 7,962 | 2,027 | 290 | 579 |

**The `splits_full/` directory has the correct patient-level splits matching the full dataset.** The default `split_manifest.json` has only 500 patients (a subset for development/testing).

**Split integrity:** Patient-level splitting ensures no patient appears across multiple partitions. No future observations are used to predict past events.

### Feature Schema

- **Total features in schema:** 1001
- **Selected features for models:** 994 (7 features excluded, likely due to missingness or redundancy)
- **Feature types:** Vital signs (HR, Temp, SBP, DBP, MAP, Resp, O2Sat), laboratory variables, demographics, missingness indicators, rolling statistics, trend features, clinical scores (shock_index, sirs_count)
- **Feature engineering:** Rolling means, standard deviations, minimums, maximums, latest values, slopes, temporal differences, availability indicators, forward-filled values

### Target Definition

- **Derived target:** `sepsis_within_6h`
- **Positive definition:** `onset <= t <= onset + 5` (sepsis within 6 hours, inclusive of onset hour)
- **Negative definition:** Before onset and for non-septic patients
- **Ineligibility:** At or after `onset + 6` (no label after 6 hours post-onset)
- **Original `SepsisLabel`:** Preserved unchanged

## 4. Model Pipeline

### Training

Each model (XGBoost, LightGBM, Extra Trees) has:
- Fixed random seed: 42
- Hyperparameters from `p2_config.json`
- 994 selected features from 1001 total
- Patient-level train/validation/test splits
- Validation-time calibration (sigmoid/isotonic)
- Training metadata with checksums

**Training configuration (from `artifacts/p2/p2_config.json`):**
```json
{
  "random_seed": 42,
  "threshold_selection_metric": "f1",
  "threshold_grid_step": 0.01,
  "calibration_method": "sigmoid",
  "train_extra_trees": true,
  "train_lightgbm": true,
  "train_xgboost": true,
  "xgboost": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, ...},
  "lightgbm": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, ...},
  "extra_trees": {"n_estimators": 200, "max_depth": null, ...}
}
```

### Evaluation

Models evaluated on identical data splits with metrics:
- AUROC, AUPRC (critical for imbalanced data)
- Sensitivity, Specificity, Precision, F1, Negative Predictive Value
- Brier score, ECE (Expected Calibration Error)
- Confusion matrix, Precision-recall curve, ROC curve, Calibration plot

**Test set metrics (7,962 patients, 303,709 observations):**
| Model | AUROC | AUPRC | ECE | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.8138 | 0.0520 | 0.3173 | 0.1634 | 0.9747 | 0.0693 | 0.0973 |
| LightGBM | 0.8128 | 0.0504 | 0.2542 | 0.2099 | 0.9633 | 0.0619 | 0.0956 |
| Extra Trees | 0.7433 | 0.0325 | 0.0008 | 0.1490 | 0.9674 | 0.0501 | 0.0750 |

### Calibration

- **Method:** Sigmoid calibration (logistic regression) fitted on validation predictions
- **Principle:** Calibration fitted on validation data only; test set remains untouched
- **Evaluation:** Reliability diagrams, Brier score, ECE, calibration intercept and slope
- **Limitation:** Calibration in retrospective dataset does not guarantee calibration in another hospital population

## 5. Prediction and Alert Pipeline

### Inference Pathway

1. Receive observation payload at `POST /api/v3/alerts/evaluate`
2. Validate input via Pydantic schemas
3. Construct feature vector using the 994 selected features (same pipeline as training)
4. Load selected model and calibrator from artifact bundle
5. Generate uncalibrated probability from model
6. Apply calibrated probability: `calibrated = sigmoid(uncalibrated)`
7. Apply threshold: `alert if calibrated >= threshold`
8. Apply deduplication: check within `dedup_window_hours` (default 6)
9. Create/update alert record in database
10. Return structured result

### Alert Engine

An alert includes:
- Unique alert identifier
- Patient pseudonymous identifier
- Prediction timestamp
- Risk score
- Risk category (based on threshold)
- Model version
- Threshold version
- Relevant available clinical context
- Alert creation time
- Acknowledgement status
- Alert lifecycle status

**Alert lifecycle:**
- `ACTIVE` → prediction meets threshold within dedup window
- `ACKNOWLEDGED` → user acknowledges alert
- `CLEARED` → probability falls below threshold or manual clearance

**Deduplication:** Predictions within 6-hour window continue existing episode; gap > 6 hours starts new episode.

### Six-Hour Early Warning Task

The central research objective is early prediction, not merely detecting sepsis after it begins.

**Prediction horizons:**
- 0–1 hours before onset
- 1–3 hours before onset
- 3–6 hours before onset

**Full intended prediction window:** Up to 6 hours before onset, using only information available at or before the prediction timestamp.

**Event-level metrics:**
- First alert time
- Time from first alert to onset
- Percentage of sepsis events detected
- Percentage detected within six hours
- Percentage detected too late
- False alerts per patient-day
- Repeated alerts
- Alert burden

**Conversion of hourly predictions to alert episodes:**
- Repeated hourly predictions at or above threshold within the deduplication window continue the same episode
- A gap larger than the deduplication window starts a new episode
- Alert persistence: once created, an episode persists until cleared by threshold breach or manual acknowledgement

## 6. Scientific Principles (P1)

- Follow the official PhysioNet/CinC Challenge 2019 label convention
- Split by patient, never by hourly observation
- Never use future observations in features (causal only)
- Fit preprocessing on training data only
- Tune thresholds on validation data only
- Keep the test set untouched until final evaluation
- Report missingness honestly; never fabricate results
- No clinical validation claims from retrospective experiments

## 7. Known Limitations

- Retrospective dataset only; no prospective validation
- Single-hospital dataset (PhysioNet 2019); generalizability limited
- Class imbalance (1.14% positive class) leads to low precision
- Low sensitivity: many sepsis events not predicted before onset
- False-alert burden: many predictions above threshold without clinical sepsis
- Calibration retrospective only; not generalizable to other populations
- Label definition based on official `SepsisLabel` with inherent uncertainty
- No external validation on independent dataset
- No clinician workflow evaluation
- Threshold not clinically validated; demonstration/configuration only

**System label: Research prototype — not approved for clinical decision-making.**

## 8. Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run P1 unit tests
pytest ml/p1_clinical/tests tests -q -o addopts=""
```

## 9. Running the Application

### Backend (P3 API)

```bash
PYTHONPATH=. python3 -m uvicorn backend.p3.app:app --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:5173>. Run P3 tests:

```bash
python3 -m pytest ml/p3_temporal/tests tests/test_p3_api.py -q -o addopts=''
cd frontend && npm test && npm run typecheck && npm run build
```

## 10. Configuration Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ICU_SENTINEL_P1_MODEL_DIR` | (none) | Path to P1 model artifact directory |
| `ICU_SENTINEL_P1_MODEL_NAME` | (none) | Model name (e.g., xgboost, lightgbm, extra_trees) |
| `ICU_SENTINEL_P1_MODEL_VERSION` | 1.0.0 | Model version |
| `ICU_SENTINEL_DATA_DIR` | data/processed | Path to processed patient data directory |
| `ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT` | false | Use demo model when no P1 artifact configured |
| `ICU_SENTINEL_ALERT_THRESHOLD` | 0.60 | Alert threshold |
| `ICU_SENTINEL_ALERT_DEDUP_WINDOW_HOURS` | 6 | Alert deduplication window |
| `ICU_SENTINEL_CORS_ORIGINS` | localhost:5173,127.0.0.1:5173,localhost:3000 | CORS origins |

## 11. Testing

**P1 clinical tests:** 198 test items in `ml/p1_clinical/tests/`
**P3 API tests:** 15 test items in `tests/test_p3_api.py`
**Frontend tests:** 18 test items in `frontend/`
**Split validation diagnostic:** `test_split_validation.py` (standalone script)

**Test commands:**
```bash
# P1 clinical tests
pytest ml/p1_clinical/tests tests -q -o addopts=""

# P3 API tests
python3 -m pytest ml/p3_temporal/tests tests/test_p3_api.py -q -o addopts=''

# Frontend tests
cd frontend && npm test

# Typecheck
cd frontend && npm run typecheck

# Build
cd frontend && npm run build
```

## 12. Reproducibility

To reproduce the full workflow:

1. **Data:** Download PhysioNet 2019 dataset, place files under `data/raw/`
2. **Generate P1 artifacts:**
   ```bash
   python scripts/generate_p1_artifacts.py
   ```
3. **Generate full splits:**
   ```bash
   python scripts/create_full_splits.py
   ```
4. **Train models:**
   ```bash
   python scripts/train_p2_matched_6000.py
   # or for full dataset:
   python scripts/train_full_cohort.py
   ```
5. **Evaluate:**
   ```bash
   python complete_final_evaluation.py
   ```
6. **Start backend:**
   ```bash
   PYTHONPATH=. python3 -m uvicorn backend.p3.app:app --host 127.0.0.1 --port 8000
   ```
7. **Start frontend:**
   ```bash
   cd frontend && npm run dev
   ```

## 13. Citation and Ownership

This repository implements **Person 1 (P1)** of a three-person architecture:
- **P1 (this work):** Data engineering, clinical baselines (SIRS/qSOFA/NEWS), classical ML (Logistic Regression, Random Forest, SVM), evaluation, and patient data APIs.
- **P2:** Extra Trees, LightGBM, XGBoost, optional CatBoost, ensembles, calibration, statistical evaluation.
- **P3:** GRU, LSTM, optional Transformer, SHAP, replay, alerts, React app.

**License:** Research use only. PhysioNet/CinC Challenge 2019 data subject to PhysioNet data use agreements.