# ICU Sentinel

Interpretable early-warning research system for predicting sepsis onset in ICU
patients, using the PhysioNet/CinC Challenge 2019 dataset.

## Ownership

This repository implements **Person 1 (P1)** of a three-person architecture:

- **P1 (this work):** Data engineering, clinical baselines (SIRS/qSOFA/NEWS),
  classical ML (Logistic Regression, Random Forest, SVM), evaluation, and
  patient data APIs.
- **P2:** Extra Trees, LightGBM, XGBoost, optional CatBoost, ensembles,
  calibration, statistical evaluation.
- **P3:** GRU, LSTM, optional Transformer, SHAP, replay, alerts, React app.

P1 exposes shared interfaces/schemas so P2 and P3 can integrate later.

## Project structure

```
icu-sentinel/
  data/            raw / processed / features / splits / reports (gitignored)
  ml/
    common/        schemas / metrics / fixtures
    p1_clinical/   ingestion / preprocessing / features / baselines /
                   classical_ml / evaluation / tests
    p2_boosting/   (P2 - not implemented by P1)
    p3_temporal/   (P3 - not implemented by P1)
  backend/
    common/
    p1_patients/   FastAPI patient endpoints
  experiments/     p1 / p2 / p3
  artifacts/       p1 / p2 / p3 (gitignored)
  docs/            p1 / p2 / p3
  scripts/
  tests/
```

## Data

The PhysioNet/CinC Challenge 2019 dataset is **not** included in this repo and
must not be committed to Git (see `.gitignore`).

- Source: https://physionet.org/content/challenge-2019/1.0.0/
- Access requires a PhysioNet credentialed account and completion of the
  required training.
- Place the downloaded training files under `data/raw/` (flat or nested;
  files must be named `p#####.psv`). The repository never creates or
  substitutes synthetic data for real results.
- Tested with Python 3.11.9. Record the actual installed versions with
  `pip freeze` rather than copying historical report versions.

## Generating P1 artefacts

```bash
# Preprocessing -> engineered features -> training/validation/test
python scripts/generate_p1_artifacts.py

# Stop after features (useful before training)
python scripts/generate_p1_artifacts.py --skip-training
```

Outputs:

- `data/processed/` per-patient cleaned parquet
- `data/splits/split_manifest.json` patient-level splits
- `data/features/combined_features.parquet` plus `data/features/patients/`
- `data/features/feature_config.json`, `feature_schema.json`, `target_policy.json`
- `data/reports/classical_ml/` validation/test metrics and selected thresholds
- `artifacts/p1/classical_ml/` fitted models

The derived target is `sepsis_within_6h` (positive for `onset..onset+5`,
negative before onset and for non-septic patients, ineligible at/after
`onset+6`). The original `SepsisLabel` is preserved unchanged.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run P1 unit tests
pytest ml/p1_clinical/tests tests -q -o addopts=""
```

`torch` is not a repository dependency. If SHAP fails to import with
`OSError: [WinError 1114] ... c10.dll`, a stray/broken CPU `torch`
installation is the cause (SHAP probes it during import). Repair or remove
that environment-only package (`pip uninstall torch torchvision`) and retry.

## Running the patient API (P1)

```bash
uvicorn backend.p1_patients.app:app --reload
```

Endpoints (read-only):
- `GET /health`
- `GET /patients`
- `GET /patients/{id}`
- `GET /patients/{id}/timeline?offset=0&limit=1000`
- `GET /patients/{id}/features?offset=0&limit=1000`

Configuration:
- `ICU_SENTINEL_PROCESSED_DIR` (default `data/processed`)
- `ICU_SENTINEL_FEATURES_FILE` (default `data/features/combined_features.parquet`)

The API never invents missing fields and never serves synthetic demo patients.
When the configured data is absent it returns HTTP 503.

## P3 research prototype (replay, SHAP, alerts)

P3 adds a retrospective Patient Replay Engine, genuine SHAP explanations and a
persisted alert workflow, plus a React/TypeScript UI. It uses the synthetic
demo cohort unless a processed-data directory and P1 artifact are configured.
Nothing in P3 is a diagnosis or a clinically validated early-warning system.

```bash
# Backend (from the repository root)
PYTHONPATH=. python3 -m uvicorn backend.p3.app:app --host 127.0.0.1 --port 8000

# Frontend (second terminal)
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:5173>. Run P3 tests with:

```bash
python3 -m pytest ml/p3_temporal/tests tests/test_p3_api.py -q -o addopts=''
cd frontend && npm test && npm run typecheck && npm run build
```

Documentation: `docs/p3/` (architecture, causality/leakage, explainability,
alerts, API reference, demo guide, implementation report).

## Scientific principles (P1)

- Follow the official PhysioNet/CinC Challenge 2019 label convention.
- Split by patient, never by hourly observation.
- Never use future observations in features (causal only).
- Fit preprocessing on training data only.
- Tune thresholds on validation data only.
- Keep the test set untouched until final evaluation.
- Report missingness honestly; never fabricate results.
- No clinical validation claims from retrospective experiments.

## License

Research use only. PhysioNet/CinC Challenge 2019 data subject to PhysioNet
data use agreements.
