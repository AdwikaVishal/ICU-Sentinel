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
- Place downloaded files under `data/raw/`.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run P1 unit tests
pytest
```

## Running the patient API (P1)

```bash
uvicorn backend.p1_patients.app:app --reload
```

Endpoints:
- `GET /patients`
- `GET /patients/{id}`
- `GET /patients/{id}/timeline`
- `GET /patients/{id}/features`

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
