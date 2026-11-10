# Deployment — ICU Sentinel

This document describes the deployment configuration, Docker setup, and CI/CD workflow for the ICU Sentinel research prototype.

## Deployment Label

**ICU Sentinel is labeled: "Research prototype — not approved for clinical decision-making."**

This label must be visible in the UI and on all documentation. No clinical validation claims are made.

## Docker Configuration

### Docker Compose (Backend + Frontend)

A `docker-compose.yml` file is provided to run both the FastAPI backend and the React frontend in containers.

**`docker-compose.yml` contents**:

```yaml
version: "3.9"

services:
  backend:
    build:
      context: ./icu-sentinel
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ICU_SENTINEL_P1_MODEL_DIR=/app/artifacts/physionet2019/full_dataset_retrain
      - ICU_SENTINEL_P1_MODEL_NAME=xgboost
      - ICU_SENTINEL_P1_MODEL_VERSION=1.0.0
      - ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT=false
      - ICU_SENTINEL_DATA_DIR=/app/data/processed_full
      - ICU_SENTINEL_ALERT_DB=/app/data/p3_alerts.db
      - ICU_SENTINEL_CORS_ORIGINS=http://localhost:5173
    volumes:
      - ./artifacts:/app/artifacts:ro
      - ./data:/app/data:ro
    depends_on:
      - frontend

  frontend:
    build:
      context: ./icu-sentinel/frontend
      dockerfile: frontend/Dockerfile
    ports:
      - "5173:5173"
    environment:
      - VITE_API_BASE_URL=http://localhost:8000
    depends_on:
      - backend
```

**`Dockerfile` (backend)**:

```dockerfile
FROM python:3.14.5-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY icu-sentinel /app/icu-sentinel

# Set working directory and run
WORKDIR /app/icu-sentinel/backend/p3

COPY pyproject.toml .

# Install project in editable mode
RUN pip install -e ".[all]"

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]
```

**`Dockerfile` (frontend)**:

```dockerfile
FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm ci --legacy-peer-deps

# Copy source and build
COPY . .
RUN npm run build

# Serve with static file server
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx"]
```

### Running with Docker Compose

```bash
# From the ICU-Sentinel root directory
docker compose up --build
```

This will start:
- Backend API at `http://localhost:8000`
- Frontend at `http://localhost:5173`

### Running Outside Docker

```bash
# Start backend
cd icu-sentinel
source .venv/bin/activate
uvicorn backend/p3/app:app --host 127.0.0.1 --port 8000

# Start frontend (separate terminal)
cd icu-sentinel/icu-sentinel/frontend
npm install
npm run dev
```

## Environment Configuration

### Required Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ICU_SENTINEL_P1_MODEL_DIR` | *not set* | Path to directory containing the model artifact (e.g., `/app/artifacts/physionet2019/full_dataset_retrain`) |
| `ICU_SENTINEL_P1_MODEL_NAME` | *not set* | Model name (e.g., `xgboost`, `lightgbm`, `extra_trees`, `logistic_regression`) |
| `ICU_SENTINEL_P1_MODEL_VERSION` | `1.0.0` | Model version string |
| `ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT` | `false` | If `false`, demo model is not used when no artifact is set (API returns 503). If `true`, demo model falls back. |
| `ICU_SENTINEL_DATA_DIR` | *not set* | Path to processed patient data directory (e.g., `/app/data/processed_full`) |
| `ICU_SENTINEL_ALERT_DB` | `data/p3_alerts.db` | Path to the alerts SQLite database |
| `ICU_SENTINEL_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000` | Comma-separated list of allowed CORS origins |
| `ICU_SENTINEL_DATA_DIR` | — | Path to processed data; required for real model loading |

### Model Loading Priority

The P3 container (`container.py:_resolve_bundle`) loads models in this order:

1. **Explicit configuration** (highest priority): If both `ICU_SENTINEL_P1_MODEL_DIR` and `ICU_SENTINEL_P1_MODEL_NAME` are set → load that model artifact
2. **Auto-detection**: If `ICU_SENTINEL_DATA_DIR` points to `processed_full` → auto-detect model in `artifacts/physionet2019/full_dataset_retrain/`
3. **Demo fallback**: If `demo_model_when_no_artifact=false` (default) → return 503 "no model loaded"; if `true` → use synthetic demo logistic regression

**Important**: The default `demo_model_when_no_artifact=false` means the operator must explicitly set `ICU_SENTINEL_P1_MODEL_DIR/NAME` to use real models. The demo model is not used by default.

### Example Production Configuration

```bash
# Export environment variables
export ICU_SENTINEL_P1_MODEL_DIR=/opt/icu-sentinel/artifacts/physionet2019/full_dataset_retrain
export ICU_SENTINEL_P1_MODEL_NAME=xgboost
export ICU_SENTINEL_P1_MODEL_VERSION=1.0.0
export ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT=false
export ICU_SENTINEL_DATA_DIR=/opt/icu-sentinel/data/processed_full
export ICU_SENTINEL_ALERT_DB=/opt/icu-sentinel/data/p3_alerts.db
export ICU_SENTINEL_CORS_ORIGINS=https://icu-sentinel.example.com,https://app.icu-sentinel.example.com

# Or in a .env file
echo "ICU_SENTINEL_P1_MODEL_DIR=/opt/icu-sentinel/artifacts/physionet2019/full_dataset_retrain" >> .env
echo "ICU_SENTINEL_P1_MODEL_NAME=xgboost" >> .env
echo "ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT=false" >> .env
```

## CI/CD Workflow

### GitHub Actions Workflows

Three workflows are defined in `.github/workflows/`:

#### 1. `lint-test.yml` — Lint and Test

```yaml
name: Lint and Test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  lint-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.14"]
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install pip setuptools wheel
          pip install -r icu-sentinel/requirements.txt
      
      - name: Lint
        run:
          source .venv/bin/activate
          ruff check icu-sentinel/
          ruff check backend/
          ruff check ml/
      
      - name: Type check
        run:
          source .venv/bin/activate
          mypy icu-sentinel/ --ignore-missing-imports
      
      - name: Run pytest
        run:
          source .venv/bin/activate
          cd icu-sentinel
          pytest -o addopts="" -p no:cov
      
      - name: Run vitest
        run:
          cd icu-sentinel/icu-sentinel/frontend
          npm test
```

#### 2. `build.yml` — Build and Lint

```yaml
name: Build and Lint

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.14.5"
      
      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install pip setuptools wheel
          pip install -r icu-sentinel/requirements.txt
      
      - name: Run ruff lint
        run:
          source .venv/bin/activate
          ruff check icu-sentinel/ backend/ ml/
      
      - name: Run mypy type check
        run:
          source .venv/bin/activate
          mypy icu-sentinel/ --ignore-missing-imports
      
      - name: Frontend build
        run: |
          cd icu-sentinel/icu-sentinel/frontend
          npm ci --legacy-peer-deps
          npm run build
```

#### 3. `import-check.yml` — Artifact Import Checks

```yaml
name: Artifact Import Checks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  import-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Verify model artifacts exist
        run: |
          if [ ! -f "artifacts/p2/xgboost/xgboost_1.0.0.joblib" ]; then
            echo "::error ::XGBoost model artifact missing"
            exit 1
          fi
          if [ ! -f "artifacts/p2/lightgbm/lightgbm_1.0.0.joblib" ]; then
            echo "::error ::LightGBM model artifact missing (expected in some configs)"
          fi
          if [ ! -f "artifacts/p2/extra_trees/extra_trees_1.0.0.joblib" ]; then
            echo "::error ::Extra Trees model artifact missing"
            exit 1
          fi
          echo "All primary model artifacts present"
      
      - name: Verify split manifests
        run: |
          if [ ! -f "icu-sentinel/data/splits/split_manifest.json" ]; then
            echo "::error ::Default split manifest missing"
            exit 1
          fi
          if [ ! -f "icu-sentinel/data/splits_full/split_manifest.json" ]; then
            echo "::error ::Full split manifest missing"
            exit 1
          fi
          echo "All split manifests present"
      
      - name: Verify evaluation metrics
        run: |
          if [ ! -f "artifacts/physionet2019/final_evaluation/test_evaluation.json" ]; then
            echo "::error ::Test evaluation metrics missing"
            exit 1
          fi
          echo "Evaluation metrics present"
```

### CI/CD Triggers

- **Push to `main`**: Runs all three workflows (lint-test, build, import-check)
- **Pull request to `main`**: Runs all three workflows for validation
- **Tag push** (e.g., `v1.0.0`): Could trigger Docker image publishing and publication (not yet implemented)

## Git Ignore Large Artifacts

The `.gitignore` already excludes:
- `artifacts/` — Large model files (~3.5 GB total)
- `*.joblib`, `*.pkl`, `*.pickle` — Model artifacts
- `*.parquet` — Processed patient data
- `*.psv` — Raw PhysioNet data
- `data/` — Under `artifacts/` exclusion

**Large files that should NOT be committed**:
- `artifacts/p2/xgboost/xgboost_1.0.0.joblib`
- `artifacts/p2/lightgbm/lightgbm_1.0.0.joblib`
- `artifacts/p2/extra_trees/extra_trees_1.0.0.joblib`
- `artifacts/physionet2019/final_evaluation/test_evaluation.json`
- `data/processed_full/` — Full processed dataset
- `data/splits_full/` — Full 39,810-patient splits

## Reproducibility

### Reproduction Checklist

- [ ] Docker Compose brings up backend + frontend successfully
- [ ] `ICU_SENTINEL_P1_MODEL_DIR` and `ICU_SENTINEL_P1_MODEL_NAME` are set for real model loading
- [ ] `ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT=false` (default, opt-in for demo)
- [ ] Backend health check passes at `http://localhost:8000/health`
- [ ] Frontend loads at `http://localhost:5173` and can fetch predictions
- [ ] `pytest -o addopts="" -p no:cov` passes for P1 tests (395/395)
- [ ] vitest passes for frontend (18/18)
- [ ] `test_split_validation.py` runs and reports "All splits match perfectly!"
- [ ] Model metrics verified from actual artifacts (AUROC: XGBoost 0.8138, LightGBM 0.8128, Extra Trees 0.7433)

### Updating Docker Images

If model artifacts or code change:

```bash
# Rebuild all images
docker compose up --build --force-recreate

# Or rebuild individually
docker compose build backend
docker compose build frontend
```

## Security and Privacy

### Data Handling

- **Raw PhysioNet data** is excluded from the repository and not served by the API
- **Processed patient data** is served via read-only FastAPI endpoints
- **No patient identifiable information** (no MRN, no SSN, no names) is exposed
- **Patient IDs** are anonymized identifiers (e.g., `p00001`, `p00002`)

### CORS Configuration

- Default `ICU_SENTINEL_CORS_ORIGINS` allows `localhost:5173`, `127.0.0.1:5173`, `localhost:3000`
- For production, set `ICU_SENTINEL_CORS_ORIGINS` to the exact production domain(s)
- Do not use `*` in production CORS configuration

### API Security

- FastAPI with no authentication for research prototype simplicity
- API is local-only by default (binds to `127.0.0.1`)
- Production deployment should add:
  - Authentication/authorization middleware
  - Rate limiting
  - HTTPS termination
  - Request validation and sanitization

## License

ICU Sentinel is a research project. See the root `LICENSE` or `pyproject.toml` for licensing details. All model artifacts are derived from PhysioNet CinC 2019 Challenge data, used under the challenge's data usage terms.

## Version History

| Version | Date | Changes |
|---|---|---|
| v1.0 | 2026-11-10 | Initial deployment documentation; Docker Compose; CI/CD workflows; `demo_model_when_no_artifact=false` default |