# API Documentation — ICU Sentinel

This document describes the FastAPI backend endpoints for the ICU Sentinel research prototype. The API serves patient data, P3 replay/prediction, and alert management functionality.

## API Base URL

- **Development**: `http://127.0.0.1:8000`
- **Production**: To be determined upon deployment

## API Version

- **Current version**: v1 (implicit, no version prefix in URLs)

## Health Check

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | `GET` | Return API status and basic information. |

**Response**:
```json
{
  "status": "ok",
  "message": "ICU Sentinel API is running"
}
```

## Patient Data Endpoints (P1)

### Get All Patients

| Endpoint | Method | Description |
|---|---|---|
| `GET /patients` | `GET` | List all patients in the data directory. |

**Query Parameters**:
- `split` (optional): Filter by split (`train`, `validation`, `test`)
- `source_dataset` (optional): Filter by source dataset (`set_a`, `set_b`)

**Response**:
```json
{
  "patients": [
    {
      "patient_id": "p00001",
      "split": "train",
      "source_dataset": "set_a",
      "has_sepsis": false
    },
    ...
  ]
}
```

### Get Specific Patient

| Endpoint | Method | Description |
|---|---|---|
| `GET /patients/{patient_id}` | `GET` | Retrieve a specific patient's parquet data, timeline, and features. |

**Response**:
```json
{
  "patient_id": "p00001",
  "hospital_id": null,
  "features": {...},
  "timeline": {
    "hours": [0, 6, 12, ...],
    "vitals": {...},
    "labs": {...}
  },
  "target": {
    "sepsis_within_6h": 0,
    "target_eligible": 1
  },
  "split": "train",
  "source_dataset": "set_a"
}
```

**Errors**:
- `404`: Patient not found
- `503`: Data directory not configured; returned when no real data is available

## P3 Replay Endpoints

### Get Patient from Repository

| Endpoint | Method | Description |
|---|---|---|
| `GET /p3/patients/{patient_id}` | `GET` | Retrieve a patient from the P3 replay repository. |

**Response**: Patient parquet data with features, target, and split information.

### Get Prediction for Patient

| Endpoint | Method | Description |
|---|---|---|
| `GET /p3/predict/{patient_id}` | `GET` | Generate a sepsis risk prediction for a specific patient. |

**Query Parameters**:
- `threshold` (optional): Probability threshold for sepsis alert (defaults to the model's fixed threshold)

**Response**:
```json
{
  "patient_id": "p00001",
  "risk_score": 0.45,  # Raw model probability
  "sepsis_within_6h": false,  # Above/below threshold
  "threshold": 0.78,  # Model-specific fixed threshold
  "model_name": "xgboost",  # Loaded model
  "calibrated": false,  # Whether calibration is applied
  "alert": false  # Whether risk_score exceeds threshold
}
```

**Errors**:
- `404`: Patient not found in repository
- `503`: No model loaded (no P1 model artifact configured via env vars)

### Get Alerts for Patient

| Endpoint | Method | Description |
|---|---|---|
| `GET /p3/alerts/{patient_id}` | `GET` | Retrieve alerts for a specific patient. |

**Response**:
```json
{
  "patient_id": "p00001",
  "alerts": [
    {
      "id": "alert_001",
      "type": "sepsis_risk",
      "risk_score": 0.82,
      "threshold": 0.78,
      "timestamp": "2024-01-15T14:30:00Z",
      "status": "active"
    }
  ],
  "total_alerts": 1
}
```

## Alert Management Endpoints

### Create Alert

| Endpoint | Method | Description |
|---|---|---|
| `POST /p3/alerts` | `POST` | Create a new alert. |

**Request**:
```json
{
  "patient_id": "p00001",
  "type": "sepsis_risk",
  "risk_score": 0.82,
  "threshold": 0.78
}
```

**Response**:
```json
{
  "id": "alert_001",
  "patient_id": "p00001",
  "type": "sepsis_risk",
  "risk_score": 0.82,
  "threshold": 0.78,
  "timestamp": "2024-01-15T14:30:00Z",
  "status": "active"
}
```

### Update / Acknowledge Alert

| Endpoint | Method | Description |
|---|---|---|
| `PATCH /p3/alerts/{alert_id}` | `PATCH` | Acknowledge or update an alert's status. |

**Request**:
```json
{
  "status": "acknowledged"
}
```

**Response**:
```json
{
  "id": "alert_001",
  "status": "acknowledged",
  "updated_at": "2024-01-15T14:35:00Z"
}
```

### Clear Alert

| Endpoint | Method | Description |
|---|---|---|
| `DELETE /p3/alerts/{alert_id}` | `DELETE` | Clear/acknowledge an alert. |

**Response**:
```json
{
  "id": "alert_001",
  "status": "clear",
  "cleared_at": "2024-01-15T14:40:00Z"
}
```

## Model Configuration Endpoints

### Get Model Status

| Endpoint | Method | Description |
|---|---|---|
| `GET /p3/model/status` | `GET` | Check which model is currently loaded and active. |

**Response**:
```json
{
  "model_loaded": true,
  "model_name": "xgboost",
  "model_source": "ICU_SENTINEL_P1_MODEL_DIR/ICU_SENTINEL_P1_MODEL_NAME",
  "demo_fallback": false,
  "risk_score_available": true
}
```

### Set Model Configuration

The model is loaded via environment variables. No API endpoint currently exists to change the model at runtime; use environment variable reconfiguration and API restart.

| Environment Variable | Description |
|---|---|
| `ICU_SENTINEL_P1_MODEL_DIR` | Path directory containing the model artifact |
| `ICU_SENTINEL_P1_MODEL_NAME` | Model name (e.g., `xgboost`, `lightgbm`, `extra_trees`, `logistic_regression`) |
| `ICU_SENTINEL_P1_MODEL_VERSION` | Model version (default: `1.0.0`) |
| `ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT` | If `false` (default now), demo model is not used when no artifact is set; `503` returned instead. If `true`, demo model falls back. |

**Model Loading Priority** (from `container.py:_resolve_bundle`):
1. **Explicit configuration**: If `ICU_SENTINEL_P1_MODEL_DIR` and `ICU_SENTINEL_P1_MODEL_NAME` are both set → load that model artifact
2. **Auto-detection**: If `data_dir` points to `processed_full` → auto-detect model in `artifacts/physionet2019/full_dataset_retrain/`
3. **Demo fallback**: If `demo_model_when_no_artifact=true` → use synthetic demo logistic regression; if `false` → return 503 "no model loaded"

**Important**: The default for `demo_model_when_no_artifact` is now `false` (changed from `true`). When no model artifact is configured, the API returns `503` instead of falling back to the demo model. This is an opt-in feature.

## Error Handling

| Status Code | Meaning |
|---|---|
| `200` | Success |
| `400` | Bad request — invalid query parameters or payload |
| `404` | Resource not found (patient, alert, model) |
| `503` | Service unavailable — no data or model configured |

All error responses include a `detail` field with a human-readable explanation.

## API Usage Examples

### Python Example

```python
import requests

# Check health
resp = requests.get("http://127.0.0.1:8000/health")
print(resp.json())

# List patients in train split
resp = requests.get("http://127.0.0.1:8000/patients?split=train")
patients = resp.json()["patients"]
print(f"Train patients: {len(patients)}")

# Get specific patient
resp = requests.get("http://127.0.0.1:8000/patients/p00001")
patient = resp.json()

# Get prediction
resp = requests.get("http://127.0.0.1:8000/p3/predict/p00001")
prediction = resp.json()
print(f"Risk score: {prediction['risk_score']}")
print(f"Alert: {prediction['alert']}")
```

### curl Examples

```bash
# Health check
curl http://127.0.0.1:8000/health

# List all patients
curl "http://127.0.0.1:8000/patients?split=train"

# Get patient prediction
curl "http://127.0.0.1:8000/p3/predict/p00001"

# Create alert
curl -X POST "http://127.0.0.1:8000/p3/alerts" \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "p00001", "type": "sepsis_risk", "risk_score": 0.82, "threshold": 0.78}'
```

## Security and Privacy

- **No patient information is exposed in URL paths** beyond patient IDs that are already part of the dataset.
- **API is for research purposes only** — labeled "Research prototype — not approved for clinical decision-making."
- **No clinical validation claims** are made; all performance metrics are retrospective.
- **IP rate-limiting** may be implemented in production deployment.
- **HTTPS** should be used in production; HTTP is acceptable for local development only.

## API Test Coverage

- P1 patient API: 395/395 tests passed (prior, exit 0)
- P3 API: tests exist in the test suite; see `ml/p3_temporal/tests/` for details
- Integration tests: synthetic/demo mode; real-model tests not yet written

## Version History

| Version | Date | Changes |
|---|---|---|
| v1 | 2026-11-10 | Initial API documentation; P3 real model integration; `demo_model_when_no_artifact=false` default |