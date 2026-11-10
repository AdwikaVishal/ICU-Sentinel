# Security and Privacy — ICU Sentinel

This document covers the security and privacy protections in place for the ICU Sentinel research prototype. The system is labeled "Research prototype — not approved for clinical decision-making" and handles data accordingly.

## Data Classification

| Data Type | Classification | Handling |
|---|---|---|
| **Raw PhysioNet data** (`data/raw/`) | Highly sensitive — PHI-adjacent | Excluded from repository via `.gitignore`; never committed; not served by API |
| **Processed patient parquet** (`data/processed_full/`) | Sensitive — contains ICU records | Served via read-only FastAPI endpoints; no personally identifiable information (PII) |
| **Model artifacts** (`artifacts/p2/`) | Internal — model files | Gitignored; distributed separately; contain no patient identifiers |
| **Split manifests** (`data/splits/`) | Internal — patient split assignments | Gitignored; contain anonymized patient IDs only |
| **Alert database** (`data/p3_alerts.db`) | Internal — alert state | SQLite database; not backed up to cloud; excluded from git |
| **Frontend UI data** | Public — research interface | No patient-identifiable data; displays risk scores and split assignments |

**Key principle**: No patient-identifiable information (name, MRN, SSN, medical record number) is stored, processed, or displayed anywhere in the system. Patient IDs are anonymized identifiers (e.g., `p00001`, `p00002`).

## Privacy Protections

### 1. No PHI in Model Features or Targets

The 994 features used by the models are derived from:
- Vital signs (heart rate, blood pressure, temperature, O2 saturation)
- Laboratory values (creatinine, lactate, WBC, platelets, etc.)
- Clinical scores (SIRS, shock index, demographics)
- Rolling statistics and change features (all retrospective)

The prediction target `sepsis_within_6h` is a binary clinical label, not patient-identifiable.

**No features derived from**:
- Patient names or identifiers
- Dates/times that could identify hospitalization periods
- Photographs or imaging data
- Free-text clinical notes

### 2. Anonymized Patient Identifiers

- Patient IDs follow the format `pXXXXX` (e.g., `p00001`, `p00042`)
- IDs are sequential and do not correspond to any external identifier
- The mapping from PhysioNet original IDs to ICU Sentinel IDs is many-to-one (multiple PhysioNet stays may map to the same anonymized ID, or IDs are simply reassigned)

### 3. Data Minimization

- The API serves only the features required for model inference
- No raw physiological waveforms or unreported data is exposed
- Timeline data is limited to hour-resolution observations, not minute-by-minute continuous data
- Feature values are sanitized: infinite values are replaced with NaN before model ingestion

### 4. No Model Output Persistence

- Model predicted probabilities are not permanently stored alongside patient identities
- Alert database tracks alert state (active/acknowledged/clear) but not the underlying patient risk scores beyond what's necessary for the alert lifecycle
- Prediction outputs are ephemeral — computed on request, not cached long-term

### 5. Repository-Level Protections

- `.gitignore` excludes `artifacts/`, `data/raw/`, `*.joblib`, `*.parquet`, `*.psv`
- `artifacts/` directory is gitignored entirely (~3.5 GB of model files)
- No sensitive data is committed to the git repository
- All large files are stored locally and referenced, not versioned

## Security Measures

### 1. Backend Security

- **FastAPI** with Pydantic input validation on all endpoints
- **CORS middleware** configured via `ICU_SENTINEL_CORS_ORIGINS` env var (default allows localhost dev origins only)
- **HTTP exception handling**: Returns `404` for not-found, `503` when data/model not available
- **No authentication** for research prototype (assumes local/trusted deployment)
- **Bind address**: Default `127.0.0.1` for local development; production should use proper networking

### 2. Dependency Security

- `requirements.txt` and `pyproject.toml` specify exact package versions
- Regular dependency updates recommended
- No unknown or unpinned third-party packages in production

### 3. Environment Variable Security

- Model artifact paths are configured via environment variables (`ICU_SENTINEL_P1_MODEL_DIR`, `ICU_SENTINEL_P1_MODEL_NAME`)
- Demo model fallback controlled by `ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT`
- Default `demo_model_when_no_artifact=false` means 503 error if no real model is configured — operator must opt-in to demo mode
- No secrets (API keys, passwords) stored in code or documentation

### 4. Alert Engine Security

- Alerts follow a lifecycle: `create → acknowledge → clear`
- No malicious input can break the alert state machine (validated via PATCH/DELETE endpoints)
- Deduplication is handled server-side; duplicate alerts are merged
- Alert database (`p3_alerts.db`) is SQLite, local to the deployment

## Data Both/And: Research vs. Clinical

### Research Use

ICU Sentinel is designed for **retrospective research** only. It:
- Uses historical PhysioNet data
- Trains and evaluates machine learning models
- Reports performance metrics (AUROC, AUPRC, etc.)
- Supports research analysis and reproduction

### Not Approved for Clinical Use

**ICU Sentinel is NOT**:
- Approved for clinical decision-making
- A substitute for clinical judgment
- Cleared by any regulatory body (FDA, CE, etc.)
- Suitable for direct patient-facing use without clinician oversight

**Must display in UI**: "Research prototype — not approved for clinical decision-making."

**Must display in documentation**: "Research prototype — not approved for clinical decision-making."

**Must not claim**: Clinical validation, diagnostic accuracy for patient care, or any treatment recommendations.

## Compliance and Ethical Considerations

### HIPAA Considerations

- The system uses de-aggregated, de-identified data from PhysioNet
- No PHI is processed, stored, or transmitted
- System is not a "covered entity" under HIPAA given the data type
- However, data handling follows privacy-by-design principles

### Research Ethics

- Data source: PhysioNet CinC 2019 Challenge — ethical approval was obtained by the challenge organizers
- No new patient data is collected
- No patients are recruited or contacted
- Results are retrospective only; no prospective claims are made

### Publication Guidelines

If citing ICU Sentinel results:
- Cite the verified artifact metrics (AUROC 0.8138 for XGBoost, etc.)
- Disclose that data is from PhysioNet CinC 2019 Challenge
- State that the system is a research prototype not approved for clinical use
- Do not claim clinical validity or readiness for deployment

## Security Checklist ✅

- [x] No PHI in features, targets, or model outputs
- [x] Anonymized patient IDs (format: pXXXXX)
- [x] Raw data excluded from git (.gitignore)
- [x] Model artifacts gitignored (~3.5 GB)
- [x] No patient-identifiable data in frontend UI
- [x] API binds to 127.0.0.1 by default (local-only)
- [x] CORS configured via env var (not wildcard in production)
- [x] `demo_model_when_no_artifact=false` (503 if no real model; operator opt-in for demo)
- [x] Alert lifecycle validated (create/acknowledge/clear)
- [x] System labeled "Research prototype — not approved for clinical decision-making"
- [x] No clinical validation claims made or implied
- [x] Environment variables for model configuration (no hardcoded paths)
- [x] Read-only API endpoints for patient data
- [x] Brier/BECE metrics reported without clinical validation claims

## Known Security/Gaps

- **No authentication**: Research prototype assumes trusted deployment; add auth for production
- **No rate limiting**: Could be added for public deployments
- **No encryption at rest**: Alert database is local SQLite; encryption not implemented
- **No audit logging**: No request-level audit trail currently
- **Demo mode without explicit opt-in risk**: Default `demo_model_when_no_artifact=false` prevents accidental demo mode, but operators should be aware

## References

- `.gitignore` — File exclusion patterns
- `pyproject.toml` — Project configuration and dependencies
- `P3_SETTINGS` — Environment variable configuration (see `backend/p3/container.py`)
- `MASTER_COMPLETION_REPORT.md` — Overall project readiness status
- PhysioNet CinC 2019 Challenge data usage terms