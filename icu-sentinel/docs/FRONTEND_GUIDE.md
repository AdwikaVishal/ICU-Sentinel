# Frontend Guide — ICU Sentinel

This document provides guidance for the ICU Sentinel React TypeScript frontend, including development, building, and usage instructions.

## Project Structure

```
icu-sentinel/frontend/
├── src/              # React source code
│   ├── components/   # Reusable UI components
│   ├── pages/        # Page components (dashboard, alerts, etc.)
│   ├── api/          # API client functions
│   ├── types/        # TypeScript type definitions
│   └── App.tsx       # Root component
├── package.json      # Dependencies and scripts
├── vite.config.ts    # Vite build configuration
└── tsconfig.json     # TypeScript configuration
```

## Getting Started

### Prerequisites

- **Node.js**: v18 or v20 (v20 recommended)
- **npm**: v10 or later (comes with Node.js)
- **Python**: 3.14.5 (for backend API, see separate backend guide)

### Installation

1. **Install dependencies**:
   ```bash
   cd icu-sentinel/frontend
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173` (default Vite dev server port).

3. **Start the backend API** (separate terminal):
   ```bash
   cd icu-sentinel
   # Ensure .venv is active
   source .venv/bin/activate
   # Start the FastAPI backend
   cd backend/p3 && uvicorn app:app --host 127.0.0.1 --port 8000
   ```
   The backend API will be available at `http://127.0.0.1:8000`.

### Environment Variables

The frontend reads API base URL from environment:

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Base URL for the FastAPI backend |

**To override**: Set `VITE_API_BASE_URL` in a `.env` file or as a shell environment variable before running `npm run dev`.

### Build for Production

```bash
npm run build
```

The built files will be output to `icu-sentinel/frontend/dist/`.

### Preview Production Build

```bash
npm run preview
```

Runs the built production server locally for testing.

## Component Library

### Available Components

The frontend uses **lucide-react** for icons and **recharts** for charts.

| Component | Description |
|---|---|
| `AlertList` | Displays a list of active/sepsis alerts with risk scores and timestamps |
| `PatientTable` | Table of patients with split info, sepsis status, and action buttons |
| `Risk Gauge` | Visual gauge showing a patient's risk score (0–1) relative to the alert threshold |
| `SplitBadges` | Colored badges indicating which split (train/val/test) a patient belongs to |
| `DemoWarning` | Banner warning that the system is in demo mode (no real model loaded) |

### Component: `RiskGauge`

Renders a visual representation of a patient's sepsis risk score.

**Props**:
| Prop | Type | Description |
|---|---|---|
| `riskScore` | `number` | Raw model probability (0 to 1) |
| `threshold` | `number` | Alert threshold (model-specific, e.g., 0.78 for XGBoost) |
| `showAlert` | `boolean` | Whether to display "ALERT" if above threshold |
| `width` | `number` (optional) | Gauge width in pixels (default: 200) |
| `height` | `number` (optional) | Gauge height in pixels (default: 100) |

**Example usage**:
```tsx
<RiskGauge riskScore={0.82} threshold={0.78} showAlert />
// Renders: gauge at 0.82, shows "ALERT" since 0.82 > 0.78
```

### Component: `SplitBadge`

Renders a colored badge indicating a patient's split assignment.

**Props**:
| Prop | Type | Description |
|---|---|---|
| `split` | `"train" | "validation" | "test"` | The patient's split |
| `size` | `"sm" | "md" | "lg"` (optional) | Badge size |

**Example usage**:
```tsx
<SplitBadge split="test" size="md" />
// Renders: red "TEST" badge
```

## Page Components

### Dashboard Page

The main page showing patient overview, alerts, and model performance summary.

**Features**:
- Patient list with split badges and sepsis status
- Real-time risk predictions from the backend API
- Active alert list with acknowledge/clear functionality
- Model information (which model is loaded, demo vs real)
- AUPRC/AUROC summary charts (using recharts)

### Alerts Page

Dedicated page for managing sepsis alerts.

**Features**:
- List of all active alerts
- Acknowledge alert button (updates alert status)
- Clear alert button (marks alert as cleared)
- Filter by patient or severity
- Alert history (archived cleared alerts)

### Patient Page

Shows detailed information for a specific patient.

**Features**:
- Patient ID and split badge
- Timeline of vital signs and features
- Risk score and alert status
- Target label (`sepsis_within_6h`)
- Feature importance SHAP summary (if explainability is wired in)

## API Client

The frontend uses a simple API client module at `src/api/client.ts`:

```tsx
// Example: Get patient prediction
const prediction = await api.getPatientPredictions("p00001");
// Example: Acknowledge an alert
await api.acknowledgeAlert("alert_001");
```

**API client functions**:
- `api.getHealth()` — Check API health
- `api.getPatients(filters?)` — List patients with optional filters
- `api.getPatientPredictions(patientId)` — Get risk prediction for a patient
- `api.createAlert(alertData)` — Create a new alert
- `api.acknowledgeAlert(alertId)` — Acknowledge an alert
- `api.clearAlert(alertId)` — Clear an alert

**Base URL**: Configured via `VITE_API_BASE_URL` environment variable.

## TypeScript Types

Key TypeScript types are defined in `src/types/api.ts`:

```typescript
// Patient summary from API
export interface PatientSummary {
  patient_id: string;
  split: "train" | "validation" | "test";
  source_dataset: "set_a" | "set_b";
  has_sepsis: boolean;
}

// Prediction result from API
export interface PredictionResult {
  patient_id: string;
  risk_score: number;      // Raw model probability (0-1)
  sepsis_within_6h: boolean; // Above/below threshold
  threshold: number;       // Model-fixed threshold
  model_name: string;      // Loaded model name
  calibrated: boolean;     // Whether calibration applied
  alert: boolean;          // Whether risk_score > threshold
}

// Alert from API
export interface Alert {
  id: string;
  patient_id: string;
  type: "sepsis_risk";
  risk_score: number;
  threshold: number;
  timestamp: string; // ISO 8601
  status: "active" | "acknowledged" | "clear";
}
```

## Development Workflow

### Code Standards

- **TypeScript**: All source code is typed (`strict: true` in tsconfig)
- **Linting**: `npm run lint` (uses eslint)
- **Formatting**: `npm run format` (uses prettier)
- **Type checking**: `npm run typecheck` (run before commits)

### Testing

- **vitest**: Unit and integration tests for React components
  - `npm test` or `npm run test:unit` runs vitest
  - 18/18 tests pass (as of the last verification)
- **Jest/Pytest**: Backend Python tests are separate (see backend documentation)

### Adding a New Page

1. Create a new component at `src/pages/NewPage.tsx`
2. Add a route in `src/App.tsx` (if using router) or import directly
3. Add API calls as needed using the `api` client
4. Add TypeScript types if interacting with API data
5. Run `npm run test:unit` to verify no regressions

### Adding a New API Endpoint (Backend)

1. Add the route in `backend/p3/app.py`
2. Define the Pydantic model in `backend/p3/schemas.py`
3. Update the API client in `frontend/src/api/client.ts` (add new function)
4. Add TypeScript types in `frontend/src/types/api.ts` (if needed)
5. Run `npm run typecheck` to verify types
6. Run `npm test` to verify no frontend regressions

## Known Issues

- **LightGBM inference**: Not available in the current environment (Windows DLL issue). The frontend will show "Extra Trees" or "Demo model" depending on configuration.
- **No real-model SHAP**: Explainability features use the demo model; real-model SHAP not yet wired.
- **Demo mode banner**: When no model artifact is configured via env vars, a banner indicates demo mode.
- **CORS**: Ensure `ICU_SENTINEL_CORS_ORIGINS` includes the frontend dev server URL (`http://localhost:5173`) or production domain.

## Deployment Checklist

- [ ] Set `ICU_SENTINEL_P1_MODEL_DIR` and `ICU_SENTINEL_P1_MODEL_NAME` env vars for real model loading
- [ ] Set `ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT=false` (default; explicit opt-in for demo)
- [ ] Configure `ICU_SENTINEL_CORS_ORIGINS` for production domain
- [ ] Build frontend: `npm run build` in `icu-sentinel/frontend/`
- [ ] Verify `VITE_API_BASE_URL` points to production backend
- [ ] Run `npm run typecheck` and `npm run lint`
- [ ] Update `.env.example` with production values (see DEPLOYMENT.md)
- [ ] System labeled "Research prototype — not approved for clinical decision-making"

## References

- `P3_API_REFERENCE.md` — Auto-generated P3 API reference (if regenerated)
- `P3_ARCHITECTURE.md` — High-level P3 architecture diagram
- `P3_DEMO_GUIDE.md` — Demo mode walkthrough
- `P3_ALERTS.md` — Alert engine documentation