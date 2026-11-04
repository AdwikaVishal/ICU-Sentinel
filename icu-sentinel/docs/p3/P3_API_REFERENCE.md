# P3 API Reference

**Base URL (development):** `http://127.0.0.1:8000`  
**OpenAPI:** `http://127.0.0.1:8000/docs`  
**Prefix:** `/api/v3`

All responses are retrospective research output. Predictions, explanations and
alerts are not diagnoses.

---

## Meta

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness. |
| `GET` | `/api/v3/health` | Versioned liveness. |
| `GET` | `/api/v3/meta` | Service, provider, data source and synthetic flag. |

```json
{
  "service": "icu-sentinel-p3",
  "version": "3.0.0",
  "provider": {
    "available": true,
    "provider_name": "model:demo_logistic_regression",
    "model": {
      "name": "demo_logistic_regression",
      "version": "demo-1.0.0",
      "source": "synthetic_demo_model",
      "is_synthetic": true
    }
  },
  "data_source": "synthetic_demo",
  "is_synthetic": true,
  "disclaimer": "Retrospective research prototype. …"
}
```

---

## Replay

| Method | Path | Notes |
|---|---|---|
| `GET` | `/patients` | List patients with provenance. |
| `GET` | `/patients/{patient_id}` | Patient summary. |
| `GET` | `/patients/{patient_id}/timeline` | Every recorded/missing hour. |
| `GET` | `/provider/status` | Active provider and model identity. |
| `GET` | `/provider/threshold` | Threshold metadata used by replay alerts. |
| `GET` | `/replay/{patient_id}?hour=t&evaluate_alerts=false` | State at hour `t`; omitted hour starts at the first hour. |
| `GET` | `/replay/{patient_id}/model-input?hour=t` | Exact causal feature vector and hash. |
| `POST` | `/replay/{patient_id}/step` | `{ "hour": t, "direction": "forward"|"backward", "evaluate_alerts": true }` |
| `POST` | `/replay/{patient_id}/jump` | `{ "hour": t, "evaluate_alerts": true }` |
| `POST` | `/replay/{patient_id}/reset` | Jump to first hour. |
| `POST` | `/replay/{patient_id}/end` | Jump to last hour. |

`ReplayState` highlights:

```json
{
  "hour": 20,
  "prediction": {
    "probability": 0.99977,
    "available": true,
    "source": "synthetic_demo_model",
    "model": { "name": "demo_logistic_regression", "is_synthetic": true },
    "model_input_hash": "…"
  },
  "observations": [
    {
      "parameter": "heart_rate",
      "display_name": "Heart rate",
      "value": 104,
      "unit": "bpm",
      "available": true
    }
  ],
  "alert": null,
  "alert_status": "not_evaluated",
  "retrospective_label": 0,
  "retrospective_label_available": true,
  "causality": {
    "max_input_hour": 20,
    "prefix_row_count": 20,
    "future_data_used": false,
    "retrospective_label_used_as_input": false
  },
  "prediction_status": "available",
  "explanation_available": true,
  "disclaimer": "Retrospective research prototype. …"
}
```

Missing hours return `prediction.available: false`,
`prediction.unavailable_reason` and empty observations. They are never rendered
as normal values.

`GET /replay/{id}/model-input` returns `feature_names`, `feature_values`,
`input_hash`, `observed_hours_used`, `prefix_row_count` and a separate
retrospective label. `SepsisLabel` is not a feature.

Errors:

- unknown patient → `404 patient_not_found`
- hour outside the timeline → `400 hour_out_of_range`
- causal features unavailable → `422 model_input_unavailable`

---

## Explanations

| Method | Path | Notes |
|---|---|---|
| `GET` | `/explanations/models` | Catalogue with explainability and method. |
| `GET` | `/explanations/global?model_name=&model_version=` | Mean abs SHAP and groups. |
| `GET` | `/explanations/features/{feature_name}` | Readable metadata. |
| `GET` | `/explanations/{patient_id}/{hour}` | Per-patient SHAP. |
| `GET` | `/explanations/{patient_id}/compare?hour_a=&hour_b=` | Signed contribution changes. |

`PatientExplanation`:

```json
{
  "status": "available",
  "patient_id": "DEMO-0004",
  "hour": 20,
  "model": { "source": "synthetic_demo_model", "is_synthetic": true },
  "explanation_method": "linear_shap",
  "output_space": "decision_function_log_odds",
  "base_value": -1.23,
  "model_output": 8.42,
  "prediction_probability": 0.99977,
  "contributions": [
    {
      "feature_name": "Lactate_rolling_mean_12h",
      "display_name": "Lactate (12 h mean)",
      "shap_value": 0.81,
      "magnitude": 0.81,
      "direction": "increases_output",
      "rank": 1
    }
  ],
  "validation": {
    "passed": true,
    "additive_error": 4.4e-16,
    "checked_output_space": "decision_function_log_odds"
  },
  "input_hash": "…",
  "reason": null
}
```

Non-available statuses:

```json
{
  "status": "unsupported",
  "reason": "SHAP explanations require a fitted model. …",
  "contributions": []
}
```

Statuses: `available`, `unavailable`, `unsupported`, `error`.
Output spaces: `decision_function_log_odds`, `probability`, `raw_model_output`.

---

## Alerts

| Method | Path | Notes |
|---|---|---|
| `GET` | `/alerts?status=&patient_id=&limit=` | Filtered episodes. |
| `GET` | `/alerts/active?limit=` | `ACTIVE` only. |
| `GET` | `/alerts/history?limit=` | `ACKNOWLEDGED` + `CLEARED`. |
| `GET` | `/alerts/threshold` | Threshold + disclosure. |
| `POST` | `/alerts/evaluate` | Evaluate one prediction. |
| `GET` | `/alerts/{alert_id}` | One episode. |
| `POST` | `/alerts/{alert_id}/acknowledge` | Acknowledge. |
| `POST` | `/alerts/{alert_id}/clear` | `{ "reason": "…" }` optional. |

Evaluate body:

```json
{
  "patient_id": "DEMO-0004",
  "hour": 20,
  "probability": 0.99,
  "model_name": "demo_logistic_regression",
  "model_version": "demo-1.0.0",
  "prediction_source": "synthetic_demo_model",
  "is_synthetic": true
}
```

There is no label field. A per-request `threshold` is allowed and is recorded
as an override.

Evaluation result actions: `created`, `updated`, `cleared`, `none`,
`unchanged`, `unavailable`. `deduplicated` is true when an ongoing episode
absorbed the prediction.

`AlertThresholdInfo` always includes:

```json
{
  "rule_id": "sepsis_risk_threshold",
  "threshold": 0.6,
  "source": "demo_configuration",
  "version": "demo-threshold-v1",
  "dedup_window_hours": 6,
  "clear_on_below_threshold": true,
  "clinically_validated": false,
  "disclosure": "Configured demonstration threshold. Not a clinically validated decision threshold and not a diagnosis."
}
```

Errors:

- unknown alert → `404 alert_not_found`
- invalid lifecycle transition → `409 invalid_alert_state`
- invalid probability/threshold → `422`
- unknown patient/hour in replay paths → `404` / `400`

All errors use:

```json
{ "error": "code", "detail": "human readable message" }
```
