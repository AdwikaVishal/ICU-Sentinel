# P3 Alert Engine

**Project:** ICU Sentinel  
**Owner:** Person 3 (P3)  
**Status:** Implemented research prototype  
**Date:** 2026-11-04

---

## 1. Purpose and safety boundary

The alert engine turns one model score into a persisted **alert episode** using
one configurable threshold rule. It is a workflow aid for retrospective
research, not a diagnosis and not a clinically validated alarm.

Hard rules:

- The ground-truth `SepsisLabel` is never an alert input.
- The threshold is never selected from labels or test-set performance.
- The threshold is configuration, not a clinical decision limit.
- An alert status is an application workflow state, not a clinical finding.
- Synthetic episodes are marked `is_synthetic: true`.

---

## 2. Trigger rule

```text
triggered = (probability >= threshold)
```

The rule is inclusive at the threshold. Probabilities must be finite and in
`[0, 1]`; invalid values are rejected by schema validation.

An unavailable prediction (`available: false`, including a missing hour)
creates no alert and returns the existing episode unchanged, if any.

---

## 3. Threshold configuration

Defaults are demonstration values:

| Setting | Default |
|---|---|
| Rule ID | `sepsis_risk_threshold` |
| Threshold | `0.60` |
| Source | `demo_configuration` |
| Version | `demo-threshold-v1` |
| Deduplication window | `6` hours |
| Clear on below threshold | `true` |
| Clinically validated | `false` |

Environment overrides:

```bash
ICU_SENTINEL_ALERT_THRESHOLD=0.55
ICU_SENTINEL_ALERT_THRESHOLD_SOURCE=demo_configuration
ICU_SENTINEL_ALERT_THRESHOLD_VERSION=demo-threshold-v2
ICU_SENTINEL_ALERT_RULE_ID=sepsis_risk_threshold
ICU_SENTINEL_ALERT_DEDUP_WINDOW_HOURS=6
ICU_SENTINEL_ALERT_CLEAR_ON_BELOW=true
ICU_SENTINEL_ALERT_THRESHOLD_VALIDATED=false
```

`ICU_SENTINEL_ALERT_THRESHOLD_VALIDATED=true` must only be used after a genuine
validation process. The API always returns `clinically_validated` and a
disclosure string. The frontend renders “Not clinically validated” for the
default demo configuration.

Per-request threshold overrides are stored on the episode with
`threshold_source: request_override` and `threshold_version: request-override`,
so a retrospective review can see which rule produced the episode.

---

## 4. Lifecycle

```text
                  score >= threshold
        (no episode) ────────────────► ACTIVE
                                         │
                          acknowledge()  │
                                         ▼
                                   ACKNOWLEDGED
                                         │
                          clear(reason)  │
                                         ▼
                                      CLEARED
```

| Status | Meaning | Appears in |
|---|---|---|
| `ACTIVE` | Triggered and not yet reviewed. | Active list |
| `ACKNOWLEDGED` | Reviewed; still able to be updated or cleared. | History |
| `CLEARED` | Terminal. Manual clear, below-threshold clear, or superseded episode. | History |

Transitions:

- `ACTIVE → ACKNOWLEDGED` via `POST /alerts/{id}/acknowledge`.
- `ACTIVE|ACKNOWLEDGED → CLEARED` via `POST /alerts/{id}/clear`.
- Acknowledging an already acknowledged episode is idempotent.
- Clearing an already cleared episode is idempotent.
- A cleared episode cannot be acknowledged, re-opened or updated.
- A later above-threshold prediction after a clear creates a new episode.

Cleared episodes are retained; the store is not a “delete on clear” log.

---

## 5. Deduplication

One ongoing episode per `(patient_id, rule_id)`.

When a triggered prediction arrives:

1. Read the latest `ACTIVE` or `ACKNOWLEDGED` episode for that patient/rule.
2. If none exists, create one with `first_hour = last_hour = hour` and
   `occurrence_count = 1`.
3. If one exists:
   - If `hour < existing.last_hour` (out-of-order/backwards replay), return
     `action: unchanged`, `deduplicated: true`. Do not roll the episode
     backwards and do not create another episode.
   - If `hour - existing.last_hour <= dedup_window_hours`, update the existing
     episode: `last_hour = max(last_hour, hour)`, new probability,
     `peak_probability = max(peak, probability)`, increment
     `occurrence_count`. Return `action: updated`, `deduplicated: true`.
   - If the gap is **larger** than the deduplication window, clear the old
     episode with reason `superseded_by_new_episode` and create a new one.

So “no duplicate active alerts” is guaranteed by the engine, even when the UI
navigates backwards or re-evaluates the same hour.

The store also treats acknowledged episodes as ongoing for episode lookup, so
acknowledging does not cause the next score to create a duplicate. The active
list only shows `ACTIVE`.

### Concurrency caveat

Deduplication is enforced by a process-level lock in `AlertEngine` plus a
query for the current episode. The SQLite schema has indexes but no unique
partial constraint on active episodes. If the backend is scaled to multiple
worker processes sharing one database, add a database-level guard for strict
cross-process deduplication.

---

## 6. Clearing policies

Clearing is explicit and auditable (`cleared_at`, `cleared_reason`).

| Trigger | Reason recorded |
|---|---|
| Manual clear from UI/API | Caller-supplied reason, e.g. `manual_clear_from_alert_centre` |
| Score falls below threshold | `probability_below_threshold` |
| New episode after a gap | `superseded_by_new_episode` |

`clear_on_below_threshold` controls only the second row. When disabled, a
below-threshold score returns `action: none` and leaves the episode active.

The Alert Centre documents the enabled policy. Clearing an alert is not a
statement that the patient is safe.

---

## 7. Persistence

`AlertStore` uses SQLite (WAL mode) at `ICU_SENTINEL_ALERT_DB`, default
`data/p3_alerts.db`.

- The parent directory is created if needed.
- Episodes survive process restarts.
- Tests use temporary databases or `:memory:`.
- The store is thread-safe and shares one connection with
  `check_same_thread=False`, guarded by an `RLock`.

Recorded fields include: patient, rule, hour, first/last hour, probability,
peak probability, threshold/source/version, model name/version, prediction
source, status, timestamps, clear reason, occurrence count and synthetic flag.

---

## 8. API contract

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v3/alerts` | All episodes, optional `status`, `patient_id`, `limit`. |
| `GET` | `/api/v3/alerts/active` | `ACTIVE` episodes only. |
| `GET` | `/api/v3/alerts/history` | `ACKNOWLEDGED` and `CLEARED` episodes. |
| `GET` | `/api/v3/alerts/threshold` | Configured threshold metadata + disclosure. |
| `POST` | `/api/v3/alerts/evaluate` | Evaluate one prediction. |
| `GET` | `/api/v3/alerts/{id}` | One episode. |
| `POST` | `/api/v3/alerts/{id}/acknowledge` | Acknowledge. |
| `POST` | `/api/v3/alerts/{id}/clear` | Clear with optional `reason`. |

Evaluation actions: `created`, `updated`, `cleared`, `none`, `unchanged`,
`unavailable`.

Errors:

- unknown episode → `404 alert_not_found`
- invalid transition (acknowledge a cleared episode) → `409 invalid_alert_state`
- invalid probability/threshold → `422`

---

## 9. Tests

`ml/p3_temporal/tests/test_alerts.py` covers threshold validation,
configuration-only thresholds, inclusive triggering, episode windows, label
exclusion, create/update/out-of-order/gap behaviour, overrides, below-threshold
clearing and policy off, acknowledge/clear semantics, idempotency,
re-triggering after clear, persistence across reopen, filters and unavailable
predictions.

`ml/p3_temporal/tests/test_integration.py` covers replay → alert → explanation
consistency, the acknowledge/clear workflow and configurable-threshold effects.
`tests/test_p3_api.py` covers the HTTP lifecycle.

Run:

```bash
python3 -m pytest ml/p3_temporal/tests/test_alerts.py \
  ml/p3_temporal/tests/test_integration.py tests/test_p3_api.py -q -o addopts=''
```
