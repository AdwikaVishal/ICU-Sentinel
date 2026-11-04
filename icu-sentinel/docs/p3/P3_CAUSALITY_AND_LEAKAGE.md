# P3 Temporal Causality and Leakage Tests

**Project:** ICU Sentinel  
**Owner:** Person 3 (P3)  
**Status:** Implemented and tested  
**Date:** 2026-11-04

---

## 1. The invariant

For a prediction at hour `t`:

> The model input may contain data from hours `<= t` only. No row, label,
> rolling statistic, forward-fill value, trend or hash from hour `> t` may
> influence the prediction.

This is a hard requirement, not a UI convention. It is enforced in the input
builder and verified by automated leakage tests.

---

## 2. Why features are built on a prefix, not sliced afterwards

A common leakage bug is:

```python
features = pipeline.transform_patient(full_stay)
row_t = features[features["hour"] == t]
```

That is wrong. P1's feature pipeline computes rolling means/maxima, trends,
hours-since-last-observation and derived scores. Computing them on the full
stay first lets future rows enter statistics at hour `t`, even if the final
row is sliced out.

P3 instead:

1. `CausalFeatureBuilder._prefix_patient()` builds a new `ProcessedPatient`
   containing only rows whose `hour <= t`.
2. It calls `FeaturePipeline.transform_patient()` on that **prefix**.
3. It selects the row for hour `t` from the resulting prefix features.

`ml/p3_temporal/replay/model_input.py` documents this rule at module level.

---

## 3. What is excluded from the input

`NON_FEATURE_COLUMNS` excludes:

- `patient_id`, `hospital_id`, `hour`, `source_file`, `source_dataset`
- `SepsisLabel` (the retrospective ground-truth label)

The label is attached to `PredictionContext.retrospective_label` for
retrospective display and evaluation only. It is:

- not in `feature_values` or `feature_names`;
- not in the input hash;
- not passed to the prediction provider;
- not accepted by `AlertEvaluationRequest`;
- never used to select an alert threshold.

`hash_prediction_input()` hashes only the causal feature values (or the
observation snapshot when no feature vector exists), plus patient ID and hour.
The hash is deliberately independent of the label.

---

## 4. Missing hours

A missing hour is not imputed into a normal value for prediction:

- `PredictionContext.observations` is empty;
- `feature_values` is `None`;
- the provider returns an explicit unavailable prediction (`missing_hour`);
- no alert is evaluated or created.

`observed_hours_used` contains only recorded hours `<= t`, which is visible in
the replay UI and model-input API.

---

## 5. Automated leakage tests

`ml/p3_temporal/tests/test_causality.py` contains the mandatory tests:

| Test | What it changes | Assertion |
|---|---|---|
| `test_hour_24_input_matches_prefix_only_recomputation` | Truncates the stay at hour 24 | Full-stay input hash and features equal a prefix-only recomputation; `prefix_row_count == 24`. |
| `test_every_future_mutation_leaves_hour_24_unchanged` | Replaces every observation after hour 24 with `-999`, `0`, `250`, `1e9` | Hour-24 probability, input hash and prediction record are unchanged. |
| `test_future_rows_do_not_enter_the_causal_input` | Appends a row at hour 300 | Input hash, observed hours and prefix row count are unchanged. |
| `test_future_labels_do_not_enter_the_causal_input` | Sets all labels to 0, then to 1 | Feature values and input hash are unchanged. |
| `test_gap_patient_only_uses_observed_prefix_rows` | Uses a patient with missing hours | Only recorded hours `<= 21` appear. |

Additional replay tests cover appended future rows, future label changes,
label exclusion from the feature vector and input hash changes when a *past*
observation changes.

These tests run in the normal suite:

```bash
python3 -m pytest ml/p3_temporal/tests/test_causality.py -q -o addopts=''
```

---

## 6. Runtime evidence in the API

Every replay response contains:

```json
{
  "causality": {
    "max_input_hour": 24,
    "prefix_row_count": 24,
    "input_observation_count": 24,
    "future_data_used": false,
    "retrospective_label_used_as_input": false,
    "model_input_hash": "…",
    "feature_schema_version": "p1-feature-schema-1.0.0"
  }
}
```

`GET /api/v3/replay/{patient_id}/model-input?hour=t` returns the exact feature
vector, input hash, observed hours and prefix size. `SepsisLabel` does not
appear in `feature_names` or `feature_values`; the retrospective label is a
separate top-level field with an explicit disclosure.

---

## 7. Explanation and alert paths

- `ExplanationService` fetches the model input at the same hour from
  `ReplayEngine.get_model_input()`.
- It refuses to explain when `context.input_hash != prediction.model_input_hash`
  and returns status `error`. This prevents explaining a different input than
  the one that produced the prediction.
- `AlertEngine.evaluate_prediction()` receives only the `PredictionResult`.
  It never queries patient labels or future rows.
- Navigating replay backwards does not create a new alert and cannot roll an
  existing episode backwards.

---

## 8. Limitations

- Causality is enforced at the model-input boundary. A future replacement
  model must consume `PredictionContext` (or an equivalent causal input) to
  inherit the guarantee.
- The demonstration model is trained on synthetic patients. This is a demo of
  the pipeline, not evidence of clinical validity.
- The P1 feature pipeline may use domain constants (for example normal ranges).
  Those are configuration constants, not future patient data.
