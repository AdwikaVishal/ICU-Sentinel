# P3 Explainable AI (SHAP)

**Project:** ICU Sentinel  
**Owner:** Person 3 (P3)  
**Status:** Implemented research prototype  
**Date:** 2026-11-04

---

## 1. What this module does

For a fitted model and a patient-hour input, P3 produces:

- a **per-patient explanation**: feature contributions, base value, model
  output, validation and an explicit status;
- a **global explanation**: mean absolute SHAP by feature and by clinical
  feature group over a background sample;
- a **model catalogue**: which registered models can be explained, with which
  method and why not when unsupported.

The service never fabricates SHAP values, never reuses global importance as a
local explanation, and never silently falls back to a different model.

---

## 2. The demonstration model

No P1/P2/P3 research artifact is required to run the demo. When no P1 artifact
is configured and the synthetic demo repository is active, the container
fits `demo_logistic_regression` (`demo-1.0.0`) on the synthetic demo patients.

This is a **real fitted scikit-learn `LogisticRegression` pipeline**. Its SHAP
values are genuine explanations of that fitted model. The model identity is
marked `is_synthetic: true` and source `synthetic_demo_model`. It is not a
clinically validated sepsis model.

A hand-written heuristic provider (`synthetic_heuristic`) is also supported for
provider-contract tests. It returns `status: unsupported` for explanations
because it has no fitted model.

---

## 3. Status contract

| Status | Meaning |
|---|---|
| `available` | A genuine SHAP explanation was produced and validated. |
| `unavailable` | The prediction or required input/background data is missing. |
| `unsupported` | The model family cannot be explained by the available exact explainers and/or has no background data. |
| `error` | Explainer construction or validation failed, or the input hash did not match the prediction. |

Every non-available status includes a human-readable `reason`. The API never
returns an empty contribution list with `status: available`.

---

## 4. Explainer selection

`ExplainerFactory` plans first, then builds lazily and caches by model
name/version.

| Model family | Method | Output space | Background |
|---|---|---|---|
| Linear classifier exposing `decision_function` | `linear_shap` (`shap.LinearExplainer`) | `decision_function_log_odds` | Required |
| scikit-learn tree ensemble | `tree_shap` (`shap.TreeExplainer`) | `probability` for classifiers | Not required |
| XGBoost / LightGBM / CatBoost booster | `tree_shap` | `probability` when supported, otherwise `raw_model_output` | Not required |
| Other model with `predict`/`predict_proba` | `kernel_shap` (`shap.KernelExplainer`) | `probability` | Required |

Rejection rules before any SHAP computation:

- no fitted model object;
- no ordered feature list;
- non-binary target;
- preprocessing changes the feature count, so contributions cannot be mapped
  back to original features;
- linear/kernel model without a background dataset;
- linear model without `decision_function`.

A rejection produces `unsupported` or `unavailable` with a reason. It never
produces placeholder values.

### Output-space detail

- **Linear SHAP** explains the estimator's `decision_function` — log-odds for
  logistic regression. This was verified against `decision_function` in the
  validation test.
- **scikit-learn tree SHAP** uses the library default output, which for
  classifier tree ensembles is a probability-like value matching
  `predict_proba`; the response labels this `probability`. Requesting
  `model_output="probability"` is rejected by SHAP for
  `tree_path_dependent`, so P3 does not pretend otherwise.
- **External boosters** try probability output and fall back to the raw model
  output, labelling whichever space was used.

The `output_space` field is shown in the API and UI so a log-odds contribution
is not mistaken for a probability change.

---

## 5. Validation

For each per-patient explanation:

1. The context input hash must equal `prediction.model_input_hash`. Otherwise
   the explanation is withheld (`error`) rather than attributed to a different
   input.
2. All model features must be present in the causal input.
3. SHAP values must be finite and have the expected shape.
4. Additivity is checked:

   ```text
   base_value + sum(shap_values) ≈ model_output
   ```

   within a small tolerance in the declared output space. The result is
   returned as `validation.passed`, `validation.additive_error` and the
   checked output space.

The demo-model smoke test reports additivity error on the order of `1e-16`.
Explanations are deterministic for the same model/input (the factory caches
the explainer).

Kernel SHAP uses `nsamples=256`, so it is approximate. Its additivity check
can legitimately fail for some inputs; the explanation then exposes the failed
validation instead of hiding it.

---

## 6. Feature metadata

`FeatureMapper` uses P1 metadata and the P1 feature schema to provide readable
names, units, categories and descriptions. Unknown features keep their
original name and are marked `metadata_available: false`; P3 does not invent a
clinical meaning.

---

## 7. Global explanations

`GlobalExplainer` computes mean absolute SHAP over the model bundle's
background sample and aggregates by clinical category. It returns:

- `features` ranked by `mean_abs_shap`;
- signed `mean_shap` for direction;
- `feature_groups` with feature counts.

If the bundle has no background data, the global explanation is `unavailable`
with a reason. Global importance describes **model reliance**, not clinical
causality.

---

## 8. What SHAP does and does not mean

SHAP answers: *“How did this fitted model's output change relative to its
baseline when these feature values were used?”*

It does **not** answer:

- whether the patient has sepsis;
- whether a feature causes sepsis;
- whether a treatment would help;
- whether the model is clinically valid.

Known limitations:

- Correlated features can split credit between them.
- Contributions are in the declared output space; log-odds contributions are
  not percentage-point probability changes.
- The background sample determines the baseline. A different synthetic
  background changes the base value.
- Kernel SHAP is sampled and approximate.
- The demo model is trained on synthetic patients, so its explanations
  describe a demonstration model, not clinical reasoning.
- SHAP cannot make an unvalidated model safe.

The UI states these limits and uses non-alarmist wording throughout.

---

## 9. Tests

`ml/p3_temporal/tests/test_explainability.py` covers:

- P1 units/readable names and unknown-feature handling;
- linear, tree and kernel explainer planning;
- missing-background `unavailable` states;
- unsupported unknown models;
- genuine additive SHAP for the demo model;
- explanation/prediction input-hash equality;
- contribution ordering and dense ranks;
- determinism;
- no fake SHAP for heuristic predictions;
- missing-hour and hash-mismatch withholding;
- unregistered-model `unsupported`;
- hourly comparison and global importance/grouping;
- catalogue explainability reporting.

Run:

```bash
python3 -m pytest ml/p3_temporal/tests/test_explainability.py -q -o addopts=''
```
