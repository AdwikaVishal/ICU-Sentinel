# Target Definition — ICU Sentinel

This document defines the prediction target, eligibility criteria, and label construction for the sepsis early-warning task.

## Prediction Target

| Attribute | Value | Description |
|---|---|---|
| `target_name` | `sepsis_within_6h` | Binary indicator: 1 if sepsis will occur within 6 hours, 0 otherwise. |
| `prediction_horizon` | 6 hours | The look-ahead window from the start of the feature observation period. |
| `positive_rate` | ~1.14% | Approximately 1.14% of eligible observations are positive (17,551 / 1,541,845). |

## Target Construction

1. **Onset detection**: `onset` = the first hour index where the original `SepsisLabel == 1` in the PhysioNet dataset.
2. **Window labeling**: For a feature window starting at hour `t`:
   - `sepsis_within_6h = 1` if there exists an hour `t' ∈ [t, t+6)` such that `SepsisLabel[t'] == 1`
   - `sepsis_within_6h = 0` if no sepsis label is present in the 6-hour window starting at `t`
3. **Ineligible cases**: If `onset + 6 <= total_hours_in_stay` is false (i.e., the patient is within 6 hours of death/discharge or the end of the record), the observation is marked `target_eligible = 0` and the target is set to NaN (not applicable).

## Eligibility Filtering

| Condition | `target_eligible` | `sepsis_within_6h` |
|---|---|---|
| Patient is within the first `onset + 6` hours of stay | 0 | NaN |
| Patient has a valid SepsisLabel in the window | 1 | 0 or 1 (as computed above) |
| Patient is beyond onset+6 hours | 1 | 0 (no future sepsis possible) |

**Key metric**: `n_eligible = 1,541,845` out of `n_rows = 1,552,210` total rows. 10,365 rows are ineligible.

## Original vs Modified Label

| Field | Description |
|---|---|
| `SepsisLabel` | The raw PhysioNet sepsis label (unmodified). |
| `sepsis_within_6h` | The modified binary target used for model training and evaluation. The original label is preserved alongside the modified target for auditability. |

## Verification

- `data/features/feature_schema.json` documents `target_column=sepsis_within_6h` and `target_eligible` flag.
- `data/features/feature_summary.json` `target_summary` confirms: `n_eligible=1,541,845`, `n_positive=17,551`, `n_negative=1,524,294`, `positive_rate_eligible=0.01138`, `original_label_modified=false`.
- P2 `target_check.target_isolation_valid=true` confirms the target is properly isolated from leakage sources.

## Class Imbalance Considerations

With a positive rate of 1.14%, the dataset is heavily imbalanced. Key implications:
- **AUROC** can be misleadingly high (a trivial classifier achieves ~0.5 AUROC by chance).
- **AUPRC** is the more informative metric — a random classifier achieves AUPRC = prevalence ≈ 0.011.
- **Precision** at any threshold will be low (~0.05–0.07 in current models), which is expected and documented.
- **F1 score** is also low (~0.06–0.09) reflecting the imbalance.
- Threshold selection uses `threshold_selection_metric=f1` with `threshold_grid_step=0.01`, optimized on the **validation split only** (never on test data).

## Citation

This target definition follows the PhysioNet/CinC 2019 Sepsis Challenge guidelines and the ICU Sentinel feature engineering protocol (see `FEATURE_ENGINEERING_DOCUMENTATION.md` and `CLASSICAL_ML_EVALUATION_PROTOCOL.md`).