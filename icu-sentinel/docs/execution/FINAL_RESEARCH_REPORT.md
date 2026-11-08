# ICU Sentinel — Final Empirical Research Report

Status: complete end-to-end P1/P2 fast evaluation on real PhysioNet/CinC 2019 data. P3 components are synthetic/demo only and were not trained or evaluated on real data.

## Scope and data contract
- Source: PhysioNet/CinC 2019 Set A and Set B `.psv` files, complete and validated.
- Dataset validation: PASS (`data/reports/dataset_integrity_report.md`).
- Patients: 40,336 total (Set A 20,336; Set B 20,000).
- Rows: 1,552,210 hourly rows.
- Septic patients: 2,932 (Set A 1,790 at 8.80%; Set B 1,142 at 5.71%).
- Positive hourly labels overall: 27,916.
- Set A and Set B are dataset partitions, not hospitals. No hospital-level generalization claims are made.

## Approved target policy
- `onset`: first hour with `SepsisLabel == 1`.
- `sepsis_within_6h`: 1 for hours `[onset, onset + 5]`; 0 before onset and for non-septic stays; ineligible at/after `onset + 6`.
- Original `SepsisLabel` is preserved.
- P1/P2 target used: `sepsis_within_6h`.

## Feature pipeline (P1)
- Full streaming feature generation completed: 40,336 patient parquet files.
- Rows: 1,552,210; model features: 994; parquet columns including identifiers/targets: 1,000.
- Validation issues: 0; causal check passed.
- Full feature schema/target policy written:
  - `data/features/feature_schema.json`
  - `data/features/feature_config.json`
  - `data/features/target_policy.json`
  - `data/features/feature_summary.json`

## Patient-level split (full cohort)
- No patient overlap between train/validation/test.
- Train: 28,234 patients / 1,083,527 rows / 2,053 septic patients / 19,577 positive hours.
- Validation: 4,034 patients / 157,069 rows / 293 septic patients / 2,786 positive hours.
- Test: 8,068 patients / 311,614 rows / 586 septic patients / 5,553 positive hours.

## Compute-constrained fast protocol
These limits were chosen for runtime/memory, not tuned on data:
- Train subsample: seeded patient-level sepsis-stratified 8,000 of 28,234 train patients (306,558 raw records; 304,511 eligible train rows; 3,481 eligible positive hours). [Corrected: P1 used 8,000, not 6,000.]
- Validation and test: full splits (validation 4,034 patients / 156,036 eligible rows / 1,753 positives; test 8,068 patients / 311,614 rows).
- Random seed: 42.
- No test-data tuning or threshold selection on test.

## P1 classical ML results (real data)
Artifacts/reports: `artifacts/p1/classical_ml_fast/` and `data/reports/classical_ml_fast/`.
- Feature selection: 994 input -> 960 output features (34 infinite `*_hours_since_last` removed; `fit_on_train_only=True`).
- Models: logistic regression (max_iter 300), random forest (30 trees, depth 8, `n_jobs=-1`), SVM (RBF, probability, 4,000-row cap).
- Test metrics at selected validation threshold (full test sensitivity/specificity/precision/Brier/rows/positives not persisted; see `docs/execution/RESEARCH_INTEGRITY_REMEDIATION.md`):

| Model | Test AUROC | Test AUPRC | Test F1 | Test threshold |
|---|---:|---:|---:|---:|
| Logistic regression | 0.7025 | 0.0296 | 0.0694 | 0.83 |
| Random forest | 0.7041 | 0.0274 | 0.0660 | 0.56 |
| SVM | 0.6300 | 0.0209 | 0.0481 | 0.02 |

Notes: logistic regression did not converge within 300 iterations (`lbfgs` iteration limit reached); this is reported, not suppressed. SVM train was subsampled to 4,000 rows (`subsampled=True`). All metrics are empirical test-set results; none were tuned on test data. P1 train cohort (8,000 patients) differs from P2 train cohort (~6,000 patients); matched comparison requires cohort reconciliation (see remediation doc).

## P2 boosting results (real data, LightGBM disabled)
Artifacts/reports: `artifacts/p2_fast/` and `data/reports/p2_fast/`.
- LightGBM disabled due to Windows native-DLL failure; only Extra Trees and XGBoost evaluated.
- Fast manifest same as P1 (6,000 train patients; full validation/test).
- Extra Trees: 80 estimators, max depth 10, `n_jobs=8`.
- XGBoost: 100 estimators, max depth 4, `n_jobs=8`, `tree_method=hist`.
- Validation thresholds selected on validation split only using F1.
- Test metrics at validation threshold:

| Model | Validation AUROC | Validation AUPRC | Validation F1 | Test AUROC | Test AUPRC | Test F1 | Test threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Extra Trees | 0.734 | 0.045 | 0.090 | 0.719 | 0.031 | 0.062 | 0.560 |
| XGBoost | 0.779 | 0.045 | 0.107 | 0.762 | 0.035 | 0.079 | 0.690 |

XGBoost had the highest empirical test AUROC (0.762) among evaluated models. All P2 metrics are one-shot test evaluations at validation-selected thresholds; no test-data tuning.

## P3 temporal components
- `ml/p3_temporal` contains alerting, replay, explainability, and causality components.
- P3 tests are synthetic/demo only: 42 non-explainability component tests passed; explainability tests deselected in this fast run.
- P3 is NOT wired to real P1/P2 model artifacts and was NOT evaluated on real patient data. No P3 clinical performance claims are made.

## Limitations
- Fast protocol used only 6,000 of 28,234 training patients due to runtime/memory constraints; full-train performance may differ.
- LightGBM unavailable on this Windows environment.
- P2 default Extra Trees/XGBoost configs were reduced for speed and are not tuned.
- Results are early-warning research prototypes, not clinical products; prospective validation is required.
- No test data was used for model selection, threshold tuning, or calibration.

## Artifacts
- P1 fast: `artifacts/p1/classical_ml_fast/`, `data/reports/classical_ml_fast/`
- P2 fast: `artifacts/p2_fast/`, `data/reports/p2_fast/`
- Features: `data/features/patients/*.parquet`, `data/features/feature_summary.json`
- Splits: `data/splits/split_manifest.json` (full), `data/splits/split_manifest_fast.json` (fast train subsample)
- Audit: `docs/execution/EXECUTION_AUDIT.md`
