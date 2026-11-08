# ICU Sentinel — MASTER EXECUTION PROMPT Completion Audit

Evidence base: actual generated artifacts only (no re-training, no artifact modification, no inferred-from-code completion). Phases/items assessed against `docs/execution/FINAL_RESEARCH_REPORT.md` and the generated data/reports + artifacts.

## Phase-level classification

| Phase | Classification | Basis |
|---|---|---|
| Dataset validation & integrity | COMPLETE | `data/reports/dataset_integrity_report.md` — overall PASS; 40,336 patients, 1,552,210 rows (Set A 20,336 / Set B 20,000). |
| P1 feature pipeline & causal schema | COMPLETE | `data/features/patients/*.parquet` (40,336 files), `data/features/feature_summary.json` (994 features, 0 issues, causal=true), `data/features/feature_schema.json`, `data/features/target_policy.json`, `data/features/feature_config.json`. |
| P1 patient-level split | COMPLETE | `data/splits/split_manifest.json` (train 28,234 / val 4,034 / test 8,068; leakage_detected=false), plus `data/splits/split_manifest_fast.json`. |
| P1 classical ML training & test eval | COMPLETE (fast) | `data/reports/classical_ml_fast/training_summary.json` (config + 960 selected features) and `data/reports/logs/p1_train_fast_stdout.log` (test AUROC/AUPRC/F1/threshold for LR/RF/SVM). |
| P2 boosting training & test eval | COMPLETE (fast, LightGBM disabled) | `data/reports/p2_fast/p2_training_summary.json` (Extra Trees + XGBoost train/validation/test metrics, thresholds, Brier). |
| P3 temporal / alerting / explainability on real data | NOT STARTED | `ml/p3_temporal` exists but is synthetic/demo only; 42 non-explainability component tests passed; not wired to real P1/P2 artifacts; no real-patient evaluation. |
| Calibration (Brier/ECE) | PARTIAL | Brier present in P2 training_summary; ECE not computed; `calibration_method=sigmoid` configured in `P2Config` but no calibrator demonstrably fitted/applied on real data. |
| Bootstrap confidence intervals | NOT STARTED | No patient-level bootstrap artifacts or results found. |
| Lead-time distribution & alerts per patient-day | NOT STARTED | No alert lead-time or alerts-per-patient-day artifacts found. |
| Alert suppression & persistence | NOT STARTED | P3 `alert_engine` exists but only synthetic/demo; not evaluated on real data. |
| SHAP on actual trained models | NOT STARTED | `ml/p3_temporal/explainability` tests use `demo_model`; no SHAP applied to real P1/P2 trained models. |
| Feature ablation | NOT STARTED | No ablation artifacts or results found. |
| Set A / Set B evaluation (not hospitals) | PARTIAL | Source dataset tracked in schema/splits; report explicitly states Set A/B are partitions, not hospitals, and makes no hospital claims; but no per-source-dataset model performance evaluation exists. |
| Integration / API / full test suite | PARTIAL | P1 395/395 passed (prior). P3: 42 passed (non-explainability; explainability deselected). P2 test run blocked by `pytest_cov` `.coverage` permission error (WinError 5) — not re-run without coverage here. API not tested. |
| Final research report | COMPLETE (fast protocol) | `docs/execution/FINAL_RESEARCH_REPORT.md` written; documents fast-protocol empirical results only. |

## Verification items 1–13

### 1. Identical patient-level split manifests and comparable training cohorts (P1 vs P2)
**Classification: PARTIAL — manifest inconsistency on disk.**

Evidence:
- Full manifest `data/splits/split_manifest.json`: train 28,234 / val 4,034 / test 8,068 patients; P2 `split_check.leakage_detected=false`, every patient in exactly one split.
- P1 CLI used `--train-patients 6000`; P2 `training_metrics` confirm train `n_rows=228,703`, `n_positive=2,607` (≈6,000 patients), full validation `n_rows=157,069`/`n_positive=2,786` and test `n_rows=311,614`/`n_positive=5,553` — matching full-manifest val/test.
- Conflict: on-disk `data/splits/split_manifest_fast.json` header reads `n_patients_train=8000`, `n_records_train=306558`, `n_sepsis_positive_hours_train=5528`, `n_patients_total=20102`, which does NOT match P2's trained train rows (228,703) nor P1's `--train-patients 6000`. `FINAL_RESEARCH_REPORT.md` also incorrectly pairs "6,000 patients" with "306,558 rows" (306,558 rows is the 8,000-patient figure).
- Conclusion: P1 and P2 used comparable cohorts in intent (6,000 train; full val/test), but identical-manifest status cannot be fully confirmed from current on-disk artifacts without regenerating/reconciling `split_manifest_fast.json`. Do not alter artifacts; flag for remediation before full-cohort run.

### 2. Exact feature counts and names used by each model
**Classification: COMPLETE — counts and names documented; they differ between P1 and P2.**

Evidence:
- P1: `data/reports/classical_ml_fast/training_summary.json` — `feature_selection_report.n_features_input=994`, `n_features_output=960`, `n_removed=34` (all `removed_infinite` = `*_hours_since_last`); `selected_features` array lists 960 names. Models trained on 960 features.
- P2: `data/reports/p2_fast/p2_training_summary.json` — `dataset_summary.n_features=994`; `feature_names` array lists all 994 features (includes `*_hours_since_last`, which P1 removed); no feature selection performed. Models trained on 994 features (infinities sanitized in pipeline).
- Note: P1 (960 selected) ≠ P2 (994 unselected). This is a documented difference, not a missing artifact.

### 3. Target definition and eligible-hour filtering
**Classification: COMPLETE.**

Evidence:
- `data/features/feature_schema.json`: `sepsis_within_6h` target; `onset` = first hour `SepsisLabel==1`; 1 for onset..onset+5; 0 before onset and non-septic; NaN/ineligible at/after onset+6; original `SepsisLabel` unchanged; `target_eligible` flag.
- `data/features/feature_summary.json` `target_policy` and `target_summary`: horizon_hours=6; n_rows=1,552,210; n_eligible=1,541,845; n_ineligible=10,365; n_positive=17,551; n_negative=1,524,294; positive_rate_eligible=0.01138; `original_label_modified=false`.
- P2 `target_column=sepsis_within_6h`; `target_check.target_isolation_valid=true`.

### 4. AUPRC, sensitivity, specificity, precision, F1 and threshold-selection methodology
**Classification: COMPLETE for methodology + P2; P1 report table is partial.**

Evidence:
- Threshold selection: `threshold_selection_metric=f1`, `threshold_grid_step=0.01`, selection on validation split only (P1 config + P2 config). No test-data tuning.
- P2 (full metrics in `data/reports/p2_fast/p2_training_summary.json` test block):
  - Extra Trees: AUROC 0.7189, AUPRC 0.0309, sensitivity 0.0660, specificity 0.9879, precision 0.0587, F1 0.0622, threshold 0.56.
  - XGBoost: AUROC 0.7621, AUPRC 0.0347, sensitivity 0.1409, specificity 0.9724, precision 0.0551, F1 0.0792, threshold 0.69.
- P1: `data/reports/logs/p1_train_fast_stdout.log` / `FINAL_RESEARCH_REPORT.md` table gives test AUROC/AUPRC/F1/threshold only (LR 0.6794/0.0263/0.0604/0.83; RF 0.6894/0.0257/0.0582/0.54; SVM 0.6053/0.0155/0.0307/0.02). P1 test sensitivity/specificity/precision are not shown in the report table (may be inside the large `training_summary.json` but not extracted here).
- Methodology documented in both configs (`threshold_selection_metric`, `threshold_grid_step`, validation-only selection).

### 5. Calibration results, Brier score and ECE
**Classification: PARTIAL.**

Evidence:
- Brier present in P2 `training_summary.json` (e.g. XGBoost test brier=0.1331; Extra Trees test brier=0.1805; train/val also present).
- ECE: not computed anywhere.
- Calibration: `P2Config.calibration_method="sigmoid"` but `ml/p2_boosting/training.py` does not fit/apply a calibrator on real data (`P2ModelResult.calibrator` fields remain unused in training). P1 has no calibration.
- Conclusion: Brier scores exist for P2; ECE and applied calibration results do not.

### 6. Patient-level bootstrap confidence intervals
**Classification: NOT STARTED.**

Evidence: No bootstrap artifacts, confidence-interval outputs, or resampling code invoked on real data found in `data/reports/` or `artifacts/`.

### 7. Patient-level / event-level sensitivity
**Classification: PARTIAL.**

Evidence: P2 reports threshold-based sensitivity on rows (Extra Trees test sensitivity 0.0660; XGBoost 0.1409; `n_rows=309,559`, `n_positive=3,498`). This is hour/row-level, not explicitly patient-level or event-level sensitivity, and no separate patient-level/event-level sensitivity analysis exists.

### 8. Lead-time distribution and alerts per patient-day
**Classification: NOT STARTED.**

Evidence: No lead-time distribution or alerts-per-patient-day artifacts found; P3 replay is synthetic/demo only.

### 9. Alert suppression and persistence evaluated
**Classification: NOT STARTED.**

Evidence: `ml/p3_temporal/alerts/` exists but only synthetic/demo; no alert suppression/persistence evaluation on real P1/P2 predictions.

### 10. SHAP explanations use actual trained models
**Classification: NOT STARTED.**

Evidence: `ml/p3_temporal/explainability/` tests (e.g. `test_explainability.py`) use `demo_model`; no SHAP explanations generated from real trained P1/P2 models.

### 11. Feature ablation performed
**Classification: NOT STARTED.**

Evidence: No ablation study artifacts or results found.

### 12. Set A/B evaluation performed without calling them hospitals
**Classification: PARTIAL.**

Evidence: `feature_schema.json` and `split_manifest*.json` carry `source_dataset` (`set_a`/`set_b`); `data/reports/dataset_integrity_report.md` and `FINAL_RESEARCH_REPORT.md` state Set A/B are dataset partitions, not hospitals, with no hospital-level generalization claims. However, no per-source-dataset model performance evaluation (Set A vs Set B test metrics) exists. The "not hospitals" disclaimer is present; actual Set A/B evaluation is not.

### 13. Integration, API and full test-suite results
**Classification: PARTIAL.**

Evidence:
- P1: 395/395 tests passed (prior, exit 0).
- P3: 42 tests passed (`ml/p3_temporal`, non-explainability, `pytest -o addopts="" -p no:cov`); explainability tests deselected; integration tests exist but synthetic/demo.
- P2: `pytest ml/p2_boosting` failed at collection due to `pytest_cov` `.coverage` permission error (`WinError 5: Access is denied`); not re-run without coverage in this session (per instruction not to alter/run beyond what's needed).
- API: no API test results found.

## GO / NO-GO recommendation

**GO to proceed to full-cohort research evaluation; NO-GO for any clinical/deployment or definitive performance claims.**

Rationale: The end-to-end P1→P2 fast pipeline is validated on real data (features generated for all 40,336 patients; real train/validation/test splits; empirical test metrics produced for LR/RF/SVM and Extra Trees/XGBoost). The pipeline can be scaled to the full 28,234-patient training cohort.

Mandatory remediation before full-cohort research evaluation:
1. Reconcile `data/splits/split_manifest_fast.json` header (8,000 train / 306,558 rows) with the actual 6,000-patient train subsample (228,703 rows) used by P1/P2; regenerate one canonical fast manifest and confirm P1/P2 use identical patient-level splits.
2. Resolve LightGBM Windows DLL failure or document its exclusion; P2 currently lacks LightGBM.
3. Add calibration (Brier is present; add ECE + actual calibrator fit) and patient-level bootstrap confidence intervals.
4. Add lead-time distribution, alerts per patient-day, and alert suppression/persistence evaluation (P3 must be wired to real trained models).
5. Add SHAP on real trained models and feature ablation.
6. Add per-source-dataset (Set A vs Set B) model evaluation while retaining the "not hospitals" framing.
7. Re-run full test suite without coverage (clear the `.coverage` permission error) including P2 and API tests.
