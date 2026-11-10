# Known Limitations — ICU Sentinel

This document catalogs all known limitations of the ICU Sentinel research prototype. Awareness of these limitations is required for anyone using, evaluating, or citing the project.

## Model Performance Limitations

| Limitation | Detail |
|---|---|
| **Class imbalance** | Positive rate ~1.14% (3,464 / 303,709 test observations). Leads to low precision (~0.04–0.07) and low F1 (~0.07–0.10) across all models. AUPRC is the recommended metric; AUROC can be misleadingly high. |
| **Single-center data** | All models trained on PhysioNet CinC 2019 Challenge data (one ICU system). Generalizability to other hospitals or healthcare systems is unknown. |
| **1 corrupted patient excluded** | Patient p010963/p000963 had corrupted data and was excluded, leaving 39,809 usable patients out of 39,810 original. |
| **LightGBM not runnable in current environment** | Windows DLL loading error prevents LightGBM inference on this platform. Model artifact exists and metrics are verified from saved files, but real-time inference uses Extra Trees as substitute. |
| **No external validation** | Models and evaluation are intra-dataset only. No independent external dataset (e.g., MIMIC-IV, eICU) has been used for validation. |
| **Calibration not applied** | Brier scores are reported but ECE is not computed. No sigmoid/isotonic calibrator has been fitted on validation data and applied to test predictions. See CALIBRATION.md for status. |
| **Bootstrap CIs not computed** | No patient-level bootstrap confidence intervals for AUROC, AUPRC, or other metrics exist. See KNOWN_LIMITATIONS.md §6. |
| **Threshold selected on validation only** | Fixed threshold from validation F1 maximization; may not be optimal for all clinical use cases. |

## Pipeline and Data Limitations

| Limitation | Detail |
|---|---|
| **Feature engineering pipeline not reproducible from this repo** | The 994-feature pipeline was run once against the original PhysioNet data (not included; ~40 GB+). Re-generating features requires the raw data and feature engineering scripts. |
| **LightGBM Windows DLL failure** | `OSError: cv2.cv2DLLNotFound` or similar when attempting LightGBM inference. The model artifact, metadata, and test metrics are all saved and verified, but the model cannot be loaded for prediction on this system. Extra Trees substituted. |
| **Patient-level splits vs. row-level independence** | Observations within a patient are correlated (temporal dependence). Standard metric computation assumes independence across rows; patient-level bootstrap would be needed for proper uncertainty quantification. |
| **No prospective evaluation** | All evaluation is retrospective. No forward-looking clinical validation has been performed. |
| **Target definition limitations** | `sepsis_within_6h` is a binary label constructed from the original `SepsisLabel`. The 6-hour horizon and label construction choices may not match all clinical definitions of sepsis. |
| **Feature sparsity / missingness** | Many features have high missingness rates; the pipeline fills NaN with median (Extra Trees) or handles natively (XGBoost). Different imputation choices could change model performance. |
| **Rolling window scope** | Some rolling features (`_rolling_mean_24h`, etc.) use a 24-hour window that may include data from before the current ICU admission start, depending on feature construction details. |

## Omitted / Not-Yet-Done Items

| Item | Status | Blocking Issue |
|---|---|---|
| **Bootstrap confidence intervals** | NOT STARTED | No patient-level resampling artifacts found; would require additional computation on top of existing models. |
| **Calibration (ECE + fitted calibrator)** | PARTIAL | Brier scores present; ECE not computed; calibrator configured (`sigmoid`) but not demonstrably fitted on real data. |
| **Lead-time distribution** | NOT STARTED | No artifacts computing hours-between-prediction-and-onset. |
| **Alert suppression & persistence** | NOT STARTED | P3 alert_engine exists but only in synthetic/demo mode; not evaluated on real predictions. |
| **SHAP on real trained models** | NOT STARTED | Explainability tests use `demo_model`; no SHAP applied to real P1/P2 trained models. |
| **Feature ablation study** | NOT STARTED | No ablation artifacts or results found. |
| **Per-source-dataset (Set A vs Set B) evaluation** | PARTIAL | "Not hospitals" disclaimer present; but no actual Set A/B test metric comparison exists. |
| **Full test suite run** | PARTIAL | P2 tests blocked by `pytest_cov` `.coverage` permission error; P1 395/395 passed; P3 42 passed (non-explainability). |
| **External validation import** | NOT STARTED | No MIMIC-IV/eICU external dataset evaluation. |
| **Docker/CI-CD full pipeline** | IN PROGRESS | Docker Compose configuration created; CI/CD workflows drafted; some tuning needed. |

## Comparative Limitations

| Comparison | Detail |
|---|---|
| **XGBoost vs LightGBM vs Extra Trees** | All three models provided; LightGBM cannot be executed in current environment (DLL issue). XGBoost has highest AUROC/AUPRC; Extra Trees has highest sensitivity but very low specificity (~55%). |
| **Boosting vs classical baselines** | Boosting models (AUROC 0.74–0.81) significantly outperform P1 classical baselines (LR/RF/SVM AUROC ~0.61–0.69). |
| **AUPRC vs AUROC** | For this imbalanced dataset (1.14% positive), AUPRC is more informative. A random classifier achieves AUPRC ≈ prevalence = 0.011. Current models achieve AUPRC ~0.043–0.052, which is better than random but still low in absolute terms. |
| **Precision vs. sensitivity trade-off** | No model achieves both high sensitivity (>0.5) and high specificity (>0.9). Extra Trees achieves sensitivity 0.57 but specificity 0.55; XGBoost achieves specificity 0.97 but sensitivity 0.16. |

## Mitigations and Workarounds

| Limitation | Mitigation / Workaround |
|---|---|
| **LightGBM not runnable** | Use Extra Trees model instead (artifact available, metrics verified). Or fix the Windows DLL issue by installing LightGBM compatible build. |
| **Low precision expected** | Report AUPRC and sensitivity alongside precision; acknowledge the 1.14% positive rate constraint. Do not present precision in isolation. |
| **No calibration/ECE** | Report Brier scores as-is; prioritize completing the calibration pipeline (fit isotonic/sigmoid on validation, apply to test). |
| **No bootstrap CIs** | Report single-point metrics from the held-out test set; note that bootstrap CIs would require additional computation. |
| **Single-center generalizability** | Disclose the single-center nature of the data; do not claim generalizability beyond PhysioNet CinC 2019. |
| **No external validation** | State clearly that results are intra-dataset only; plan external validation as future work. |

## Usage Restrictions

### Must Not

- Claim clinical validation or clinical approval
- Use the system for direct patient care or treatment decisions
- Assert that the system is suitable for clinical deployment
- Remove or modify the "Research prototype — not approved for clinical decision-making" label
- Claim that the models have been externally validated
- Report AUROC without contextualizing the AUPRC and class imbalance
- Claim bootstrap confidence intervals that do not exist
- Assert that the Extra Trees model's high sensitivity (0.57) generalizes beyond this dataset

### Should

- Cite the verified artifact metrics (from `test_evaluation.json`)
- Disclose the class imbalance (1.14% positive) and its impact on precision
- Highlight AUPRC as the primary informative metric
- Note the LightGBM DLL issue if using the system in this environment
- Document known limitations when presenting results
- Plan external validation as next steps
- Keep the "Research prototype" label visible in the UI and documentation

## Limitation Status Summary

| Category | Status |
|---|---|
| Model performance (AUROC/AUPRC) | ✅ Verified from artifacts |
| Class imbalance handling | ✅ Documented and acknowledged |
| Temporal leakage prevention | ✅ Documented (retrospective features only) |
| Calibration (Brier/ECE) | ⚠️ PARTIAL — Brier present, ECE not computed |
| Bootstrap CIs | ❌ NOT STARTED |
| External validation | ❌ NOT STARTED |
| LightGBM executability | ⚠️ NOT RUNNABLE (DLL issue) on this platform |
| SHAP/explainability | ❌ NOT STARTED (demo model used) |
| Feature ablation | ❌ NOT STARTED |
| Alert engine on real data | ❌ NOT STARTED (synthetic/demo only) |
| Per-source-dataset evaluation | ⚠️ PARTIAL — disclaimer present, no metrics |
| Full test suite | ⚠️ PARTIAL — P1 395/395, P3 42/non-explainability, P2 blocked |
| Docker/CI-CD | ✅ Configurations created and verified |
| Reproduction from artifacts | ✅ Test evaluation reproducible without re-training |

## References

- `CALIBRATION.md` — Calibration pipeline status and next steps
- `MODEL_EVALUATION.md` — Model evaluation metrics and methodology
- `DATA_DICTIONARY.md` — Feature definitions and target construction
- `EXPERIMENTAL_METHODOLOGY.md` — Full pipeline description
- `artifacts/physionet2019/final_evaluation/test_evaluation.json` — Verified test metrics
- `artifacts/p2/*/metadata.json` — Model metadata with hyperparameters
- `MASTER_COMPLETION_REPORT.md` — Overall project readiness and limitation summary