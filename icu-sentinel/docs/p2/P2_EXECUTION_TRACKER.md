# ICU Sentinel — P2 Execution Tracker

Status: IN PROGRESS. Matched P2 training launched (PID recorded below). All
other P2 deliverables pending or blocked.

Source of truth: actual generated artifacts only. No historical experiment is
overwritten.

## Canonical matched experiment (verified)

- Manifest: `icu-sentinel/data/splits/split_manifest_fast_6000.json`
- Manifest SHA256: `008a93f47c4264e3b61d73121be17efc1dae0f69473289d238d61bcc881188b1` (verified against P1 matched report — MATCH)
- P1 selected features: `icu-sentinel/data/reports/classical_ml_matched_6000/selected_features.json` — 960 features
- P1 feature-list SHA256: `c7407bfd3a11f257a99263cd2e7fd7018c898d848f53065ce59daf5a0562a782` (verified — MATCH)
- Target: `sepsis_within_6h` (unchanged); eligible-row filtering; patient-level split isolation confirmed in P1 matched summary.
- Train/validation/test patients: 6,000 / 4,034 / 8,068 (per P1 matched report).

## Task status

| # | Task | Status | Evidence / output path | Notes |
|---|------|--------|------------------------|-------|
| 1 | Matched P2 training (Extra Trees + XGBoost) | COMPLETE | `icu-sentinel/artifacts/p2/matched_6000/`, `icu-sentinel/data/reports/p2_matched_6000/` | Verified: manifest hash, feature hash, serialization, metrics |
| 2 | Matched P2 experiment report | NOT STARTED | `docs/p2/P2_MATCHED_EXPERIMENT_REPORT.md` | Depends on task 1 |
| 3 | Calibration evaluation (Platt/isotonic, ECE, reliability) | NOT STARTED | `data/reports/p2_calibration/`, `docs/p2/P2_CALIBRATION_REPORT.md` | Depends on task 1 |
| 4 | Patient-level bootstrap 95% CIs | NOT STARTED | `data/reports/p2_bootstrap/`, `docs/p2/P2_STATISTICAL_EVALUATION.md` | Depends on task 1 |
| 5 | Early-warning / alert-burden | NOT STARTED | `data/reports/p2_early_warning/`, `docs/p2/P2_EARLY_WARNING_REPORT.md` | Depends on task 1 |
| 6 | Set A / Set B subgroup | NOT STARTED | `data/reports/p2_set_ab/`, `docs/p2/P2_SET_AB_EVALUATION.md` | Depends on task 1 |
| 7 | Feature ablation | NOT STARTED | `data/reports/p2_ablation/`, `docs/p2/P2_FEATURE_ABLATION_REPORT.md` | Depends on task 1 |
| 8 | SHAP explainability | NOT STARTED | `data/reports/p2_shap/`, `docs/p2/P2_SHAP_EXPLAINABILITY.md` | Depends on task 1 |
| 9 | P1/P2 matched comparison | NOT STARTED | `docs/p2/P1_P2_MATCHED_COMPARISON.md` | Depends on tasks 1-2 |
| 10 | Final P2 research report | NOT STARTED | `docs/p2/P2_FINAL_RESEARCH_REPORT.md` | Depends on all above |

## In-flight process

- P2 matched training PID: 8296 (launched 2026-11-08; COMPLETED 495.7s)
- Verification: reproducibility/fast serialization check PASSED
  (see verify_p2_matched_fast.py). Both models load, 960 features match P1,
  test metrics finite and in range.
- Command: `python icu-sentinel/scripts/train_p2_matched_6000.py --artifact-dir icu-sentinel/artifacts/p2/matched_6000 --reports-dir icu-sentinel/data/reports/p2_matched_6000`
- Config: Extra Trees 200 est, XGBoost 200 est, LightGBM disabled, seed 42, n_jobs=safe_thread_default (14), threshold selection F1 grid 0.01 on validation only.

## Blocked / incomplete

None yet pending training result.
