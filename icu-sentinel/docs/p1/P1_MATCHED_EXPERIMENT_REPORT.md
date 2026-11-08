# ICU Sentinel — P1 Matched Experiment Report (6,000-Patient Cohort)

Status: complete. Matched P1 baseline trained and evaluated on the canonical 6,000-patient fast cohort for like-for-like comparison with P2.

## Experiment summary

| Item | Value |
|---|---|
| Training cohort | 6,000 stratified patients (canonical manifest) |
| Validation cohort | 4,034 patients (full split) |
| Test cohort | 8,068 patients (full split) |
| Unique patients | 18,102 (no overlap across splits) |
| Target | `sepsis_within_6h` (strict 6h pre-onset) |
| Target isolation | valid (no leak into features) |
| Split leakage | none detected |
| Feature selection | training-only; 994 → 960 features (34 removed) |
| Removed features | all `*_hours_since_last` (infinite on never-observed) |
| Models trained | Logistic Regression, Random Forest, SVM |
| Random seed | 42 |
| Source | real (PhysioNet/CinC 2019) |

## Canonical split manifest

- Path: `data/splits/split_manifest_fast_6000.json`
- SHA256: `008a93f47c4264e3b61d73121be17efc1dae0f69473289d238d61bcc881188b1`
- This manifest was generated as a **separate file**; the existing `split_manifest.json` and `split_manifest_fast.json` were **not** modified.

| Split | Patients | Records | Sepsis patients | Positive hours |
|---|---:|---:|---:|---:|
| Train | 6,000 | 230,232 | 436 | 4,136 |
| Validation | 4,034 | 157,069 | 293 | 2,786 |
| Test | 8,068 | 311,614 | 586 | 5,553 |

## Reproducibility command

```bash
python scripts/train_p1_matched_6000.py \
  --manifest data/splits/split_manifest_fast_6000.json \
  --artifact-dir artifacts/p1/classical_ml_matched_6000 \
  --reports-dir data/reports/classical_ml_matched_6000 \
  --seed 42
```

The script consumes the canonical manifest directly (it does not regenerate or overwrite any existing split manifest). It calls `run_training_protocol` with `source="real"`, trains LR/RF/SVM, selects thresholds on validation only (F1, grid 0.01), and evaluates frozen models once on test.

## Test results (one-shot evaluation at validation-selected threshold)

Test set: 8,068 patients, 309,559 eligible rows, 3,498 positive hours.

| Model | Test AUROC | Test AUPRC | Test sensitivity | Test specificity | Test precision | Test F1 | Test Brier | Test threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.6794 | 0.0263 | 0.1655 | 0.9506 | 0.0369 | 0.0604 | 0.1819 | 0.83 |
| Random forest | 0.6894 | 0.0257 | 0.1286 | 0.9624 | 0.0376 | 0.0582 | 0.1319 | 0.54 |
| SVM (4,000-row cap) | 0.6053 | 0.0155 | 0.1241 | 0.9204 | 0.0175 | 0.0307 | 0.0112 | 0.02 |

Confusion matrices (test):

| Model | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| Logistic regression | 579 | 15,108 | 290,953 | 2,919 |
| Random forest | 450 | 11,507 | 294,554 | 3,048 |
| SVM | 434 | 24,353 | 281,708 | 3,064 |

## Train / validation metrics (persisted in `training_summary.json`)

Rows/positives computed from stored confusion matrices.

| Model | Split | Rows | Positives | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | train | 304,511 | 3,481 | 0.8209 | 0.0464 | 0.7443 | 0.7530 | 0.0337 | 0.0644 | 0.1784 |
| Logistic regression | validation | 156,036 | 1,753 | 0.6726 | 0.0278 | 0.1934 | 0.9501 | 0.0421 | 0.0692 | 0.1831 |
| Random forest | train | 304,511 | 3,481 | 0.9255 | 0.4143 | 0.7771 | 0.9115 | 0.0922 | 0.1649 | 0.1399 |
| Random forest | validation | 156,036 | 1,753 | 0.7107 | 0.0284 | 0.1603 | 0.9566 | 0.0403 | 0.0644 | 0.1328 |
| SVM | train | 4,000 | 46 | 0.9988 | 0.9815 | 0.9783 | 0.9977 | 0.8333 | 0.9000 | 0.0102 |
| SVM | validation | 156,036 | 1,753 | 0.6090 | 0.0167 | 0.1398 | 0.9223 | 0.0200 | 0.0350 | 0.0111 |

SVM train is subsampled to 4,000 rows (`svm_max_train_rows=4000`); logistic regression did not converge within 300 iterations (`lbfgs` iteration limit reached — reported in stderr, not suppressed).

## Artifacts

Reports: `data/reports/classical_ml_matched_6000/`
- `training_summary.json` — full config, per-model train/validation/test metrics, thresholds, split counts, manifest hash/path, feature-selection report, overlap/target-isolation checks
- `selected_features.json` — 960 kept feature names, `fit_on_train_only=true`
- `thresholds.json` — validation-selected thresholds (F1, grid 0.01), `test_set_used_for_selection=false`
- `feature_selection_report.json` — 34 removed features (all `*_hours_since_last`, infinite)

Artifacts: `artifacts/p1/classical_ml_matched_6000/`
- `logistic_regression/` — `logistic_regression_1.0.0.joblib`, `_features.json`, `_metadata.json`
- `random_forest/` — `random_forest_1.0.0.joblib`, `_features.json`, `_metadata.json`
- `svm/` — `svm_1.0.0.joblib`, `_features.json`, `_metadata.json`
- `registry/` — `registry.json`

Note: model metadata files carry placeholder values (`source="synthetic"`, `random_seed=0`, `n_rows_train=0`) that do not reflect the actual real-data run; rely on `training_summary.json` and the run log for provenance.

## Feature policy (matched to P2 for comparison)

- P1 matched experiment uses **960 features** (same as the prior P1 run).
- P2 used **994 features** (retains `*_hours_since_last`, sanitizes infinities).
- The 34 removed P1 features are all `*_hours_since_last` missingness features, which are infinite when a variable has never been observed. P1 removes them; P2 keeps and sanitizes them.
- No test information was used in feature selection (`fit_on_train_only=true`).

## Limitations

- Fast protocol: only 6,000 of 28,234 train patients; full-train performance may differ.
- Logistic regression did not converge (max_iter=300).
- SVM train subsampled to 4,000 rows.
- Model metadata placeholder values are unreliable for provenance.
- These are early-warning research results, not clinical deployment evidence.
- No hospital-level generalisation claims (Set A/Set B are dataset partitions).
