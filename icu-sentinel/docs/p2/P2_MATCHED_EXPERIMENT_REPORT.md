# ICU Sentinel — P2 Matched Experiment Report (6,000-Patient Cohort)

Status: complete. Matched Extra Trees and XGBoost trained and evaluated on the
canonical 6,000-patient fast cohort for like-for-like comparison with P1.
LightGBM disabled (Windows DLL failure, pre-existing).

## Experiment summary

| Item | Value |
|---|---|
| Training cohort | 6,000 stratified patients (canonical manifest) |
| Validation cohort | 4,034 patients (full split) |
| Test cohort | 8,068 patients (full split) |
| Unique patients | 18,102 (no overlap across splits) |
| Target | `sepsis_within_6h` (strict 6h pre-onset) |
| Target isolation | valid (no leak into features) |
| Split leakage | none detected (`leakage_detected=false`) |
| Feature selection | None applied by P2; restricted to P1's exact 960 features |
| Features | 960 (exact ordered P1 list; `fit_on_train_only` in P1) |
| Models trained | Extra Trees, XGBoost (LightGBM disabled) |
| Random seed | 42 |
| Source | real (PhysioNet/CinC 2019) |
| Training duration | 495.7 s (~8.3 min) |
| Threads | Extra Trees n_jobs=12, XGBoost n_jobs=12 (16 logical CPUs) |

## Canonical split manifest

- Path: `icu-sentinel/data/splits/split_manifest_fast_6000.json`
- SHA256: `008a93f47c4264e3b61d73121be17efc1dae0f69473289d238d61bcc881188b1` (verified against P1 matched report — MATCH)
- P1 selected features: `icu-sentinel/data/reports/classical_ml_matched_6000/selected_features.json` — 960 features
- P1 feature-list SHA256: `c7407bfd3a11f257a99263cd2e7fd7018c898d848f53065ce59daf5a0562a782` (verified — MATCH)

| Split | Patients | Records | Sepsis patients | Positive hours |
|---|---:|---:|---:|---:|
| Train | 6,000 | 228,703 eligible | — | 2,607 |
| Validation | 4,034 | 156,036 eligible | — | 1,753 |
| Test | 8,068 | 309,559 eligible | — | 3,498 |

Counts agree exactly with the P1 matched experiment test cohort
(`n_rows=309,559`, `n_positive=3,498`).

## Reproducibility command

```bash
python scripts/train_p2_matched_6000.py \
  --manifest data/splits/split_manifest_fast_6000.json \
  --features-dir data/features \
  --p1-features data/reports/classical_ml_matched_6000/selected_features.json \
  --artifact-dir artifacts/p2/matched_6000 \
  --reports-dir data/reports/p2_matched_6000 \
  --seed 42 \
  --extra-trees-estimators 200 \
  --xgboost-estimators 200 \
  --n-jobs 12
```

The script verifies the manifest hash and P1 feature-list hash before training,
restricts every split to exactly P1's 960 ordered features, selects thresholds
on validation only (F1, grid 0.01), and evaluates frozen models once on test.

## Model configurations (persisted in `p2_config.json` and `p2_training_summary.json`)

**Extra Trees** (`ExtraTreesClassifier` inside a `SimpleImputer(median)` pipeline)
- `n_estimators=200`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`
- `max_features="sqrt"`, `class_weight="balanced"`, `n_jobs=12`, `random_state=42`

**XGBoost** (`XGBClassifier` inside a `SimpleImputer(median)` pipeline)
- `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`, `min_child_weight=1.0`
- `subsample=0.9`, `colsample_bytree=0.9`, `reg_alpha=0.0`, `reg_lambda=1.0`
- `n_jobs=12`, `tree_method="hist"`, `objective="binary:logistic"`, `eval_metric="logloss"`
- `scale_pos_weight=negatives/positives` (train-only), `random_state=42`

Threshold selection: `metric=F1`, `grid_step=0.01`, split=validation only.
`test_set_used_for_selection=false`.

## Test results (one-shot evaluation at validation-selected threshold)

Test set: 8,068 patients, 309,559 eligible rows, 3,498 positive hours.

| Model | Test AUROC | Test AUPRC | Test sensitivity | Test specificity | Test precision | Test F1 | Test Brier | Test threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Extra Trees | 0.7047 | 0.0263 | 0.1875 | 0.9468 | 0.0387 | 0.0642 | 0.01119 | 0.04 |
| XGBoost | 0.7241 | 0.0320 | 0.0961 | 0.9783 | 0.0481 | 0.0641 | 0.10628 | 0.69 |

Confusion matrices (test):

| Model | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| Extra Trees | 656 | 16,274 | 289,787 | 2,842 |
| XGBoost | 336 | 6,643 | 299,418 | 3,162 |

## Train / validation metrics (persisted in `p2_training_summary.json`)

| Model | Split | Rows | Positives | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | Brier | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Extra Trees | train | 228,703 | 2,607 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9992 | 0.9996 | ~0.0000 | 0.5 |
| Extra Trees | validation | 156,036 | 1,753 | 0.7062 | 0.0250 | 0.1905 | 0.9422 | 0.0361 | 0.0607 | 0.01115 | 0.04 |
| XGBoost | train | 228,703 | 2,607 | 0.9686 | 0.4589 | 0.9260 | 0.8901 | 0.0885 | 0.1616 | 0.0997 | 0.5 |
| XGBoost | validation | 156,036 | 1,753 | 0.7387 | 0.0332 | 0.1158 | 0.9779 | 0.0562 | 0.0756 | 0.10729 | 0.69 |

Note: Extra Trees train metrics are near-perfect (consistent with an unregularised,
fully-grown tree ensemble on imputed median features); validation/test metrics are
the operative results. XGBoost train shows mild overfitting relative to validation.

## Integrity checks (all PASS)

- Manifest SHA256 matches P1 matched experiment (MATCH).
- P1 feature-list SHA256 matches (960 features, exact order) (MATCH).
- Test row count (309,559) and positive count (3,498) match P1 matched test cohort.
- Split overlap: `leakage_detected=false`, `n_overlapping_patients=0`,
  `every_patient_in_exactly_one_split=true` (18,102 patients).
- Target isolation: `target_isolation_valid=true`, no issues.
- No test-set fitting or tuning: thresholds selected on validation only.
- Model serialization: both `extra_trees_1.0.0.joblib` and
  `xgboost_1.0.0.joblib` load successfully; selected feature lists equal P1's
  960 features; metrics finite and within [0, 1].

## Artifacts

Reports: `icu-sentinel/data/reports/p2_matched_6000/`
- `p2_training_summary.json` — full config, per-model train/validation/test
  metrics, thresholds, split counts, manifest hash/path, feature hash,
  training duration, effective threads.
- `p2_thresholds.json` — validation-selected thresholds (F1, grid 0.01),
  `test_set_used_for_selection=false`.
- `p2_experiment.json` — full experiment record.

Artifacts: `icu-sentinel/artifacts/p2/matched_6000/`
- `extra_trees/` — `extra_trees_1.0.0.joblib`, `_features.json`, `_metadata.json`
- `xgboost/` — `xgboost_1.0.0.joblib`, `_features.json`, `_metadata.json`
- `p2_config.json` — full P2 configuration (LightGBM disabled).

Logs: `icu-sentinel/data/reports/logs/p2_matched_6000_{stdout,stderr}.log`

## Limitations

- Fast protocol: only 6,000 of 28,234 training patients; full-train performance may differ.
- LightGBM unavailable on this Windows environment (pre-existing DLL failure); Extra Trees + XGBoost evaluated only.
- Extra Trees train near-perfect metrics indicate limited regularisation; use validation/test for claims.
- Default hyperparameters (software defaults / reduced for speed), not tuned on data.
- Thresholds selected on validation F1; different operating points trade sensitivity vs precision.
- Results are early-warning research prototypes, not clinical products; no clinical validation or deployment claims.
- Set A and Set B are dataset partitions, not hospitals; no hospital-level generalisation claims.
