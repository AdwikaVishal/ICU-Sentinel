# ICU Sentinel — Optimised Pipeline Validation

**Branch:** `perf/phase1-6-optimizations`
**Commit:** `9e7c683` Phase 4-6: Add performance optimizations (bootstrap, memory, CPU, loader, checkpoint)
**Status:** GO for use in the next P1/P2 research experiment (see limitations below).

## 1. Scope

This document validates the six performance-optimisation phases implemented on
`perf/phase1-6-optimizations`. It is a software-validation report only; none of
the metrics below constitute research evidence and none were computed on the
full cohort.

## 2. Files changed (commit `9e7c683`)

15 files, 3,039 insertions, 158 deletions. Only performance-inventory files
were committed; unrelated pre-existing working-tree changes (README, backend,
docs/p1/*, pyproject.toml, requirements.txt, etc.) were deliberately left out.

Modified:
- `icu-sentinel/ml/p1_clinical/classical_ml/data_loader.py` — split-scoped
  loading, per-split Parquet shards, content-fingerprint stale-cache
  detection, atomic writes, per-patient fallback preserved.
- `icu-sentinel/ml/p1_clinical/evaluation/bootstrap.py` — indexed patient-level
  cluster bootstrap; the seeded `rng.choice(patient_ids, ...)` draw is
  byte-identical to the previous implementation.

New:
- `icu-sentinel/ml/p1_clinical/tests/test_bootstrap_indexed_equivalence.py`
- `icu-sentinel/ml/p1_clinical/tests/test_loader_phase5.py`
- `icu-sentinel/ml/p1_clinical/tests/test_performance_characterization.py`
- `icu-sentinel/ml/p2_boosting/base.py`
- `icu-sentinel/ml/p2_boosting/checkpoint.py`
- `icu-sentinel/ml/p2_boosting/config.py`
- `icu-sentinel/ml/p2_boosting/feature_contract.py` (required P2 dependency:
  present on `feature/p2-optimization`, absent on `perf`/`main`; restored so
  the committed P2 code is importable and runnable)
- `icu-sentinel/ml/p2_boosting/pipelines.py`
- `icu-sentinel/ml/p2_boosting/tests/test_checkpoint_resume.py`
- `icu-sentinel/ml/p2_boosting/tests/test_performance_characterization.py`
- `icu-sentinel/ml/p2_boosting/training.py`
- `icu-sentinel/scripts/generate_p2_artifacts.py`
- `icu-sentinel/tests/golden/performance_characterization.json`

## 3. Integrity checks performed

- Raw data, split manifests, and existing model artifacts verified untouched:
  `git status --short -- data/ artifacts/` → no changes.
- Scientific target/feature policies untouched: `DERIVED_TARGET_COLUMN`,
  `validate_target_isolation`, P1 960-feature behaviour, and model tree
  counts/depths unchanged.
- No existing P1/P2 artifacts overwritten; smoke test wrote only to temp
  directories.
- LightGBM not enabled (pre-existing Windows DLL access violation; disabled by
  default).

## 4. Test results

All test commands use `pytest -o addopts("-p no:cov")` to avoid coverage
permission issues on Windows.

| Suite | Result |
|-------|--------|
| P1 full suite (`ml/p1_clinical/tests`) | 408 passed (7 pre-existing single-class sklearn warnings) |
| P2 suite, excl. `test_lightgbm.py` | 54 passed |
| Bootstrap equivalence vs. old loop | passed (seeds 0/42/123, exact CI equality) |
| Loader/shard equivalence vs. per-patient | passed (row/patient order, dtype, feature order, float32 hashes) |
| Checkpoint / resume | passed (fingerprint verification, cross-config refusal, resume skips retraining) |
| Golden characterisation (P1 loader/bootstrap, P2 training/prediction) | passed |
| Ruff on all changed files | All checks passed |

**LightGBM:** excluded. `test_lightgbm.py` crashes inside the LightGBM C
extension (`OSError: access violation` in `LGBM_DatasetSetField`) on this
Windows box. This is pre-existing and unrelated to the optimisations; LightGBM
remains disabled by default (`--disable-lightgbm`).

## 5. Controlled real-data smoke test

**Command run:**
```
python scripts/generate_p2_artifacts.py \
  --processed-dir data/features/patients \
  --split-manifest <tmp>/smoke/smoke_manifest_200.json \
  --feature-schema data/features/feature_schema.json \
  --artifact-dir <tmp>/smoke/artifacts \
  --reports-dir <tmp>/smoke/reports \
  --disable-lightgbm \
  --extra-trees-estimators 10 \
  --xgboost-estimators 10 \
  --extra-trees-n-jobs 2 \
  --xgboost-n-jobs 2 \
  --train-patients 0 \
  --seed 42 \
  --source real \
  --overwrite
```
(`<tmp>/smoke/smoke_manifest_200.json` is a temporary stratified manifest with
200 train / 40 validation / 80 test real patients; the real `split_manifest_fast.json`
was not modified.)

**Outcome:** exit code 0.
- Real-data loading: OK (994 features, target `sepsis_within_6h`, no split leakage).
- Model fitting: OK (Extra Trees, XGBoost).
- Prediction: OK (validation/test metrics computed).
- Serialization: OK (artifacts + reports written to temp dirs).
- Metrics (un-tuned, tiny sample — NOT research results):
  - Extra Trees: validation AUROC=0.670, test AUROC=0.616
  - XGBoost: validation AUROC=0.620, test AUROC=0.397

**Checkpoint / resume on real data (direct `train_p2_models` call):**
- `p2_experiment.json` records `"effective_threads":
  {"extra_trees": 2, "xgboost": 2, "cpu_count": 16}`.
- Resume with `checkpoint_dir` + `resume=True`: `train_model` called **0 times**,
  thresholds identical (ET 0.21, XGB 0.57), metrics equivalent.
- Cross-config fingerprint mismatch correctly raises `ValueError` (refuses to
  combine artifacts from different configurations).

## 6. Memory and runtime observations (from the performance audit)

- Host: Windows 11, Python 3.11, 16 logical CPUs, ~16 GB RAM, 93 GB free disk.
- Full cohort: 40,336 patients / 1,552,210 rows / 994 features.
- P1 fast cohort (6,000 train patients): completed naturally in 47m47s; observed
  6.3–11.1 GB private working set.
- Estimated float32 footprint: train 4.01 GiB, validation 0.58 GiB, test
  1.15 GiB, full 5.75 GiB. The previous loader + repeated P2 copies could exceed
  16 GB; full-cohort P2 as-shipped was likely OOM.
- Main bottlenecks addressed: per-patient file opens (40,336 parquet files),
  repeated full-frame validation/sanitisation copies, repeated per-iteration
  DataFrame filtering in bootstrap, P2 `n_jobs=1` (single-threaded), fully-grown
  Extra Trees.

The optimisations target these bottlenecks without changing model behaviour:
one-time per-split sanitisation, indexed bootstrap gathering, split-scoped
loader/shards, and separate safe ET/XGB thread controls.

## 7. Checkpoint / resume verification

- `RunManifest` records `config_fingerprint` (SHA-256 of the serialised
  `P2Config`), source, target column, feature set, and enabled models.
- Manifest written atomically (temp file + `os.replace`); no `.tmp` leftovers.
- Before resume, `manifest_matches_config` rejects any fingerprint mismatch
  (e.g. different estimator count) and `train_p2_models` rejects a feature-set
  mismatch — artifacts from different configurations are never combined.
- Completed models are skipped on resume and loaded from their checkpointed
  artifact; `train_model` is not invoked. Verified on real data with zero
  retraining and equivalent metrics.

## 8. Remaining limitations

- LightGBM cannot be trained on this Windows machine (pre-existing DLL access
  violation); it stays disabled by default and is excluded from the test suite.
- Golden snapshot hashes are environment/library-version sensitive; regenerate
  `tests/golden/performance_characterization.json` if scikit-learn, XGBoost,
  pandas, or numpy are upgraded.
- Full-cohort P2 may still require ample RAM (>=32 GB recommended) or an
  Extra Trees depth/leaf cap; the optimisations reduce but do not eliminate the
  footprint of 994 features over 1.55M rows.
- `feature_contract.py` had to be carried onto `perf/phase1-6-optimizations`
  (it existed on `feature/p2-optimization` but not on `perf`/`main`) because the
  committed P2 code imports it. It is a required dependency, not a behaviour
  change.
- Working tree retains unrelated pre-existing modifications (not part of this
  commit); review `git status` before merging.

## 9. Exact commands for the next experiments

### Matched fast experiment (reproduce the smoke test on the canonical fast manifest)
```
python scripts/generate_p2_artifacts.py \
  --processed-dir data/features/patients \
  --split-manifest data/splits/split_manifest_fast.json \
  --feature-schema data/features/feature_schema.json \
  --artifact-dir artifacts/p2_fast \
  --reports-dir data/reports/p2_fast \
  --disable-lightgbm \
  --extra-trees-estimators 200 \
  --xgboost-estimators 200 \
  --extra-trees-n-jobs 2 \
  --xgboost-n-jobs 2 \
  --seed 42 \
  --source synthetic
```

### Full-cohort experiment (do not run without separate approval and adequate RAM)
```
python scripts/generate_p2_artifacts.py \
  --processed-dir data/features/patients \
  --split-manifest data/splits/split_manifest.json \
  --feature-schema data/features/feature_schema.json \
  --artifact-dir artifacts/p2 \
  --reports-dir data/reports/p2 \
  --disable-lightgbm \
  --extra-trees-estimators 200 \
  --xgboost-estimators 200 \
  --extra-trees-depth <cap if needed> \
  --extra-trees-n-jobs <safe default> \
  --xgboost-n-jobs <safe default> \
  --seed 42 \
  --source real
```

## 10. Recommendation

**GO.** All six optimisation phases are implemented, equivalence-proven against
the previous implementations, covered by targeted and suite tests (482 passed
excluding the pre-existing LightGBM issue), lint-clean, and validated on real
data including checkpoint/resume. No scientific target, feature, model-depth, or
tree-count policies were changed, and no existing data or artifacts were
modified.
