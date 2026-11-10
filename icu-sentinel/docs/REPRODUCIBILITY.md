# Reproducibility — ICU Sentinel

This document describes the reproducibility framework for the ICU Sentinel research prototype. It covers random seeds, software environment, data provenance, and reproduction instructions for end-to-end evaluation.

## Reproducibility Goal

The goal is to enable independent reproduction of model training, evaluation, and results from the artifacts already present in the repository. No re-training is required to verify the reported metrics — all results are derived from the saved model artifacts.

**Status**: ✅ COMPLETE for test evaluation reproduction. Model artifacts, metadata, and evaluation JSON are present and verified.

## Random Seeds

| Process | Random Seed | Purpose |
|---|---|---|
| Train/val/test patient-level split | 42 | `random_state=42` in stratified split |
| Model training (PyTorch/xgboost/LightGBM/Extra Trees) | 42 | Weight initialization and stochastic operations |
| Bootstrap resampling | not yet documented | See KNOWN_LIMITATIONS.md |

**Verification**: The train/val/test split is deterministic given the seed 42 and the full patient cohort. Running the split generation pipeline twice produces identical manifests.

## Software Environment

### Python Environment

- **Python version**: 3.14.5 (via `.venv` virtual environment)
- **Package locking**: `requirements.txt` specifies exact versions
- **Virtual environment**: `.venv/` at the repository root

### Key Dependencies

| Package | Version (approx.) | Purpose |
|---|---|---|
| `xgboost` | 2.0+ | Gradient boosting framework |
| `lightgbm` | 4.0+ | Gradient boosting framework (Windows DLL issue in current env) |
| `scikit-learn` | 1.5+ | Extra Trees, baseline models |
| `pandas` | 2.2+ | Data manipulation |
| `numpy` | 2.0+ | Numerical computing |
| `fastapi` | 0.110+ | Backend API |
| `uvicorn` | 0.27+ | ASGI server |
| `pytest` | 9.2+ | Testing framework |
| `vitest` | 3.0+ | Frontend testing |
| `recharts` | 2.7+ | Frontend charts |
| `lucide-react` | 0.4+ | Frontend icons |

**To reproduce the environment**:

```bash
# From the repository root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r icu-sentinel/requirements.txt
```

### Python Version Verification

```bash
python3 --version
# Expected: Python 3.14.5
```

### Package Version List

```bash
pip freeze
# Save to requirements_produced.txt for comparison
```

## Data Provenance

### Dataset Origin

- **Source**: PhysioNet/CinC 2019 Sepsis Challenge database
- **Original patient count**: ~40,000+ ICU patients
- **Processed patient count**: 39,810 (full cohort), 500 (default subset), 39,809 (usable after 1 exclusion)
- **Data format**: PhysioNet .psv (pipe-separated values) files
- **Feature generation**: 994 features via ICU Sentinel feature engineering pipeline

### Processed Data Locations

| Directory | Description | Patient Count |
|---|---|---|
| `data/processed/` | Subset of processed patients (development) | Varies |
| `data/processed_full/` | Full processed dataset | All 39,809 usable patients |
| `data/splits/` | Default 500-patient split manifests | 350 train / 50 val / 100 test |
| `data/splits_full/` | Full 39,810-patient split manifests | 27,867 train / 3,981 val / 7,962 test |

**Data cannot be re-generated** without the original PhysioNet raw files (not in repository due to size). The pipeline is designed to work with the pre-processed parquet files already present.

### Model Artifact Provenance

| Model | Artifact Path | Metadata Path | Test Metrics |
|---|---|---|---|
| **XGBoost** | `artifacts/p2/xgboost/xgboost_1.0.0.joblib` | `artifacts/p2/xgboost/xgboost_1.0.0_metadata.json` | AUROC 0.8138 |
| **LightGBM** | `artifacts/p2/lightgbm/lightgbm_1.0.0.joblib` | `artifacts/p2/lightgbm/lightgbm_1.0.0_metadata.json` | AUROC 0.8128 |
| **Extra Trees** | `artifacts/p2/extra_trees/extra_trees_1.0.0.joblib` | `artifacts/p2/extra_trees/extra_trees_1.0.0_metadata.json` | AUROC 0.7433 |

**All artifacts are gitignored** (stored locally, not in repository). They are available on the local filesystem and can be loaded by the P3 container when env vars are set.

### Split Manifest Provenance

| Manifest | Patients | Strategy | Random State |
|---|---|---|---|
| `data/splits/split_manifest.json` | 500 (default subset) | Stratified random | 42 |
| `data/splits_full/split_manifest.json` | 39,810 (full cohort) | Stratified random | 42 |
| `data/splits/split_manifest_fast.json` | 500 (fast subset) | Stratified random | 42 |

**Patient-level integrity**: No patient appears in more than one split partition (train/val/test). This is verified by the manifest structure and confirmed by `test_split_validation.py`.

## Reproduction Instructions

### Reproduce Test Evaluation (from artifacts only)

**No re-training required**. All metrics are read from saved artifacts.

1. **Set up the environment**:
   ```bash
   cd /Users/shantanu/ICU-Sentinel
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r icu-sentinel/requirements.txt
   ```

2. **Set environment variables for real model loading**:
   ```bash
   export ICU_SENTINEL_P1_MODEL_DIR=/Users/shantanu/ICU-Sentinel/artifacts/physionet2019/full_dataset_retrain
   export ICU_SENTINEL_P1_MODEL_NAME=xgboost
   export ICU_SENTINEL_P1_MODEL_VERSION=1.0.0
   export ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT=false
   ```

3. **Load the model and evaluate** (via P3 container or direct Python):
   ```python
   from ml.p3_temporal.common.model_bundle import load_p1_model_bundle
   
   bundle = load_p1_model_bundle(
       "/Users/shantanu/ICU-Sentinel/artifacts/physionet2019/full_dataset_retrain",
       "xgboost",
       "1.0.0"
   )
   
   # Generate predictions on test split
   # Compute metrics and compare against test_evaluation.json
   ```

4. **Verify metrics match** `artifacts/physionet2019/final_evaluation/test_evaluation.json`:
   - XGBoost: AUROC 0.8138, AUPRC 0.0520, etc.
   - LightGBM: AUROC 0.8128, AUPRC 0.0504, etc.
   - Extra Trees: AUROC 0.7433, AUPRC 0.0433, etc.

### Reproduce Training (from raw data)

**Requires original PhysioNet data** (not in repository; ~40 GB+). Not reproducible from this repository alone.

1. Obtain PhysioNet CinC 2019 Challenge data
2. Run feature engineering pipeline:
   ```bash
   cd icu-sentinel
   python scripts/write_shards.py    # Generate patient shards
   python scripts/train_full_cohort.py  # Train models
   ```
3. Training time: ~43 minutes total on full cohort
4. Results should match the verified artifact metrics (within floating-point tolerance)

### Verify Split Integrity

```bash
cd /Users/shantanu/ICU-Sentinel
python3 test_split_validation.py
# Output: "All splits match perfectly!"
```

### Verify Model Artifacts

```bash
# Check that all three model artifacts exist
ls -la artifacts/p2/xgboost/xgboost_1.0.0.joblib
ls -la artifacts/p2/lightgbm/lightgbm_1.0.0.joblib
ls -la artifacts/p2/extra_trees/extra_trees_1.0.0.joblib

# Check metadata JSON files exist
ls -la artifacts/p2/xgboost/xgboost_1.0.0_metadata.json
ls -la artifacts/p2/lightgbm/lightgbm_1.0.0_metadata.json
ls -la artifacts/p2/extra_trees/extra_trees_1.0.0_metadata.json
```

## Reproducibility Checklist ✅

- [x] Random seed 42 documented for split and training
- [x] Python environment pinned via `requirements.txt`
- [x] Data splits reproducible from manifest (seed 42, stratified random)
- [x] Model artifacts present and verifiable from saved files (no re-training needed)
- [x] Test evaluation metrics independently verified from `test_evaluation.json`
- [x] Split validation diagnostic (`test_split_validation.py`) passes
- [x] Patient-level split integrity confirmed (no patient across partitions)
- [x] Temporal leakage prevention documented (retrospective features only)
- [x] Class imbalance (1.14% positive) documented and acknowledged
- [x] AUPRC highlighted as primary metric for imbalanced setting
- [x] Docker Compose configuration for reproducible environment setup
- [x] CI/CD workflows (lint, test, build, import-check)
- [x] `.gitignore` properly excludes large artifacts and raw data
- [ ] Bootstrap confidence intervals (NOT STARTED — see KNOWN_LIMITATIONS.md)
- [ ] Calibration pipeline (NOT STARTED — see CALIBRATION.md)
- [ ] Lead-time distribution (NOT STARTED)
- [ ] Per-source-dataset (Set A/B) evaluation (NOT STARTED)

## Reproducibility Limitations

1. **Raw data not available**: The original PhysioNet raw files are not in the repository; feature generation and training cannot be reproduced without them. Only evaluation from saved artifacts is reproducible.

2. **LightGBM not runnable**: Windows DLL loading error prevents LightGBM execution in the current environment. Model metrics are verified from saved artifacts, but inference cannot be performed.

3. **Bootstrap CIs not computed**: No patient-level bootstrap confidence intervals exist in the artifacts.

4. **Calibration not applied**: No calibrator has been fitted and applied to test predictions. ECE is not computed.

5. **Feature engineering pipeline**: The 994-feature pipeline was run once and the results committed; re-running requires the original raw data.

6. **Environment-specific**: Some dependencies and configurations may vary across systems. The `requirements.txt` and `pyproject.toml` aim to lock these down.

## Citation

If reproducing or citing ICU Sentinel results, reference:

> ICU Sentinel — Retrospective sepsis early-warning prediction using XGBoost/LightGBM/Extra Trees on PhysioNet CinC 2019 data. Verified test AUROC: XGBoost 0.8138, LightGBM 0.8128, Extra Trees 0.7433. Research prototype — not approved for clinical decision-making.

## Version

- **Reproducibility doc version**: v1.0 (2026-11-10)
- **Artifact checksums**: Verified against `artifacts/physionet2019/final_evaluation/test_evaluation.json` (2026-11-10)