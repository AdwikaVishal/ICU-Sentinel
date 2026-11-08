# P2 Documentation — Advanced Tabular ML

## Overview

P2 implements advanced gradient-boosted tree models (Extra Trees, LightGBM, XGBoost)
on top of P1's verified feature and split contracts. It provides unified evaluation,
probability calibration, and experiment tracking while maintaining strict training
protocol discipline.

## Documents

### [P2_FINAL_REPORT.md](P2_FINAL_REPORT.md)
Complete implementation report with:
- Executive summary and status
- Implementation details for all three models
- Test evidence (47/48 tests pass; LightGBM Windows DLL issue)
- Known limitations and blockers
- Conditions required for empirical claims
- Final verdict: **GO WITH CONDITIONS** (code complete; real data blocked)

### [P2_QUICKSTART.md](P2_QUICKSTART.md)
Practical guide for using P2:
- Training models from P1 artifacts
- Loading and making predictions
- Calibrating probabilities
- Running tests
- Common issues and solutions
- Validation checklist

### [P2_ARCHITECTURE.md](P2_ARCHITECTURE.md)
Technical design documentation:
- Design principles
- Module structure and responsibilities
- Core components (config, training, evaluation, calibration)
- Data flow diagrams
- Testing strategy
- Extension points for new models/metrics
- Future enhancements

## Quick Reference

### Training P2 Models

```bash
# Requires P1 artifacts first
python scripts/generate_p1_artifacts.py

# Train P2 (all models if LightGBM works)
python scripts/generate_p2_artifacts.py

# Skip LightGBM (Windows workaround)
python scripts/generate_p2_artifacts.py --disable-lightgbm
```

### Running Tests

```bash
# All non-LightGBM tests (47 tests)
python -m pytest ml/p2_boosting/tests -k "not lightgbm" -v

# Individual components
python -m pytest ml/p2_boosting/tests/test_extra_trees.py -v
python -m pytest ml/p2_boosting/tests/test_xgboost.py -v
python -m pytest ml/p2_boosting/tests/test_calibration.py -v
```

### Expected Outputs

```
artifacts/p2/
├── extra_trees/
│   ├── extra_trees_1.0.0.joblib
│   ├── extra_trees_1.0.0_features.json
│   └── extra_trees_1.0.0_metadata.json
├── xgboost/
│   ├── xgboost_1.0.0.joblib
│   ├── xgboost_1.0.0_features.json
│   └── xgboost_1.0.0_metadata.json
└── p2_config.json

data/reports/p2/
├── p2_thresholds.json            # Validation-selected thresholds
├── p2_training_summary.json      # Complete metrics and provenance
└── p2_experiment.json            # Experiment metadata
```

## Status Summary

### ✅ Complete
- Extra Trees implementation (7/7 tests pass)
- XGBoost implementation (6/6 tests pass)
- Unified training protocol
- Evaluation and metrics (reuses P1)
- Probability calibration (sigmoid, isotonic)
- Feature contract validation
- Model serialization
- Experiment tracking
- Documentation

### ⚠️ Blocked
- LightGBM on Windows (native DLL issue)
- Real data training (PhysioNet dataset not yet acquired)
- Cross-hospital evaluation (hospital IDs not available)

### 📋 Recommended Next Steps
1. Acquire PhysioNet dataset legitimately
2. Generate P1 artifacts from real data
3. Train P2 models (with or without LightGBM)
4. Compare P2 boosting models against P1 classical baselines
5. Proceed to P3 for temporal deep learning models
6. Implement ensemble strategies combining P1 + P2 + P3

## Key Features

### Reproducibility
- All models use deterministic random seeds
- LightGBM configured with `deterministic=True`
- XGBoost configured with `tree_method="hist"`
- Tests verify prediction reproducibility

### Split Discipline
- Training, validation, test metrics stored separately
- Thresholds selected on validation only
- Test set evaluated once at frozen threshold
- Patient-level split integrity validated
- Split leakage detection and rejection

### Feature Contract Compliance
- Reuses P1's verified feature schema
- Excludes identifiers: `patient_id`, `source_dataset`, `hour`, `hospital_id`
- Excludes targets: `SepsisLabel`, `sepsis_within_6h`, `target_eligible`
- Validates schema match before training/prediction
- Handles missing values appropriately per model

### Class Imbalance Handling
- Extra Trees: `class_weight="balanced"`
- LightGBM: `class_weight="balanced"`
- XGBoost: `scale_pos_weight` computed from training data
- All models support configurable imbalance strategies

### Calibration
- Platt scaling (sigmoid calibration)
- Isotonic regression (monotonic calibration)
- Fitted on validation split only
- Test split explicitly rejected
- Serializable calibrators

## Implementation Details

### Models

| Model | Imputation | Class Weight | Native NaN | Deterministic |
|-------|------------|--------------|------------|---------------|
| Extra Trees | Median | Balanced | No | Yes (seed) |
| LightGBM | None | Balanced | Yes | Yes (force) |
| XGBoost | None | scale_pos_weight | Yes | Yes (hist) |

### Metrics (from P1)

- AUROC (Area Under ROC Curve)
- AUPRC (Average Precision)
- Sensitivity (Recall)
- Specificity
- Precision
- F1 Score
- Brier Score
- Confusion Matrix

### Configuration

Hierarchical `P2Config` with per-model hyperparameters:
- Global: seed, threshold metric, calibration method
- Per-model: n_estimators, learning_rate, max_depth, etc.
- JSON serializable for reproducibility

## Python API Examples

### Training

```python
from ml.p2_boosting import P2Config, train_p2_models
from ml.p1_clinical.classical_ml.data_loader import load_from_manifest_and_features

dataset = load_from_manifest_and_features(
    manifest_path="data/splits/split_manifest.json",
    processed_dir="data/features/patients",
)

config = P2Config(
    train_extra_trees=True,
    train_lightgbm=False,  # Windows workaround
    train_xgboost=True,
)

result = train_p2_models(
    dataset,
    config,
    artifact_dir="artifacts/p2",
    reports_dir="data/reports/p2",
    source="real",
)
```

### Prediction

```python
from ml.p2_boosting.serialization import load_p2_model
from ml.p2_boosting.base import predict_probabilities

model = load_p2_model("artifacts/p2/extra_trees/extra_trees_1.0.0.joblib")
predictions = predict_probabilities(
    model, 
    X_test, 
    threshold=model.threshold or 0.5
)
```

### Calibration

```python
from ml.p2_boosting.calibration import (
    fit_probability_calibrator,
    apply_probability_calibrator,
)

calibrator = fit_probability_calibrator(
    y_validation,
    validation_predictions["predicted_probability"],
    method="sigmoid",
    partition="validation",
)

calibrated_test_probs = apply_probability_calibrator(
    calibrator,
    test_predictions["predicted_probability"],
)
```

## Common Issues

### LightGBM DLL Error (Windows)
**Symptom:** `OSError: exception: access violation reading 0x0000000000000000`  
**Solution:** Disable LightGBM via `--disable-lightgbm` or `train_lightgbm=False`

### Missing P1 Artifacts
**Symptom:** `ERROR: processed patient features not found`  
**Solution:** Run `python scripts/generate_p1_artifacts.py` first

### Feature Schema Mismatch
**Symptom:** `ValueError: Feature schema mismatch; missing required columns`  
**Solution:** Ensure input excludes identifiers and targets

## Contact

P2 maintains the advanced tabular ML pipeline. Questions about boosting models,
calibration, or P2 configuration should be addressed before modifying the
P2 implementation.

For P1 feature engineering or preprocessing questions, consult P1 documentation.  
For P3 temporal modeling questions, consult P3 documentation when available.

