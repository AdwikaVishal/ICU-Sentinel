# P2 Quick Start Guide

## Prerequisites

P2 requires P1 artifacts to exist. Run P1 first:

```bash
# Generate P1 artifacts (requires real PhysioNet data under data/raw/)
python scripts/generate_p1_artifacts.py
```

## Training P2 Models

### With All Models (if LightGBM works)

```bash
python scripts/generate_p2_artifacts.py
```

### Without LightGBM (Windows workaround)

```bash
python scripts/generate_p2_artifacts.py --disable-lightgbm
```

### Custom Configuration

```bash
python scripts/generate_p2_artifacts.py \
    --processed-dir data/features/patients \
    --split-manifest data/splits/split_manifest.json \
    --feature-schema data/features/feature_schema.json \
    --artifact-dir artifacts/p2 \
    --reports-dir data/reports/p2 \
    --source real
```

## Expected Outputs

After successful P2 training:

```
artifacts/p2/
  extra_trees/
    extra_trees_1.0.0.joblib
    extra_trees_1.0.0_features.json
    extra_trees_1.0.0_metadata.json
  xgboost/
    xgboost_1.0.0.joblib
    xgboost_1.0.0_features.json
    xgboost_1.0.0_metadata.json
  p2_config.json

data/reports/p2/
  p2_thresholds.json              # Validation-selected thresholds
  p2_training_summary.json        # Complete training record
  p2_experiment.json              # Experiment metadata
```

## Using P2 Models in Code

### Training Models

```python
from ml.p2_boosting.config import P2Config
from ml.p2_boosting.training import train_p2_from_artifacts

config = P2Config(
    train_extra_trees=True,
    train_lightgbm=False,  # Disable if DLL issues
    train_xgboost=True,
)

result = train_p2_from_artifacts(
    processed_dir="data/features/patients",
    split_manifest_path="data/splits/split_manifest.json",
    config=config,
    artifact_dir="artifacts/p2",
    reports_dir="data/reports/p2",
    source="real",
)

print(f"Trained models: {list(result.models.keys())}")
print(f"Validation thresholds: {result.thresholds}")
```

### Loading and Predicting

```python
import pandas as pd
from ml.p2_boosting.serialization import load_p2_model
from ml.p2_boosting.base import predict_probabilities

# Load a trained model
model = load_p2_model("artifacts/p2/extra_trees/extra_trees_1.0.0.joblib")

# Load test data (must match feature schema)
X_test = pd.read_parquet("data/features/patients/patient_001.parquet")
X_test = X_test[model.selected_features]

# Predict with validation threshold
threshold = model.threshold or 0.5
predictions = predict_probabilities(model, X_test, threshold=threshold)

print(predictions.head())
# Columns: predicted_probability, predicted_class
```

### Evaluating Predictions

```python
from ml.p2_boosting.evaluation import evaluate_predictions

metrics = evaluate_predictions(
    y_true=y_test,
    y_prob=predictions["predicted_probability"],
    threshold=threshold,
    model_name="extra_trees",
    split="test",
)

print(f"AUROC: {metrics['metrics']['auroc']:.3f}")
print(f"AUPRC: {metrics['metrics']['auprc']:.3f}")
print(f"F1: {metrics['metrics']['f1']:.3f}")
```

### Calibrating Probabilities

```python
from ml.p2_boosting.calibration import (
    fit_probability_calibrator,
    apply_probability_calibrator,
)

# Fit calibrator on validation predictions
calibrator = fit_probability_calibrator(
    y_validation,
    validation_predictions["predicted_probability"],
    method="sigmoid",  # or "isotonic"
    partition="validation",
)

# Apply to test predictions
calibrated_probs = apply_probability_calibrator(
    calibrator,
    test_predictions["predicted_probability"],
)

# Calibrator can be saved/loaded
calibrator.save("artifacts/p2/extra_trees_calibrator.joblib")
```

## Running P2 Tests

### All Non-LightGBM Tests

```bash
python -m pytest ml/p2_boosting/tests -k "not lightgbm" -v
```

### Individual Model Tests

```bash
# Extra Trees
python -m pytest ml/p2_boosting/tests/test_extra_trees.py -v

# XGBoost
python -m pytest ml/p2_boosting/tests/test_xgboost.py -v

# Calibration
python -m pytest ml/p2_boosting/tests/test_calibration.py -v

# Evaluation
python -m pytest ml/p2_boosting/tests/test_evaluation.py -v
```

## Configuration Options

### P2Config

```python
from ml.p2_boosting.config import (
    P2Config,
    ExtraTreesConfig,
    XGBoostConfig,
)

config = P2Config(
    random_seed=42,
    threshold_selection_metric="f1",  # or "precision", "recall", "balanced_accuracy"
    threshold_grid_step=0.01,
    calibration_method="sigmoid",  # or "isotonic"
    
    # Enable/disable models
    train_extra_trees=True,
    train_lightgbm=False,
    train_xgboost=True,
    
    # Per-model hyperparameters
    extra_trees=ExtraTreesConfig(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
    ),
    xgboost=XGBoostConfig(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
    ),
)

# Save/load config
config.save("p2_config.json")
loaded = P2Config.load("p2_config.json")
```

## Known Issues

### LightGBM DLL Error (Windows)

**Symptom:**
```
OSError: exception: access violation reading 0x0000000000000000
```

**Solution:** Disable LightGBM:
- CLI: `--disable-lightgbm`
- Code: `P2Config(train_lightgbm=False)`

### Missing P1 Artifacts

**Symptom:**
```
ERROR: processed patient features not found
```

**Solution:** Run P1 first: `python scripts/generate_p1_artifacts.py`

### Feature Schema Mismatch

**Symptom:**
```
ValueError: Feature schema mismatch; missing required columns
```

**Solution:** Ensure input data matches the P1 feature schema and excludes
identifiers/targets:
- Exclude: `patient_id`, `source_dataset`, `hour`, `hospital_id`
- Exclude: `SepsisLabel`, `sepsis_within_6h`, `target_eligible`

## Validation Checklist

Before reporting P2 results:

- [ ] Used P1 artifacts from real PhysioNet data (not synthetic fixtures)
- [ ] Verified patient-level split integrity (no leakage)
- [ ] Confirmed thresholds selected on validation only
- [ ] Test set evaluated once at frozen threshold
- [ ] Source labeled correctly (`real` or `synthetic`)
- [ ] Synthetic results explicitly labeled as software validation only

## Next Steps

After P2 is trained:

1. Review validation and test metrics in `data/reports/p2/`
2. Compare P2 boosting models against P1 classical baselines
3. Proceed to P3 for temporal deep learning models (GRU/LSTM)
4. Consider ensemble strategies combining P1 + P2 + P3
5. Evaluate calibration quality using reliability diagrams
6. Implement cross-hospital evaluation when real hospital IDs are available

