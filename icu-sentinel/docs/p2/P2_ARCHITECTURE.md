# P2 Architecture — ICU Sentinel

## Overview

P2 (Advanced Tabular ML) provides three gradient-boosted tree ensembles—Extra Trees,
LightGBM, and XGBoost—on top of P1's verified feature and split contracts. It adds
unified evaluation, probability calibration, and lightweight experiment tracking while
maintaining strict adherence to the training protocol (fit on train, select on
validation, evaluate once on test).

---

## Design Principles

1. **Reuse P1 Contracts:** P2 never re-preprocesses or re-splits data. It consumes
   P1's patient-level features and split manifests directly.

2. **Unified Interface:** All three models expose the same `P2ModelResult` contract
   and prediction interface.

3. **Explicit Split Discipline:** Training, validation, and test metrics are stored
   separately. Threshold selection is validation-only by construction.

4. **No Silent Fallbacks:** Invalid configurations raise `ValueError` rather than
   silently defaulting or skipping steps.

5. **Serializable Provenance:** Configurations, metadata, and experiment records
   are JSON-serializable with explicit `source` labels.

---

## Module Structure

```
ml/p2_boosting/
├── __init__.py              Public API (train_p2_models, P2Config, evaluate_predictions)
├── base.py                  P2ModelResult dataclass, predict_probabilities
├── config.py                P2Config, ExtraTreesConfig, LightGBMConfig, XGBoostConfig
├── pipelines.py             Unified sklearn pipeline builder for all models
├── training.py              train_p2_models orchestration, split validation
├── evaluation.py            evaluate_predictions, select_validation_threshold
├── calibration.py           fit_probability_calibrator, apply_probability_calibrator
├── feature_contract.py      P1 feature schema integration, validation
├── serialization.py         save_p2_model, load_p2_model (reuses P1 conventions)
├── experiment.py            JSON experiment records with json_safe conversion
├── extra_trees.py           Extra Trees wrapper
├── lightgbm_model.py        LightGBM wrapper
├── xgboost_model.py         XGBoost wrapper
└── tests/                   48 tests (39 pass without LightGBM)
    ├── fixtures.py          Synthetic dataset and config generators
    ├── test_extra_trees.py
    ├── test_xgboost.py
    ├── test_lightgbm.py     (Windows DLL issue)
    ├── test_calibration.py
    ├── test_evaluation.py
    ├── test_feature_contract.py
    ├── test_config.py
    └── test_training.py
```

---

## Core Components

### 1. P2ModelResult (base.py)

The unified result container for all P2 models:

```python
@dataclass
class P2ModelResult:
    model_name: str
    model_version: str = "1.0.0"
    model: Any = None                        # Fitted sklearn pipeline
    preprocessing_pipeline: Any = None       # Imputer (Extra Trees only)
    selected_features: list[str]             # Feature names in order
    hyperparameters: dict                    # Model config
    training_metrics: dict                   # Train split metrics
    validation_metrics: dict                 # Validation split metrics
    test_metrics: dict                       # Test split metrics (one-shot)
    threshold: float | None                  # Validation-selected threshold
    random_seed: int = 42
    training_timestamp: str
    n_patients_train: int
    n_rows_train: int
    n_positive_train: int
    n_negative_train: int
    class_weight: str | None
    calibration_method: str | None
    calibrator: Any = None
    source: str = "synthetic"
```

Key guarantees:
- `training_metrics`, `validation_metrics`, and `test_metrics` are always separate
- `threshold` is selected on validation only
- `source` explicitly labels synthetic vs. real data

### 2. P2Config (config.py)

Hierarchical configuration with per-model hyperparameters:

```python
@dataclass
class P2Config:
    random_seed: int = 42
    threshold_selection_metric: str = "f1"
    threshold_grid_step: float = 0.01
    calibration_method: str = "sigmoid"
    
    train_extra_trees: bool = True
    train_lightgbm: bool = True
    train_xgboost: bool = True
    
    extra_trees: ExtraTreesConfig = field(default_factory=ExtraTreesConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
```

Supports JSON serialization:
```python
config.save("p2_config.json")
loaded = P2Config.load("p2_config.json")
```

### 3. Training Orchestration (training.py)

The `train_p2_models` function implements the complete protocol:

```python
def train_p2_models(
    dataset: LoadedDataset,
    config: P2Config | None = None,
    *,
    artifact_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    source: str | None = None,
    overwrite: bool = False,
) -> P2TrainingResult:
```

**Validation steps:**
1. Verify enabled models are supported
2. Require `sepsis_within_6h` as target (reject `SepsisLabel`)
3. Check patient-level split overlap (reject if leakage detected)
4. Validate target isolation (reject if target columns in features)
5. Require validation split (reject if missing)

**Training steps:**
1. Fit each enabled model on training data
2. Compute training metrics at threshold 0.5 (recorded but not used)
3. Evaluate on validation at threshold 0.5
4. Select threshold on validation using configured metric (default: F1)
5. Re-evaluate validation at selected threshold
6. If test exists, evaluate once at validation threshold

**Persistence:**
- Save fitted models to `artifact_dir/{model_name}/`
- Save `p2_config.json` to `artifact_dir/`
- Save `p2_thresholds.json` to `reports_dir/`
- Save `p2_training_summary.json` to `reports_dir/`
- Save `p2_experiment.json` to `reports_dir/`

### 4. Unified Pipeline Builder (pipelines.py)

All models use scikit-learn-compatible pipelines:

**Extra Trees:**
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", ExtraTreesClassifier(...))
])
```

**LightGBM:**
```python
Pipeline([
    ("classifier", LGBMClassifier(...))
])
# Native NaN handling; no imputer needed
```

**XGBoost:**
```python
Pipeline([
    ("classifier", XGBClassifier(...))
])
# Native NaN handling; scale_pos_weight computed from y_train
```

All classifiers:
- Use reproducible random seeds
- Support class imbalance (balanced weights or scale_pos_weight)
- Return two-column predict_proba output

### 5. Feature Contract (feature_contract.py)

Reuses P1's identifier and target definitions:

```python
P2_METADATA_COLUMNS = ("patient_id", "source_dataset", "hour", "hospital_id")
P2_TARGET_COLUMNS = ("SepsisLabel", "sepsis_within_6h", "target_eligible")
P2_DEFAULT_TARGET = "sepsis_within_6h"
```

**Validation functions:**
- `resolve_feature_names()`: exclude identifiers/targets from column list
- `validate_feature_names()`: ensure no leakage, no duplicates
- `validate_feature_frame()`: check schema match, numeric types, non-empty
- `sanitize_infinities()`: replace ±inf with NaN (trees handle NaN)
- `validate_training_target()`: binary, both classes, no NaN
- `validate_prediction_target()`: binary evaluation target

### 6. Evaluation (evaluation.py)

Reuses P1's shared metrics (`ml.common.metrics.compute_binary_metrics`):

```python
def evaluate_predictions(
    y_true, y_prob, *, threshold=0.5, model_name=None, split=None
) -> dict:
```

Returns:
- `auroc`, `auprc`, `sensitivity`, `specificity`, `precision`, `f1`, `brier`
- Confusion matrix: `tp`, `tn`, `fp`, `fn`
- Metadata: `model_name`, `split`, `threshold`, `n_rows`, `n_positive`

Threshold selection:
```python
def select_validation_threshold(
    y_true, y_prob, *, metric="f1", grid_step=0.01, partition="validation"
) -> float:
```

- Reuses P1's `select_threshold_on_validation` implementation
- Explicitly rejects `partition="test"`
- Grid search over [0.0, 1.0] with configurable step

### 7. Calibration (calibration.py)

Probability calibration fitted on validation only:

**Supported methods:**
- `sigmoid`: Platt scaling (logistic regression on logits)
- `isotonic`: Isotonic regression (monotonic mapping)

**Fitting:**
```python
calibrator = fit_probability_calibrator(
    y_validation, validation_probs,
    method="sigmoid",
    partition="validation"
)
```

Rejects `partition="test"` by construction.

**Application:**
```python
calibrated_probs = apply_probability_calibrator(calibrator, test_probs)
```

Calibrators are serializable:
```python
calibrator.save("calibrator.joblib")
loaded = ProbabilityCalibrator.load("calibrator.joblib")
```

### 8. Serialization (serialization.py)

Reuses P1's `ModelMetadata` and `save_model_artifact`:

```python
save_p2_model(result, artifact_dir, overwrite=False)
loaded_result = load_p2_model(model_path)
```

Saved artifacts:
- `{model_name}_{version}.joblib`: fitted pipeline + preprocessing + metadata
- `{model_name}_{version}_features.json`: selected feature names
- `{model_name}_{version}_metadata.json`: training provenance

---

## Data Flow

```
P1 Artifacts
│
├── data/features/patients/*.parquet   ──┐
├── data/splits/split_manifest.json    ──┤
└── data/features/feature_schema.json  ──┤
                                          │
                                          v
                          train_p2_from_artifacts()
                                          │
                ┌─────────────────────────┼─────────────────────────┐
                │                         │                         │
                v                         v                         v
          Load train split         Load validation           Load test split
                │                         │                         │
                v                         v                         v
          Fit models on           Evaluate at 0.5           Evaluate once
          training data           Select threshold          at validation
                │                         │                  threshold
                │                         │                         │
                └─────────────────────────┴─────────────────────────┘
                                          │
                                          v
                                 P2TrainingResult
                                          │
                ┌─────────────────────────┼─────────────────────────┐
                │                         │                         │
                v                         v                         v
        artifacts/p2/              reports/p2/            reports/p2/
        {model}/                   p2_thresholds.json     p2_training_summary.json
        {model}.joblib             p2_experiment.json
```

---

## Testing Strategy

### Unit Tests (fixtures.py)

Synthetic datasets with known properties:
- 6 features (f0...f5)
- Linear decision boundary: `y = (f0 + 0.5*f1 + noise > 0)`
- ~50% positive class
- Deliberate missingness in f0 (every 11th row)
- 120 rows per split (train/validation/test)
- Unique patient IDs per split (no leakage)

### Test Coverage

| Category | Tests | Coverage |
|----------|-------|----------|
| Model behavior | 13 | Training, prediction, reproducibility, serialization |
| Config management | 3 | Serialization, enabled models, defaults |
| Calibration | 7 | Sigmoid/isotonic fitting, application, partition validation |
| Evaluation | 6 | Metrics, threshold selection, validation enforcement |
| Feature contract | 10 | Name validation, frame validation, target isolation |
| Training protocol | 4 | Split discipline, leakage rejection, artifact writing |

All tests:
- Use synthetic data (no real patient data)
- Verify reproducibility (same seed → same predictions)
- Test error paths (invalid input → clear ValueError)
- Confirm serialization round-trips

---

## Extension Points

### Adding a New Model

1. Add config dataclass to `config.py`:
   ```python
   @dataclass
   class CatBoostConfig:
       n_estimators: int = 200
       learning_rate: float = 0.05
       ...
   ```

2. Add field to `P2Config`:
   ```python
   train_catboost: bool = True
   catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
   ```

3. Add pipeline builder to `pipelines.py`:
   ```python
   if model_name == "catboost":
       return Pipeline([("classifier", CatBoostClassifier(...))])
   ```

4. Add wrapper to `catboost_model.py`:
   ```python
   def train_catboost(X, y, config, ...):
       return train_model("catboost", X, y, config, ...)
   ```

5. Add tests to `tests/test_catboost.py`

### Adding Custom Metrics

Extend `ml.common.metrics.compute_binary_metrics` or provide a custom
evaluation function:

```python
def evaluate_with_custom_metric(y_true, y_prob, threshold):
    base_metrics = evaluate_predictions(y_true, y_prob, threshold=threshold)
    custom_metric = compute_custom_metric(y_true, y_prob)
    base_metrics["metrics"]["custom"] = custom_metric
    return base_metrics
```

### Adding Ensemble Methods

Create a new module `ml/p2_boosting/ensemble.py`:

```python
def train_stacked_ensemble(
    base_models: dict[str, P2ModelResult],
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> P2ModelResult:
    # Stack predictions from base models
    # Train meta-learner on validation
    # Return ensemble P2ModelResult
```

---

## Limitations and Future Work

### Current Limitations

1. **No Hyperparameter Tuning:** Defaults are software defaults, not optimized
2. **No Ensemble Methods:** Only individual models, no stacking/voting
3. **No Feature Selection:** Uses all P1 features (duplicate detection in P1)
4. **No Cross-Hospital Evaluation:** Hospital IDs not available in dataset
5. **LightGBM Windows DLL:** Native library issue on Windows

### Future Enhancements

1. **Automated Tuning:**
   - Optuna/GridSearch on validation split
   - Early stopping with validation loss
   - Multi-objective optimization (AUROC + calibration)

2. **Ensemble Methods:**
   - Weighted averaging of model probabilities
   - Stacked generalization (meta-learner)
   - Voting classifiers with optimal weights

3. **Advanced Calibration:**
   - Temperature scaling
   - Beta calibration
   - Reliability diagrams for calibration assessment

4. **Cross-Hospital Evaluation:**
   - Leave-one-hospital-out cross-validation
   - Domain adaptation techniques
   - Hospital-stratified bootstrapping

5. **Feature Engineering:**
   - Automated feature selection (RFE, permutation importance)
   - Interaction features
   - Temporal aggregations

6. **Production Readiness:**
   - ONNX export for deployment
   - Model monitoring and drift detection
   - A/B testing framework
   - Real-time prediction API

---

## References

- scikit-learn pipelines: https://scikit-learn.org/stable/modules/compose.html
- LightGBM: https://lightgbm.readthedocs.io/
- XGBoost: https://xgboost.readthedocs.io/
- Calibration: Platt (1999), Zadrozny & Elkan (2002)
- P1 Final Report: `docs/p1/P1_FINAL_REPORT.md`
- P1 Feature Handoff: `docs/p1/P2_P3_FEATURE_HANDOFF.md`

