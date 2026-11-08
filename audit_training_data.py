"""Audit training data to understand preprocessing expectations."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent / "icu-sentinel"
sys.path.insert(0, str(_ROOT))

from ml.p1_clinical.classical_ml.data_loader import (
    load_split_aware_dataset,
    SplitManifest,
    FeatureSchema,
)
from ml.p1_clinical.preprocessing.targets import DERIVED_TARGET_COLUMN

MANIFEST = _ROOT / "data" / "splits" / "split_manifest_fast_6000.json"
PROCESSED = _ROOT / "data" / "features" / "patients"
SCHEMA = _ROOT / "data" / "features" / "feature_schema.json"

print('=' * 70)
print('LOADING TRAINING DATA SAMPLE (via per-patient, no shard)')
print('=' * 70)

manifest = SplitManifest.load(MANIFEST)
schema = FeatureSchema.load(SCHEMA)

# Load FIRST 50 training patients only (fast sample)
train_manifest_sample = SplitManifest(
    patients=[p for p in manifest.patients if p.split == 'train'][:50],
    n_patients_train=50,
    n_patients_validation=0,
    n_patients_test=0,
)

dataset = load_split_aware_dataset(
    processed_dir=PROCESSED,
    split_manifest=train_manifest_sample,
    feature_schema=schema,
    source="real",
    target_column=DERIVED_TARGET_COLUMN,
    splits=("train",),
    use_shards=False,  # Per-patient loading
)

train_sample = dataset.get_split("train")
print(f'\nTrain sample loaded: {train_sample.n_rows} rows, {train_sample.n_patients} patients')

X_train = train_sample.X

print(f'\n=== DATA CHARACTERISTICS (Training Sample) ===')
print(f'Shape: {X_train.shape}')
print(f'Dtype: {X_train.dtypes.unique()}')

# Check for infinities
n_inf = np.isinf(X_train.values).sum()
n_nan = np.isnan(X_train.values).sum()
n_total = X_train.size

print(f'\nInfinities: {n_inf:,} ({100*n_inf/n_total:.4f}%)')
print(f'NaNs: {n_nan:,} ({100*n_nan/n_total:.4f}%)')
print(f'Total values: {n_total:,}')

if n_inf > 0:
    inf_cols = [col for col in X_train.columns if np.isinf(X_train[col]).any()]
    print(f'\nColumns with infinities ({len(inf_cols)}):')
    for col in inf_cols[:10]:
        n_inf_col = np.isinf(X_train[col]).sum()
        print(f'  {col}: {n_inf_col} ({100*n_inf_col/len(X_train):.2f}%)')
    if len(inf_cols) > 10:
        print(f'  ... and {len(inf_cols)-10} more')

if n_nan > 0:
    nan_cols = [col for col in X_train.columns if np.isnan(X_train[col]).any()]
    print(f'\nColumns with NaNs ({len(nan_cols)}):')
    for col in nan_cols[:10]:
        n_nan_col = np.isnan(X_train[col]).sum()
        print(f'  {col}: {n_nan_col} ({100*n_nan_col/len(X_train):.2f}%)')
    if len(nan_cols) > 10:
        print(f'  ... and {len(nan_cols)-10} more')

print('\n' + '=' * 70)
print('LOADING TEST DATA (from shard)')
print('=' * 70)

# Load test shard
test_dataset = load_split_aware_dataset(
    processed_dir=PROCESSED,
    split_manifest=manifest,
    feature_schema=schema,
    source="real",
    target_column=DERIVED_TARGET_COLUMN,
    splits=("test",),
    use_shards=True,
)

test_data = test_dataset.get_split("test")
print(f'\nTest data loaded: {test_data.n_rows} rows, {test_data.n_patients} patients')

X_test = test_data.X

print(f'\n=== DATA CHARACTERISTICS (Test Set) ===')
print(f'Shape: {X_test.shape}')
print(f'Dtype: {X_test.dtypes.unique()}')

# Check for infinities
n_inf_test = np.isinf(X_test.values).sum()
n_nan_test = np.isnan(X_test.values).sum()
n_total_test = X_test.size

print(f'\nInfinities: {n_inf_test:,} ({100*n_inf_test/n_total_test:.4f}%)')
print(f'NaNs: {n_nan_test:,} ({100*n_nan_test/n_total_test:.4f}%)')
print(f'Total values: {n_total_test:,}')

print('\n' + '=' * 70)
print('COMPARISON')
print('=' * 70)
print(f'Train sample inf rate: {100*n_inf/n_total:.4f}%')
print(f'Test set inf rate: {100*n_inf_test/n_total_test:.4f}%')
print(f'Train sample NaN rate: {100*n_nan/n_total:.4f}%')
print(f'Test set NaN rate: {100*n_nan_test/n_total_test:.4f}%')

print('\n=== CONCLUSION ===')
if n_inf > 0 or n_nan > 0:
    print('Training data contains infinities/NaNs - SimpleImputer would be applied')
if n_inf_test > 0 or n_nan_test > 0:
    print('Test data contains infinities/NaNs - needs sanitization')
