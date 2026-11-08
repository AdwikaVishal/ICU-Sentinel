"""Compare P1 and P2 feature lists."""
import json
from pathlib import Path

# Load P1 features
p1_path = Path('icu-sentinel/artifacts/p1/classical_ml_matched_6000/logistic_regression_features.json')
with open(p1_path) as f:
    p1_features = json.load(f)

# Load P2 features
p2_path = Path('icu-sentinel/artifacts/p2_fast/xgboost/xgboost_1.0.0_features.json')
with open(p2_path) as f:
    p2_data = json.load(f)
    p2_features = p2_data['selected_features'] if isinstance(p2_data, dict) else p2_data

print('=' * 70)
print('P1 vs P2 FEATURE COMPARISON')
print('=' * 70)
print(f'P1 features: {len(p1_features)}')
print(f'P2 features: {len(p2_features)}')
print(f'Difference: {len(p2_features) - len(p1_features)}')

# Find differences
p1_set = set(p1_features)
p2_set = set(p2_features)

only_p2 = p2_set - p1_set
only_p1 = p1_set - p2_set

print(f'\nFeatures only in P2 ({len(only_p2)}):')
for feat in sorted(only_p2):
    print(f'  {feat}')

print(f'\nFeatures only in P1 ({len(only_p1)}):')
for feat in sorted(only_p1):
    print(f'  {feat}')

# Check if it's *_hours_since_last
hours_since_p1 = [f for f in p1_features if f.endswith('_hours_since_last')]
hours_since_p2 = [f for f in p2_features if f.endswith('_hours_since_last')]

print(f'\n*_hours_since_last features:')
print(f'  P1: {len(hours_since_p1)}')
print(f'  P2: {len(hours_since_p2)}')

if len(hours_since_p1) != len(hours_since_p2):
    print(f'\n  Only in P2: {sorted(set(hours_since_p2) - set(hours_since_p1))}')
    print(f'  Only in P1: {sorted(set(hours_since_p1) - set(hours_since_p2))}')
