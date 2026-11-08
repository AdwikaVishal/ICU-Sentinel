"""Audit P2 model artifacts for preprocessing pipelines and structure."""
import joblib
import json
import numpy as np
from pathlib import Path

# Inspect Extra Trees artifact
et_path = Path('icu-sentinel/artifacts/p2_fast/extra_trees/extra_trees_1.0.0.joblib')
et_artifact = joblib.load(et_path)

print('=' * 70)
print('EXTRA TREES ARTIFACT STRUCTURE')
print('=' * 70)
print(f'Type: {type(et_artifact)}')
if isinstance(et_artifact, dict):
    print(f'Keys: {list(et_artifact.keys())}')
    
    model = et_artifact.get('model')
    print(f'\nModel type: {type(model)}')
    print(f'Model class: {model.__class__.__name__ if model else "None"}')
    
    if hasattr(model, 'n_features_in_'):
        print(f'Model n_features_in_: {model.n_features_in_}')
    
    pp = et_artifact.get('preprocessing_pipeline')
    print(f'\nPreprocessing pipeline type: {type(pp)}')
    if pp is not None:
        print(f'Pipeline class: {pp.__class__.__name__}')
        if hasattr(pp, 'steps'):
            print(f'Pipeline steps:')
            for name, transformer in pp.steps:
                print(f'  - {name}: {transformer.__class__.__name__}')
                if hasattr(transformer, 'strategy'):
                    print(f'      strategy: {transformer.strategy}')
                if hasattr(transformer, 'statistics_'):
                    print(f'      statistics shape: {transformer.statistics_.shape if hasattr(transformer.statistics_, "shape") else len(transformer.statistics_)}')
    else:
        print('Preprocessing pipeline: None')
    
    meta = et_artifact.get('metadata', {})
    if meta:
        print(f'\nMetadata:')
        for k, v in meta.items():
            if not isinstance(v, (list, np.ndarray)):
                print(f'  {k}: {v}')

# Inspect XGBoost artifact  
xgb_path = Path('icu-sentinel/artifacts/p2_fast/xgboost/xgboost_1.0.0.joblib')
xgb_artifact = joblib.load(xgb_path)

print('\n' + '=' * 70)
print('XGBOOST ARTIFACT STRUCTURE')
print('=' * 70)
print(f'Type: {type(xgb_artifact)}')
if isinstance(xgb_artifact, dict):
    print(f'Keys: {list(xgb_artifact.keys())}')
    
    model = xgb_artifact.get('model')
    print(f'\nModel type: {type(model)}')
    print(f'Model class: {model.__class__.__name__ if model else "None"}')
    
    if hasattr(model, 'n_features_in_'):
        print(f'Model n_features_in_: {model.n_features_in_}')
    
    pp = xgb_artifact.get('preprocessing_pipeline')
    print(f'\nPreprocessing pipeline type: {type(pp)}')
    if pp is not None:
        print(f'Pipeline class: {pp.__class__.__name__}')
        if hasattr(pp, 'steps'):
            print(f'Pipeline steps:')
            for name, transformer in pp.steps:
                print(f'  - {name}: {transformer.__class__.__name__}')
                if hasattr(transformer, 'strategy'):
                    print(f'      strategy: {transformer.strategy}')
                if hasattr(transformer, 'statistics_'):
                    print(f'      statistics shape: {transformer.statistics_.shape if hasattr(transformer.statistics_, "shape") else len(transformer.statistics_)}')
    else:
        print('Preprocessing pipeline: None')
    
    meta = xgb_artifact.get('metadata', {})
    if meta:
        print(f'\nMetadata:')
        for k, v in meta.items():
            if not isinstance(v, (list, np.ndarray)):
                print(f'  {k}: {v}')

# Check feature lists
print('\n' + '=' * 70)
print('FEATURE LISTS')
print('=' * 70)

et_features_path = Path('icu-sentinel/artifacts/p2_fast/extra_trees/extra_trees_1.0.0_features.json')
with open(et_features_path) as f:
    et_features = json.load(f)
    
if isinstance(et_features, dict) and 'selected_features' in et_features:
    et_features = et_features['selected_features']
    
print(f'Extra Trees features: {len(et_features)}')
print(f'First 5: {et_features[:5]}')
print(f'Last 5: {et_features[-5:]}')

xgb_features_path = Path('icu-sentinel/artifacts/p2_fast/xgboost/xgboost_1.0.0_features.json')
with open(xgb_features_path) as f:
    xgb_features = json.load(f)
    
if isinstance(xgb_features, dict) and 'selected_features' in xgb_features:
    xgb_features = xgb_features['selected_features']
    
print(f'\nXGBoost features: {len(xgb_features)}')
print(f'First 5: {xgb_features[:5]}')
print(f'Last 5: {xgb_features[-5:]}')

print(f'\nFeatures match: {et_features == xgb_features}')
