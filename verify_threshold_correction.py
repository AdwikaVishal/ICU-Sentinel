"""Verify P2 threshold correction
Quick verification that thresholds are loaded correctly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

_ROOT = Path(__file__).parent / "icu-sentinel"
sys.path.insert(0, str(_ROOT))

from ml.p1_clinical.classical_ml.data_loader import (
    load_split_aware_dataset,
    SplitManifest,
    FeatureSchema,
)
from ml.p1_clinical.preprocessing.targets import DERIVED_TARGET_COLUMN

# Configuration
MANIFEST_PATH = _ROOT / "data" / "splits" / "split_manifest_fast_6000.json"
PROCESSED_DIR = _ROOT / "data" / "features" / "patients"
SCHEMA_PATH = _ROOT / "data" / "features" / "feature_schema.json"
P2_ARTIFACTS = _ROOT / "artifacts" / "p2_fast"
P2_REPORTS = _ROOT / "data" / "reports" / "p2_fast"
TARGET_COLUMN = DERIVED_TARGET_COLUMN

MODELS = {
    "extra_trees": {
        "path": P2_ARTIFACTS / "extra_trees" / "extra_trees_1.0.0.joblib",
        "features_path": P2_ARTIFACTS / "extra_trees" / "extra_trees_1.0.0_features.json",
    },
    "xgboost": {
        "path": P2_ARTIFACTS / "xgboost" / "xgboost_1.0.0.joblib",
        "features_path": P2_ARTIFACTS / "xgboost" / "xgboost_1.0.0_features.json",
    },
}


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def main():
    print("=" * 70)
    print("P2 THRESHOLD CORRECTION VERIFICATION")
    print("=" * 70)
    
    # Load threshold file
    print("\n[1] Loading validation-selected thresholds...")
    thresholds_path = P2_REPORTS / "p2_thresholds.json"
    with open(thresholds_path) as f:
        thresholds_data = json.load(f)
    
    print(f"  Selection split: {thresholds_data['selection_split']}")
    print(f"  Selection metric: {thresholds_data['selection_metric']}")
    print(f"  Test set used: {thresholds_data['test_set_used_for_selection']}")
    print("  Thresholds:")
    for model_name, threshold in thresholds_data["thresholds"].items():
        print(f"    {model_name}: {threshold:.4f}")
    
    # Load test data
    print("\n[2] Loading test data...")
    manifest = SplitManifest.load(MANIFEST_PATH)
    schema = FeatureSchema.load(SCHEMA_PATH)
    
    dataset = load_split_aware_dataset(
        processed_dir=PROCESSED_DIR,
        split_manifest=manifest,
        feature_schema=schema,
        source="real",
        target_column=TARGET_COLUMN,
        splits=("test",),
        use_shards=True,
    )
    
    test_data = dataset.get_split("test")
    X_test = test_data.X
    y_test = test_data.y.values
    
    print(f"  Test samples: {len(y_test)}")
    print(f"  Test positives: {y_test.sum()} ({100*y_test.mean():.2f}%)")
    
    # Evaluate each model with validation thresholds
    print("\n[3] Evaluating models with validation-selected thresholds...")
    
    results = {}
    for model_name, model_config in MODELS.items():
        print(f"\n  [{model_name.upper()}]")
        
        model_artifact = joblib.load(model_config["path"])
        if isinstance(model_artifact, dict) and 'model' in model_artifact:
            model = model_artifact['model']
        else:
            model = model_artifact
        
        with open(model_config["features_path"]) as f:
            features_data = json.load(f)
            if isinstance(features_data, dict) and "selected_features" in features_data:
                model_features = features_data["selected_features"]
            else:
                model_features = features_data
        
        X_test_model = X_test[model_features]
        X_test_model = X_test_model.replace([np.inf, -np.inf], np.finfo(np.float32).max)
        X_test_model = X_test_model.fillna(0)
        
        y_prob = model.predict_proba(X_test_model)[:, 1]
        
        # Get validation-selected threshold
        val_threshold = thresholds_data["thresholds"][model_name]
        
        # Compute metrics with validation threshold
        metrics_val = compute_metrics(y_test, y_prob, val_threshold)
        
        # Compute metrics with 0.5 (for comparison)
        metrics_50 = compute_metrics(y_test, y_prob, 0.5)
        
        print(f"    Validation threshold: {val_threshold:.4f}")
        print(f"      AUROC: {metrics_val['auroc']:.4f}")
        print(f"      AUPRC: {metrics_val['auprc']:.4f}")
        print(f"      F1: {metrics_val['f1']:.4f}")
        print(f"      Precision: {metrics_val['precision']:.4f}")
        print(f"      Recall: {metrics_val['recall']:.4f}")
        
        print(f"    Threshold 0.5 (for comparison):")
        print(f"      AUROC: {metrics_50['auroc']:.4f}")
        print(f"      AUPRC: {metrics_50['auprc']:.4f}")
        print(f"      F1: {metrics_50['f1']:.4f}")
        print(f"      Precision: {metrics_50['precision']:.4f}")
        print(f"      Recall: {metrics_50['recall']:.4f}")
        
        results[model_name] = {
            "validation_threshold": metrics_val,
            "threshold_0.5": metrics_50,
        }
    
    # Save corrected results
    print("\n[4] Saving corrected results...")
    output_path = P2_REPORTS / "p2_advanced_results_corrected.json"
    
    corrected_results = {
        "manifest": str(MANIFEST_PATH),
        "n_patients_test": int(test_data.n_patients),
        "n_rows_test": int(test_data.n_rows),
        "n_positive_test": int(test_data.n_positive),
        "prevalence_test": float(test_data.n_positive / test_data.n_rows),
        "n_features": len(X_test.columns),
        "target_column": TARGET_COLUMN,
        "random_seed": 42,
        "threshold_correction_applied": True,
        "models": {
            model_name: {
                "metrics": results[model_name]["validation_threshold"],
            }
            for model_name in results
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(corrected_results, f, indent=2)
    
    print(f"  Saved to: {output_path}")
    
    # Compare with old results
    print("\n[5] Comparison with old (incorrect) results...")
    old_results_path = P2_REPORTS / "p2_advanced_results.json"
    if old_results_path.exists():
        with open(old_results_path) as f:
            old_results = json.load(f)
        
        for model_name in results:
            if model_name in old_results["models"]:
                old_metrics = old_results["models"][model_name]["metrics"]
                new_metrics = corrected_results["models"][model_name]["metrics"]
                
                print(f"\n  [{model_name.upper()}]")
                print(f"    Old threshold: {old_metrics['threshold']:.4f}")
                print(f"    New threshold: {new_metrics['threshold']:.4f}")
                print(f"    F1 change: {old_metrics['f1']:.4f} -> {new_metrics['f1']:.4f}")
                print(f"    Precision change: {old_metrics['precision']:.4f} -> {new_metrics['precision']:.4f}")
                print(f"    Recall change: {old_metrics['recall']:.4f} -> {new_metrics['recall']:.4f}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
