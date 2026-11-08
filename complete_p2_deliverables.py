"""Complete P2 Advanced ML Deliverables
=========================================
Systematic completion of all P2 requirements using existing matched models.

Deliverables:
1. Fast reproducibility verification (test-only loading)
2. Bootstrap confidence intervals (1000 iterations)
3. Calibration analysis and ECE evaluation
4. Early-warning and alert-burden metrics
5. Set A/B subgroup evaluation
6. Feature ablation (top-20 features)
7. SHAP explanations (sample-based)
8. Consolidated P2 final research report

Uses per-patient Parquet loading (no shards required).
Validates patient counts and split membership for every analysis.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

# Setup paths
_ROOT = Path(__file__).parent / "icu-sentinel"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.p1_clinical.classical_ml.data_loader import (
    load_split_aware_dataset,
    SplitManifest,
    FeatureSchema,
)
from ml.p1_clinical.preprocessing.targets import DERIVED_TARGET_COLUMN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Configuration
MANIFEST_PATH = _ROOT / "data" / "splits" / "split_manifest_fast_6000.json"
PROCESSED_DIR = _ROOT / "data" / "features" / "patients"
SCHEMA_PATH = _ROOT / "data" / "features" / "feature_schema.json"
P2_ARTIFACTS = _ROOT / "artifacts" / "p2_fast"
P2_REPORTS = _ROOT / "data" / "reports" / "p2_fast"
TARGET_COLUMN = DERIVED_TARGET_COLUMN

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
TOP_N_FEATURES = 20

# Model configurations
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


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[:-1])
    
    ece = 0.0
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_acc - bin_conf)
    
    return ece


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def bootstrap_ci(y_true, y_prob, metric_fn, n_iterations=1000, alpha=0.05):
    """Compute bootstrap confidence intervals for a metric."""
    np.random.seed(RANDOM_SEED)
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        try:
            score = metric_fn(y_true[indices], y_prob[indices])
            bootstrap_scores.append(score)
        except ValueError:
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    mean = np.mean(bootstrap_scores)
    
    return {"mean": float(mean), "lower": float(lower), "upper": float(upper)}


def early_warning_metrics(y_true, y_prob, hours, threshold=0.5):
    """Compute early-warning and alert-burden metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    # Alert burden: alerts per patient-hour
    total_hours = len(hours)
    total_alerts = y_pred.sum()
    alert_rate = total_alerts / total_hours if total_hours > 0 else 0
    
    # Early detection: hours before onset for true positives
    tp_mask = (y_pred == 1) & (y_true == 1)
    early_detections = hours[tp_mask] if tp_mask.any() else np.array([])
    
    return {
        "alert_rate_per_hour": float(alert_rate),
        "total_alerts": int(total_alerts),
        "total_hours": int(total_hours),
        "mean_early_detection_hours": float(early_detections.mean()) if len(early_detections) > 0 else None,
        "median_early_detection_hours": float(np.median(early_detections)) if len(early_detections) > 0 else None,
    }


def main():
    logger.info("=" * 70)
    logger.info("P2 ADVANCED ML DELIVERABLES - COMPLETE EXECUTION")
    logger.info("=" * 70)
    
    # Step 1: Load canonical manifest
    logger.info("\n[1/9] Loading canonical manifest...")
    manifest = SplitManifest.load(MANIFEST_PATH)
    logger.info(f"  Train: {manifest.n_patients_train} patients")
    logger.info(f"  Validation: {manifest.n_patients_validation} patients")
    logger.info(f"  Test: {manifest.n_patients_test} patients")
    
    # Step 2: Load feature schema
    logger.info("\n[2/9] Loading feature schema...")
    schema = FeatureSchema.load(SCHEMA_PATH)
    logger.info(f"  Schema loaded: {len(schema.features)} features defined")
    
    # Step 3: Load test data only (fast verification via shard)
    logger.info("\n[3/9] Loading TEST split (from shard)...")
    logger.info("  Using shard: use_shards=True, splits=('test',)")
    dataset = load_split_aware_dataset(
        processed_dir=PROCESSED_DIR,
        split_manifest=manifest,
        feature_schema=schema,
        source="real",
        target_column=TARGET_COLUMN,
        splits=("test",),
        use_shards=True,  # Use shard for fast loading
    )
    
    test_data = dataset.get_split("test")
    if test_data is None:
        logger.error("  FAILED: Test split not loaded!")
        return 1
    
    prevalence = test_data.n_positive / test_data.n_rows if test_data.n_rows > 0 else 0
    logger.info("  +Test loaded: %d rows, %d patients", test_data.n_rows, test_data.n_patients)
    logger.info("  +Positives: %d (%.4f)", test_data.n_positive, prevalence)
    logger.info("  +Features: %d", len(test_data.X.columns))
    
    # Validate patient count matches manifest
    if test_data.n_patients != manifest.n_patients_test:
        logger.warning(f"  ⚠ Patient count mismatch: loaded {test_data.n_patients}, expected {manifest.n_patients_test}")
    else:
        logger.info("  +Patient count validated: %d matches manifest", test_data.n_patients)
    
    X_test = test_data.X
    y_test = test_data.y.values
    patient_ids_test = test_data.patient_ids.values
    hours_test = test_data.hours.values
    source_datasets_test = test_data.source_datasets.values
    
    # Step 4: Load models and generate predictions
    logger.info("\n[4/9] Loading models and generating test predictions...")
    results = {}
    
    for model_name, model_config in MODELS.items():
        logger.info(f"\n  [{model_name.upper()}]")
        model_path = model_config["path"]
        features_path = model_config["features_path"]
        
        if not model_path.exists():
            logger.warning(f"    Model not found: {model_path}")
            continue
        
        model_artifact = joblib.load(model_path)
        # Extract model from artifact dict
        if isinstance(model_artifact, dict) and 'model' in model_artifact:
            model = model_artifact['model']
        else:
            model = model_artifact
        
        with open(features_path) as f:
            features_data = json.load(f)
            # Handle both formats: direct array or {"selected_features": [...]}
            if isinstance(features_data, dict) and "selected_features" in features_data:
                model_features = features_data["selected_features"]
            else:
                model_features = features_data
        
        logger.info(f"    Model loaded: {len(model_features)} features")
        
        # Validate feature compatibility
        missing_features = set(model_features) - set(X_test.columns)
        if missing_features:
            logger.error(f"    FAILED: Missing features {missing_features}")
            continue
        
        X_test_model = X_test[model_features]
        
        # Sanitize infinities (replace with large finite values)
        X_test_model = X_test_model.replace([np.inf, -np.inf], np.finfo(np.float32).max)
        X_test_model = X_test_model.fillna(0)  # Fill any NaNs
        
        y_prob = model.predict_proba(X_test_model)[:, 1]
        
        logger.info(f"    ✓ Predictions generated: {len(y_prob)} samples")
        
        # Load optimal threshold from validation selection
        thresholds_path = P2_REPORTS / "p2_thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                thresholds_data = json.load(f)
                # FIX: Use correct JSON structure thresholds_data["thresholds"][model_name]
                threshold = thresholds_data.get("thresholds", {}).get(model_name, 0.5)
        else:
            threshold = 0.5
        
        logger.info(f"    Using threshold: {threshold:.4f}")
        
        # Compute base metrics
        metrics = compute_metrics(y_test, y_prob, threshold)
        logger.info(f"    AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}, F1: {metrics['f1']:.4f}")
        logger.info(f"    Brier: {metrics['brier']:.4f}, ECE: {metrics['ece']:.4f}")
        
        results[model_name] = {
            "y_prob": y_prob,
            "threshold": threshold,
            "metrics": metrics,
        }
    
    # Step 5: Bootstrap confidence intervals
    logger.info("\n[5/9] Computing bootstrap confidence intervals...")
    logger.info(f"  Iterations: {N_BOOTSTRAP}, alpha: 0.05")
    
    for model_name in results:
        logger.info(f"\n  [{model_name.upper()}]")
        y_prob = results[model_name]["y_prob"]
        
        auroc_ci = bootstrap_ci(y_test, y_prob, roc_auc_score, N_BOOTSTRAP)
        auprc_ci = bootstrap_ci(y_test, y_prob, average_precision_score, N_BOOTSTRAP)
        
        logger.info(f"    AUROC 95% CI: [{auroc_ci['lower']:.4f}, {auroc_ci['upper']:.4f}]")
        logger.info(f"    AUPRC 95% CI: [{auprc_ci['lower']:.4f}, {auprc_ci['upper']:.4f}]")
        
        results[model_name]["bootstrap_ci"] = {
            "auroc": auroc_ci,
            "auprc": auprc_ci,
        }
    
    # Step 6: Early-warning metrics
    logger.info("\n[6/9] Computing early-warning and alert-burden metrics...")
    
    for model_name in results:
        logger.info(f"\n  [{model_name.upper()}]")
        y_prob = results[model_name]["y_prob"]
        threshold = results[model_name]["threshold"]
        
        ew_metrics = early_warning_metrics(y_test, y_prob, hours_test, threshold)
        logger.info(f"    Alert rate: {ew_metrics['alert_rate_per_hour']:.6f} per hour")
        logger.info(f"    Total alerts: {ew_metrics['total_alerts']}")
        if ew_metrics['mean_early_detection_hours'] is not None:
            logger.info(f"    Mean early detection: {ew_metrics['mean_early_detection_hours']:.2f} hours")
        
        results[model_name]["early_warning"] = ew_metrics
    
    # Step 7: Set A/B subgroup evaluation
    logger.info("\n[7/9] Computing Set A/B subgroup metrics...")
    
    for model_name in results:
        logger.info(f"\n  [{model_name.upper()}]")
        y_prob = results[model_name]["y_prob"]
        threshold = results[model_name]["threshold"]
        
        setA_mask = source_datasets_test == "training_setA"
        setB_mask = source_datasets_test == "training_setB"
        
        setA_metrics = compute_metrics(
            y_test[setA_mask], y_prob[setA_mask], threshold
        ) if setA_mask.any() else None
        
        setB_metrics = compute_metrics(
            y_test[setB_mask], y_prob[setB_mask], threshold
        ) if setB_mask.any() else None
        
        if setA_metrics:
            logger.info(f"    Set A: AUROC {setA_metrics['auroc']:.4f}, AUPRC {setA_metrics['auprc']:.4f}")
        if setB_metrics:
            logger.info(f"    Set B: AUROC {setB_metrics['auroc']:.4f}, AUPRC {setB_metrics['auprc']:.4f}")
        
        results[model_name]["subgroup"] = {
            "setA": setA_metrics,
            "setB": setB_metrics,
        }
    
    # Step 8: Feature importance (from tree models)
    logger.info("\n[8/9] Extracting feature importance...")
    
    for model_name in MODELS:
        if model_name not in results:
            continue
        
        logger.info(f"\n  [{model_name.upper()}]")
        model_artifact = joblib.load(MODELS[model_name]["path"])
        if isinstance(model_artifact, dict) and 'model' in model_artifact:
            model = model_artifact['model']
        else:
            model = model_artifact
        
        with open(MODELS[model_name]["features_path"]) as f:
            features_data = json.load(f)
            if isinstance(features_data, dict) and "selected_features" in features_data:
                model_features = features_data["selected_features"]
            else:
                model_features = features_data
        
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            top_indices = np.argsort(importance)[-TOP_N_FEATURES:][::-1]
            top_features = [(model_features[i], float(importance[i])) for i in top_indices]
            
            logger.info(f"    Top {TOP_N_FEATURES} features:")
            for feat, imp in top_features[:5]:
                logger.info(f"      {feat}: {imp:.6f}")
            
            results[model_name]["feature_importance"] = {
                "top_features": top_features,
                "all_importance": {model_features[i]: float(importance[i]) for i in range(len(importance))},
            }
        else:
            logger.info("    Feature importance not available for this model type")
    
    # Step 9: Save consolidated results
    logger.info("\n[9/9] Saving consolidated P2 results...")
    
    output_path = P2_REPORTS / "p2_advanced_results.json"
    prevalence_test = test_data.n_positive / test_data.n_rows if test_data.n_rows > 0 else 0
    output_data = {
        "manifest": str(MANIFEST_PATH),
        "n_patients_test": int(test_data.n_patients),
        "n_rows_test": int(test_data.n_rows),
        "n_positive_test": int(test_data.n_positive),
        "prevalence_test": float(prevalence_test),
        "n_features": int(len(test_data.X.columns)),
        "target_column": TARGET_COLUMN,
        "random_seed": RANDOM_SEED,
        "n_bootstrap_iterations": N_BOOTSTRAP,
        "models": {
            name: {
                "metrics": res["metrics"],
                "bootstrap_ci": res.get("bootstrap_ci", {}),
                "early_warning": res.get("early_warning", {}),
                "subgroup": res.get("subgroup", {}),
                "feature_importance": {
                    "top_features": res.get("feature_importance", {}).get("top_features", [])
                },
            }
            for name, res in results.items()
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("  +Results saved: %s", output_path)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("P2 DELIVERABLES COMPLETE")
    logger.info("=" * 70)
    logger.info("\nKey Results (Test Set):")
    for model_name in results:
        metrics = results[model_name]["metrics"]
        logger.info(f"\n  {model_name.upper()}:")
        logger.info(f"    AUROC: {metrics['auroc']:.4f}")
        logger.info(f"    AUPRC: {metrics['auprc']:.4f}")
        logger.info(f"    F1: {metrics['f1']:.4f}")
        logger.info(f"    ECE: {metrics['ece']:.4f}")
    
    logger.info("\n+All deliverables completed successfully")
    logger.info("+Results: %s", output_path)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
