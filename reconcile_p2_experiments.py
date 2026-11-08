"""Reconcile all P2 experiments - Extract key configuration and results"""
import json
import hashlib
from pathlib import Path

# Define experiment locations
EXPERIMENTS = {
    "p2_fast": {
        "artifacts": Path("icu-sentinel/artifacts/p2_fast"),
        "reports": Path("icu-sentinel/data/reports/p2_fast"),
    },
    "p2_matched_6000": {
        "artifacts": Path("icu-sentinel/artifacts/p2"),  # Note: different path
        "reports": Path("icu-sentinel/data/reports/p2_matched_6000"),
    },
}

def hash_features(features):
    """Generate hash of feature list"""
    return hashlib.sha256(json.dumps(sorted(features)).encode()).hexdigest()[:16]

def extract_experiment_info(exp_name, paths):
    """Extract all key information from an experiment"""
    info = {"name": exp_name}
    
    # Read training summary
    summary_path = paths["reports"] / "p2_training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        
        info["config"] = summary.get("config", {})
        info["feature_count"] = len(summary.get("feature_names", []))
        info["feature_hash"] = hash_features(summary.get("feature_names", []))
        info["manifest_hash"] = summary.get("manifest_sha256", "N/A")[:16]
        info["random_seed"] = summary.get("config", {}).get("random_seed", "N/A")
        info["source"] = summary.get("source", "N/A")
        
        # Extract hyperparameters
        et_config = summary.get("config", {}).get("extra_trees", {})
        xgb_config = summary.get("config", {}).get("xgboost", {})
        
        info["et_n_estimators"] = et_config.get("n_estimators", "N/A")
        info["et_max_depth"] = et_config.get("max_depth", "null")
        info["et_class_weight"] = et_config.get("class_weight", "N/A")
        
        info["xgb_n_estimators"] = xgb_config.get("n_estimators", "N/A")
        info["xgb_max_depth"] = xgb_config.get("max_depth", "N/A")
        info["xgb_learning_rate"] = xgb_config.get("learning_rate", "N/A")
        
        # Extract thresholds
        info["thresholds"] = summary.get("thresholds", {})
        
        # Extract test metrics
        test_metrics = summary.get("test_metrics", {})
        if "extra_trees" in test_metrics:
            et_test = test_metrics["extra_trees"].get("metrics", {})
            info["et_test_auroc"] = et_test.get("auroc", "N/A")
            info["et_test_auprc"] = et_test.get("auprc", "N/A")
            info["et_test_brier"] = et_test.get("brier", "N/A")
            info["et_test_threshold"] = test_metrics["extra_trees"].get("threshold", "N/A")
        
        if "xgboost" in test_metrics:
            xgb_test = test_metrics["xgboost"].get("metrics", {})
            info["xgb_test_auroc"] = xgb_test.get("auroc", "N/A")
            info["xgb_test_auprc"] = xgb_test.get("auprc", "N/A")
            info["xgb_test_brier"] = xgb_test.get("brier", "N/A")
            info["xgb_test_threshold"] = test_metrics["xgboost"].get("threshold", "N/A")
    
    # Read experiment.json if different
    exp_path = paths["reports"] / "p2_experiment.json"
    if exp_path.exists():
        with open(exp_path) as f:
            exp = json.load(f)
        
        # Check if this has different config
        exp_config = exp.get("config", {})
        if exp_config:
            info["alt_config"] = {
                "et_n_estimators": exp_config.get("extra_trees", {}).get("n_estimators"),
                "xgb_n_estimators": exp_config.get("xgboost", {}).get("n_estimators"),
            }
        
        # Check if has different test metrics
        if "test_metrics" in exp:
            info["alt_test_metrics"] = {}
            tm = exp["test_metrics"]
            if "extra_trees" in tm:
                info["alt_test_metrics"]["et_auroc"] = tm["extra_trees"].get("auroc")
                info["alt_test_metrics"]["et_auprc"] = tm["extra_trees"].get("auprc")
            if "xgboost" in tm:
                info["alt_test_metrics"]["xgb_auroc"] = tm["xgboost"].get("auroc")
                info["alt_test_metrics"]["xgb_auprc"] = tm["xgboost"].get("auprc")
    
    # Check model metadata
    et_meta = paths["artifacts"] / "extra_trees" / "extra_trees_1.0.0_metadata.json"
    if et_meta.exists():
        with open(et_meta) as f:
            meta = json.load(f)
        info["et_timestamp"] = meta.get("training_timestamp", "N/A")
        info["et_n_features"] = len(meta.get("selected_feature_names", []))
    
    xgb_meta = paths["artifacts"] / "xgboost" / "xgboost_1.0.0_metadata.json"
    if xgb_meta.exists():
        with open(xgb_meta) as f:
            meta = json.load(f)
        info["xgb_timestamp"] = meta.get("training_timestamp", "N/A")
        info["xgb_n_features"] = len(meta.get("selected_feature_names", []))
    
    return info

def main():
    print("=" * 100)
    print("P2 EXPERIMENT RECONCILIATION")
    print("=" * 100)
    
    results = {}
    for name, paths in EXPERIMENTS.items():
        print(f"\nExtracting: {name}...")
        results[name] = extract_experiment_info(name, paths)
    
    # Save full results
    output_path = Path("icu-sentinel/data/reports/remediation/p2_experiment_reconciliation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Full results saved to: {output_path}")
    
    # Print comparison table
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    
    for name, info in results.items():
        print(f"\n### {name.upper()} ###")
        print(f"Features: {info.get('feature_count', 'N/A')}")
        print(f"Feature Hash: {info.get('feature_hash', 'N/A')}")
        print(f"Manifest Hash: {info.get('manifest_hash', 'N/A')}")
        print(f"Random Seed: {info.get('random_seed', 'N/A')}")
        print(f"Source: {info.get('source', 'N/A')}")
        
        print(f"\nExtra Trees:")
        print(f"  n_estimators: {info.get('et_n_estimators', 'N/A')}")
        print(f"  max_depth: {info.get('et_max_depth', 'N/A')}")
        print(f"  class_weight: {info.get('et_class_weight', 'N/A')}")
        print(f"  Threshold: {info.get('et_test_threshold', 'N/A')}")
        print(f"  Test AUROC: {info.get('et_test_auroc', 'N/A')}")
        print(f"  Test AUPRC: {info.get('et_test_auprc', 'N/A')}")
        print(f"  Test Brier: {info.get('et_test_brier', 'N/A')}")
        
        print(f"\nXGBoost:")
        print(f"  n_estimators: {info.get('xgb_n_estimators', 'N/A')}")
        print(f"  max_depth: {info.get('xgb_max_depth', 'N/A')}")
        print(f"  learning_rate: {info.get('xgb_learning_rate', 'N/A')}")
        print(f"  Threshold: {info.get('xgb_test_threshold', 'N/A')}")
        print(f"  Test AUROC: {info.get('xgb_test_auroc', 'N/A')}")
        print(f"  Test AUPRC: {info.get('xgb_test_auprc', 'N/A')}")
        print(f"  Test Brier: {info.get('xgb_test_brier', 'N/A')}")
        
        if "alt_test_metrics" in info:
            print(f"\nALTERNATE TEST METRICS (from p2_experiment.json):")
            alt = info["alt_test_metrics"]
            print(f"  ET AUROC: {alt.get('et_auroc', 'N/A')}")
            print(f"  ET AUPRC: {alt.get('et_auprc', 'N/A')}")
            print(f"  XGB AUROC: {alt.get('xgb_auroc', 'N/A')}")
            print(f"  XGB AUPRC: {alt.get('xgb_auprc', 'N/A')}")

if __name__ == "__main__":
    main()
