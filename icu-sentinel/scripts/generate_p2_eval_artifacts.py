#!/usr/bin/env python3
"""Generate additional P2 evaluation artifacts: calibration, bootstrap, ROC/PR curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ICU_SENTINEL_ROOT = _REPO_ROOT / "icu-sentinel"
if str(_ICU_SENTINEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ICU_SENTINEL_ROOT))

from ml.p1_clinical.evaluation.calibration import (
    CalibrationResult,
    compute_calibration,
)
from ml.p2_boosting.calibration import (
    ProbabilityCalibrator,
    fit_probability_calibrator,
    apply_probability_calibrator,
)
from ml.p2_boosting.evaluation import evaluate_predictions
from ml.p2_boosting.serialization import load_p2_model
from ml.p2_boosting.training import train_p2_from_artifacts
from ml.p1_clinical.evaluation.bootstrap import compute_bootstrap_confidence_intervals
from ml.p1_clinical.evaluation.classification_metrics import compute_binary_metrics
from sklearn.metrics import roc_curve, precision_recall_curve


def generate_calibration_artifacts(
    artifact_dir: Path,
    reports_dir: Path,
    processed_dir: Path,
    split_manifest_path: Path,
) -> dict:
    """Fit calibrators on validation and evaluate on test."""
    
    result = train_p2_from_artifacts(
        processed_dir=processed_dir,
        split_manifest_path=split_manifest_path,
        feature_schema_path=None,
        artifact_dir=None,
        reports_dir=None,
        source="real",
        overwrite=False,
    )
    
    from ml.p2_boosting.base import predict_probabilities
    from ml.p1_clinical.classical_ml.data_loader import load_from_manifest_and_features
    
    dataset = load_from_manifest_and_features(
        manifest_path=split_manifest_path,
        processed_dir=processed_dir,
        schema_path=None,
        source="real",
    )
    
    val_split = dataset.get_split("validation")
    test_split = dataset.get_split("test")
    
    feature_names = result.feature_names
    
    calibration_results = {}
    
    for model_name, model_result in result.models.items():
        threshold = model_result.threshold
        
        # Validation predictions - need to sanitize infinities
        val_X = val_split.X[feature_names].replace([np.inf, -np.inf], np.nan)
        val_probs = predict_probabilities(
            model_result, val_X, threshold=threshold, already_sanitized=True
        )["predicted_probability"]
        val_y = val_split.y
        
        # Test predictions - need to sanitize infinities
        test_X = test_split.X[feature_names].replace([np.inf, -np.inf], np.nan)
        test_probs = predict_probabilities(
            model_result, test_X, threshold=threshold, already_sanitized=True
        )["predicted_probability"]
        test_y = test_split.y
        
        # Uncalibrated calibration on test
        uncal_cal = compute_calibration(
            test_y, test_probs, model_name, "1.0.0", "test", source="real"
        )
        
        # Fit calibrators on validation
        calibrators = {}
        for method in ("sigmoid", "isotonic"):
            try:
                calibrator = fit_probability_calibrator(
                    val_y, val_probs, method=method, partition="validation"
                )
                # Apply to test
                cal_probs = apply_probability_calibrator(calibrator, test_probs)
                cal_result = compute_calibration(
                    test_y, cal_probs, model_name, "1.0.0", "test", source="real"
                )
                calibrators[method] = {
                    "calibrated": True,
                    "ece": cal_result.expected_calibration_error,
                    "mce": cal_result.max_calibration_error,
                }
            except Exception as e:
                calibrators[method] = {
                    "calibrated": False,
                    "error": str(e),
                }
        
        calibration_results[model_name] = {
            "uncalibrated": {
                "ece": uncal_cal.expected_calibration_error,
                "mce": uncal_cal.max_calibration_error,
            },
            "calibrators": calibrators,
            "threshold": threshold,
        }
    
    with open(artifact_dir / "calibration_results.json", "w") as f:
        json.dump(calibration_results, f, indent=2, allow_nan=False)
    
    # Generate calibration report
    with open(reports_dir / "P2_CALIBRATION.md", "w") as f:
        f.write("# P2 Model Calibration Analysis\n\n")
        for model_name, cal in calibration_results.items():
            f.write(f"## {model_name}\n\n")
            f.write(f"- **Validation threshold**: {cal['threshold']:.3f}\n\n")
            f.write("### Uncalibrated (Test)\n")
            f.write(f"- ECE: {cal['uncalibrated']['ece']:.4f}\n")
            f.write(f"- MCE: {cal['uncalibrated']['mce']:.4f}\n\n")
            f.write("### Calibrated (Test)\n")
            for method, res in cal["calibrators"].items():
                if res.get("calibrated"):
                    f.write(f"- **{method}**: ECE={res['ece']:.4f}, MCE={res['mce']:.4f}\n")
                else:
                    f.write(f"- **{method}**: Failed - {res.get('error', 'unknown')}\n")
            f.write("\n")
    
    return calibration_results


def generate_roc_pr_artifacts(
    artifact_dir: Path,
    processed_dir: Path,
    split_manifest_path: Path,
) -> dict:
    """Generate ROC and PR curve data for all models."""
    
    from ml.p2_boosting.training import train_p2_from_artifacts
    from ml.p2_boosting.base import predict_probabilities
    from ml.p1_clinical.classical_ml.data_loader import load_from_manifest_and_features
    
    result = train_p2_from_artifacts(
        processed_dir=processed_dir,
        split_manifest_path=split_manifest_path,
        feature_schema_path=None,
        artifact_dir=None,
        reports_dir=None,
        source="real",
        overwrite=False,
    )
    
    dataset = load_from_manifest_and_features(
        manifest_path=split_manifest_path,
        processed_dir=processed_dir,
        schema_path=None,
        source="real",
    )
    
    test_split = dataset.get_split("test")
    feature_names = result.feature_names
    test_y = test_split.y
    
    roc_pr_data = {}
    
    for model_name, model_result in result.models.items():
        test_X = test_split.X[feature_names].replace([np.inf, -np.inf], np.nan)
        test_probs = predict_probabilities(
            model_result, test_X, threshold=model_result.threshold, already_sanitized=True
        )["predicted_probability"]
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(test_y, test_probs)
        # Filter out infinite thresholds
        finite_mask = np.isfinite(thresholds)
        fpr = fpr[finite_mask]
        tpr = tpr[finite_mask]
        thresholds = thresholds[finite_mask]
        # PR curve
        precision, recall, pr_thresholds = precision_recall_curve(test_y, test_probs)
        # pr_thresholds has len(precision) - 1, so we filter precision/recall to match
        finite_mask_pr = np.isfinite(pr_thresholds)
        precision = precision[:-1][finite_mask_pr]  # Last element has no threshold
        recall = recall[:-1][finite_mask_pr]
        pr_thresholds = pr_thresholds[finite_mask_pr]
        
        roc_pr_data[model_name] = {
            "roc": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
            "pr": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist(),
            },
        }
    
    # Save ROC data
    (artifact_dir / "roc_data").mkdir(exist_ok=True)
    (artifact_dir / "pr_data").mkdir(exist_ok=True)
    
    for model_name, data in roc_pr_data.items():
        with open(artifact_dir / "roc_data" / f"{model_name}.json", "w") as f:
            json.dump(data["roc"], f, allow_nan=False)
        with open(artifact_dir / "pr_data" / f"{model_name}.json", "w") as f:
            json.dump(data["pr"], f, allow_nan=False)
    
    return roc_pr_data


def generate_bootstrap_artifacts(
    artifact_dir: Path,
    reports_dir: Path,
    processed_dir: Path,
    split_manifest_path: Path,
    n_iterations: int = 100,
) -> dict:
    """Run patient-level bootstrap confidence intervals."""
    
    from ml.p2_boosting.training import train_p2_from_artifacts
    from ml.p2_boosting.base import predict_probabilities
    from ml.p1_clinical.classical_ml.data_loader import load_from_manifest_and_features
    
    result = train_p2_from_artifacts(
        processed_dir=processed_dir,
        split_manifest_path=split_manifest_path,
        feature_schema_path=None,
        artifact_dir=None,
        reports_dir=None,
        source="real",
        overwrite=False,
    )
    
    dataset = load_from_manifest_and_features(
        manifest_path=split_manifest_path,
        processed_dir=processed_dir,
        schema_path=None,
        source="real",
    )
    
    test_split = dataset.get_split("test")
    feature_names = result.feature_names
    test_y = test_split.y
    
    # Build prediction DataFrame for bootstrap
    bootstrap_data = []
    for model_name, model_result in result.models.items():
        test_X = test_split.X[feature_names].replace([np.inf, -np.inf], np.nan)
        test_probs = predict_probabilities(
            model_result, test_X, threshold=model_result.threshold, already_sanitized=True
        )["predicted_probability"]
        test_preds = (test_probs >= model_result.threshold).astype(int)
        
        for i in range(len(test_y)):
            bootstrap_data.append({
                "patient_id": test_split.patient_ids.iloc[i],
                "true_label": int(test_y.iloc[i]),
                "model": model_name,
                "predicted_probability": float(test_probs[i]),
                "predicted_class": int(test_preds[i]),
            })
    
    bootstrap_df = pd.DataFrame(bootstrap_data)
    
    # Run bootstrap for each model
    all_results = {}
    for model_name in result.models.keys():
        model_df = bootstrap_df[bootstrap_df["model"] == model_name].drop(columns=["model"])
        if len(model_df) == 0:
            continue
        
        bootstrap_result = compute_bootstrap_confidence_intervals(
            predictions_df=model_df,
            model_name=model_name,
            model_version="1.0.0",
            split="test",
            metric_names=["auroc", "auprc", "sensitivity", "specificity", "precision", "f1", "brier"],
            n_iterations=n_iterations,
            confidence_level=0.95,
            random_seed=42,
            source="real",
        )
        
        all_results[model_name] = bootstrap_result.to_dict() if hasattr(bootstrap_result, 'to_dict') else bootstrap_result
    
    with open(artifact_dir / "bootstrap_results.json", "w") as f:
        json.dump(all_results, f, indent=2, allow_nan=False)
    
    # Save CI summary as CSV
    ci_rows = []
    for model_name, res in all_results.items():
        if "metrics" in res:
            for metric_data in res["metrics"]:
                ci_rows.append({
                    "model": model_name,
                    "metric": metric_data.get("metric_name", "unknown"),
                    "point_estimate": metric_data.get("point_estimate", float("nan")),
                    "ci_lower": metric_data.get("ci_lower", float("nan")),
                    "ci_upper": metric_data.get("ci_upper", float("nan")),
                    "n_iterations": metric_data.get("n_iterations", n_iterations),
                })
    
    pd.DataFrame(ci_rows).to_csv(artifact_dir / "bootstrap_confidence_intervals.csv", index=False)
    
    # Generate report
    with open(reports_dir / "P2_STATISTICAL_VALIDATION.md", "w") as f:
        f.write("# P2 Statistical Validation - Bootstrap Confidence Intervals\n\n")
        f.write(f"**Method**: Patient-level bootstrap (n={n_iterations} iterations, 95% CI)\n\n")
        for model_name, res in all_results.items():
            f.write(f"## {model_name}\n\n")
            if "metrics" in res:
                for metric_data in res["metrics"]:
                    metric_name = metric_data.get("metric_name", "unknown")
                    pe = metric_data.get("point_estimate", float("nan"))
                    lo = metric_data.get("ci_lower", float("nan"))
                    hi = metric_data.get("ci_upper", float("nan"))
                    f.write(f"- **{metric_name}**: {pe:.4f} [{lo:.4f}, {hi:.4f}]\n")
            f.write("\n")
    
    return all_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "p2",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_ICU_SENTINEL_ROOT / "data" / "reports" / "p2",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=_ICU_SENTINEL_ROOT / "data" / "features" / "patients",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=_ICU_SENTINEL_ROOT / "data" / "splits" / "split_manifest.json",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating calibration artifacts...")
    generate_calibration_artifacts(args.artifact_dir, args.reports_dir, args.processed_dir, args.split_manifest)
    
    print("Generating ROC/PR artifacts...")
    generate_roc_pr_artifacts(args.artifact_dir, args.processed_dir, args.split_manifest)
    
    print(f"Generating bootstrap artifacts ({args.bootstrap_iterations} iterations)...")
    generate_bootstrap_artifacts(args.artifact_dir, args.reports_dir, args.processed_dir, args.split_manifest, args.bootstrap_iterations)
    
    print("All P2 evaluation artifacts generated!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())