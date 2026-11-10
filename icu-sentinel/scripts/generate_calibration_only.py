#!/usr/bin/env python3
"""Generate calibration artifacts for trained models."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ICU_SENTINEL_ROOT = _REPO_ROOT / "icu-sentinel"
if str(_ICU_SENTINEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ICU_SENTINEL_ROOT))

from ml.p1_clinical.evaluation.calibration import compute_calibration
from ml.p2_boosting.calibration import fit_probability_calibrator, apply_probability_calibrator
from ml.p2_boosting.base import predict_probabilities
from ml.p2_boosting.serialization import load_p2_model
from ml.p2_boosting.training import train_p2_from_artifacts
from ml.p1_clinical.classical_ml.data_loader import load_from_manifest_and_features


def main():
    artifact_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _REPO_ROOT / "artifacts" / "physionet2019" / "full_dataset_retrain"
    processed_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else _ICU_SENTINEL_ROOT / "data" / "features_full" / "patients"
    split_manifest_path = Path(sys.argv[3]) if len(sys.argv) > 3 else _ICU_SENTINEL_ROOT / "data" / "splits_full" / "split_manifest.json"
    reports_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else _ICU_SENTINEL_ROOT / "data" / "reports_full" / "p2_full_retrain"
    
    print(f"Loading dataset...")
    dataset = load_from_manifest_and_features(
        manifest_path=split_manifest_path,
        processed_dir=processed_dir,
        schema_path=None,
        source="real",
    )
    
    val_split = dataset.get_split("validation")
    test_split = dataset.get_split("test")
    
    print(f"Loading training results...")
    result = train_p2_from_artifacts(
        processed_dir=processed_dir,
        split_manifest_path=split_manifest_path,
        feature_schema_path=None,
        artifact_dir=None,
        reports_dir=None,
        source="real",
        overwrite=False,
    )
    
    feature_names = result.feature_names
    
    calibration_results = {}
    
    for model_name, model_result in result.models.items():
        print(f"Processing {model_name}...")
        threshold = model_result.threshold
        
        # Validation predictions
        val_X = val_split.X[feature_names].replace([np.inf, -np.inf], np.nan)
        val_probs = predict_probabilities(
            model_result, val_X, threshold=threshold, already_sanitized=True
        )["predicted_probability"]
        val_y = val_split.y
        
        # Test predictions
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
                print(f"  {method}: ECE={cal_result.expected_calibration_error:.4f}")
            except Exception as e:
                print(f"  {method}: FAILED - {e}")
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
        print(f"  Uncalibrated ECE: {uncal_cal.expected_calibration_error:.4f}")
    
    # Save calibration results
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "calibration_results.json", "w") as f:
        json.dump(calibration_results, f, indent=2, allow_nan=False)
    
    # Generate calibration report
    reports_dir.mkdir(parents=True, exist_ok=True)
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
    
    print("Calibration artifacts generated!")
    return 0


if __name__ == "__main__":
    sys.exit(main())