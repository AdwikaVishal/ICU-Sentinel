"""Reproducibility check for matched P2 artifacts: reload saved models and
re-predict on the test split, comparing to stored test metrics."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_ML_ROOT = _REPO / "icu-sentinel"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

import numpy as np

from ml.p1_clinical.classical_ml.data_loader import load_from_manifest_and_features
from ml.p2_boosting.serialization import load_p2_model
from ml.p2_boosting.base import predict_probabilities
from ml.p2_boosting.evaluation import evaluate_predictions

MANIFEST = _ML_ROOT / "data" / "splits" / "split_manifest_fast_6000.json"
FEATURES = _ML_ROOT / "data" / "reports" / "classical_ml_matched_6000" / "selected_features.json"
ARTIFACT_DIR = _ML_ROOT / "artifacts" / "p2" / "matched_6000"
SUMMARY = _ML_ROOT / "data" / "reports" / "p2_matched_6000" / "p2_training_summary.json"


def main() -> int:
    p1_feat = json.load(open(FEATURES))
    p1_features = list(p1_feat["selected_features"])
    feature_hash = hashlib.sha256(
        json.dumps(p1_features, sort_keys=True).encode()
    ).hexdigest()
    assert feature_hash == "c7407bfd3a11f257a99263cd2e7fd7018c898d848f53065ce59daf5a0562a782", feature_hash
    print(f"P1 feature hash OK: {feature_hash} ({len(p1_features)} features)")

    manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert manifest_hash == "008a93f47c4264e3b61d73121be17efc1dae0f69473289d238d61bcc881188b1", manifest_hash
    print(f"Manifest hash OK: {manifest_hash}")

    dataset = load_from_manifest_and_features(
        manifest_path=MANIFEST,
        processed_dir=_ML_ROOT / "data" / "features" / "patients",
        schema_path=_ML_ROOT / "data" / "features" / "feature_schema.json",
        source="real",
    )
    test = dataset.get_split("test")
    assert test is not None and len(test.X) > 0
    X_test = test.X[p1_features]
    y_test = test.y
    print(f"Test split: {len(test.X)} rows, {test.n_patients} patients, "
          f"{int((y_test==1).sum())} positives")

    summary = json.load(open(SUMMARY))
    stored = summary["test_metrics"]
    thresholds = summary["thresholds"]

    all_ok = True
    for name in ("extra_trees", "xgboost"):
        model_result = load_p2_model(ARTIFACT_DIR / name / f"{name}_1.0.0.joblib")
        assert model_result.model is not None, f"{name} model not loaded"
        assert list(model_result.selected_features) == p1_features, f"{name} feature mismatch"
        thr = float(thresholds["thresholds"][name])
        preds = predict_probabilities(model_result, X_test, threshold=thr, already_sanitized=False)
        metrics = evaluate_predictions(y_test, preds["predicted_probability"], threshold=thr,
                                       model_name=name, split="test")
        m = metrics["metrics"]; sm = stored[name]["metrics"]
        print(f"\n--- {name} (threshold={thr}) ---")
        for key in ("auroc", "auprc", "sensitivity", "specificity", "precision", "f1", "brier"):
            a, b = m[key], sm[key]
            ok = np.isfinite(a) and abs(a - b) < 1e-6
            all_ok &= ok
            print(f"  {key}: recomputed={a:.6f} stored={b:.6f} match={ok}")
        for key in ("n_rows", "n_positive"):
            a, b = metrics[key], stored[name][key]
            ok = a == b
            all_ok &= ok
            print(f"  {key}: recomputed={a} stored={b} match={ok}")
        cm_ok = (metrics["confusion_matrix"] == stored[name]["confusion_matrix"])
        all_ok &= cm_ok
        print(f"  confusion_matrix match={cm_ok}")
        # metric range checks
        for key in ("auroc", "auprc", "sensitivity", "specificity", "precision", "f1", "brier"):
            assert 0.0 <= m[key] <= 1.0, f"{name} {key} out of range: {m[key]}"

    print(f"\nALL REPRODUCIBILITY CHECKS PASS: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
