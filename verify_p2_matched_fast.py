"""Fast serialization reproducibility check for matched P2 artifacts.
Loads saved joblib models and verifies feature lists match P1's 960 features.
Does NOT re-load the full patient parquet dataset (too slow)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_ML_ROOT = Path("icu-sentinel").resolve()
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from ml.p2_boosting.serialization import load_p2_model

ARTIFACT_DIR = _ML_ROOT / "artifacts" / "p2" / "matched_6000"
FEATURES = _ML_ROOT / "data" / "reports" / "classical_ml_matched_6000" / "selected_features.json"
SUMMARY = _ML_ROOT / "data" / "reports" / "p2_matched_6000" / "p2_training_summary.json"

EXPECTED_FEATURE_HASH = "c7407bfd3a11f257a99263cd2e7fd7018c898d848f53065ce59daf5a0562a782"


def main() -> int:
    p1_feat = json.load(open(FEATURES))
    p1_features = list(p1_feat["selected_features"])
    feature_hash = hashlib.sha256(
        json.dumps(p1_features, sort_keys=True).encode()
    ).hexdigest()
    assert feature_hash == EXPECTED_FEATURE_HASH, feature_hash
    print(f"P1 feature hash OK: {feature_hash} ({len(p1_features)} features)")

    summary = json.load(open(SUMMARY))
    thresholds = summary["thresholds"]
    stored_models = summary["models"]
    all_ok = True

    for name in ("extra_trees", "xgboost"):
        joblib_path = ARTIFACT_DIR / name / f"{name}_1.0.0.joblib"
        feats_path = ARTIFACT_DIR / name / f"{name}_1.0.0_features.json"
        meta_path = ARTIFACT_DIR / name / f"{name}_1.0.0_metadata.json"
        for p in (joblib_path, feats_path, meta_path):
            assert p.exists() and p.stat().st_size > 0, f"missing/empty: {p}"

        model_result = load_p2_model(joblib_path)
        assert model_result.model is not None, f"{name}: model not loaded"
        assert list(model_result.selected_features) == p1_features, (
            f"{name}: feature list mismatch vs P1"
        )
        assert model_result.random_seed == 42
        assert model_result.model_name == name
        assert model_result.model_version == "1.0.0"

        thr = float(thresholds[name])
        sm = stored_models[name]
        assert abs(sm["threshold"] - thr) < 1e-9, f"{name}: threshold mismatch"
        assert sm["n_features"] == len(p1_features)

        # metric sanity (from stored summary)
        tm = summary["test_metrics"][name]["metrics"]
        for key in ("auroc", "auprc", "sensitivity", "specificity", "precision", "f1", "brier"):
            v = tm[key]
            assert 0.0 <= v <= 1.0, f"{name}: {key}={v} out of range"
        assert tm["auroc"] > 0.5, f"{name}: AUROC not > 0.5: {tm['auroc']}"
        assert tm["brier"] >= 0.0, f"{name}: negative Brier {tm['brier']}"

        print(f"  {name}: LOADED OK | features={len(model_result.selected_features)} "
              f"| threshold={thr} | seed={model_result.random_seed} | "
              f"test AUROC={tm['auroc']:.4f} AUPRC={tm['auprc']:.4f} "
              f"F1={tm['f1']:.4f} Brier={tm['brier']:.5f}")

    # config check
    config = json.load(open(ARTIFACT_DIR / "p2_config.json"))
    assert config["train_lightgbm"] is False
    assert config["train_extra_trees"] is True
    assert config["train_xgboost"] is True
    assert config["extra_trees"]["n_estimators"] == 200
    assert config["xgboost"]["n_estimators"] == 200
    assert config["random_seed"] == 42
    print("P2 config OK: LightGBM disabled, ET/XGB 200 est, seed 42")

    print("\nALL SERIALIZATION / FEATURE / METRIC CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
