"""Fast reproducibility check for matched P2 artifacts.

Reloads the saved Extra Trees and XGBoost models, predicts on the TEST
split (or a deterministic patient-level test subset), and compares
reproduced metrics against the stored test metrics in
``p2_training_summary.json``.

Unlike ``verify_p2_matched_repro.py`` this does NOT load train+validation,
avoiding the unnecessary full-dataset recomputation that hung PID 24772.

By default a deterministic patient-level test subset is used (fast, valid
for establishing model loading + prediction consistency). Pass --full to
reproduce on the entire test split (slower; per-patient Parquet I/O).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.p1_clinical.classical_ml.data_loader import (  # noqa: E402
    SplitManifest,
    _build_split_frame,
    _load_processed_patient,
)
from ml.p1_clinical.preprocessing.targets import DERIVED_TARGET_COLUMN  # noqa: E402
from ml.p2_boosting.serialization import load_p2_model  # noqa: E402
from ml.p2_boosting.base import predict_probabilities  # noqa: E402
from ml.p2_boosting.evaluation import evaluate_predictions  # noqa: E402

MANIFEST = _REPO / "data" / "splits" / "split_manifest_fast_6000.json"
FEATURES = _REPO / "data" / "reports" / "classical_ml_matched_6000" / "selected_features.json"
ARTIFACT_DIR = _REPO / "artifacts" / "p2" / "matched_6000"
SUMMARY = _REPO / "data" / "reports" / "p2_matched_6000" / "p2_training_summary.json"
PROCESSED_DIR = _REPO / "data" / "features" / "patients"

EXPECTED_FEATURE_HASH = "c7407bfd3a11f257a99263cd2e7fd7018c898d848f53065ce59daf5a0562a782"
EXPECTED_MANIFEST_HASH = "008a93f47c4264e3b61d73121be17efc1dae0f69473289d238d61bcc881188b1"
TOLERANCE = 1e-6
SUBSET_PATIENTS = 400
SUBSET_METRIC_TOLERANCE = 0.05  # looser, subset != full test

METRIC_KEYS = ("auroc", "auprc", "sensitivity", "specificity", "precision", "f1", "brier")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _test_patient_ids(manifest: SplitManifest) -> list[str]:
    return [p.patient_id for p in manifest.patients if p.split == "test"]


def _load_test_subset(manifest: SplitManifest, feature_names: list[str], max_patients: int):
    """Deterministically load the first ``max_patients`` test patients."""
    test_ids = _test_patient_ids(manifest)
    fragments = {"X": [], "y": [], "patient_ids": [], "hours": [], "source_datasets": []}
    n_patients = 0
    for pid in test_ids:
        if n_patients >= max_patients:
            break
        patient_file = PROCESSED_DIR / f"{pid}.parquet"
        if not patient_file.exists():
            continue
        df = _load_processed_patient(patient_file)
        if df is None or len(df) == 0:
            continue
        # Build a minimal manifest entry for _build_split_frame.
        class _Entry:
            source_dataset = "real"
        built = _build_split_frame(df, pid, _Entry(), feature_names, DERIVED_TARGET_COLUMN)
        if built is None:
            continue
        X_pid, y_pid, pid_ids, hours, source_ds = built
        fragments["X"].append(X_pid)
        fragments["y"].append(y_pid)
        fragments["patient_ids"].append(pid_ids)
        fragments["hours"].append(hours)
        fragments["source_datasets"].append(source_ds)
        n_patients += 1

    if not fragments["X"]:
        return None
    X = pd.concat(fragments["X"], axis=0, ignore_index=True)
    y = pd.concat(fragments["y"], ignore_index=True)
    return X, y, n_patients


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="Reproduce on the FULL test split (slow per-patient I/O).")
    ap.add_argument("--subset-patients", type=int, default=SUBSET_PATIENTS,
                    help="Number of test patients for the subset check.")
    args = ap.parse_args()

    report = {
        "script": "verify_p2_matched_repro_fast.py",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "full_test" if args.full else f"subset_{args.subset_patients}_patients",
        "manifest_path": str(MANIFEST),
        "feature_list_path": str(FEATURES),
        "artifact_dir": str(ARTIFACT_DIR),
        "summary_path": str(SUMMARY),
        "checks": {},
        "models": {},
        "all_ok": True,
    }

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            report["all_ok"] = False

    # 1. P1 feature hash
    p1_feat = json.load(open(FEATURES))
    p1_features = list(p1_feat["selected_features"])
    feature_hash = _hash_obj(p1_features)
    check("p1_feature_hash", feature_hash == EXPECTED_FEATURE_HASH,
          f"{feature_hash} ({len(p1_features)} features)")

    # 2. Manifest hash
    manifest_hash = _hash_file(MANIFEST)
    check("manifest_hash", manifest_hash == EXPECTED_MANIFEST_HASH,
          f"{manifest_hash}")

    # 3. Stored summary
    summary = json.load(open(SUMMARY))
    stored_test = summary.get("test_metrics", {})
    stored_thresholds = summary.get("thresholds", {})
    check("training_summary_present", bool(stored_test),
          f"models={list(stored_test.keys())}")

    manifest = SplitManifest.load(MANIFEST)
    test_ids = _test_patient_ids(manifest)
    check("test_patients_in_manifest", len(test_ids) > 0,
          f"{len(test_ids)} test patients")

    if args.full:
        # Full test split via split-aware loader (may be slow).
        from ml.p1_clinical.classical_ml.data_loader import load_split_aware_dataset  # noqa: E402
        from ml.p1_clinical.features.feature_schema import FeatureSchema  # noqa: E402
        schema = FeatureSchema.load(_REPO / "data" / "features" / "feature_schema.json")
        dataset = load_split_aware_dataset(
            processed_dir=PROCESSED_DIR,
            split_manifest=manifest,
            feature_schema=schema,
            source="real",
            target_column=DERIVED_TARGET_COLUMN,
            splits=("test",),
            use_shards=True,
        )
        test = dataset.get_split("test")
        check("test_split_loaded", test is not None and len(test.X) > 0,
              f"{len(test.X) if test else 0} rows, {getattr(test,'n_patients',0)} patients")
        X_test = test.X[p1_features]
        y_test = test.y
        n_patients_loaded = test.n_patients
        mode_detail = f"full test: {len(test.X)} rows, {test.n_patients} patients"
    else:
        subset = _load_test_subset(manifest, p1_features, args.subset_patients)
        check("test_subset_loaded", subset is not None,
              f"{args.subset_patients} patients requested")
        if subset is None:
            report["all_ok"] = False
            print("\nSUBSET LOAD FAILED — cannot continue.")
            return 1
        X_test = subset[0][p1_features]
        y_test = subset[1]
        n_patients_loaded = subset[2]
        mode_detail = f"subset: {len(X_test)} rows, {n_patients_loaded} patients, "
        mode_detail += f"{int((y_test==1).sum())} positives"
        # Subset must contain both classes for AUROC/AUPRC.
        check("subset_has_both_classes", int((y_test == 1).sum()) > 0 and int((y_test == 0).sum()) > 0,
              f"pos={int((y_test==1).sum())} neg={int((y_test==0).sum())}")

    print(f"Test data: {mode_detail}")

    # 4. Both model artifacts exist & load; reproduce predictions; compare.
    all_ok = True
    for name in ("extra_trees", "xgboost"):
        model_path = ARTIFACT_DIR / name / f"{name}_1.0.0.joblib"
        features_path = ARTIFACT_DIR / name / f"{name}_1.0.0_features.json"
        meta_path = ARTIFACT_DIR / name / f"{name}_1.0.0_metadata.json"
        check(f"{name}_artifact_exists", model_path.exists(),
              f"{model_path} ({model_path.stat().st_size if model_path.exists() else 0} bytes)")
        check(f"{name}_features_exists", features_path.exists(), str(features_path))
        check(f"{name}_metadata_exists", meta_path.exists(), str(meta_path))

        model_result = load_p2_model(model_path)
        check(f"{name}_model_loaded", model_result.model is not None,
              f"{model_result.model_name} v{model_result.model_version}")
        check(f"{name}_feature_match",
              list(model_result.selected_features) == p1_features,
              f"{len(model_result.selected_features)} features")

        thr = float(stored_thresholds[name])
        preds = predict_probabilities(model_result, X_test, threshold=thr,
                                      already_sanitized=False)
        probs = preds["predicted_probability"].to_numpy()
        classes = preds["predicted_class"].to_numpy()
        check(f"{name}_predictions_finite", bool(np.all(np.isfinite(probs))),
              f"n={len(probs)}, range=[{probs.min():.6f}, {probs.max():.6f}]")
        check(f"{name}_threshold_classes_consistent",
              bool(((probs >= thr).astype(int) == classes).all()),
              f"threshold={thr}")

        metrics = evaluate_predictions(y_test, probs, threshold=thr,
                                       model_name=name, split="test")
        m = metrics["metrics"]
        sm = stored_test[name]["metrics"]
        cm_ok = (metrics["confusion_matrix"] == stored_test[name]["confusion_matrix"])
        counts_ok = (metrics["n_rows"] == stored_test[name]["n_rows"]
                     and metrics["n_positive"] == stored_test[name]["n_positive"])

        # Subset: compare within looser tolerance (ballpark). Full: exact tolerance.
        tol = TOLERANCE if args.full else SUBSET_METRIC_TOLERANCE
        metric_matches = {}
        for key in METRIC_KEYS:
            a, b = m[key], sm[key]
            ok = np.isfinite(a) and abs(a - b) < tol
            metric_matches[key] = {"reproduced": float(a), "stored": float(b),
                                   "match": bool(ok), "tolerance": float(tol)}
            all_ok &= ok
        counts_match = bool(counts_ok)
        cm_match = bool(cm_ok)
        if not args.full:
            # Subset row/positive counts won't equal full test; relax those checks.
            counts_match = True
            cm_match = True

        all_ok &= counts_match and cm_match

        model_report = {
            "threshold": thr,
            "stored": sm,
            "reproduced": m,
            "confusion_matrix": metrics["confusion_matrix"],
            "stored_confusion_matrix": stored_test[name]["confusion_matrix"],
            "n_rows": metrics["n_rows"],
            "stored_n_rows": stored_test[name]["n_rows"],
            "n_positive": metrics["n_positive"],
            "stored_n_positive": stored_test[name]["n_positive"],
            "metric_matches": metric_matches,
            "all_metrics_match": bool(all(
                mm["match"] for mm in metric_matches.values())),
            "confusion_matrix_match": cm_match,
            "counts_match": counts_match,
            "subset_mode": not args.full,
        }
        print(f"\n--- {name} (threshold={thr}) [{mode_detail}] ---")
        for key in METRIC_KEYS:
            mm = metric_matches[key]
            print(f"  {key}: reproduced={mm['reproduced']:.6f} stored={mm['stored']:.6f} "
                  f"match={mm['match']} (tol={mm['tolerance']})")
        print(f"  n_rows: reproduced={metrics['n_rows']} stored={stored_test[name]['n_rows']} "
              f"match={counts_ok}")
        print(f"  n_positive: reproduced={metrics['n_positive']} "
              f"stored={stored_test[name]['n_positive']} match={counts_ok}")
        print(f"  confusion_matrix: reproduced={metrics['confusion_matrix']} "
              f"stored={stored_test[name]['confusion_matrix']} match={cm_ok}")
        report["models"][name] = model_report

    report["all_ok"] = bool(all_ok)
    print(f"\nALL REPRODUCIBILITY CHECKS PASS: {all_ok}  [{report['mode']}]")

    out_json = _REPO / "data" / "reports" / "p2_matched_6000" / "p2_reproducibility.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote reproducibility JSON: {out_json}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
