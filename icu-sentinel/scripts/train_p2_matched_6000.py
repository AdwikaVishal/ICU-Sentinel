# ruff: noqa: E402
"""Run a matched P2 experiment on the canonical 6,000-patient fast cohort.

Trains Extra Trees and XGBoost on exactly the same 960 features, patient
splits, target definition, and evaluation partitions as the matched P1
experiment. LightGBM is disabled.

This script does NOT modify any existing split manifest or P1/P2 artifact.
Outputs are written to fresh directories under artifacts/p2/matched_6000/
and data/reports/p2_matched_6000/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.p1_clinical.classical_ml.data_loader import load_from_manifest_and_features
from ml.p2_boosting.config import P2Config, safe_thread_default
from ml.p2_boosting.training import train_p2_models

EXPECTED_MANIFEST_HASH = (
    "008a93f47c4264e3b61d73121be17efc1dae0f69473289d238d61bcc881188b1"
)
EXPECTED_FEATURE_HASH = (
    "c7407bfd3a11f257a99263cd2e7fd7018c898d848f53065ce59daf5a0562a782"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "data" / "splits" / "split_manifest_fast_6000.json",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "features",
    )
    parser.add_argument(
        "--p1-features",
        type=Path,
        default=_REPO_ROOT
        / "data"
        / "reports"
        / "classical_ml_matched_6000"
        / "selected_features.json",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "p2" / "matched_6000",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "reports" / "p2_matched_6000",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extra-trees-estimators",
        type=int,
        default=200,
        help="Extra Trees estimators (P2 default: 200).",
    )
    parser.add_argument(
        "--xgboost-estimators",
        type=int,
        default=200,
        help="XGBoost estimators (P2 default: 200).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=safe_thread_default(),
        help="Parallel jobs for Extra Trees and XGBoost "
        "(safe_thread_default for this machine).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    if not args.p1_features.exists():
        print(f"ERROR: P1 features not found: {args.p1_features}", file=sys.stderr)
        return 2

    # --- Verify manifest hash ---
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if manifest_hash != EXPECTED_MANIFEST_HASH:
        print(
            f"ERROR: manifest hash mismatch: {manifest_hash} "
            f"!= {EXPECTED_MANIFEST_HASH}",
            file=sys.stderr,
        )
        return 2

    # --- Load and verify P1's exact 960-feature list ---
    p1_feat = json.load(open(args.p1_features))
    p1_features = list(p1_feat["selected_features"])
    p1_feature_hash = hashlib.sha256(
        json.dumps(p1_features, sort_keys=True).encode()
    ).hexdigest()
    if p1_feature_hash != EXPECTED_FEATURE_HASH:
        print(
            f"ERROR: P1 feature list hash mismatch: {p1_feature_hash} "
            f"!= {EXPECTED_FEATURE_HASH}",
            file=sys.stderr,
        )
        return 2

    print(f"Manifest: {args.manifest}")
    print(f"Manifest SHA256: {manifest_hash}")
    print(f"P1 features: {args.p1_features}")
    print(f"P1 feature count: {len(p1_features)}")
    print(f"P1 feature-list SHA256: {p1_feature_hash}")
    print("Target: sepsis_within_6h (unchanged)")
    print("Models: Extra Trees, XGBoost (LightGBM disabled)")
    print(f"Random seed: {args.seed}")
    print(f"n_jobs: {args.n_jobs}")
    print(f"Extra Trees estimators: {args.extra_trees_estimators}")
    print(f"XGBoost estimators: {args.xgboost_estimators}")
    print(f"Output artifacts: {args.artifact_dir}")
    print(f"Output reports: {args.reports_dir}")

    # --- Load dataset (all 994 schema features), then restrict to P1's 960 ---
    dataset = load_from_manifest_and_features(
        manifest_path=args.manifest,
        processed_dir=args.features_dir / "patients",
        schema_path=args.features_dir / "feature_schema.json",
        source="real",
    )

    missing = [f for f in p1_features if f not in dataset.feature_names]
    if missing:
        print(
            f"ERROR: {len(missing)} P1 features missing from dataset: "
            f"{missing[:10]}",
            file=sys.stderr,
        )
        return 2

    # Restrict every split to exactly P1's 960 features.
    for split_name in ("train", "validation", "test"):
        split = dataset.get_split(split_name)
        if split is not None:
            split.X = split.X[p1_features]

    dataset.feature_names = p1_features
    dataset.n_features = len(p1_features)

    # --- Configure P2: Extra Trees + XGBoost only ---
    config = P2Config(
        random_seed=args.seed,
        train_extra_trees=True,
        train_lightgbm=False,
        train_xgboost=True,
    )
    config.extra_trees.n_estimators = args.extra_trees_estimators
    config.extra_trees.n_jobs = args.n_jobs
    config.xgboost.n_estimators = args.xgboost_estimators
    config.xgboost.n_jobs = args.n_jobs

    print(f"\nEnabled models: {config.enabled_models()}")
    print(f"Features for training: {len(p1_features)} (exact P1 match)")

    start = time.time()
    result = train_p2_models(
        dataset,
        config,
        artifact_dir=args.artifact_dir,
        reports_dir=args.reports_dir,
        source="real",
        overwrite=False,
    )
    duration = time.time() - start

    # --- Persist extra experiment metadata on top of P2's own summary ---
    summary_path = args.reports_dir / "p2_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_path.exists():
        summary = json.load(open(summary_path))
    else:
        summary = result.to_dict()

    summary["manifest_path"] = str(args.manifest)
    summary["manifest_sha256"] = manifest_hash
    summary["feature_list_sha256"] = p1_feature_hash
    summary["p1_feature_count"] = len(p1_features)
    summary["feature_match_with_p1"] = True
    summary["training_duration_seconds"] = duration
    summary["matched_p1_experiment"] = True
    summary["p2_config"] = config.to_dict()

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMatched P2 training complete: {len(result.models)} models")
    print(f"Features used: {result.n_features_total}")
    print(f"Training duration: {duration:.1f}s")
    print(f"Reports: {args.reports_dir}")
    print(f"Artifacts: {args.artifact_dir}")
    for name, metrics in result.test_metrics.items():
        m = metrics.get("metrics", {})
        print(
            f"  TEST {name}: AUROC={m.get('auroc', float('nan')):.4f} "
            f"AUPRC={m.get('auprc', float('nan')):.4f} "
            f"F1={m.get('f1', float('nan')):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
