# ruff: noqa: E402
"""Run the P1 classical-ML protocol with a documented compute-fast setup.

The full training split has ~1.08M hourly rows. To finish a real-data
empirical run on a 16 GB workstation, this script:

- uses a seeded, patient-level, sepsis-stratified subsample of the
  training split (default 8,000 patients);
- keeps the full validation and test splits for evaluation;
- reduces Random Forest to 30 trees / depth 8;
- caps Logistic Regression at 300 iterations;
- caps RBF-SVM training at 4,000 stratified rows.

These are compute constraints, not tuning choices. They are written into
the generated reports so the final protocol is auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sklearn.model_selection import train_test_split

from ml.p1_clinical.classical_ml.model_config import ModelConfig
from ml.p1_clinical.classical_ml.training import run_training_protocol


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "features",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "splits",
    )
    parser.add_argument("--train-patients", type=int, default=8000)
    parser.add_argument("--rf-trees", type=int, default=30)
    parser.add_argument("--rf-depth", type=int, default=8)
    parser.add_argument("--lr-max-iter", type=int, default=300)
    parser.add_argument("--svm-max-rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _make_fast_manifest(
    manifest_path: Path,
    output_path: Path,
    n_train: int,
    seed: int,
) -> dict:
    with open(manifest_path) as f:
        manifest = json.load(f)

    train = [p for p in manifest["patients"] if p["split"] == "train"]
    other = [p for p in manifest["patients"] if p["split"] != "train"]
    if n_train >= len(train):
        selected_train = train
    else:
        labels = [bool(p["has_sepsis"]) for p in train]
        selected_train, _ = train_test_split(
            train,
            train_size=n_train,
            random_state=seed,
            stratify=labels,
        )

    selected = selected_train + other
    selected.sort(key=lambda p: p["patient_id"])

    manifest["patients"] = selected
    manifest["n_patients_total"] = len(selected)
    manifest["n_patients_train"] = len(selected_train)
    manifest["n_records_train"] = sum(p["n_records"] for p in selected_train)
    manifest["n_sepsis_patients_train"] = sum(
        1 for p in selected_train if p["has_sepsis"]
    )
    manifest["n_sepsis_positive_hours_train"] = sum(
        p["n_sepsis_positive_hours"] for p in selected_train
    )
    manifest["fast_protocol"] = {
        "enabled": True,
        "train_patients_requested": n_train,
        "train_patients_used": len(selected_train),
        "random_state": seed,
        "stratified_by": "has_sepsis",
        "validation_and_test": "full",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest["fast_protocol"]


def main() -> int:
    args = _parse_args()
    manifest_path = args.splits_dir / "split_manifest.json"
    fast_manifest_path = args.splits_dir / "split_manifest_fast.json"

    if not manifest_path.exists():
        print(f"ERROR: missing {manifest_path}", file=sys.stderr)
        return 2
    if not (args.features_dir / "patients").exists():
        print(
            f"ERROR: missing feature directory {args.features_dir / 'patients'}",
            file=sys.stderr,
        )
        return 2

    fast_info = _make_fast_manifest(
        manifest_path,
        fast_manifest_path,
        n_train=args.train_patients,
        seed=args.seed,
    )
    print(
        "Fast manifest:",
        json.dumps(fast_info, sort_keys=True),
        flush=True,
    )

    config = ModelConfig(
        random_seed=args.seed,
        rf_n_estimators=args.rf_trees,
        rf_max_depth=args.rf_depth,
        rf_n_jobs=-1,
        lr_max_iter=args.lr_max_iter,
        svm_max_train_rows=args.svm_max_rows,
        train_logistic_regression=True,
        train_random_forest=True,
        train_svm=True,
    )

    reports_dir = _REPO_ROOT / "data" / "reports" / "classical_ml_fast"
    artifact_dir = _REPO_ROOT / "artifacts" / "p1" / "classical_ml_fast"
    config.save(reports_dir / "model_config.json")

    result = run_training_protocol(
        processed_dir=args.features_dir / "patients",
        split_manifest_path=fast_manifest_path,
        feature_schema_path=args.features_dir / "feature_schema.json",
        config=config,
        artifact_dir=artifact_dir,
        reports_dir=reports_dir,
        source="real",
    )

    print(
        f"Training complete: {len(result.models)} models, "
        f"{len(result.selected_features)} selected features",
        flush=True,
    )
    for name, metrics in result.test_metrics.items():
        print(
            f"TEST {name}: AUROC={metrics.get('auroc', float('nan')):.4f} "
            f"AUPRC={metrics.get('auprc', float('nan')):.4f} "
            f"F1={metrics.get('f1', float('nan')):.4f} "
            f"threshold={result.thresholds[name]:.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
