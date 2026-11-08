# ruff: noqa: E402
"""Run a matched P1 baseline on the canonical 6,000-patient fast cohort.

This script deliberately does NOT reuse ``scripts/train_p1_fast.py``: that
script hardcodes ``data/splits/split_manifest_fast.json`` and regenerates
(overwrites) it via ``_make_fast_manifest``. Here we consume the canonical
manifest ``data/splits/split_manifest_fast_6000.json`` directly and write all
outputs to fresh, non-overwriting directories.

Protocol:
- Train: 6,000 stratified patients (canonical manifest)
- Validation / Test: full canonical splits (4,034 / 8,068 patients)
- Target: sepsis_within_6h (strict 6h pre-onset)
- Feature selection: training-only, removes 34 all-infinite hours_since_last
- Threshold selection: validation-only (F1, grid 0.01)
- Test: frozen one-shot evaluation, no tuning
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.p1_clinical.classical_ml.model_config import ModelConfig
from ml.p1_clinical.classical_ml.training import run_training_protocol
from ml.p1_clinical.preprocessing.patient_splits import SplitManifest


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
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "data" / "splits" / "split_manifest_fast_6000.json",
        help="Canonical split manifest (not overwritten).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "p1" / "classical_ml_matched_6000",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "reports" / "classical_ml_matched_6000",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _manifest_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _add_split_counts(report_path: Path, manifest_path: Path) -> None:
    """Inject per-split patient/row counts into the saved training summary."""
    manifest = json.load(open(manifest_path))
    splits = {}
    for s in ["train", "validation", "test"]:
        ps = [p for p in manifest["patients"] if p["split"] == s]
        splits[s] = {
            "n_patients": len(ps),
            "n_records": sum(p["n_records"] for p in ps),
            "n_sepsis_patients": sum(1 for p in ps if p["has_sepsis"]),
            "n_sepsis_positive_hours": sum(
                p["n_sepsis_positive_hours"] for p in ps
            ),
        }
    summary = json.load(open(report_path))
    summary["split_counts"] = splits
    summary["manifest_path"] = str(manifest_path)
    summary["manifest_sha256"] = _manifest_hash(manifest_path)
    summary["manifest_generated_at"] = datetime.now(UTC).isoformat()
    # Attach row/patient counts to each model's metrics blocks where available.
    for model_name, model_block in summary.get("models", {}).items():
        for split_key in ["training_metrics", "validation_metrics"]:
            m = model_block.get(split_key, {})
            if m:
                cm = m.get("confusion_matrix", {})
                m["n_rows"] = int(cm.get("tp", 0) + cm.get("tn", 0) + cm.get("fp", 0) + cm.get("fn", 0))
                m["n_positive"] = int(cm.get("tp", 0) + cm.get("fn", 0))
    for model_name, m in summary.get("test_metrics", {}).items():
        cm = m.get("confusion_matrix", {})
        m["n_rows"] = int(cm.get("tp", 0) + cm.get("tn", 0) + cm.get("fp", 0) + cm.get("fn", 0))
        m["n_positive"] = int(cm.get("tp", 0) + cm.get("fn", 0))
        m["n_patients"] = splits["test"]["n_patients"]
    for split_key, split_name in [("validation_metrics", "validation"), ("training_metrics", "train")]:
        for model_name, m in summary.get(split_key, {}).items():
            if "n_patients" not in m:
                m["n_patients"] = splits[split_name]["n_patients"]
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)


def main() -> int:
    args = _parse_args()

    if not args.manifest.exists():
        print(f"ERROR: canonical manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest = SplitManifest.load(args.manifest)
    train_ids = [p.patient_id for p in manifest.patients if p.split == "train"]
    print(f"Manifest: {args.manifest}")
    print(f"SHA256: {_manifest_hash(args.manifest)}")
    print(f"Train patients: {len(train_ids)}")
    print(f"Features dir: {args.features_dir / 'patients'}")

    config = ModelConfig(
        random_seed=args.seed,
        rf_n_estimators=30,
        rf_max_depth=8,
        rf_n_jobs=-1,
        lr_max_iter=300,
        svm_max_train_rows=4000,
        train_logistic_regression=True,
        train_random_forest=True,
        train_svm=True,
    )

    result = run_training_protocol(
        processed_dir=args.features_dir / "patients",
        split_manifest_path=args.manifest,
        feature_schema_path=args.features_dir / "feature_schema.json",
        config=config,
        artifact_dir=args.artifact_dir,
        reports_dir=args.reports_dir,
        source="real",
    )

    report_path = args.reports_dir / "training_summary.json"
    _add_split_counts(report_path, args.manifest)

    print(f"\nMatched P1 training complete: {len(result.models)} models")
    print(f"Selected features: {len(result.selected_features)}")
    print(f"Reports: {args.reports_dir}")
    print(f"Artifacts: {args.artifact_dir}")
    for name, metrics in result.test_metrics.items():
        cm = metrics.get("confusion_matrix", {})
        n_rows = cm.get("tp", 0) + cm.get("tn", 0) + cm.get("fp", 0) + cm.get("fn", 0)
        print(
            f"  TEST {name}: AUROC={metrics.get('auroc', float('nan')):.4f} "
            f"AUPRC={metrics.get('auprc', float('nan')):.4f} "
            f"F1={metrics.get('f1', float('nan')):.4f} "
            f"thr={result.thresholds[name]:.2f} rows={n_rows} "
            f"pos={cm.get('tp', 0) + cm.get('fn', 0)}"
        )
    fs = result.feature_selection_report
    print(
        f"  Feature selection: {fs.n_features_input} -> {fs.n_features_output} "
        f"(removed_infinite={len(fs.removed_infinite)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
