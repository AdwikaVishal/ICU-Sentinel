# ruff: noqa: E402
"""Generate P1 artefacts from the official PhysioNet/CinC 2019 dataset.

This script never downloads data and never creates synthetic replacements.
It exits with a clear message when the real dataset is unavailable.

Expected raw layout (either flat or nested under data/raw):
    data/raw/**/p00101.psv ... (one pipe-delimited file per patient)

Usage:
    python scripts/generate_p1_artifacts.py
    python scripts/generate_p1_artifacts.py --raw-dir data/raw --limit 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ICU_SENTINEL_ROOT = _REPO_ROOT / "icu-sentinel"
if str(_ICU_SENTINEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ICU_SENTINEL_ROOT))

from ml.p1_clinical.classical_ml.model_config import ModelConfig
from ml.p1_clinical.classical_ml.training import run_training_protocol
from ml.p1_clinical.features.feature_pipeline import (
    run_feature_pipeline_streaming,
)
from ml.p1_clinical.preprocessing.patient_splits import (
    iter_processed_patients_from_directory,
)
from ml.p1_clinical.preprocessing.pipeline import run_pipeline
from ml.p1_clinical.preprocessing.targets import TargetPolicy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=_REPO_ROOT / "data" / "raw")
    parser.add_argument(
        "--processed-dir", type=Path, default=_REPO_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=_REPO_ROOT / "data" / "splits"
    )
    parser.add_argument(
        "--features-dir", type=Path, default=_REPO_ROOT / "data" / "features"
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=_REPO_ROOT / "data" / "reports"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=_REPO_ROOT / "artifacts" / "p1"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional patient cap for a smoke run (never used for final results).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Stop after preprocessing and feature generation.",
    )
    return parser.parse_args()


def _count_raw_files(raw_dir: Path) -> int:
    return sum(1 for _ in raw_dir.rglob("*.psv")) if raw_dir.is_dir() else 0


def main() -> int:
    args = _parse_args()
    n_raw = _count_raw_files(args.raw_dir)
    if n_raw == 0:
        print(
            "ERROR: no .psv patient files found under "
            f"{args.raw_dir}.\n"
            "This project does not ship the PhysioNet/CinC Challenge 2019 "
            "dataset and will not fabricate it.\n"
            "Obtain it legitimately from "
            "https://physionet.org/content/challenge-2019/1.0.0/ "
            "(credentialed access), then place the training set files under "
            f"{args.raw_dir} and re-run this script.",
            file=sys.stderr,
        )
        return 2

    print(f"Discovered {n_raw} raw .psv files under {args.raw_dir}")
    preprocessing = run_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        splits_dir=args.splits_dir,
        reports_dir=args.reports_dir,
        limit=args.limit,
        retain_processed=False,
        retain_validation_dataframes=False,
    )
    print(
        f"Preprocessing complete: {preprocessing.n_patients_processed} patients "
        f"processed, {preprocessing.n_patients_failed} failed"
    )
    if preprocessing.n_patients_processed == 0:
        print("ERROR: preprocessing produced no patients.", file=sys.stderr)
        return 3

    feature_result = run_feature_pipeline_streaming(
        iter_processed_patients_from_directory(args.processed_dir),
        output_dir=args.features_dir,
        save_patient_features=True,
    )
    print(
        f"Feature generation complete: {feature_result.n_patients} patients, "
        f"{feature_result.n_rows} rows"
    )

    with open(args.features_dir / "target_policy.json", "w") as f:
        json.dump(TargetPolicy().to_dict(), f, indent=2)

    if args.skip_training:
        print("Skipping training as requested.")
        return 0

    training = run_training_protocol(
        processed_dir=args.features_dir / "patients",
        split_manifest_path=args.splits_dir / "split_manifest.json",
        feature_schema_path=args.features_dir / "feature_schema.json",
        config=ModelConfig(),
        artifact_dir=args.artifact_dir / "classical_ml",
        reports_dir=args.reports_dir / "classical_ml",
        source="real",
    )
    print(
        f"Training complete: {len(training.models)} models; "
        f"validation/test metrics written under {args.reports_dir / 'classical_ml'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
