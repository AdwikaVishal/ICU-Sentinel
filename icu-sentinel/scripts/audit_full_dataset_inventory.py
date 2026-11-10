# ruff: noqa: E402
"""Build a complete per-patient inventory and processing funnel.

This script inspects the *entire* locally available PhysioNet/CinC 2019
training partition and every downstream artifact, producing one machine
readable row per patient file:

    Downloaded -> Parsed -> Structurally valid -> Target available
        -> Features generated -> Eligible for training

Outputs (under ``--out-dir``):
    patient_inventory.csv   one row per patient file
    processing_summary.json funnel counts and percentages
    exclusion_report.csv    excluded patients and reasons
    dataset_manifest.json   cohort-level totals for the final dataset

No patient-identifiable information is emitted: patient identifiers are the
pseudonymous PhysioNet record names (``p000001`` ...).

Usage:
    python scripts/audit_full_dataset_inventory.py
    python scripts/audit_full_dataset_inventory.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXPECTED_PATIENTS = 40_336
EXPECTED_SET_A = 20_336
EXPECTED_SET_B = 20_000
CLINICAL_COLUMN_COUNT = 40


def _patient_id(path: str) -> str:
    return Path(path).stem


def _inspect(task: tuple[str, str, str]) -> dict:
    """Inspect one raw patient file plus its downstream artifacts."""
    raw_path, processed_dir, features_dir = task
    pid = _patient_id(raw_path)
    source_set = Path(raw_path).parent.name

    record = {
        "patient_id": pid,
        "source_set": source_set,
        "raw_exists": True,
        "raw_bytes": 0,
        "readable": False,
        "parsing_success": False,
        "n_observations_raw": 0,
        "n_clinical_columns": 0,
        "missing_value_pct": float("nan"),
        "timestamp_valid": False,
        "label_available": False,
        "has_sepsis": False,
        "processed_exists": False,
        "n_rows_processed": 0,
        "features_exists": False,
        "n_rows_features": 0,
        "n_eligible_rows": 0,
        "n_positive_rows": 0,
        "eligible": False,
        "exclusion_stage": "",
        "exclusion_reason": "",
    }

    try:
        record["raw_bytes"] = Path(raw_path).stat().st_size
        frame = pd.read_csv(raw_path, sep="|")
        record["readable"] = True
    except Exception as exc:  # noqa: BLE001
        record["exclusion_stage"] = "parsed"
        record["exclusion_reason"] = f"unreadable: {type(exc).__name__}"
        return record

    if len(frame) == 0:
        record["exclusion_stage"] = "parsed"
        record["exclusion_reason"] = "empty file (no observations)"
        return record

    record["parsing_success"] = True
    record["n_observations_raw"] = int(len(frame))
    record["n_clinical_columns"] = int(len(frame.columns))
    clinical = frame.drop(columns=[c for c in ("SepsisLabel",) if c in frame])
    total_cells = clinical.shape[0] * clinical.shape[1]
    record["missing_value_pct"] = (
        float(clinical.isna().to_numpy().sum() / total_cells) if total_cells else 1.0
    )

    # Timestamps: ICULOS must exist, be numeric and strictly increasing.
    if "ICULOS" in frame.columns:
        iculos = pd.to_numeric(frame["ICULOS"], errors="coerce")
        record["timestamp_valid"] = bool(
            iculos.notna().all() and iculos.is_monotonic_increasing
        )

    if "SepsisLabel" in frame.columns:
        labels = pd.to_numeric(frame["SepsisLabel"], errors="coerce")
        valid_labels = labels.dropna()
        record["label_available"] = bool(
            len(valid_labels) == len(labels) and valid_labels.isin([0, 1]).all()
        )
        record["has_sepsis"] = bool((valid_labels == 1).any())
    else:
        labels = None

    if not record["timestamp_valid"]:
        record["exclusion_stage"] = "validated"
        record["exclusion_reason"] = "invalid or missing ICULOS timestamps"
    elif not record["label_available"]:
        record["exclusion_stage"] = "target"
        record["exclusion_reason"] = "SepsisLabel missing or non-binary"

    processed_path = Path(processed_dir) / f"{pid}.parquet"
    if processed_path.exists():
        record["processed_exists"] = True
        record["n_rows_processed"] = int(pq.ParquetFile(processed_path).metadata.num_rows)
    elif not record["exclusion_reason"]:
        record["exclusion_stage"] = "validated"
        record["exclusion_reason"] = "processed parquet missing"

    feature_path = Path(features_dir) / "patients" / f"{pid}.parquet"
    if feature_path.exists():
        record["features_exists"] = True
        record["n_rows_features"] = int(pq.ParquetFile(feature_path).metadata.num_rows)
        try:
            targets = pd.read_parquet(
                feature_path, columns=["sepsis_within_6h", "target_eligible"]
            )
        except Exception as exc:  # noqa: BLE001
            record["exclusion_stage"] = "features"
            record["exclusion_reason"] = f"feature target unreadable: {type(exc).__name__}"
            return record
        eligible = targets["target_eligible"].astype(bool)
        target = targets["sepsis_within_6h"]
        record["n_eligible_rows"] = int(eligible.sum())
        record["n_positive_rows"] = int((target == 1).sum())
        if record["n_eligible_rows"] == 0:
            record["exclusion_stage"] = "target"
            record["exclusion_reason"] = "no eligible prediction hours"
        elif record["n_rows_features"] == 0:
            record["exclusion_stage"] = "features"
            record["exclusion_reason"] = "empty feature table"
        else:
            record["eligible"] = True
    elif not record["exclusion_reason"]:
        record["exclusion_stage"] = "features"
        record["exclusion_reason"] = "features not generated"

    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=_REPO_ROOT / "data" / "raw"
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=_REPO_ROOT / "data" / "processed_full"
    )
    parser.add_argument(
        "--features-dir", type=Path, default=_REPO_ROOT / "data" / "features_full"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT.parent
        / "artifacts"
        / "physionet2019"
        / "full_dataset_v2"
        / "dataset",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    raw_paths = sorted(str(p) for p in args.raw_dir.rglob("*.psv"))
    if not raw_paths:
        print(f"ERROR: no .psv files under {args.raw_dir}", file=sys.stderr)
        return 2

    print(f"Inventorying {len(raw_paths)} raw patient files ...", flush=True)
    tasks = [(p, str(args.processed_dir), str(args.features_dir)) for p in raw_paths]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, record in enumerate(executor.map(_inspect, tasks, chunksize=64), start=1):
            rows.append(record)
            if i % 2000 == 0 or i == len(tasks):
                print(f"  inspected {i}/{len(tasks)}", flush=True)

    inventory = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.out_dir / "patient_inventory.csv", index=False)

    excluded = inventory[~inventory["eligible"]].copy()
    excluded[
        [
            "patient_id",
            "source_set",
            "exclusion_stage",
            "exclusion_reason",
            "readable",
            "n_observations_raw",
            "processed_exists",
            "features_exists",
        ]
    ].to_csv(args.out_dir / "exclusion_report.csv", index=False)

    n_downloaded = len(inventory)
    n_parsed = int(inventory["parsing_success"].sum())
    n_valid = int(
        (inventory["parsing_success"] & inventory["timestamp_valid"]).sum()
    )
    n_label = int(
        (
            inventory["parsing_success"]
            & inventory["timestamp_valid"]
            & inventory["label_available"]
        ).sum()
    )
    n_processed = int(inventory["processed_exists"].sum())
    n_features = int(inventory["features_exists"].sum())
    n_eligible = int(inventory["eligible"].sum())

    funnel_stages = [
        ("expected", EXPECTED_PATIENTS),
        ("downloaded", n_downloaded),
        ("parsed", n_parsed),
        ("validated", n_valid),
        ("target_available", n_label),
        ("processed", n_processed),
        ("features_generated", n_features),
        ("eligible_for_training", n_eligible),
    ]
    funnel = [
        {
            "stage": name,
            "count": count,
            "pct_of_expected": round(100.0 * count / EXPECTED_PATIENTS, 3),
            "pct_of_downloaded": round(100.0 * count / n_downloaded, 3),
        }
        for name, count in funnel_stages
    ]

    reason_counts = (
        excluded.groupby(["exclusion_stage", "exclusion_reason"], dropna=False)
        .size()
        .reset_index(name="n_patients")
        .sort_values("n_patients", ascending=False)
    )
    by_set = (
        inventory.groupby("source_set")
        .agg(
            n_files=("patient_id", "size"),
            n_parsed=("parsing_success", "sum"),
            n_processed=("processed_exists", "sum"),
            n_features=("features_exists", "sum"),
            n_eligible=("eligible", "sum"),
        )
        .reset_index()
    )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_version": "physionet2019-1.0.0-training-open",
        "expected_files": {
            "total": EXPECTED_PATIENTS,
            "training_setA": EXPECTED_SET_A,
            "training_setB": EXPECTED_SET_B,
        },
        "funnel": funnel,
        "by_source_set": by_set.to_dict(orient="records"),
        "exclusion_reasons": reason_counts.to_dict(orient="records"),
        "counts": {
            "downloaded": n_downloaded,
            "missing_files": EXPECTED_PATIENTS - n_downloaded,
            "parsed": n_parsed,
            "validated": n_valid,
            "target_available": n_label,
            "processed": n_processed,
            "features_generated": n_features,
            "eligible_for_training": n_eligible,
            "excluded": n_downloaded - n_eligible,
        },
    }
    with open(args.out_dir / "processing_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    eligible = inventory[inventory["eligible"]]
    manifest = {
        "generated_at": summary["generated_at"],
        "dataset_version": summary["dataset_version"],
        "random_seed": 42,
        "total_eligible_patients": n_eligible,
        "total_observations_raw": int(inventory["n_observations_raw"].sum()),
        "total_observations_processed": int(inventory["n_rows_processed"].sum()),
        "total_observations_features": int(inventory["n_rows_features"].sum()),
        "total_eligible_prediction_rows": int(eligible["n_eligible_rows"].sum()),
        "total_positive_prediction_rows": int(eligible["n_positive_rows"].sum()),
        "sepsis_prevalence_rows": round(
            float(
                eligible["n_positive_rows"].sum()
                / max(1, eligible["n_eligible_rows"].sum())
            ),
            6,
        ),
        "sepsis_patients": int(eligible["has_sepsis"].sum()),
        "sepsis_patient_prevalence": round(
            float(eligible["has_sepsis"].mean()), 6
        ),
        "excluded_patients": n_downloaded - n_eligible,
        "missing_expected_files": EXPECTED_PATIENTS - n_downloaded,
        "processing_configuration": {
            "raw_dir": str(args.raw_dir),
            "processed_dir": str(args.processed_dir),
            "features_dir": str(args.features_dir),
            "target": "sepsis_within_6h",
            "target_horizon_hours": 6,
            "imputation": "none at feature level (model-native NaN / train-fit median)",
        },
    }
    with open(args.out_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(json.dumps(summary["counts"], indent=2))
    print(f"Inventory written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
