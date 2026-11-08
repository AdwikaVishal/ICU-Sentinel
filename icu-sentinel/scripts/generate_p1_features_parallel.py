# ruff: noqa: E402
"""Generate P1 features from processed patient parquet files in parallel.

This script is a faster alternative to the sequential feature phase in
``generate_p1_artifacts.py``. Each worker loads one processed patient,
computes causal features, and writes the per-patient feature parquet.
The parent aggregates target/validation statistics and optionally builds
the combined feature parquet in bounded batches.

Usage:
    python scripts/generate_p1_features_parallel.py --workers 12
    python scripts/generate_p1_features_parallel.py --workers 12 --write-combined
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from ml.p1_clinical.features.feature_pipeline import FeaturePipeline
from ml.p1_clinical.preprocessing.schemas import CLINICAL_VARIABLES, ProcessedPatient
from ml.p1_clinical.preprocessing.targets import (
    DERIVED_TARGET_COLUMN,
    TARGET_ELIGIBLE_COLUMN,
    TargetPolicy,
)

_WORKER_PIPELINE: FeaturePipeline | None = None


def _init_worker() -> None:
    global _WORKER_PIPELINE
    _WORKER_PIPELINE = FeaturePipeline()


def _patient_from_frame(df: pd.DataFrame) -> ProcessedPatient:
    patient_id = str(df["patient_id"].iloc[0])
    hospital_value = df["hospital_id"].iloc[0]
    return ProcessedPatient(
        patient_id=patient_id,
        hospital_id=str(hospital_value) if pd.notna(hospital_value) else None,
        hour=df["hour"].tolist(),
        clinical={v: df[v].tolist() for v in CLINICAL_VARIABLES if v in df.columns},
        sepsis_label=[int(x) for x in df["SepsisLabel"].tolist()],
        source_file=str(df["source_file"].iloc[0]),
        source_dataset=str(df["source_dataset"].iloc[0]),
        n_rows=len(df),
    )


def _process_file(task: tuple[str, str]) -> tuple:
    path_str, output_dir_str = task
    if _WORKER_PIPELINE is None:
        raise RuntimeError("worker pipeline was not initialised")

    df = pd.read_parquet(path_str)
    patient = _patient_from_frame(df)
    features = _WORKER_PIPELINE.transform_patient(patient)
    out_path = Path(output_dir_str) / f"{patient.patient_id}.parquet"
    features.to_parquet(out_path, index=False)

    target = features[DERIVED_TARGET_COLUMN]
    eligible = features[TARGET_ELIGIBLE_COLUMN].astype(bool)
    report = _WORKER_PIPELINE._validate(features)

    return (
        patient.patient_id,
        len(features),
        int(eligible.sum()),
        int((~eligible).sum()),
        int((target == 1).sum()),
        int((target == 0).sum()),
        int(target.isna().sum()),
        int(report["n_issues"]),
        report["issues"][:3],
    )


def _write_combined(output_dir: Path, batch_size: int = 500) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted((output_dir / "patients").glob("*.parquet"))
    combined_path = output_dir / "combined_features.parquet"
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None

    for start in range(0, len(files), batch_size):
        frames = [pd.read_parquet(p) for p in files[start:start + batch_size]]
        table = pa.Table.from_pandas(pd.concat(frames, ignore_index=True), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(combined_path, table.schema)
            schema = table.schema
        else:
            table = table.cast(schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()
    return combined_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "features",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(12, (os.cpu_count() or 4) - 1)),
    )
    parser.add_argument(
        "--write-combined",
        action="store_true",
        help="Build combined_features.parquet after per-patient features.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing per-patient feature parquet files first.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    processed_paths = sorted(args.processed_dir.glob("*.parquet"))
    if not processed_paths:
        print(f"ERROR: no processed parquet files under {args.processed_dir}", file=sys.stderr)
        return 2

    patients_dir = args.features_dir / "patients"
    patients_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for path in patients_dir.glob("*.parquet"):
            path.unlink()

    pipeline = FeaturePipeline()
    pipeline.save_config(args.features_dir / "feature_config.json")
    pipeline.save_schema(args.features_dir / "feature_schema.json")
    with open(args.features_dir / "target_policy.json", "w") as f:
        json.dump(TargetPolicy().to_dict(), f, indent=2)

    total = len(processed_paths)
    print(
        f"Computing features for {total} patients with {args.workers} workers",
        flush=True,
    )

    tasks = [(str(path), str(patients_dir)) for path in processed_paths]
    n_patients = 0
    n_rows = 0
    n_eligible = 0
    n_ineligible = 0
    n_positive = 0
    n_negative = 0
    n_missing_target = 0
    n_issues = 0
    issue_examples: list[str] = []

    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
    ) as executor:
        for result in executor.map(_process_file, tasks, chunksize=25):
            (
                _patient_id,
                rows,
                eligible,
                ineligible,
                positive,
                negative,
                missing_target,
                issues,
                examples,
            ) = result
            n_patients += 1
            n_rows += rows
            n_eligible += eligible
            n_ineligible += ineligible
            n_positive += positive
            n_negative += negative
            n_missing_target += missing_target
            n_issues += issues
            if examples and len(issue_examples) < 20:
                issue_examples.extend(examples)

            if n_patients % 500 == 0 or n_patients == total:
                elapsed = time.perf_counter() - started
                rate = n_patients / elapsed if elapsed else 0.0
                eta = (total - n_patients) / rate if rate else 0.0
                print(
                    f"[{n_patients}/{total}] {rate:.1f} patients/s "
                    f"ETA {eta / 60:.1f} min",
                    flush=True,
                )

    sample_path = next(iter(sorted(patients_dir.glob("*.parquet"))), None)
    feature_cols: list[str] = []
    if sample_path is not None:
        sample = pd.read_parquet(sample_path)
        id_cols = {
            "patient_id",
            "source_dataset",
            "hour",
            "SepsisLabel",
            DERIVED_TARGET_COLUMN,
            TARGET_ELIGIBLE_COLUMN,
        }
        feature_cols = [c for c in sample.columns if c not in id_cols]

    policy = TargetPolicy()
    summary = {
        "n_patients": n_patients,
        "n_rows": n_rows,
        "n_features": len(feature_cols),
        "feature_categories": FeaturePipeline._categorize_features(feature_cols),
        "causal": True,
        "no_imputation": True,
        "no_future_leakage": True,
        "parallel_workers": args.workers,
        "target_policy": policy.to_dict(),
        "target_summary": {
            "target_name": policy.target_name,
            "available": n_patients > 0,
            "horizon_hours": policy.horizon_hours,
            "policy": policy.to_dict(),
            "n_rows": n_rows,
            "n_eligible": n_eligible,
            "n_ineligible": n_ineligible,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_missing_target": n_missing_target,
            "positive_rate_eligible": (
                n_positive / n_eligible if n_eligible else float("nan")
            ),
            "original_label_modified": False,
        },
        "validation_report": {
            "n_issues": n_issues,
            "issues": issue_examples,
            "causal_check_passed": n_issues == 0,
        },
        "config": pipeline.config.to_dict(),
    }
    with open(args.features_dir / "feature_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Feature generation complete: {n_patients} patients, {n_rows} rows, "
        f"{len(feature_cols)} features, {n_issues} validation issues",
        flush=True,
    )

    if args.write_combined:
        combined_path = _write_combined(args.features_dir)
        print(f"Combined features written to {combined_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
