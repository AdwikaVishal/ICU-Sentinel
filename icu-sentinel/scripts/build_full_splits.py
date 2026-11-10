# ruff: noqa: E402
"""Rebuild reproducible patient-level splits for the full eligible cohort.

Uses the existing :class:`PatientSplitter` (stratified by patient sepsis
status, fixed seed) so the split protocol is identical to earlier phases,
but applied to every patient that has engineered features.

Outputs under ``--out-dir``:
    split_manifest.json           full manifest (existing schema)
    train_patient_ids.json
    validation_patient_ids.json
    test_patient_ids.json
    split_manifest.csv            flat, one row per patient
    split_summary.json            cohort-level split statistics

Usage:
    python scripts/build_full_splits.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from ml.p1_clinical.preprocessing.patient_splits import (
    PatientSplitter,
    check_split_leakage,
)
from ml.p1_clinical.preprocessing.schemas import ProcessedPatient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=_REPO_ROOT / "data" / "features_full"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_REPO_ROOT / "data" / "splits_full_v2"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    patients_dir = args.features_dir / "patients"
    paths = sorted(patients_dir.glob("*.parquet"))
    if not paths:
        print(f"ERROR: no feature files under {patients_dir}", file=sys.stderr)
        return 2

    print(f"Reading {len(paths)} feature files for split construction ...", flush=True)
    patients: list[ProcessedPatient] = []
    per_patient_rows: list[dict] = []
    for i, path in enumerate(paths, start=1):
        frame = pd.read_parquet(
            path,
            columns=[
                "patient_id",
                "source_dataset",
                "SepsisLabel",
                "sepsis_within_6h",
                "target_eligible",
            ],
        )
        pid = str(frame["patient_id"].iloc[0])
        labels = [int(x) for x in frame["SepsisLabel"].tolist()]
        eligible = frame["target_eligible"].astype(bool)
        target = frame["sepsis_within_6h"]
        source_dataset = str(frame["source_dataset"].iloc[0])
        patients.append(
            ProcessedPatient(
                patient_id=pid,
                hospital_id=source_dataset,
                hour=list(range(len(frame))),
                clinical={},
                sepsis_label=labels,
                source_file=f"{pid}.psv",
                source_dataset=source_dataset,
                n_rows=len(frame),
            )
        )
        per_patient_rows.append(
            {
                "patient_id": pid,
                "source_dataset": source_dataset,
                "n_records": int(len(frame)),
                "n_eligible_rows": int(eligible.sum()),
                "n_positive_rows": int((target == 1).sum()),
                "has_sepsis": bool(max(labels) == 1),
            }
        )
        if i % 5000 == 0 or i == len(paths):
            print(f"  read {i}/{len(paths)}", flush=True)

    splitter = PatientSplitter(
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.seed,
        stratify_by_label=True,
        strategy="stratified_random",
    )
    manifest = splitter.split(patients)
    leakage = check_split_leakage(manifest)

    extra = pd.DataFrame(per_patient_rows)
    manifest_frame = pd.DataFrame(
        [
            {
                "patient_id": p.patient_id,
                "split": p.split,
                "source_dataset": p.source_dataset,
                "n_records": p.n_records,
                "has_sepsis": p.has_sepsis,
            }
            for p in manifest.patients
        ]
    ).merge(extra, on="patient_id", suffixes=("", "_x"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest.save(args.out_dir / "split_manifest.json")
    manifest_frame.to_csv(args.out_dir / "split_manifest.csv", index=False)

    split_ids = {
        split: sorted(
            p.patient_id for p in manifest.patients if p.split == split
        )
        for split in ("train", "validation", "test")
    }
    for split, ids in split_ids.items():
        with open(args.out_dir / f"{split}_patient_ids.json", "w") as f:
            json.dump(ids, f, indent=2)

    summary = {
        "strategy": manifest.strategy,
        "random_state": manifest.random_state,
        "test_size": manifest.test_size,
        "validation_size": manifest.validation_size,
        "stratify_by_label": manifest.stratify_by_label,
        "n_patients_total": manifest.n_patients_total,
        "leakage_check": leakage,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        sub = manifest_frame[manifest_frame["split"] == split]
        summary["splits"][split] = {
            "n_patients": int(len(sub)),
            "n_sepsis_patients": int(sub["has_sepsis"].sum()),
            "sepsis_patient_prevalence": round(float(sub["has_sepsis"].mean()), 6),
            "n_observations": int(sub["n_records"].sum()),
            "n_eligible_rows": int(sub["n_eligible_rows"].sum()),
            "n_positive_rows": int(sub["n_positive_rows"].sum()),
            "positive_rate": round(
                float(sub["n_positive_rows"].sum() / max(1, sub["n_eligible_rows"].sum())),
                6,
            ),
            "source_datasets": sub["source_dataset"].value_counts().to_dict(),
        }
    with open(args.out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["splits"], indent=2))
    print(f"Leakage detected: {leakage['leakage_detected']}")
    print(f"Splits written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
