#!/usr/bin/env python3
"""Ultra-fast dataset inventory using only filesystem checks."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed_full"
FEATURES_DIR = REPO_ROOT / "data" / "features_full" / "patients"
SPLITS_DIR = REPO_ROOT / "data" / "splits_full"
REPORTS_DIR = REPO_ROOT / "data" / "reports_full"

EXPECTED_COUNTS = {
    "training_setA": 20336,
    "training_setB": 20000,
}


def main():
    print("Generating Ultra-Fast Dataset Inventory")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    # Load split manifest once into a dict
    split_manifest_path = SPLITS_DIR / "split_manifest.json"
    split_assignments = {}
    if split_manifest_path.exists():
        with open(split_manifest_path) as f:
            manifest = json.load(f)
        for p in manifest.get("patients", []):
            split_assignments[p["patient_id"]] = p["split"]
    print(f"Loaded {len(split_assignments)} split assignments")
    
    # Get file lists from filesystem (much faster than globbing in loop)
    print("Scanning raw files...")
    raw_files_a = set(f.stem for f in (RAW_DIR / "training_setA").glob("*.psv"))
    raw_files_b = set(f.stem for f in (RAW_DIR / "training_setB").glob("*.psv"))
    print(f"  training_setA: {len(raw_files_a)} files")
    print(f"  training_setB: {len(raw_files_b)} files")
    
    print("Scanning processed files...")
    processed_files = set(f.stem for f in PROCESSED_DIR.glob("*.parquet"))
    print(f"  processed_full: {len(processed_files)} files")
    
    print("Scanning feature files...")
    feature_files = set(f.stem for f in FEATURES_DIR.glob("*.parquet"))
    print(f"  features_full: {len(feature_files)} files")
    
    # Combine all patient IDs
    all_patients_a = [(pid, "training_setA") for pid in sorted(raw_files_a)]
    all_patients_b = [(pid, "training_setB") for pid in sorted(raw_files_b)]
    all_patients = all_patients_a + all_patients_b
    
    print(f"Total raw patient files: {len(all_patients)}")
    
    # Build inventory rows
    inventory_rows = []
    
    for i, (patient_id, source_dataset) in enumerate(all_patients):
        if i % 10000 == 0:
            print(f"  Processing {i}/{len(all_patients)}...")
        
        # Check file existence at each stage
        raw_exists = patient_id in raw_files_a if source_dataset == "training_setA" else patient_id in raw_files_b
        processed_exists = patient_id in processed_files
        feature_exists = patient_id in feature_files
        
        # Split assignment
        split = split_assignments.get(patient_id, "unknown")
        
        # Determine eligibility
        eligible = raw_exists and processed_exists and feature_exists
        exclusion_reasons = []
        if not raw_exists:
            exclusion_reasons.append("raw_file_missing")
        if not processed_exists:
            exclusion_reasons.append("processed_file_missing")
        if not feature_exists:
            exclusion_reasons.append("feature_file_missing")
        
        inventory_rows.append({
            "patient_id": patient_id,
            "source_dataset": source_dataset,
            "split": split,
            "raw_file_exists": raw_exists,
            "processed_file_exists": processed_exists,
            "feature_file_exists": feature_exists,
            "eligible": eligible,
            "exclusion_reason": "; ".join(exclusion_reasons) if exclusion_reasons else "",
        })
    
    # Create DataFrame and save
    import pandas as pd
    inventory_df = pd.DataFrame(inventory_rows)
    
    output_path = REPORTS_DIR / "patient_inventory.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(output_path, index=False)
    print(f"\nInventory saved to: {output_path}")
    
    # Print funnel summary
    print("\n" + "=" * 70)
    print("PROCESSING FUNNEL (Ultra-Fast)")
    print("=" * 70)
    
    total = len(inventory_df)
    
    stages = [
        ("Expected (PhysioNet)", sum(EXPECTED_COUNTS.values())),
        ("Downloaded (raw files)", total),
        ("Raw files exist", int(inventory_df["raw_file_exists"].sum())),
        ("Processed files exist", int(inventory_df["processed_file_exists"].sum())),
        ("Feature files exist", int(inventory_df["feature_file_exists"].sum())),
        ("ELIGIBLE FOR TRAINING", int(inventory_df["eligible"].sum())),
    ]
    
    for label, count in stages:
        pct = (count / stages[0][1] * 100) if stages[0][1] > 0 else 0
        print(f"  {label:35s}: {count:6d} ({pct:5.1f}%)")
    
    print("=" * 70)
    
    # Exclusion breakdown
    excluded = inventory_df[~inventory_df["eligible"]]
    if len(excluded) > 0:
        print(f"\nEXCLUSION BREAKDOWN ({len(excluded)} patients):")
        reason_counts = {}
        for _, row in excluded.iterrows():
            for reason in row["exclusion_reason"].split("; "):
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason:50s}: {count:6d}")
    
    # By source dataset
    print("\nBY SOURCE DATASET:")
    for ds in ["training_setA", "training_setB"]:
        subset = inventory_df[inventory_df["source_dataset"] == ds]
        eligible_count = int(subset["eligible"].sum())
        print(f"  {ds:15s}: {len(subset):6d} total, {eligible_count:6d} eligible ({eligible_count/len(subset)*100:.1f}%)")
    
    # By split
    print("\nBY SPLIT:")
    for split in ["train", "validation", "test", "unknown"]:
        subset = inventory_df[inventory_df["split"] == split]
        eligible_count = int(subset["eligible"].sum())
        if len(subset) > 0:
            print(f"  {split:12s}: {len(subset):6d} total, {eligible_count:6d} eligible ({eligible_count/len(subset)*100:.1f}%)")
    
    # Save summary JSON
    summary = {
        "inventory_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_expected": sum(EXPECTED_COUNTS.values()),
        "total_raw_files": total,
        "funnel": {label: count for label, count in stages},
        "exclusion_breakdown": reason_counts if len(excluded) > 0 else {},
        "by_source_dataset": {
            ds: {
                "total": int(len(inventory_df[inventory_df["source_dataset"] == ds])),
                "eligible": int(inventory_df[inventory_df["source_dataset"] == ds]["eligible"].sum()),
            }
            for ds in ["training_setA", "training_setB"]
        },
        "by_split": {
            split: {
                "total": int(len(inventory_df[inventory_df["split"] == split])),
                "eligible": int(inventory_df[inventory_df["split"] == split]["eligible"].sum()),
            }
            for split in ["train", "validation", "test", "unknown"]
            if len(inventory_df[inventory_df["split"] == split]) > 0
        },
    }
    
    summary_path = REPORTS_DIR / "inventory_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())