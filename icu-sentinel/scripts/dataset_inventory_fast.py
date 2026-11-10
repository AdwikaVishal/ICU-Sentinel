#!/usr/bin/env python3
"""Fast dataset inventory using file existence checks only."""

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


def get_split_assignment(patient_id: str, split_manifest_path: Path) -> str:
    """Get split assignment from manifest."""
    if not split_manifest_path.exists():
        return "unknown"
    
    try:
        with open(split_manifest_path) as f:
            manifest = json.load(f)
        
        for p in manifest.get("patients", []):
            if p["patient_id"] == patient_id:
                return p["split"]
    except Exception:
        pass
    
    return "unknown"


def main():
    print("Generating Fast Dataset Inventory")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    # Load split manifest
    split_manifest_path = SPLITS_DIR / "split_manifest.json"
    
    # Collect all patient IDs from raw files
    all_patients = []
    for dataset in ["training_setA", "training_setB"]:
        dataset_path = RAW_DIR / dataset
        if dataset_path.exists():
            for f in sorted(dataset_path.glob("*.psv")):
                all_patients.append({
                    "patient_id": f.stem,
                    "source_dataset": dataset,
                })
    
    print(f"Total raw patient files: {len(all_patients)}")
    
    # Process each patient - FAST: only check file existence
    inventory_rows = []
    
    for i, p in enumerate(all_patients):
        if i % 10000 == 0:
            print(f"  Processing {i}/{len(all_patients)}...")
        
        patient_id = p["patient_id"]
        source_dataset = p["source_dataset"]
        
        # Check file existence at each stage
        raw_file = RAW_DIR / source_dataset / f"{patient_id}.psv"
        processed_file = PROCESSED_DIR / f"{patient_id}.parquet"
        feature_file = FEATURES_DIR / f"{patient_id}.parquet"
        
        raw_exists = raw_file.exists()
        raw_size = raw_file.stat().st_size if raw_exists else 0
        
        processed_exists = processed_file.exists()
        processed_size = processed_file.stat().st_size if processed_exists else 0
        
        feature_exists = feature_file.exists()
        feature_size = feature_file.stat().st_size if feature_exists else 0
        
        # Split assignment
        split = get_split_assignment(patient_id, split_manifest_path)
        
        # Determine eligibility (based on file existence and non-zero size)
        eligible = True
        exclusion_reasons = []
        
        if not raw_exists:
            eligible = False
            exclusion_reasons.append("raw_file_missing")
        elif raw_size == 0:
            eligible = False
            exclusion_reasons.append("raw_file_empty")
        
        if not processed_exists:
            eligible = False
            exclusion_reasons.append("processed_file_missing")
        elif processed_size == 0:
            eligible = False
            exclusion_reasons.append("processed_file_empty")
        
        if not feature_exists:
            eligible = False
            exclusion_reasons.append("feature_file_missing")
        elif feature_size == 0:
            eligible = False
            exclusion_reasons.append("feature_file_empty")
        
        inventory_rows.append({
            "patient_id": patient_id,
            "source_dataset": source_dataset,
            "split": split,
            "raw_file_exists": raw_exists,
            "raw_file_size": raw_size,
            "processed_file_exists": processed_exists,
            "processed_file_size": processed_size,
            "feature_file_exists": feature_exists,
            "feature_file_size": feature_size,
            "eligible": eligible,
            "exclusion_reason": "; ".join(exclusion_reasons) if exclusion_reasons else "",
        })
    
    # Create DataFrame and save
    import pandas as pd
    inventory_df = pd.DataFrame(inventory_rows)
    
    output_path = REPORTS_DIR / "patient_inventory_fast.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(output_path, index=False)
    print(f"\nInventory saved to: {output_path}")
    
    # Print funnel summary
    print("\n" + "=" * 70)
    print("PROCESSING FUNNEL (Fast - File Existence Only)")
    print("=" * 70)
    
    total = len(inventory_df)
    
    stages = [
        ("Expected (PhysioNet)", sum(EXPECTED_COUNTS.values())),
        ("Downloaded (raw files)", total),
        ("Raw files exist & non-empty", int(inventory_df["raw_file_exists"].sum() & (inventory_df["raw_file_size"] > 0))),
        ("Processed files exist & non-empty", int(inventory_df["processed_file_exists"].sum() & (inventory_df["processed_file_size"] > 0))),
        ("Feature files exist & non-empty", int(inventory_df["feature_file_exists"].sum() & (inventory_df["feature_file_size"] > 0))),
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
    
    summary_path = REPORTS_DIR / "inventory_summary_fast.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())