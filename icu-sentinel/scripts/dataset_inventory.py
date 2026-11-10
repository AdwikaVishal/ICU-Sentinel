#!/usr/bin/env python3
"""Complete dataset inventory for PhysioNet Challenge 2019.

Creates a machine-readable CSV with one row per patient file containing:
- Patient identifier
- File existence at each stage
- Parsing success
- Number of observations
- Number of available features
- Missing-value percentage
- Timestamp validity
- Target-label availability
- Eligibility status
- Exclusion reason
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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


def get_patient_id_from_raw(filepath: Path) -> str:
    """Extract patient ID from raw .psv filename."""
    return filepath.stem


def validate_raw_file(filepath: Path) -> dict:
    """Validate a raw .psv file."""
    result = {
        "file_exists": False,
        "file_readable": False,
        "file_size_bytes": 0,
        "has_header": False,
        "n_columns": 0,
        "has_sepsis_label": False,
        "n_rows": 0,
        "parse_error": None,
    }
    
    if not filepath.exists():
        result["parse_error"] = "file_not_found"
        return result
    
    result["file_exists"] = True
    result["file_size_bytes"] = filepath.stat().st_size
    
    if result["file_size_bytes"] == 0:
        result["parse_error"] = "empty_file"
        return result
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            if not header:
                result["parse_error"] = "empty_header"
                return result
            
            columns = header.split("|")
            result["has_header"] = True
            result["n_columns"] = len(columns)
            result["has_sepsis_label"] = "SepsisLabel" in columns
            
            # Count rows
            row_count = 1  # header row
            for _ in f:
                row_count += 1
            result["n_rows"] = row_count - 1  # exclude header
            result["file_readable"] = True
            
    except Exception as e:
        result["parse_error"] = f"read_error: {str(e)[:100]}"
    
    return result


def validate_processed_file(filepath: Path) -> dict:
    """Validate a processed parquet file."""
    result = {
        "file_exists": False,
        "file_readable": False,
        "n_rows": 0,
        "n_columns": 0,
        "has_required_columns": False,
        "missing_pct": 0.0,
        "iculos_monotonic": False,
        "has_sepsis_label": False,
        "parse_error": None,
    }
    
    if not filepath.exists():
        result["parse_error"] = "file_not_found"
        return result
    
    result["file_exists"] = True
    
    try:
        df = pd.read_parquet(filepath)
        result["file_readable"] = True
        result["n_rows"] = len(df)
        result["n_columns"] = len(df.columns)
        result["has_required_columns"] = all(c in df.columns for c in ["patient_id", "hour", "SepsisLabel", "ICULOS"])
        result["has_sepsis_label"] = "SepsisLabel" in df.columns
        result["missing_pct"] = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100 if len(df) > 0 else 0
        
        # Check ICULOS monotonic
        if "ICULOS" in df.columns and len(df) > 1:
            icol = pd.to_numeric(df["ICULOS"], errors="coerce")
            result["iculos_monotonic"] = bool((icol.diff().dropna() >= 0).all())
        
    except Exception as e:
        result["parse_error"] = f"read_error: {str(e)[:100]}"
    
    return result


def validate_feature_file(filepath: Path) -> dict:
    """Validate a feature parquet file."""
    result = {
        "file_exists": False,
        "file_readable": False,
        "n_rows": 0,
        "n_features": 0,
        "has_target": False,
        "has_target_eligible": False,
        "n_eligible_rows": 0,
        "n_positive_rows": 0,
        "parse_error": None,
    }
    
    if not filepath.exists():
        result["parse_error"] = "file_not_found"
        return result
    
    result["file_exists"] = True
    
    try:
        df = pd.read_parquet(filepath)
        result["file_readable"] = True
        result["n_rows"] = len(df)
        
        # Count features (exclude identifier and target columns)
        id_cols = {"patient_id", "source_dataset", "hour", "SepsisLabel", "sepsis_within_6h", "target_eligible"}
        feature_cols = [c for c in df.columns if c not in id_cols]
        result["n_features"] = len(feature_cols)
        result["has_target"] = "sepsis_within_6h" in df.columns
        result["has_target_eligible"] = "target_eligible" in df.columns
        
        if "target_eligible" in df.columns:
            result["n_eligible_rows"] = int(df["target_eligible"].astype(bool).sum())
        if "sepsis_within_6h" in df.columns:
            result["n_positive_rows"] = int((df["sepsis_within_6h"] == 1).sum())
            
    except Exception as e:
        result["parse_error"] = f"read_error: {str(e)[:100]}"
    
    return result


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
    print("Generating Complete Dataset Inventory")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Raw data: {RAW_DIR}")
    print(f"Processed: {PROCESSED_DIR}")
    print(f"Features: {FEATURES_DIR}")
    
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
                    "raw_file": f,
                })
    
    print(f"Total raw patient files: {len(all_patients)}")
    
    # Process each patient
    inventory_rows = []
    
    for i, p in enumerate(all_patients):
        if i % 5000 == 0:
            print(f"  Processing {i}/{len(all_patients)}...")
        
        patient_id = p["patient_id"]
        source_dataset = p["source_dataset"]
        
        # Raw validation
        raw_result = validate_raw_file(p["raw_file"])
        
        # Processed validation
        processed_file = PROCESSED_DIR / f"{patient_id}.parquet"
        processed_result = validate_processed_file(processed_file)
        
        # Feature validation
        feature_file = FEATURES_DIR / f"{patient_id}.parquet"
        feature_result = validate_feature_file(feature_file)
        
        # Split assignment
        split = get_split_assignment(patient_id, split_manifest_path)
        
        # Determine eligibility
        eligible = True
        exclusion_reasons = []
        
        if not raw_result["file_exists"]:
            eligible = False
            exclusion_reasons.append("raw_file_missing")
        elif not raw_result["file_readable"]:
            eligible = False
            exclusion_reasons.append(f"raw_file_unreadable: {raw_result['parse_error']}")
        elif raw_result["n_rows"] == 0:
            eligible = False
            exclusion_reasons.append("raw_file_empty")
        elif not raw_result["has_sepsis_label"]:
            eligible = False
            exclusion_reasons.append("missing_sepsis_label")
        
        if not processed_result["file_exists"]:
            eligible = False
            exclusion_reasons.append("processed_file_missing")
        elif not processed_result["file_readable"]:
            eligible = False
            exclusion_reasons.append(f"processed_file_unreadable: {processed_result['parse_error']}")
        elif processed_result["n_rows"] == 0:
            eligible = False
            exclusion_reasons.append("processed_file_empty")
        elif not processed_result["has_required_columns"]:
            eligible = False
            exclusion_reasons.append("processed_missing_required_columns")
        
        if not feature_result["file_exists"]:
            eligible = False
            exclusion_reasons.append("feature_file_missing")
        elif not feature_result["file_readable"]:
            eligible = False
            exclusion_reasons.append(f"feature_file_unreadable: {feature_result['parse_error']}")
        elif feature_result["n_rows"] == 0:
            eligible = False
            exclusion_reasons.append("feature_file_empty")
        elif not feature_result["has_target"]:
            eligible = False
            exclusion_reasons.append("feature_missing_target")
        elif not feature_result["has_target_eligible"]:
            eligible = False
            exclusion_reasons.append("feature_missing_target_eligible")
        elif feature_result["n_eligible_rows"] == 0:
            eligible = False
            exclusion_reasons.append("feature_no_eligible_rows")
        
        inventory_rows.append({
            "patient_id": patient_id,
            "source_dataset": source_dataset,
            "split": split,
            "raw_file_exists": raw_result["file_exists"],
            "raw_file_readable": raw_result["file_readable"],
            "raw_n_rows": raw_result["n_rows"],
            "raw_has_sepsis_label": raw_result["has_sepsis_label"],
            "raw_parse_error": raw_result["parse_error"],
            "processed_file_exists": processed_result["file_exists"],
            "processed_file_readable": processed_result["file_readable"],
            "processed_n_rows": processed_result["n_rows"],
            "processed_n_columns": processed_result["n_columns"],
            "processed_has_required_columns": processed_result["has_required_columns"],
            "processed_missing_pct": round(processed_result["missing_pct"], 2),
            "processed_iculos_monotonic": processed_result["iculos_monotonic"],
            "processed_parse_error": processed_result["parse_error"],
            "feature_file_exists": feature_result["file_exists"],
            "feature_file_readable": feature_result["file_readable"],
            "feature_n_rows": feature_result["n_rows"],
            "feature_n_features": feature_result["n_features"],
            "feature_has_target": feature_result["has_target"],
            "feature_has_target_eligible": feature_result["has_target_eligible"],
            "feature_n_eligible_rows": feature_result["n_eligible_rows"],
            "feature_n_positive_rows": feature_result["n_positive_rows"],
            "feature_parse_error": feature_result["parse_error"],
            "eligible": eligible,
            "exclusion_reason": "; ".join(exclusion_reasons) if exclusion_reasons else "",
        })
    
    # Create DataFrame and save
    inventory_df = pd.DataFrame(inventory_rows)
    
    output_path = REPORTS_DIR / "patient_inventory.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(output_path, index=False)
    print(f"\nInventory saved to: {output_path}")
    
    # Print funnel summary
    print("\n" + "=" * 70)
    print("PROCESSING FUNNEL")
    print("=" * 70)
    
    total = len(inventory_df)
    
    stages = [
        ("Expected (PhysioNet)", sum(EXPECTED_COUNTS.values())),
        ("Downloaded (raw files)", total),
        ("Raw readable & valid", int(inventory_df["raw_file_readable"].sum())),
        ("Processed file exists", int(inventory_df["processed_file_exists"].sum())),
        ("Processed readable & valid", int(inventory_df["processed_file_readable"].sum())),
        ("Feature file exists", int(inventory_df["feature_file_exists"].sum())),
        ("Feature readable & valid", int(inventory_df["feature_file_readable"].sum())),
        ("Feature has target & eligible", int(inventory_df["feature_has_target"].sum() & inventory_df["feature_has_target_eligible"].sum())),
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