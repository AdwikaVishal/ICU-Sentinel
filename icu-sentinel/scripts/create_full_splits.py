#!/usr/bin/env python3
"""Create new patient-level splits for all eligible patients."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed_full"
FEATURES_DIR = REPO_ROOT / "data" / "features_full" / "patients"
SPLITS_DIR = REPO_ROOT / "data" / "splits_full"
REPORTS_DIR = REPO_ROOT / "data" / "reports_full"

VALID_SPLITS = ("train", "validation", "test")


def load_processed_patient(patient_id: str) -> dict | None:
    """Load a processed patient parquet file and return sepsis info."""
    filepath = PROCESSED_DIR / f"{patient_id}.parquet"
    if not filepath.exists():
        return None
    
    try:
        df = pd.read_parquet(filepath)
        if len(df) == 0:
            return None
        
        n_records = len(df)
        n_sepsis_positive = int((df["SepsisLabel"] == 1).sum()) if "SepsisLabel" in df.columns else 0
        n_sepsis_negative = n_records - n_sepsis_positive
        has_sepsis = n_sepsis_positive > 0
        source_file = str(df["source_file"].iloc[0]) if "source_file" in df.columns else f"{patient_id}.psv"
        source_dataset = str(df["source_dataset"].iloc[0]) if "source_dataset" in df.columns else "unknown"
        hospital_id = str(df["hospital_id"].iloc[0]) if "hospital_id" in df.columns and pd.notna(df["hospital_id"].iloc[0]) else None
        
        return {
            "patient_id": patient_id,
            "hospital_id": hospital_id,
            "n_records": n_records,
            "n_sepsis_positive_hours": n_sepsis_positive,
            "n_sepsis_negative_hours": n_sepsis_negative,
            "has_sepsis": has_sepsis,
            "source_file": source_file,
            "source_dataset": source_dataset,
        }
    except Exception as e:
        print(f"Error loading {patient_id}: {e}")
        return None


def create_split_manifest(
    patient_ids: list[str],
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
    stratify_by_label: bool = True,
) -> dict:
    """Create split manifest for all eligible patients."""
    
    print(f"Loading patient info for {len(patient_ids)} patients...")
    patients_info = []
    for i, pid in enumerate(patient_ids):
        if i % 5000 == 0:
            print(f"  Loaded {i}/{len(patient_ids)}...")
        info = load_processed_patient(pid)
        if info:
            patients_info.append(info)
    
    print(f"Successfully loaded {len(patients_info)} patients")
    
    if len(patients_info) == 0:
        raise ValueError("No valid patients found")
    
    # Sort for reproducibility
    patients_info.sort(key=lambda x: x["patient_id"])
    
    ids = [p["patient_id"] for p in patients_info]
    has_sepsis = [p["has_sepsis"] for p in patients_info]
    
    # First split: train+val vs test
    train_val_ids, test_ids = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_state,
        stratify=has_sepsis if stratify_by_label and len(set(has_sepsis)) > 1 else None,
    )
    
    # Second split: train vs validation from train_val
    train_val_id_set = set(train_val_ids)
    train_val_ids_list = [p["patient_id"] for p in patients_info if p["patient_id"] in train_val_id_set]
    train_val_sepsis = [p["has_sepsis"] for p in patients_info if p["patient_id"] in train_val_id_set]
    
    val_size_adjusted = validation_size / (1.0 - test_size)
    
    if len(train_val_ids_list) > 1:
        train_ids, val_ids = train_test_split(
            train_val_ids_list,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_sepsis if stratify_by_label and len(set(train_val_sepsis)) > 1 else None,
        )
    else:
        train_ids = train_val_ids_list
        val_ids = []
    
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    # Assign splits
    entries = []
    for p in patients_info:
        if p["patient_id"] in train_set:
            split = "train"
        elif p["patient_id"] in val_set:
            split = "validation"
        elif p["patient_id"] in test_set:
            split = "test"
        else:
            split = "train"  # Should not happen
        
        entries.append({
            "patient_id": p["patient_id"],
            "hospital_id": p["hospital_id"],
            "split": split,
            "n_records": p["n_records"],
            "n_sepsis_positive_hours": p["n_sepsis_positive_hours"],
            "n_sepsis_negative_hours": p["n_sepsis_negative_hours"],
            "has_sepsis": p["has_sepsis"],
            "source_file": p["source_file"],
            "source_dataset": p["source_dataset"],
        })
    
    # Build manifest
    train_entries = [e for e in entries if e["split"] == "train"]
    val_entries = [e for e in entries if e["split"] == "validation"]
    test_entries = [e for e in entries if e["split"] == "test"]
    
    manifest = {
        "strategy": "stratified_random",
        "random_state": random_state,
        "test_size": test_size,
        "validation_size": validation_size,
        "stratify_by_label": stratify_by_label,
        "n_patients_total": len(entries),
        "n_patients_train": len(train_entries),
        "n_patients_validation": len(val_entries),
        "n_patients_test": len(test_entries),
        "n_records_train": sum(e["n_records"] for e in train_entries),
        "n_records_validation": sum(e["n_records"] for e in val_entries),
        "n_records_test": sum(e["n_records"] for e in test_entries),
        "n_sepsis_patients_train": sum(1 for e in train_entries if e["has_sepsis"]),
        "n_sepsis_patients_validation": sum(1 for e in val_entries if e["has_sepsis"]),
        "n_sepsis_patients_test": sum(1 for e in test_entries if e["has_sepsis"]),
        "n_sepsis_positive_hours_train": sum(e["n_sepsis_positive_hours"] for e in train_entries),
        "n_sepsis_positive_hours_validation": sum(e["n_sepsis_positive_hours"] for e in val_entries),
        "n_sepsis_positive_hours_test": sum(e["n_sepsis_positive_hours"] for e in test_entries),
        "patients": entries,
    }
    
    return manifest


def save_split_manifests(manifest: dict, output_dir: Path) -> dict[str, Path]:
    """Save the full manifest and per-split patient ID lists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    saved = {"manifest": manifest_path}
    for split in VALID_SPLITS:
        ids = [p["patient_id"] for p in manifest["patients"] if p["split"] == split]
        split_path = output_dir / f"{split}_patient_ids.json"
        with open(split_path, "w") as f:
            json.dump({
                "split": split,
                "n_patients": len(ids),
                "patient_ids": ids,
                "source": "real",
            }, f, indent=2)
        saved[split] = split_path
    
    return saved


def check_split_leakage(manifest: dict) -> dict:
    """Check for patient-level leakage between splits."""
    splits = {s: set(p["patient_id"] for p in manifest["patients"] if p["split"] == s) for s in VALID_SPLITS}
    
    overlaps = {}
    for i, s1 in enumerate(VALID_SPLITS):
        for s2 in list(VALID_SPLITS)[i + 1:]:
            inter = sorted(splits[s1] & splits[s2])
            if inter:
                overlaps[f"{s1}_vs_{s2}"] = inter
    
    n_overlapping = sum(len(v) for v in overlaps.values())
    
    # Check every patient appears exactly once
    split_counts = {}
    for p in manifest["patients"]:
        split_counts[p["patient_id"]] = split_counts.get(p["patient_id"], 0) + 1
    multi_split = {pid: cnt for pid, cnt in split_counts.items() if cnt > 1}
    
    # Class distribution
    split_sepsis_rates = {}
    for split in VALID_SPLITS:
        split_patients = [p for p in manifest["patients"] if p["split"] == split]
        if split_patients:
            split_sepsis_rates[split] = round(
                sum(1 for p in split_patients if p["has_sepsis"]) / len(split_patients), 4
            )
        else:
            split_sepsis_rates[split] = 0.0
    
    return {
        "leakage_detected": n_overlapping > 0 or len(multi_split) > 0,
        "n_overlapping_patients": n_overlapping,
        "overlaps": overlaps,
        "patients_in_multiple_splits": multi_split,
        "every_patient_in_exactly_one_split": len(multi_split) == 0,
        "split_sepsis_rates": split_sepsis_rates,
        "n_patients_total": manifest["n_patients_total"],
        "n_patients_assigned": len(split_counts),
    }


def main():
    print("Creating New Patient-Level Splits for Full Eligible Cohort")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    # Get all patient IDs from feature files (all eligible patients)
    feature_files = list(FEATURES_DIR.glob("*.parquet"))
    patient_ids = [f.stem for f in feature_files]
    print(f"Found {len(patient_ids)} eligible patients with feature files")
    
    # Create split manifest
    manifest = create_split_manifest(
        patient_ids,
        test_size=0.2,
        validation_size=0.1,
        random_state=42,
        stratify_by_label=True,
    )
    
    # Check for leakage
    leakage = check_split_leakage(manifest)
    print("\nLeakage Check:")
    print(f"  Leakage detected: {leakage['leakage_detected']}")
    print(f"  Overlapping patients: {leakage['n_overlapping_patients']}")
    print(f"  Every patient in exactly one split: {leakage['every_patient_in_exactly_one_split']}")
    print(f"  Split sepsis rates: {leakage['split_sepsis_rates']}")
    
    if leakage["leakage_detected"]:
        print("ERROR: Split leakage detected!")
        return 1
    
    # Save manifests
    saved = save_split_manifests(manifest, SPLITS_DIR)
    print(f"\nSaved manifests:")
    for name, path in saved.items():
        print(f"  {name}: {path}")
    
    # Save leakage report
    leakage_path = REPORTS_DIR / "split_leakage_report.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(leakage_path, "w") as f:
        json.dump(leakage, f, indent=2)
    print(f"  leakage: {leakage_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Total patients:    {manifest['n_patients_total']:6d}")
    print(f"  Train:             {manifest['n_patients_train']:6d} ({manifest['n_patients_train']/manifest['n_patients_total']*100:.1f}%)")
    print(f"  Validation:        {manifest['n_patients_validation']:6d} ({manifest['n_patients_validation']/manifest['n_patients_total']*100:.1f}%)")
    print(f"  Test:              {manifest['n_patients_test']:6d} ({manifest['n_patients_test']/manifest['n_patients_total']*100:.1f}%)")
    print(f"  Train records:     {manifest['n_records_train']:6d}")
    print(f"  Validation records: {manifest['n_records_validation']:6d}")
    print(f"  Test records:      {manifest['n_records_test']:6d}")
    print(f"  Septic patients - Train: {manifest['n_sepsis_patients_train']}, Val: {manifest['n_sepsis_patients_validation']}, Test: {manifest['n_sepsis_patients_test']}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())