#!/usr/bin/env python3
"""Audit PhysioNet dataset for missing and invalid files.

This script:
1. Lists expected files from S3 (authoritative source)
2. Compares with local files
3. Validates file integrity (readable, non-empty, proper format)
4. Generates missing file manifest and audit report
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
REPORTS_DIR = REPO_ROOT / "data" / "reports" / "download"
AWS_CLI = REPO_ROOT / "aws" / "dist" / "aws"
S3_BUCKET = "s3://physionet-open/challenge-2019/1.0.0/training/"

# Expected totals from PhysioNet documentation
EXPECTED_COUNTS = {
    "training_setA": 20336,
    "training_setB": 20000,
}


def list_s3_files(dataset: str) -> set[str]:
    """List all .psv files in S3 for a dataset."""
    print(f"Listing S3 files for {dataset}...")
    s3_path = f"{S3_BUCKET}{dataset}/"
    
    try:
        result = subprocess.run(
            [str(AWS_CLI), "s3", "ls", s3_path, "--no-sign-request", "--recursive"],
            capture_output=True,
            text=True,
            check=True,
        )
        
        files = set()
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                s3_key = parts[3]
                filename = Path(s3_key).name
                if filename.endswith(".psv"):
                    files.add(filename)
        
        print(f"  Found {len(files)} files in S3 for {dataset}")
        return files
    
    except subprocess.CalledProcessError as e:
        print(f"ERROR listing S3 files: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def list_local_files(dataset_path: Path) -> dict[str, Path]:
    """List all local .psv files and return {filename: full_path}."""
    if not dataset_path.exists():
        return {}
    
    files = {}
    for f in dataset_path.glob("*.psv"):
        files[f.name] = f
    
    return files


def validate_file(filepath: Path) -> tuple[bool, str]:
    """Validate that a .psv file is readable and has proper structure.
    
    Returns: (is_valid, reason_if_invalid)
    """
    if not filepath.exists():
        return False, "file_not_found"
    
    if filepath.stat().st_size == 0:
        return False, "empty_file"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            if not header:
                return False, "empty_header"
            
            columns = header.split("|")
            if "SepsisLabel" not in columns:
                return False, "missing_sepsis_label"
            
            # Try to read one data row
            first_row = f.readline().strip()
            if not first_row:
                return False, "no_data_rows"
            
            row_values = first_row.split("|")
            if len(row_values) != len(columns):
                return False, "column_count_mismatch"
        
        return True, ""
    
    except Exception as e:
        return False, f"read_error:{str(e)[:50]}"


def audit_dataset(dataset: str) -> dict:
    """Audit a single dataset (setA or setB)."""
    print(f"\n{'=' * 60}")
    print(f"Auditing {dataset}")
    print(f"{'=' * 60}")
    
    dataset_path = RAW_DIR / dataset
    
    # Get authoritative list from S3
    s3_files = list_s3_files(dataset)
    expected_count = EXPECTED_COUNTS[dataset]
    
    if len(s3_files) != expected_count:
        print(f"WARNING: S3 lists {len(s3_files)} files but expected {expected_count}")
    
    # Get local files
    local_files = list_local_files(dataset_path)
    print(f"Local files: {len(local_files)}")
    
    # Find missing files
    missing_files = sorted(s3_files - set(local_files.keys()))
    print(f"Missing files: {len(missing_files)}")
    
    # Find unexpected files (shouldn't happen but check)
    unexpected_files = sorted(set(local_files.keys()) - s3_files)
    if unexpected_files:
        print(f"WARNING: {len(unexpected_files)} unexpected local files not in S3")
    
    # Validate existing files
    print(f"Validating {len(local_files)} existing files...")
    invalid_files = {}
    valid_count = 0
    
    for i, (filename, filepath) in enumerate(local_files.items(), 1):
        if i % 5000 == 0:
            print(f"  Validated {i}/{len(local_files)} files...")
        
        is_valid, reason = validate_file(filepath)
        if is_valid:
            valid_count += 1
        else:
            invalid_files[filename] = reason
            print(f"  INVALID: {filename} - {reason}")
    
    print(f"Valid files: {valid_count}")
    print(f"Invalid files: {len(invalid_files)}")
    
    return {
        "dataset": dataset,
        "expected_count": expected_count,
        "s3_file_count": len(s3_files),
        "local_file_count": len(local_files),
        "valid_file_count": valid_count,
        "invalid_file_count": len(invalid_files),
        "missing_file_count": len(missing_files),
        "unexpected_file_count": len(unexpected_files),
        "missing_files": missing_files,
        "invalid_files": invalid_files,
        "unexpected_files": unexpected_files,
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("PhysioNet Dataset Audit")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"AWS CLI: {AWS_CLI}")
    print(f"Raw data: {RAW_DIR}")
    
    # Audit both datasets
    audit_results = {}
    for dataset in ["training_setA", "training_setB"]:
        audit_results[dataset] = audit_dataset(dataset)
    
    # Generate missing file manifest for Set B
    setB_missing = audit_results["training_setB"]["missing_files"]
    if setB_missing:
        missing_manifest = REPORTS_DIR / "missing_files_setB.txt"
        with open(missing_manifest, 'w') as f:
            f.write("\n".join(setB_missing) + "\n")
        print(f"\nMissing file manifest: {missing_manifest}")
        print(f"  {len(setB_missing)} missing files listed")
    
    # Generate JSON audit report
    audit_report = {
        "audit_timestamp": datetime.utcnow().isoformat() + "Z",
        "methodology": "S3 listing comparison with local files, integrity validation",
        "datasets": audit_results,
        "total_expected": sum(EXPECTED_COUNTS.values()),
        "total_local": sum(r["local_file_count"] for r in audit_results.values()),
        "total_valid": sum(r["valid_file_count"] for r in audit_results.values()),
        "total_invalid": sum(r["invalid_file_count"] for r in audit_results.values()),
        "total_missing": sum(r["missing_file_count"] for r in audit_results.values()),
    }
    
    audit_json = REPORTS_DIR / "download_audit.json"
    with open(audit_json, 'w') as f:
        json.dump(audit_report, f, indent=2)
    print(f"\nAudit report: {audit_json}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total expected:  {audit_report['total_expected']:>6}")
    print(f"Total local:     {audit_report['total_local']:>6}")
    print(f"Total valid:     {audit_report['total_valid']:>6}")
    print(f"Total invalid:   {audit_report['total_invalid']:>6}")
    print(f"Total missing:   {audit_report['total_missing']:>6}")
    print(f"{'=' * 60}")
    
    if audit_report['total_missing'] == 0 and audit_report['total_invalid'] == 0:
        print("✓ Dataset is COMPLETE and VALID")
        return 0
    else:
        print("✗ Dataset recovery required")
        return 1


if __name__ == "__main__":
    sys.exit(main())
