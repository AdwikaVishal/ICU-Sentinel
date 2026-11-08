#!/usr/bin/env python3
"""Fast audit: identify missing files without deep validation."""

import json
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
REPORTS_DIR = REPO_ROOT / "data" / "reports" / "download"
AWS_CLI = REPO_ROOT / "aws" / "dist" / "aws"
S3_BUCKET = "s3://physionet-open/challenge-2019/1.0.0/training/"

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
            timeout=120,
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
        
        print(f"  Found {len(files)} files in S3")
        return files
    
    except subprocess.TimeoutExpired:
        print(f"ERROR: S3 listing timed out", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR listing S3: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def list_local_files(dataset_path: Path) -> set[str]:
    """List all local .psv filenames."""
    if not dataset_path.exists():
        return set()
    
    return {f.name for f in dataset_path.glob("*.psv")}


def quick_check_files(dataset_path: Path, sample_files: list[str]) -> dict:
    """Quick check on a sample of files for obvious issues."""
    issues = {}
    
    for filename in sample_files[:100]:  # Check first 100
        filepath = dataset_path / filename
        try:
            size = filepath.stat().st_size
            if size == 0:
                issues[filename] = "empty"
            elif size < 100:
                issues[filename] = "suspiciously_small"
        except Exception as e:
            issues[filename] = f"stat_error:{str(e)[:30]}"
    
    return issues


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("PhysioNet Dataset Fast Audit")
    print(f"Timestamp: {datetime.now(UTC).isoformat()}")
    print(f"Raw data: {RAW_DIR}\n")
    
    audit_results = {}
    
    for dataset in ["training_setA", "training_setB"]:
        print(f"{'=' * 60}")
        print(f"Auditing {dataset}")
        print(f"{'=' * 60}")
        
        dataset_path = RAW_DIR / dataset
        
        # Get S3 files
        s3_files = list_s3_files(dataset)
        expected = EXPECTED_COUNTS[dataset]
        
        if len(s3_files) != expected:
            print(f"WARNING: S3 has {len(s3_files)} files, expected {expected}")
        
        # Get local files
        local_files = list_local_files(dataset_path)
        print(f"Local files: {len(local_files)}")
        
        # Find missing
        missing_files = sorted(s3_files - local_files)
        print(f"Missing files: {len(missing_files)}")
        
        # Quick integrity check on sample
        sample_issues = quick_check_files(dataset_path, sorted(local_files)[:100])
        if sample_issues:
            print(f"Sample check found {len(sample_issues)} issues in first 100 files")
        
        audit_results[dataset] = {
            "dataset": dataset,
            "expected_count": expected,
            "s3_count": len(s3_files),
            "local_count": len(local_files),
            "missing_count": len(missing_files),
            "missing_files": missing_files,
            "sample_issues": sample_issues,
        }
        print()
    
    # Write missing files manifest for Set B
    setB_missing = audit_results["training_setB"]["missing_files"]
    if setB_missing:
        manifest_path = REPORTS_DIR / "missing_files_setB.txt"
        with open(manifest_path, 'w') as f:
            f.write("\n".join(setB_missing) + "\n")
        print(f"Missing file manifest: {manifest_path}")
        print(f"  Listed {len(setB_missing)} missing files\n")
    
    # Write JSON audit
    audit_report = {
        "audit_timestamp": datetime.now(UTC).isoformat(),
        "methodology": "S3 listing comparison, sample integrity check",
        "validation_level": "fast",
        "datasets": audit_results,
        "total_expected": sum(EXPECTED_COUNTS.values()),
        "total_local": sum(r["local_count"] for r in audit_results.values()),
        "total_missing": sum(r["missing_count"] for r in audit_results.values()),
    }
    
    audit_path = REPORTS_DIR / "download_audit.json"
    with open(audit_path, 'w') as f:
        json.dump(audit_report, f, indent=2)
    print(f"Audit report: {audit_path}\n")
    
    # Summary
    print(f"{'=' * 60}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total expected:  {audit_report['total_expected']:>6}")
    print(f"Total local:     {audit_report['total_local']:>6}")
    print(f"Total missing:   {audit_report['total_missing']:>6}")
    
    for dataset, result in audit_results.items():
        if result['sample_issues']:
            print(f"\n{dataset} sample issues: {len(result['sample_issues'])}")
            for fname, issue in list(result['sample_issues'].items())[:5]:
                print(f"  {fname}: {issue}")
    
    print(f"{'=' * 60}")
    
    if audit_report['total_missing'] > 0:
        print("✗ Dataset incomplete - recovery required")
        return 1
    else:
        print("✓ All expected files present locally")
        return 0


if __name__ == "__main__":
    sys.exit(main())
