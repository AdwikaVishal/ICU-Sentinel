#!/usr/bin/env python3
"""Identify missing files by comparing expected patient IDs with local files.

Uses the known PhysioNet numbering pattern:
- Set A: p000001.psv through p020336.psv
- Set B: p100001.psv through p120000.psv
"""

import json
from datetime import datetime, UTC
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
REPORTS_DIR = REPO_ROOT / "data" / "reports" / "download"

# PhysioNet patient ID ranges
SET_A_RANGE = (1, 20336)  # p000001 through p020336
SET_B_RANGE = (100001, 120000)  # p100001 through p120000


def generate_expected_filenames(start_id: int, end_id: int) -> set[str]:
    """Generate expected .psv filenames for a patient ID range."""
    return {f"p{pid:06d}.psv" for pid in range(start_id, end_id + 1)}


def get_local_files(dataset_path: Path) -> set[str]:
    """Get set of local .psv filenames."""
    if not dataset_path.exists():
        return set()
    return {f.name for f in dataset_path.glob("*.psv")}


def quick_validate_sample(dataset_path: Path, files: list[str], n: int = 50) -> dict:
    """Quick validation of a sample of files."""
    issues = {}
    for filename in files[:n]:
        filepath = dataset_path / filename
        try:
            size = filepath.stat().st_size
            if size == 0:
                issues[filename] = "empty_file"
            elif size < 50:
                issues[filename] = "suspiciously_small"
            else:
                # Quick check: can we read the header?
                try:
                    with open(filepath, 'r') as f:
                        header = f.readline()
                        if "SepsisLabel" not in header:
                            issues[filename] = "missing_sepsis_label_column"
                except Exception as e:
                    issues[filename] = f"read_error"
        except Exception as e:
            issues[filename] = f"stat_error"
    return issues


def audit_dataset(name: str, id_range: tuple[int, int]) -> dict:
    """Audit a dataset by comparing expected IDs with local files."""
    print(f"\n{'=' * 60}")
    print(f"Auditing {name}")
    print(f"{'=' * 60}")
    
    dataset_path = RAW_DIR / name
    expected_count = id_range[1] - id_range[0] + 1
    
    print(f"Expected patient IDs: {id_range[0]} to {id_range[1]} ({expected_count} files)")
    
    # Generate expected filenames
    expected_files = generate_expected_filenames(id_range[0], id_range[1])
    print(f"Expected files: {len(expected_files)}")
    
    # Get local files
    local_files = get_local_files(dataset_path)
    print(f"Local files: {len(local_files)}")
    
    # Find missing
    missing_files = sorted(expected_files - local_files)
    print(f"Missing files: {len(missing_files)}")
    
    # Find unexpected (shouldn't be any)
    unexpected_files = sorted(local_files - expected_files)
    if unexpected_files:
        print(f"WARNING: {len(unexpected_files)} unexpected files")
        for f in unexpected_files[:10]:
            print(f"  Unexpected: {f}")
    
    # Quick validation of sample
    print(f"Validating sample of {min(50, len(local_files))} files...")
    sample_issues = quick_validate_sample(dataset_path, sorted(local_files), 50)
    if sample_issues:
        print(f"  Found {len(sample_issues)} issues in sample:")
        for fname, issue in list(sample_issues.items())[:5]:
            print(f"    {fname}: {issue}")
    else:
        print(f"  Sample validation passed")
    
    return {
        "dataset": name,
        "expected_count": expected_count,
        "local_count": len(local_files),
        "missing_count": len(missing_files),
        "unexpected_count": len(unexpected_files),
        "sample_issues_count": len(sample_issues),
        "missing_files": missing_files,
        "unexpected_files": unexpected_files,
        "sample_issues": sample_issues,
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("PhysioNet Dataset Missing File Identification")
    print(f"Timestamp: {datetime.now(UTC).isoformat()}")
    print(f"Raw data directory: {RAW_DIR}")
    
    # Audit both datasets
    results = {}
    results["training_setA"] = audit_dataset("training_setA", SET_A_RANGE)
    results["training_setB"] = audit_dataset("training_setB", SET_B_RANGE)
    
    # Write missing file manifest for Set B
    setB_missing = results["training_setB"]["missing_files"]
    if setB_missing:
        manifest_path = REPORTS_DIR / "missing_files_setB.txt"
        with open(manifest_path, 'w') as f:
            f.write("\n".join(setB_missing) + "\n")
        print(f"\n✓ Missing file manifest written: {manifest_path}")
        print(f"  {len(setB_missing)} missing files")
    
    # Write Set A manifest too if needed
    setA_missing = results["training_setA"]["missing_files"]
    if setA_missing:
        manifest_path = REPORTS_DIR / "missing_files_setA.txt"
        with open(manifest_path, 'w') as f:
            f.write("\n".join(setA_missing) + "\n")
        print(f"\n✓ Missing file manifest written: {manifest_path}")
        print(f"  {len(setA_missing)} missing files")
    
    # Write JSON audit report
    audit_report = {
        "audit_timestamp": datetime.now(UTC).isoformat(),
        "methodology": "Expected patient ID enumeration vs local files",
        "patient_id_ranges": {
            "training_setA": {"start": SET_A_RANGE[0], "end": SET_A_RANGE[1]},
            "training_setB": {"start": SET_B_RANGE[0], "end": SET_B_RANGE[1]},
        },
        "datasets": results,
        "total_expected": sum(r["expected_count"] for r in results.values()),
        "total_local": sum(r["local_count"] for r in results.values()),
        "total_missing": sum(r["missing_count"] for r in results.values()),
        "total_unexpected": sum(r["unexpected_count"] for r in results.values()),
    }
    
    audit_path = REPORTS_DIR / "download_audit.json"
    with open(audit_path, 'w') as f:
        json.dump(audit_report, f, indent=2)
    print(f"\n✓ Audit report written: {audit_path}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total expected:    {audit_report['total_expected']:>6,}")
    print(f"Total local:       {audit_report['total_local']:>6,}")
    print(f"Total missing:     {audit_report['total_missing']:>6,}")
    print(f"Total unexpected:  {audit_report['total_unexpected']:>6,}")
    print(f"{'=' * 60}")
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Expected: {result['expected_count']:>6,}")
        print(f"  Local:    {result['local_count']:>6,}")
        print(f"  Missing:  {result['missing_count']:>6,}")
        if result['sample_issues_count'] > 0:
            print(f"  Sample issues: {result['sample_issues_count']}")
    
    if audit_report['total_missing'] == 0:
        print(f"\n✓ Dataset is COMPLETE")
        return 0
    else:
        print(f"\n✗ Dataset recovery required")
        return 1


if __name__ == "__main__":
    sys.exit(main())
