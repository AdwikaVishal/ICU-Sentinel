#!/usr/bin/env python3
"""Build combined features parquet using pyarrow dataset (parallel)."""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ICU_SENTINEL_ROOT = _REPO_ROOT / "icu-sentinel"

def main():
    features_dir = _ICU_SENTINEL_ROOT / "data" / "features_full"
    patients_dir = features_dir / "patients"
    combined_path = features_dir / "combined_features.parquet"
    
    print("Creating dataset from patient files...")
    # Use pyarrow dataset for parallel reading
    dataset = ds.dataset(patients_dir, format="parquet", partitioning="hive")
    print(f"Dataset: {dataset.count_rows()} rows, {len(dataset.schema)} columns")
    
    print("Writing combined features...")
    # Write in one go using dataset scanner
    scanner = dataset.scanner(use_threads=True)
    table = scanner.to_table()
    
    print(f"Read {table.num_rows} rows, {len(table.schema)} columns")
    
    # Write atomically
    import os
    tmp_path = combined_path.with_name(combined_path.name + ".tmp")
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, combined_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    
    print(f"Combined features written: {table.num_rows} rows -> {combined_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())