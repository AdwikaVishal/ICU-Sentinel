#!/usr/bin/env python3
"""Build combined features parquet from existing per-patient feature files."""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ICU_SENTINEL_ROOT = _REPO_ROOT / "icu-sentinel"

def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    """Write a Parquet file atomically via temp file + os.replace."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main():
    features_dir = _ICU_SENTINEL_ROOT / "data" / "features_full"
    patients_dir = features_dir / "patients"
    combined_path = features_dir / "combined_features.parquet"
    
    print("Scanning patient feature files...")
    patient_files = sorted(patients_dir.glob("*.parquet"))
    print(f"Found {len(patient_files)} patient feature files")
    
    if len(patient_files) == 0:
        print("ERROR: No patient feature files found")
        return 1
    
    # Read in batches and write combined
    batch_size = 1000
    writer = None
    schema = None
    total_rows = 0
    
    for i in range(0, len(patient_files), batch_size):
        batch_files = patient_files[i:i + batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(patient_files) + batch_size - 1)//batch_size} ({len(batch_files)} files)...")
        
        frames = []
        for f in batch_files:
            try:
                df = pd.read_parquet(f)
                frames.append(df)
            except Exception as e:
                print(f"    WARNING: Failed to read {f.name}: {e}")
        
        if not frames:
            continue
            
        batch_df = pd.concat(frames, ignore_index=True)
        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        
        if writer is None:
            writer = pq.ParquetWriter(combined_path, table.schema)
            schema = table.schema
        else:
            table = table.cast(schema)
        
        writer.write_table(table)
        total_rows += len(batch_df)
    
    if writer is not None:
        writer.close()
    
    print(f"Combined features written: {total_rows} rows -> {combined_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())