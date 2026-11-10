#!/usr/bin/env python3
"""Write split shards from combined features parquet (much faster)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ICU_SENTINEL_ROOT = _REPO_ROOT / "icu-sentinel"
if str(_ICU_SENTINEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ICU_SENTINEL_ROOT))

from ml.p1_clinical.preprocessing.patient_splits import SplitManifest, VALID_SPLITS


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
    manifest_path = _ICU_SENTINEL_ROOT / "data" / "splits_full" / "split_manifest.json"
    combined_path = _ICU_SENTINEL_ROOT / "data" / "features_full" / "combined_features.parquet"
    shard_dir = _ICU_SENTINEL_ROOT / "data" / "features_full" / "shards"
    
    print("Loading manifest...")
    manifest = SplitManifest.load(manifest_path)
    
    # Build patient_id -> split mapping
    patient_split = {}
    for p in manifest.patients:
        patient_split[p.patient_id] = p.split
    
    print("Loading combined features...")
    # Read combined features
    table = pq.read_table(combined_path)
    df = table.to_pandas()
    print(f"Loaded {len(df)} rows")
    
    # Add split column
    df['split'] = df['patient_id'].map(patient_split)
    
    # Filter out unknown splits
    df = df[df['split'].isin(VALID_SPLITS)]
    print(f"Rows with valid splits: {len(df)}")
    
    # Write shards per split
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    for split in VALID_SPLITS:
        split_df = df[df['split'] == split].copy()
        if len(split_df) == 0:
            print(f"  {split}: 0 rows, skipping")
            continue
        
        # Drop the split column for storage
        split_df = split_df.drop(columns=['split'])
        
        table = pa.Table.from_pandas(split_df, preserve_index=False)
        shard_path = shard_dir / f"{split}.parquet"
        _write_parquet_atomic(shard_path, table)
        print(f"  {split}: {len(split_df)} rows -> {shard_path}")
        
        # Write fingerprint
        fingerprint_path = shard_dir / f"{split}.fingerprint.json"
        # Simple fingerprint: row count + column names
        fingerprint = {
            "n_rows": len(split_df),
            "columns": list(split_df.columns),
        }
        import hashlib
        fp_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        fingerprint["fingerprint"] = fp_hash
        with open(fingerprint_path, "w") as f:
            json.dump(fingerprint, f, indent=2)
    
    print("Shards written successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())