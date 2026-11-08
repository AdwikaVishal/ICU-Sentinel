"""Generate ONLY test split shard for fast P2 deliverables.
This is a minimal, validated shard operation targeting only the test split."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent / "icu-sentinel"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.p1_clinical.classical_ml.data_loader import (
    write_split_shards,
    SplitManifest,
    FeatureSchema,
)
from ml.p1_clinical.preprocessing.targets import DERIVED_TARGET_COLUMN

MANIFEST = _ROOT / "data" / "splits" / "split_manifest_fast_6000.json"
PROCESSED = _ROOT / "data" / "features" / "patients"
SCHEMA = _ROOT / "data" / "features" / "feature_schema.json"
SHARD_DIR = _ROOT / "data" / "features" / "patients" / "shards"

print("=" * 70)
print("GENERATING TEST-ONLY SHARD")
print("=" * 70)

manifest = SplitManifest.load(MANIFEST)
schema = FeatureSchema.load(SCHEMA)

print(f"\nManifest: {MANIFEST}")
print(f"  Test patients: {manifest.n_patients_test}")
print(f"\nShard directory: {SHARD_DIR}")
print(f"Processing ONLY test split...")

rows = write_split_shards(
    processed_dir=PROCESSED,
    split_manifest=manifest,
    feature_schema=schema,
    source="real",
    target_column=DERIVED_TARGET_COLUMN,
    splits=("test",),  # ONLY test split
    shard_dir=SHARD_DIR,
)

print(f"\n✓ Test shard complete:")
print(f"  Rows: {rows.get('test', 0)}")
print(f"  File: {SHARD_DIR / 'test.parquet'}")
print("\nTest shard ready for fast P2 analyses.")
