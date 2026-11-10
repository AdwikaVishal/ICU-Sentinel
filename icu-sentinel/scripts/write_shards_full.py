#!/usr/bin/env python3
"""Write split shards for faster loading."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ICU_SENTINEL_ROOT = _REPO_ROOT / "icu-sentinel"
if str(_ICU_SENTINEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ICU_SENTINEL_ROOT))

from ml.p1_clinical.classical_ml.data_loader import write_split_shards
from ml.p1_clinical.preprocessing.patient_splits import SplitManifest


from ml.p1_clinical.features.feature_schema import FeatureSchema

def main():
    manifest_path = _ICU_SENTINEL_ROOT / "data" / "splits_full" / "split_manifest.json"
    processed_dir = _ICU_SENTINEL_ROOT / "data" / "features_full" / "patients"
    schema_path = _ICU_SENTINEL_ROOT / "data" / "features_full" / "feature_schema.json"
    
    print("Loading manifest...")
    manifest = SplitManifest.load(manifest_path)
    
    print("Loading feature schema...")
    feature_schema = FeatureSchema.load(schema_path) if schema_path.exists() else None
    
    print("Writing split shards...")
    rows = write_split_shards(
        processed_dir=processed_dir,
        split_manifest=manifest,
        feature_schema=feature_schema,
        source="real",
    )
    
    print(f"Shards written: {rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())