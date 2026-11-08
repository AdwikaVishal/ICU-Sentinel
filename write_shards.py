"""Write per-split Parquet shards for the canonical matched 6000 cohort.
One-time slow load (per-patient parquet). After this, split loading is fast
(single parquet read per split) for calibration, bootstrap, early-warning,
and Set A/B analyses."""
from __future__ import annotations

import sys
from pathlib import Path

_ML_ROOT = Path("icu-sentinel").resolve()
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from ml.p1_clinical.classical_ml.data_loader import (
    write_split_shards,
    load_split_aware_dataset,
    check_split_overlap,
    validate_target_isolation,
)
from ml.p1_clinical.classical_ml.data_loader import SplitManifest, FeatureSchema
from ml.p1_clinical.preprocessing.targets import DERIVED_TARGET_COLUMN

MANIFEST = _ML_ROOT / "data" / "splits" / "split_manifest_fast_6000.json"
PROCESSED = _ML_ROOT / "data" / "features" / "patients"
SCHEMA = _ML_ROOT / "data" / "features" / "feature_schema.json"
SHARD_DIR = _ML_ROOT / "data" / "features" / "patients" / "shards"


def main() -> int:
    manifest = SplitManifest.load(MANIFEST)
    schema = FeatureSchema.load(SCHEMA)
    print(f"Writing shards to {SHARD_DIR}")
    rows = write_split_shards(
        processed_dir=PROCESSED,
        split_manifest=manifest,
        feature_schema=schema,
        source="real",
        target_column=DERIVED_TARGET_COLUMN,
        shard_dir=SHARD_DIR,
    )
    print("Shard rows:", rows)
    total = sum(rows.values())
    print(f"Total shard rows: {total}")

    # Verify fast reload from shards
    ds = load_split_aware_dataset(
        processed_dir=PROCESSED,
        split_manifest=manifest,
        feature_schema=schema,
        source="real",
        target_column=DERIVED_TARGET_COLUMN,
        use_shards=True,
        shard_dir=SHARD_DIR,
    )
    for s in ("train", "validation", "test"):
        sd = ds.get_split(s)
        print(f"  {s}: rows={sd.n_rows} patients={sd.n_patients} "
              f"positives={sd.n_positive} "
              f"has_source_ds={sd.source_datasets is not None}")

    overlap = check_split_overlap(ds)
    target = validate_target_isolation(ds)
    print("overlap:", overlap)
    print("target_isolation:", target)
    print("Shard write + fast-reload verify complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
