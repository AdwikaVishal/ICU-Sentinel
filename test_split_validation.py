"""Diagnostic: validate split labels in manifest match loader expectations."""
from pathlib import Path
import json
import sys

_ML_ROOT = Path("icu-sentinel").resolve()
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from ml.p1_clinical.preprocessing.patient_splits import VALID_SPLITS
from ml.p1_clinical.classical_ml.data_loader import SplitManifest

MANIFEST = _ML_ROOT / "data" / "splits" / "split_manifest_fast.json"

print("=" * 70)
print("SPLIT VALIDATION DIAGNOSTIC")
print("=" * 70)

print(f"\nLoader expects (VALID_SPLITS): {VALID_SPLITS}")

manifest = SplitManifest.load(MANIFEST)
print(f"\nManifest loaded: {MANIFEST}")
print(f"  n_patients_train: {manifest.n_patients_train}")
print(f"  n_patients_validation: {manifest.n_patients_validation}")
print(f"  n_patients_test: {manifest.n_patients_test}")

# Extract unique splits from manifest
manifest_splits = set()
for entry in manifest.patients:
    manifest_splits.add(entry.split)

print(f"\nManifest contains splits: {sorted(manifest_splits)}")

# Check for mismatch
expected = set(VALID_SPLITS)
found = manifest_splits
missing = expected - found
extra = found - expected

print(f"\n{'✓' if not missing and not extra else '✗'} Validation Result:")
if missing:
    print(f"  MISSING in manifest: {sorted(missing)}")
if extra:
    print(f"  EXTRA in manifest (not in VALID_SPLITS): {sorted(extra)}")
if not missing and not extra:
    print(f"  All splits match perfectly!")

# Show sample entries per split
print(f"\nSample patient entries (first 3 per split):")
for split in sorted(manifest_splits):
    entries = [e for e in manifest.patients if e.split == split][:3]
    print(f"  {split}: {[e.patient_id for e in entries]}")

print("\n" + "=" * 70)
