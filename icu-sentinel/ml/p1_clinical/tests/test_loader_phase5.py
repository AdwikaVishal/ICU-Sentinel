"""Fixture-level equivalence tests for Phase 5 loader/shard optimisation.

Uses synthetic fixtures only. Verifies that per-split Parquet shards produce
byte-for-byte identical results (row order, patient order, target filtering,
feature order, float32 values) to the original per-patient loading path,
including stale-cache fallback and split-scoped loading.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from ml.p1_clinical.classical_ml.data_loader import (
    VALID_SPLITS,
    LoadedDataset,
    load_split_aware_dataset,
    write_split_shards,
)
from ml.p1_clinical.tests.test_training import (
    _make_small_manifest,
    _write_synthetic_patients,
)


def _sha_array(values) -> str:
    arr = np.ascontiguousarray(values)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _load_both(tmp_path, manifest, splits=None, source="synthetic"):
    processed_dir = _write_synthetic_patients(tmp_path, manifest)
    shard_dir = tmp_path / "shards"
    counts = write_split_shards(
        processed_dir,
        manifest,
        source=source,
        shard_dir=shard_dir,
        splits=splits,
    )
    assert set(counts) == set(splits or VALID_SPLITS)
    # Atomic writes left no .tmp files behind.
    assert not any(p.suffix == ".tmp" for p in shard_dir.iterdir())
    per_patient = load_split_aware_dataset(
        processed_dir, manifest, source=source, use_shards=False, splits=splits
    )
    from_shards = load_split_aware_dataset(
        processed_dir,
        manifest,
        source=source,
        use_shards=True,
        shard_dir=shard_dir,
        splits=splits,
    )
    return per_patient, from_shards


def _assert_split_equal(a: LoadedDataset, b: LoadedDataset, split: str) -> None:
    sa = getattr(a, split)
    sb = getattr(b, split)
    assert sa is not None and sb is not None, f"split {split} missing"
    assert list(sa.X.columns) == list(sb.X.columns)
    assert sa.X.dtypes.equals(sb.X.dtypes)
    np.testing.assert_array_equal(sa.X.to_numpy(), sb.X.to_numpy())
    np.testing.assert_array_equal(sa.y.to_numpy(), sb.y.to_numpy())
    np.testing.assert_array_equal(sa.patient_ids.to_numpy(), sb.patient_ids.to_numpy())
    np.testing.assert_array_equal(sa.hours.to_numpy(), sb.hours.to_numpy())
    assert _sha_array(sa.X.to_numpy()) == _sha_array(sb.X.to_numpy())
    assert _sha_array(sa.y.to_numpy()) == _sha_array(sb.y.to_numpy())


def test_shard_equivalence_to_per_patient_all_splits(tmp_path):
    manifest = _make_small_manifest(tmp_path)
    per_patient, from_shards = _load_both(tmp_path, manifest)
    assert list(from_shards.feature_names) == list(per_patient.feature_names)
    assert from_shards.n_features == per_patient.n_features
    for split in VALID_SPLITS:
        _assert_split_equal(per_patient, from_shards, split)


def test_shard_row_and_patient_order_preserved(tmp_path):
    manifest = _make_small_manifest(tmp_path)
    per_patient, from_shards = _load_both(tmp_path, manifest)
    for split in VALID_SPLITS:
        sa = getattr(per_patient, split)
        sb = getattr(from_shards, split)
        assert list(sa.patient_ids) == list(sb.patient_ids)
        assert list(sa.hours) == list(sb.hours)


def test_split_scoped_loading_subset(tmp_path):
    manifest = _make_small_manifest(tmp_path)
    per_patient, from_shards = _load_both(
        tmp_path, manifest, splits=("train", "test")
    )
    # Only requested splits populated.
    assert from_shards.train is not None
    assert from_shards.test is not None
    assert from_shards.validation is None
    # Per-patient path respects splits identically.
    assert per_patient.train is not None
    assert per_patient.test is not None
    assert per_patient.validation is None
    _assert_split_equal(per_patient, from_shards, "train")
    _assert_split_equal(per_patient, from_shards, "test")


def test_stale_shard_falls_back_to_per_patient(tmp_path):
    manifest = _make_small_manifest(tmp_path)
    processed_dir = _write_synthetic_patients(tmp_path, manifest)
    shard_dir = tmp_path / "shards"
    write_split_shards(processed_dir, manifest, source="synthetic", shard_dir=shard_dir)

    # Corrupt every fingerprint so shards are detected as stale.
    for split in VALID_SPLITS:
        fp = shard_dir / f"{split}.fingerprint.json"
        fp.write_text(json.dumps({"fingerprint": "stale"}), encoding="utf-8")

    per_patient = load_split_aware_dataset(
        processed_dir, manifest, source="synthetic", use_shards=False
    )
    from_shards = load_split_aware_dataset(
        processed_dir,
        manifest,
        source="synthetic",
        use_shards=True,
        shard_dir=shard_dir,
    )
    assert list(from_shards.feature_names) == list(per_patient.feature_names)
    for split in VALID_SPLITS:
        _assert_split_equal(per_patient, from_shards, split)


def test_shard_files_are_valid_parquet(tmp_path):
    import pandas as pd

    manifest = _make_small_manifest(tmp_path)
    processed_dir = _write_synthetic_patients(tmp_path, manifest)
    shard_dir = tmp_path / "shards"
    write_split_shards(processed_dir, manifest, source="synthetic", shard_dir=shard_dir)
    for split in VALID_SPLITS:
        path = shard_dir / f"{split}.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) > 0
        assert "patient_id" in df.columns
        assert "hour" in df.columns
