"""Split-aware data loading for P1 classical ML models.

Loads processed patient data from Part 3/4 output and aligns it with
patient-level train/validation/test splits. Ensures no patient overlap
between splits and that SepsisLabel is preserved as the target only.

Phase 5 optimisation: in addition to the original per-patient loading path,
the loader can read per-split Parquet shards (one file per split) written by
:func:`write_split_shards`. Shards are validated with a content fingerprint
(stale-cache detection) and written atomically. When shards are absent or
stale, the loader falls back to the original per-patient path, which is kept
byte-for-byte behaviour-compatible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..features.feature_schema import FeatureSchema
from ..preprocessing.patient_splits import (
    VALID_SPLITS,
    SplitManifest,
)
from ..preprocessing.targets import (
    DERIVED_TARGET_COLUMN,
    TARGET_ELIGIBLE_COLUMN,
    ensure_sepsis_target,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.classical_ml.data_loader")

# Columns that must never be used as model features.
IDENTIFIER_COLUMNS: list[str] = [
    "patient_id",
    "source_dataset",
    "hour",
    "source_file",
]
TARGET_COLUMN: str = "SepsisLabel"
TARGET_COLUMNS: list[str] = [
    TARGET_COLUMN,
    DERIVED_TARGET_COLUMN,
    TARGET_ELIGIBLE_COLUMN,
]

_SHARD_FILENAME = "{split}.parquet"
_SHARD_FINGERPRINT_FILENAME = "{split}.fingerprint.json"


@dataclass
class SplitData:
    """Data for a single split (train/validation/test)."""

    split: str
    X: pd.DataFrame  # features only
    y: pd.Series  # target (SepsisLabel)
    patient_ids: pd.Series
    hours: pd.Series
    source_datasets: pd.Series | None = None
    target_column: str = DERIVED_TARGET_COLUMN
    n_patients: int = 0
    n_rows: int = 0
    n_positive: int = 0
    n_negative: int = 0
    positive_rate: float = 0.0

    def __post_init__(self) -> None:
        self.n_rows = len(self.X)
        self.n_patients = self.patient_ids.nunique()
        if len(self.y) > 0:
            self.n_positive = int((self.y == 1).sum())
            self.n_negative = int((self.y == 0).sum())
            self.positive_rate = self.n_positive / len(self.y)
        else:
            self.n_positive = 0
            self.n_negative = 0
            self.positive_rate = 0.0


@dataclass
class LoadedDataset:
    """Complete loaded dataset with all splits."""

    train: SplitData | None = None
    validation: SplitData | None = None
    test: SplitData | None = None
    feature_names: list[str] = field(default_factory=list)
    n_features: int = 0
    n_patients_total: int = 0
    source: str = "synthetic"
    target_column: str = DERIVED_TARGET_COLUMN

    def get_split(self, split: str) -> SplitData | None:
        if split == "train":
            return self.train
        if split == "validation":
            return self.validation
        if split == "test":
            return self.test
        return None

    def summary(self) -> dict:
        return {
            "n_patients_total": self.n_patients_total,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names),
            "splits": {
                s: (
                    {
                        "n_patients": getattr(self, s).n_patients,
                        "n_rows": getattr(self, s).n_rows,
                        "n_positive": getattr(self, s).n_positive,
                        "n_negative": getattr(self, s).n_negative,
                        "positive_rate": getattr(self, s).positive_rate,
                    }
                    if getattr(self, s) is not None
                    else None
                )
                for s in VALID_SPLITS
            },
            "source": self.source,
            "target_column": self.target_column,
        }


def _load_processed_patient(path: Path) -> pd.DataFrame | None:
    """Load a single processed patient parquet file."""
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def _is_feature_column(col: str) -> bool:
    """Return True if a column is a model feature (not identifier/target)."""
    if col in IDENTIFIER_COLUMNS:
        return False
    return col not in TARGET_COLUMNS


def _manifest_fingerprint(
    split_manifest: SplitManifest,
    source: str,
    target_column: str,
    feature_names: list[str],
    splits: tuple[str, ...],
) -> str:
    """Content fingerprint for split-scoped shard staleness detection.

    Covers the manifest patient/split assignments, source label, target
    column, feature names and the requested splits so any change invalidates
    the shards.
    """
    payload = {
        "source": source,
        "target_column": target_column,
        "splits": list(splits),
        "feature_names": list(feature_names),
        "patients": [
            {
                "patient_id": p.patient_id,
                "split": p.split,
                "n_records": p.n_records,
            }
            for p in split_manifest.patients
            if p.split in splits
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _shard_dir_for(processed_path: Path, shard_dir: str | Path | None) -> Path:
    if shard_dir is not None:
        return Path(shard_dir)
    return processed_path / "shards"


def _shard_parquet_path(shard_dir: Path, split: str) -> Path:
    return shard_dir / _SHARD_FILENAME.format(split=split)


def _shard_fingerprint_path(shard_dir: Path, split: str) -> Path:
    return shard_dir / _SHARD_FINGERPRINT_FILENAME.format(split=split)


def _shard_fresh(shard_dir: Path, split: str, fingerprint: str) -> bool:
    """Return True if the split shard exists and its fingerprint matches."""
    parquet_path = _shard_parquet_path(shard_dir, split)
    fingerprint_path = _shard_fingerprint_path(shard_dir, split)
    if not parquet_path.exists() or not fingerprint_path.exists():
        return False
    try:
        stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return stored.get("fingerprint") == fingerprint


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Write a Parquet file atomically via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _discover_feature_names(
    processed_path: Path, feature_schema: FeatureSchema | None
) -> list[str]:
    """Discover feature columns (identifiers/targets excluded), sorted."""
    all_patient_files = sorted(processed_path.glob("*.parquet"))
    if not all_patient_files:
        return []
    first_df = _load_processed_patient(all_patient_files[0])
    if first_df is None or len(first_df) == 0:
        return []
    if "patient_id" not in first_df.columns:
        first_df["patient_id"] = all_patient_files[0].stem
    first_df = ensure_sepsis_target(first_df)
    discovered = [c for c in first_df.columns if _is_feature_column(c)]
    if feature_schema is not None:
        allowed = {
            f
            for f in feature_schema.feature_names()
            if f not in IDENTIFIER_COLUMNS and f not in TARGET_COLUMNS
        }
        discovered = [f for f in discovered if f in allowed]
    return sorted(discovered)


def _build_split_frame(
    df: pd.DataFrame,
    pid: str,
    entry: Any,
    feature_names: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series] | None:
    """Extract and validate one patient's split frame.

    Returns ``(X, y, patient_ids, hours, source_datasets)`` or ``None`` when
    the patient has no usable rows. This mirrors the historical per-patient
    processing exactly (row order, eligible filtering, float32 casting).
    """
    if "patient_id" not in df.columns:
        df = df.copy()
        df["patient_id"] = pid
    df = ensure_sepsis_target(df)

    # Only eligible rows carry a valid target.
    df = df[df[TARGET_ELIGIBLE_COLUMN].astype(bool)].copy()
    df = df.dropna(subset=[target_column])
    if len(df) == 0:
        return None

    missing = [c for c in ([target_column] + feature_names) if c not in df.columns]
    if missing:
        logger.warning("Patient %s missing columns %s; skipping", pid, missing)
        return None

    try:
        X_pid = df[feature_names].astype(np.float32)
    except (TypeError, ValueError):
        X_pid = (
            df[feature_names]
            .apply(pd.to_numeric, errors="coerce")
            .astype(np.float32)
        )
    y_pid = df[target_column].copy()
    patient_ids_pid = pd.Series([pid] * len(df), dtype="object")
    hours_pid = (
        df["hour"].copy()
        if "hour" in df.columns
        else pd.Series(range(len(df)))
    )
    source_ds_pid = (
        df["source_dataset"].copy()
        if "source_dataset" in df.columns
        else pd.Series([entry.source_dataset] * len(df))
    )
    return X_pid, y_pid, patient_ids_pid, hours_pid, source_ds_pid


def _concat_split(
    split: str,
    fragments: dict[str, list],
    feature_names: list[str],
    target_column: str,
) -> SplitData | None:
    """Concatenate accumulated per-patient fragments into one SplitData."""
    if not fragments["X"]:
        return None
    return SplitData(
        split=split,
        X=pd.concat(fragments["X"], axis=0, ignore_index=True),
        y=pd.concat(fragments["y"], ignore_index=True),
        patient_ids=pd.concat(fragments["patient_ids"], ignore_index=True),
        hours=pd.concat(fragments["hours"], ignore_index=True),
        source_datasets=pd.concat(fragments["source_datasets"], ignore_index=True),
        target_column=target_column,
    )


def _load_from_patients(
    processed_path: Path,
    split_manifest: SplitManifest,
    feature_schema: FeatureSchema | None,
    source: str,
    target_column: str,
    splits: tuple[str, ...],
) -> LoadedDataset:
    """Original per-patient loading path (behaviour-preserving)."""
    feature_names = _discover_feature_names(processed_path, feature_schema)
    n_features = len(feature_names)
    logger.info("Discovered %d feature columns", n_features)

    if split_manifest is None:
        logger.warning("No split manifest provided; returning empty dataset.")
        return LoadedDataset(
            feature_names=feature_names,
            n_features=n_features,
            n_patients_total=0,
            source=source,
            target_column=target_column,
        )

    fragments: dict[str, dict[str, list]] = {
        split: {"X": [], "y": [], "patient_ids": [], "hours": [], "source_datasets": []}
        for split in splits
    }
    all_patient_ids: set[str] = set()

    for entry in split_manifest.patients:
        pid = entry.patient_id
        split = entry.split
        if split not in fragments:
            logger.warning("Patient %s has unknown split %r; skipping", pid, split)
            continue
        patient_file = processed_path / f"{pid}.parquet"

        if not patient_file.exists():
            logger.warning("Processed file missing for patient %s", pid)
            continue

        df = _load_processed_patient(patient_file)
        if df is None or len(df) == 0:
            continue

        built = _build_split_frame(df, pid, entry, feature_names, target_column)
        if built is None:
            continue
        X_pid, y_pid, patient_ids_pid, hours_pid, source_ds_pid = built

        fragment = fragments[split]
        fragment["X"].append(X_pid)
        fragment["y"].append(y_pid)
        fragment["patient_ids"].append(patient_ids_pid)
        fragment["hours"].append(hours_pid)
        fragment["source_datasets"].append(source_ds_pid)
        all_patient_ids.add(pid)

    splits_result: dict[str, SplitData | None] = {}
    for split in splits:
        splits_result[split] = _concat_split(
            split, fragments[split], feature_names, target_column
        )

    return LoadedDataset(
        train=splits_result.get("train"),
        validation=splits_result.get("validation"),
        test=splits_result.get("test"),
        feature_names=feature_names,
        n_features=n_features,
        n_patients_total=len(all_patient_ids),
        source=source,
        target_column=target_column,
    )


def _frame_for_shard(
    split_data: SplitData, feature_names: list[str], target_column: str
) -> pd.DataFrame:
    """Assemble a single-row-ordered frame for shard storage."""
    frame = pd.concat(
        [
            split_data.X[feature_names].reset_index(drop=True),
            split_data.y.reset_index(drop=True).rename(target_column),
            split_data.patient_ids.reset_index(drop=True).rename("patient_id"),
            split_data.hours.reset_index(drop=True).rename("hour"),
        ],
        axis=1,
    )
    if split_data.source_datasets is not None:
        frame["source_dataset"] = split_data.source_datasets.reset_index(drop=True)
    return frame


def _read_split_shard(
    shard_dir: Path,
    split: str,
    feature_names: list[str],
    target_column: str,
) -> SplitData | None:
    """Read one split shard and rebuild a SplitData in stored row order."""
    parquet_path = _shard_parquet_path(shard_dir, split)
    if not parquet_path.exists():
        return None
    frame = pd.read_parquet(parquet_path)
    if len(frame) == 0:
        return SplitData(
            split=split,
            X=pd.DataFrame(columns=feature_names),
            y=pd.Series(dtype=float),
            patient_ids=pd.Series(dtype=object),
            hours=pd.Series(dtype=int),
            target_column=target_column,
        )
    X = frame[feature_names].astype(np.float32, copy=False)
    y = frame[target_column].copy()
    patient_ids = frame["patient_id"].copy()
    hours = frame["hour"].copy()
    source_ds = frame["source_dataset"] if "source_dataset" in frame.columns else None
    return SplitData(
        split=split,
        X=X,
        y=y,
        patient_ids=patient_ids,
        hours=hours,
        source_datasets=source_ds,
        target_column=target_column,
    )


def _write_split_shards(
    processed_path: Path,
    shard_dir: Path,
    split_manifest: SplitManifest,
    feature_names: list[str],
    source: str,
    target_column: str,
    splits: tuple[str, ...],
) -> dict[str, int]:
    """Build and atomically write per-split Parquet shards plus fingerprints.

    Returns a mapping of split -> row count. Shards are only written for
    splits with at least one usable row. Callers use :func:`load_split_aware_dataset`
    with ``use_shards=True`` to read them back.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows_per_split: dict[str, int] = {}
    fragments: dict[str, dict[str, list]] = {
        split: {"X": [], "y": [], "patient_ids": [], "hours": [], "source_datasets": []}
        for split in splits
    }
    for entry in split_manifest.patients:
        split = entry.split
        if split not in fragments:
            continue
        patient_file = processed_path / f"{entry.patient_id}.parquet"
        if not patient_file.exists():
            continue
        df = _load_processed_patient(patient_file)
        if df is None or len(df) == 0:
            continue
        built = _build_split_frame(
            df, entry.patient_id, entry, feature_names, target_column
        )
        if built is None:
            continue
        X_pid, y_pid, pid_ids, hours, source_ds = built
        fragment = fragments[split]
        fragment["X"].append(X_pid)
        fragment["y"].append(y_pid)
        fragment["patient_ids"].append(pid_ids)
        fragment["hours"].append(hours)
        fragment["source_datasets"].append(source_ds)

    for split in splits:
        fragment = fragments[split]
        if not fragment["X"]:
            continue
        split_data = _concat_split(
            split, fragment, feature_names, target_column
        )
        if split_data is None:
            continue
        frame = _frame_for_shard(split_data, feature_names, target_column)
        _write_parquet_atomic(_shard_parquet_path(shard_dir, split), frame)
        fingerprint = _manifest_fingerprint(
            split_manifest, source, target_column, feature_names, splits
        )
        fingerprint_path = _shard_fingerprint_path(shard_dir, split)
        tmp_fp = fingerprint_path.with_name(fingerprint_path.name + ".tmp")
        try:
            tmp_fp.write_text(
                json.dumps({"fingerprint": fingerprint}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp_fp, fingerprint_path)
        finally:
            if tmp_fp.exists():
                tmp_fp.unlink(missing_ok=True)
        rows_per_split[split] = int(len(frame))
    return rows_per_split


def _try_load_shards(
    shard_dir: Path,
    split_manifest: SplitManifest,
    splits: tuple[str, ...],
    source: str,
    target_column: str,
    feature_names: list[str] | None = None,
) -> LoadedDataset | None:
    """Try loading all requested splits from fresh shards.

    Returns None if any shard is missing/stale (caller falls back to
    per-patient loading). When feature_names is None they are discovered
    from the shard fingerprint context (handled by the caller).
    """
    if feature_names is None:
        return None
    fingerprint = _manifest_fingerprint(
        split_manifest, source, target_column, feature_names, splits
    )
    split_data: dict[str, SplitData | None] = {}
    for split in splits:
        if not _shard_fresh(shard_dir, split, fingerprint):
            return None
        split_data[split] = _read_split_shard(
            shard_dir, split, feature_names, target_column
        )
    n_patients = sum(
        int(split_data[s].n_patients) for s in splits if split_data[s] is not None
    )
    return LoadedDataset(
        train=split_data.get("train"),
        validation=split_data.get("validation"),
        test=split_data.get("test"),
        feature_names=list(feature_names),
        n_features=len(feature_names),
        n_patients_total=n_patients,
        source=source,
        target_column=target_column,
    )


def load_split_aware_dataset(
    processed_dir: str | Path,
    split_manifest: SplitManifest,
    feature_schema: FeatureSchema | None = None,
    source: str = "synthetic",
    target_column: str = DERIVED_TARGET_COLUMN,
    *,
    splits: tuple[str, ...] | None = None,
    use_shards: bool = True,
    shard_dir: str | Path | None = None,
) -> LoadedDataset:
    """Load processed patient data aligned with patient-level splits.

    Parameters
    ----------
    processed_dir : str | Path
        Directory containing per-patient parquet files from Part 4.
    split_manifest : SplitManifest
        Patient-level split assignments from Part 3.
    feature_schema : FeatureSchema, optional
        Feature schema to validate column compatibility.
    source : str
        Data source label ("synthetic" or "real").
    target_column : str
        Target column to load. Defaults to the derived strict 6h target
        ``sepsis_within_6h``.
    splits : tuple of str, optional
        Subset of ``VALID_SPLITS`` to load. Defaults to all splits
        (train/validation/test). Split-scoped loading avoids touching
        patients/files for unrequested splits.
    use_shards : bool, default=True
        If True, attempt to read per-split Parquet shards. Shards are only
        used when their content fingerprint matches (stale-cache detection);
        otherwise the loader falls back to per-patient loading.
    shard_dir : str | Path, optional
        Directory holding shard files. Defaults to ``processed_dir/shards``.

    Returns
    -------
    LoadedDataset
        Train/validation/test SplitData objects with feature/target
        separation. No patient appears in more than one split.
    """
    processed_path = Path(processed_dir)
    splits = tuple(VALID_SPLITS) if splits is None else tuple(splits)
    shard_path = _shard_dir_for(processed_path, shard_dir)

    feature_names = _discover_feature_names(processed_path, feature_schema)
    n_features = len(feature_names)
    logger.info("Discovered %d feature columns", n_features)

    if split_manifest is None:
        logger.warning("No split manifest provided; returning empty dataset.")
        return LoadedDataset(
            feature_names=feature_names,
            n_features=n_features,
            n_patients_total=0,
            source=source,
            target_column=target_column,
        )

    # Fast path: per-split shards when fresh.
    if use_shards:
        shard_result = _try_load_shards(
            shard_path,
            split_manifest,
            splits,
            source,
            target_column,
            feature_names,
        )
        if shard_result is not None:
            logger.info("Loaded split-scoped shards from %s", shard_path)
            return shard_result
        logger.debug("Shards missing or stale; falling back to per-patient loading")

    # Fallback: original per-patient loading (unchanged behaviour).
    return _load_from_patients(
        processed_path,
        split_manifest,
        feature_schema,
        source,
        target_column,
        splits,
    )


def write_split_shards(
    processed_dir: str | Path,
    split_manifest: SplitManifest,
    feature_schema: FeatureSchema | None = None,
    source: str = "synthetic",
    target_column: str = DERIVED_TARGET_COLUMN,
    *,
    splits: tuple[str, ...] | None = None,
    shard_dir: str | Path | None = None,
) -> dict[str, int]:
    """Public entrypoint to build per-split shard Parquet files.

    Returns a mapping of split -> row count. Intended for offline use; the
    loader falls back to per-patient reading when shards are missing or
    stale. No existing P1 artifacts are overwritten outside ``shard_dir``.
    """
    processed_path = Path(processed_dir)
    splits = tuple(VALID_SPLITS) if splits is None else tuple(splits)
    shard_path = _shard_dir_for(processed_path, shard_dir)
    feature_names = _discover_feature_names(processed_path, feature_schema)
    return _write_split_shards(
        processed_path,
        shard_path,
        split_manifest,
        feature_names,
        source,
        target_column,
        splits,
    )


def load_from_manifest_and_features(
    manifest_path: str | Path,
    processed_dir: str | Path,
    schema_path: str | Path | None = None,
    source: str = "synthetic",
    target_column: str = DERIVED_TARGET_COLUMN,
) -> LoadedDataset:
    """Convenience: load split manifest and feature schema, then dataset."""
    manifest = SplitManifest.load(manifest_path)
    schema = None
    if schema_path is not None:
        schema = FeatureSchema.load(schema_path)
    return load_split_aware_dataset(
        processed_dir=processed_dir,
        split_manifest=manifest,
        feature_schema=schema,
        source=source,
        target_column=target_column,
    )


def check_split_overlap(dataset: LoadedDataset) -> dict:
    """Check for patient overlap between splits.

    Returns a dict with leakage detection results.
    """
    split_patients: dict[str, set[str]] = {}
    for split in VALID_SPLITS:
        sd = dataset.get_split(split)
        if sd is not None:
            split_patients[split] = set(sd.patient_ids.unique())
        else:
            split_patients[split] = set()

    overlaps: dict[str, list[str]] = {}
    splits_list = list(VALID_SPLITS)
    for i, s1 in enumerate(splits_list):
        for s2 in splits_list[i + 1 :]:
            inter = sorted(split_patients[s1] & split_patients[s2])
            if inter:
                overlaps[f"{s1}_vs_{s2}"] = inter

    # Every patient should appear in exactly one split.
    patient_split_counts: dict[str, int] = {}
    for split, pids in split_patients.items():
        for pid in pids:
            patient_split_counts[pid] = (
                patient_split_counts.get(pid, 0) + 1
            )
    multi_split = {pid: c for pid, c in patient_split_counts.items() if c > 1}

    return {
        "leakage_detected": len(overlaps) > 0 or len(multi_split) > 0,
        "n_overlapping_patients": sum(len(v) for v in overlaps.values()),
        "overlaps": overlaps,
        "patients_in_multiple_splits": multi_split,
        "every_patient_in_exactly_one_split": len(multi_split) == 0,
        "n_patients_total": dataset.n_patients_total,
        "n_patients_assigned": len(patient_split_counts),
    }


def validate_target_isolation(dataset: LoadedDataset) -> dict:
    """Validate that SepsisLabel is only in the target column, not features.

    Returns a dict with validation results.
    """
    issues: list[str] = []
    for split in VALID_SPLITS:
        sd = dataset.get_split(split)
        if sd is None:
            continue
        # Check no identifier column leaked into features.
        for col in IDENTIFIER_COLUMNS:
            if col in sd.X.columns:
                issues.append(
                    f"{split}: identifier column '{col}' found in features"
                )
        # Check no target column leaked into features.
        for col in TARGET_COLUMNS:
            if col in sd.X.columns:
                issues.append(
                    f"{split}: target column '{col}' found in features"
                )
        # Check target values are 0/1.
        y = sd.y.dropna()
        bad = y[~y.isin([0, 1])]
        if len(bad) > 0:
            issues.append(
                f"{split}: {len(bad)} row(s) with invalid "
                f"{sd.target_column} values"
            )

    return {
        "target_isolation_valid": len(issues) == 0,
        "issues": issues,
    }
