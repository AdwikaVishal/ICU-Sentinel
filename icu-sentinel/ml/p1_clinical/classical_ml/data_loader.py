"""Split-aware data loading for P1 classical ML models.

Loads processed patient data from Part 3/4 output and aligns it with
patient-level train/validation/test splits. Ensures no patient overlap
between splits and that SepsisLabel is preserved as the target only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..features.feature_schema import FeatureSchema
from ..preprocessing.patient_splits import (
    VALID_SPLITS,
    SplitManifest,
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


@dataclass
class SplitData:
    """Data for a single split (train/validation/test)."""

    split: str
    X: pd.DataFrame  # features only
    y: pd.Series  # target (SepsisLabel)
    patient_ids: pd.Series
    hours: pd.Series
    source_datasets: pd.Series | None = None
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
    return col != TARGET_COLUMN


def load_split_aware_dataset(
    processed_dir: str | Path,
    split_manifest: SplitManifest,
    feature_schema: FeatureSchema | None = None,
    source: str = "synthetic",
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

    Returns
    -------
    LoadedDataset
        Train/validation/test SplitData objects with feature/target
        separation. No patient appears in more than one split.
    """
    processed_path = Path(processed_dir)
    feature_names: list[str] = []

    # Discover feature columns from available parquet files.
    all_patient_files = sorted(processed_path.glob("*.parquet"))
    if not all_patient_files:
        logger.warning("No processed patient files found in %s", processed_path)
        return LoadedDataset(
            feature_names=feature_names,
            n_features=0,
            n_patients_total=0,
            source=source,
        )

    # Load the first file to discover columns.
    first_df = _load_processed_patient(all_patient_files[0])
    if first_df is None or len(first_df) == 0:
        return LoadedDataset(
            feature_names=feature_names,
            n_features=0,
            n_patients_total=0,
            source=source,
        )

    # Identify feature columns (exclude identifiers and target).
    discovered_features = [c for c in first_df.columns if _is_feature_column(c)]

    # Cross-reference with feature schema if provided.
    if feature_schema is not None:
        set(feature_schema.tabular_features().__class__ and [])
        schema_feature_names = set(feature_schema.feature_names())
        # Keep only schema features that are not identifiers/target.
        schema_feature_names = {
            f
            for f in schema_feature_names
            if f not in IDENTIFIER_COLUMNS and f != TARGET_COLUMN
        }
        discovered_features = [
            f for f in discovered_features if f in schema_feature_names
        ]
        logger.info(
            "Feature schema cross-reference: %d of %d discovered features "
            "match schema",
            len(discovered_features),
            len(set(discovered_features) | schema_feature_names),
        )

    feature_names = sorted(discovered_features)
    n_features = len(feature_names)
    logger.info("Discovered %d feature columns", n_features)

    # Build per-split data.
    if split_manifest is None:
        logger.warning("No split manifest provided; returning empty dataset.")
        return LoadedDataset(
            feature_names=feature_names,
            n_features=n_features,
            n_patients_total=0,
            source=source,
        )

    splits: dict[str, SplitData | None] = {
        "train": None,
        "validation": None,
        "test": None,
    }
    all_patient_ids: set[str] = set()

    for entry in split_manifest.patients:
        pid = entry.patient_id
        split = entry.split
        patient_file = processed_path / f"{pid}.parquet"

        if not patient_file.exists():
            logger.warning("Processed file missing for patient %s", pid)
            continue

        df = _load_processed_patient(patient_file)
        if df is None or len(df) == 0:
            continue

        # Ensure required columns exist.
        missing = [c for c in ([TARGET_COLUMN] + feature_names) if c not in df.columns]
        if missing:
            logger.warning(
                "Patient %s missing columns %s; skipping", pid, missing
            )
            continue

        # Extract features, target, identifiers.
        X_pid = df[feature_names].copy()
        y_pid = df[TARGET_COLUMN].copy()
        patient_ids_pid = pd.Series([pid] * len(df))
        hours_pid = df["hour"].copy() if "hour" in df.columns else pd.Series(
            range(len(df))
        )
        source_ds_pid = (
            df["source_dataset"].copy()
            if "source_dataset" in df.columns
            else pd.Series([entry.source_dataset] * len(df))
        )

        # Accumulate into the appropriate split.
        current = splits.get(split)
        if current is None:
            splits[split] = SplitData(
                split=split,
                X=X_pid,
                y=y_pid,
                patient_ids=patient_ids_pid,
                hours=hours_pid,
                source_datasets=source_ds_pid,
            )
        else:
            splits[split] = SplitData(
                split=split,
                X=pd.concat([current.X, X_pid], ignore_index=True),
                y=pd.concat([current.y, y_pid], ignore_index=True),
                patient_ids=pd.concat(
                    [current.patient_ids, patient_ids_pid], ignore_index=True
                ),
                hours=pd.concat([current.hours, hours_pid], ignore_index=True),
                source_datasets=pd.concat(
                    [current.source_datasets, source_ds_pid], ignore_index=True
                ),
            )

        all_patient_ids.add(pid)

    # Assign splits to LoadedDataset.
    result = LoadedDataset(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        feature_names=feature_names,
        n_features=n_features,
        n_patients_total=len(all_patient_ids),
        source=source,
    )
    return result


def load_from_manifest_and_features(
    manifest_path: str | Path,
    processed_dir: str | Path,
    schema_path: str | Path | None = None,
    source: str = "synthetic",
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
        # Check target is not in features.
        if TARGET_COLUMN in sd.X.columns:
            issues.append(
                f"{split}: target column '{TARGET_COLUMN}' found in features"
            )
        # Check target values are 0/1.
        y = sd.y.dropna()
        bad = y[~y.isin([0, 1])]
        if len(bad) > 0:
            issues.append(
                f"{split}: {len(bad)} row(s) with invalid SepsisLabel values"
            )

    return {
        "target_isolation_valid": len(issues) == 0,
        "issues": issues,
    }
