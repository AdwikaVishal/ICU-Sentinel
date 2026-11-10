"""Patient-level data splitting for the P1 preprocessing pipeline.

Splits patients (not individual hourly rows) into train/validation/test
sets. All observations belonging to a patient remain in a single split,
with no patient overlap between splits. Supports reproducible random
seeds and stratified splitting by sepsis label.

Source dataset identity (Set A vs Set B from the official Challenge)
is preserved as a hospital identifier to enable future cross-hospital
evaluation architecture, without claiming cross-hospital validation has
been performed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .schemas import (
    CLINICAL_VARIABLES,
    ProcessedPatient,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.preprocessing.patient_splits")

VALID_SPLITS = ("train", "validation", "test")


@dataclass
class PatientSplitEntry:
    """A single patient's split assignment."""

    patient_id: str
    hospital_id: str | None
    split: str  # train / validation / test
    n_records: int
    n_sepsis_positive_hours: int
    n_sepsis_negative_hours: int
    has_sepsis: bool
    source_file: str
    source_dataset: str


@dataclass
class SplitManifest:
    """Complete split manifest for all patients."""

    strategy: str
    random_state: int
    test_size: float
    validation_size: float
    stratify_by_label: bool
    n_patients_total: int
    n_patients_train: int
    n_patients_validation: int
    n_patients_test: int
    n_records_train: int
    n_records_validation: int
    n_records_test: int
    n_sepsis_patients_train: int
    n_sepsis_patients_validation: int
    n_sepsis_patients_test: int
    n_sepsis_positive_hours_train: int
    n_sepsis_positive_hours_validation: int
    n_sepsis_positive_hours_test: int
    patients: list[PatientSplitEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "random_state": self.random_state,
            "test_size": self.test_size,
            "validation_size": self.validation_size,
            "stratify_by_label": self.stratify_by_label,
            "n_patients_total": self.n_patients_total,
            "n_patients_train": self.n_patients_train,
            "n_patients_validation": self.n_patients_validation,
            "n_patients_test": self.n_patients_test,
            "n_records_train": self.n_records_train,
            "n_records_validation": self.n_records_validation,
            "n_records_test": self.n_records_test,
            "n_sepsis_patients_train": self.n_sepsis_patients_train,
            "n_sepsis_patients_validation": self.n_sepsis_patients_validation,
            "n_sepsis_patients_test": self.n_sepsis_patients_test,
            "n_sepsis_positive_hours_train": self.n_sepsis_positive_hours_train,
            "n_sepsis_positive_hours_validation": self.n_sepsis_positive_hours_validation,
            "n_sepsis_positive_hours_test": self.n_sepsis_positive_hours_test,
            "patients": [
                {
                    "patient_id": p.patient_id,
                    "hospital_id": p.hospital_id,
                    "split": p.split,
                    "n_records": p.n_records,
                    "n_sepsis_positive_hours": p.n_sepsis_positive_hours,
                    "n_sepsis_negative_hours": p.n_sepsis_negative_hours,
                    "has_sepsis": p.has_sepsis,
                    "source_file": p.source_file,
                    "source_dataset": p.source_dataset,
                }
                for p in self.patients
            ],
        }

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> SplitManifest:
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)
        patients = [
            PatientSplitEntry(
                patient_id=p["patient_id"],
                hospital_id=p["hospital_id"],
                split=p["split"],
                n_records=p["n_records"],
                n_sepsis_positive_hours=p["n_sepsis_positive_hours"],
                n_sepsis_negative_hours=p["n_sepsis_negative_hours"],
                has_sepsis=p["has_sepsis"],
                source_file=p["source_file"],
                source_dataset=p["source_dataset"],
            )
            for p in data["patients"]
        ]
        return cls(
            strategy=data["strategy"],
            random_state=data["random_state"],
            test_size=data["test_size"],
            validation_size=data["validation_size"],
            stratify_by_label=data["stratify_by_label"],
            n_patients_total=data["n_patients_total"],
            n_patients_train=data["n_patients_train"],
            n_patients_validation=data["n_patients_validation"],
            n_patients_test=data["n_patients_test"],
            n_records_train=data["n_records_train"],
            n_records_validation=data["n_records_validation"],
            n_records_test=data["n_records_test"],
            n_sepsis_patients_train=data["n_sepsis_patients_train"],
            n_sepsis_patients_validation=data["n_sepsis_patients_validation"],
            n_sepsis_patients_test=data["n_sepsis_patients_test"],
            n_sepsis_positive_hours_train=data["n_sepsis_positive_hours_train"],
            n_sepsis_positive_hours_validation=data["n_sepsis_positive_hours_validation"],
            n_sepsis_positive_hours_test=data["n_sepsis_positive_hours_test"],
            patients=patients,
        )

    def patient_ids_by_split(self, split: str) -> list[str]:
        return [p.patient_id for p in self.patients if p.split == split]

    def overlap_between_splits(self) -> dict[str, list[str]]:
        """Return any patient IDs appearing in more than one split."""
        splits = {s: set(self.patient_ids_by_split(s)) for s in VALID_SPLITS}
        overlaps: dict[str, list[str]] = {}
        for i, s1 in enumerate(VALID_SPLITS):
            for s2 in list(VALID_SPLITS)[i + 1:]:
                inter = sorted(splits[s1] & splits[s2])
                if inter:
                    overlaps[f"{s1}_vs_{s2}"] = inter
        return overlaps


class PatientSplitter:
    """Splits patients into train/validation/test splits."""

    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42,
        stratify_by_label: bool = True,
        strategy: str = "stratified_random",
    ) -> None:
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        if not (0 <= validation_size < 1):
            raise ValueError("validation_size must be between 0 and 1")
        if test_size + validation_size >= 1:
            raise ValueError("test_size + validation_size must be < 1")
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.stratify_by_label = stratify_by_label
        self.strategy = strategy

    def _stratified_or_plain_split(
        self,
        ids: list[str],
        labels: list[bool],
        test_size: float,
    ) -> tuple[list[str], list[str]]:
        """Attempt stratified split; fall back to plain split if not possible."""
        try:
            if self.stratify_by_label and len(set(labels)) > 1 and len(ids) >= 2:
                return train_test_split(
                    ids,
                    test_size=test_size,
                    random_state=self.random_state,
                    stratify=labels,
                )
        except ValueError:
            pass
        return train_test_split(
            ids,
            test_size=test_size,
            random_state=self.random_state,
        )

    def split(
        self,
        patients: list[ProcessedPatient],
    ) -> SplitManifest:
        """Split a list of ProcessedPatient into train/validation/test.

        All observations of a patient remain in one split.
        """
        # Build patient-level records.
        entries: list[PatientSplitEntry] = []
        for p in patients:
            labels = p.sepsis_label
            n_pos = sum(1 for x in labels if x == 1)
            n_neg = sum(1 for x in labels if x == 0)
            entries.append(
                PatientSplitEntry(
                    patient_id=p.patient_id,
                    hospital_id=p.hospital_id,
                    split="",  # assigned below
                    n_records=p.n_rows,
                    n_sepsis_positive_hours=n_pos,
                    n_sepsis_negative_hours=n_neg,
                    has_sepsis=n_pos > 0,
                    source_file=p.source_file,
                    source_dataset=p.source_dataset,
                )
            )

        if len(entries) == 0:
            return self._empty_manifest(entries)

        # Deterministic ordering by patient_id for reproducibility.
        entries.sort(key=lambda e: e.patient_id)

        ids = [e.patient_id for e in entries]
        has_sepsis = [e.has_sepsis for e in entries]

        # First split: train+val vs test.
        train_val_ids, test_ids = self._stratified_or_plain_split(
            ids, has_sepsis, self.test_size
        )

        # Second split: train vs validation from train_val.
        val_size_adjusted = self.validation_size / (1.0 - self.test_size)
        train_val_id_set = set(train_val_ids)
        train_val_ids_list = [e.patient_id for e in entries if e.patient_id in train_val_id_set]
        train_val_sepsis = [
            e.has_sepsis for e in entries if e.patient_id in train_val_id_set
        ]

        if len(train_val_ids_list) > 1:
            train_ids, val_ids = self._stratified_or_plain_split(
                train_val_ids_list, train_val_sepsis, val_size_adjusted
            )
        else:
            train_ids = list(train_val_ids_list)
            val_ids = []

        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)

        # Assign splits.
        for e in entries:
            if e.patient_id in train_set:
                e.split = "train"
            elif e.patient_id in val_set:
                e.split = "validation"
            elif e.patient_id in test_set:
                e.split = "test"
            else:
                # Should not happen, but safeguard.
                e.split = "train"
                logger.warning(
                    "Patient %s not assigned to any split; defaulting to train",
                    e.patient_id,
                )

        return self._build_manifest(entries)

    def _build_manifest(self, entries: list[PatientSplitEntry]) -> SplitManifest:
        train = [e for e in entries if e.split == "train"]
        val = [e for e in entries if e.split == "validation"]
        test = [e for e in entries if e.split == "test"]

        return SplitManifest(
            strategy=self.strategy,
            random_state=self.random_state,
            test_size=self.test_size,
            validation_size=self.validation_size,
            stratify_by_label=self.stratify_by_label,
            n_patients_total=len(entries),
            n_patients_train=len(train),
            n_patients_validation=len(val),
            n_patients_test=len(test),
            n_records_train=sum(e.n_records for e in train),
            n_records_validation=sum(e.n_records for e in val),
            n_records_test=sum(e.n_records for e in test),
            n_sepsis_patients_train=sum(1 for e in train if e.has_sepsis),
            n_sepsis_patients_validation=sum(1 for e in val if e.has_sepsis),
            n_sepsis_patients_test=sum(1 for e in test if e.has_sepsis),
            n_sepsis_positive_hours_train=sum(e.n_sepsis_positive_hours for e in train),
            n_sepsis_positive_hours_validation=sum(e.n_sepsis_positive_hours for e in val),
            n_sepsis_positive_hours_test=sum(e.n_sepsis_positive_hours for e in test),
            patients=entries,
        )

    def _empty_manifest(self, entries: list[PatientSplitEntry]) -> SplitManifest:
        return SplitManifest(
            strategy=self.strategy,
            random_state=self.random_state,
            test_size=self.test_size,
            validation_size=self.validation_size,
            stratify_by_label=self.stratify_by_label,
            n_patients_total=0,
            n_patients_train=0,
            n_patients_validation=0,
            n_patients_test=0,
            n_records_train=0,
            n_records_validation=0,
            n_records_test=0,
            n_sepsis_patients_train=0,
            n_sepsis_patients_validation=0,
            n_sepsis_patients_test=0,
            n_sepsis_positive_hours_train=0,
            n_sepsis_positive_hours_validation=0,
            n_sepsis_positive_hours_test=0,
            patients=entries,
        )


def check_split_leakage(manifest: SplitManifest) -> dict:
    """Check for patient-level leakage between splits.

    Returns a dict with leakage detection results.
    """
    overlaps = manifest.overlap_between_splits()
    n_overlapping = sum(len(v) for v in overlaps.values())

    # Check that every patient appears in exactly one split.
    split_counts: dict[str, int] = {}
    for p in manifest.patients:
        split_counts[p.patient_id] = split_counts.get(p.patient_id, 0) + 1
    multi_split = {pid: cnt for pid, cnt in split_counts.items() if cnt > 1}

    # Check class distribution consistency.
    split_sepsis_rates: dict[str, float] = {}
    for split in VALID_SPLITS:
        split_patients = [p for p in manifest.patients if p.split == split]
        if split_patients:
            split_sepsis_rates[split] = round(
                sum(1 for p in split_patients if p.has_sepsis) / len(split_patients), 4
            )
        else:
            split_sepsis_rates[split] = 0.0

    return {
        "leakage_detected": n_overlapping > 0 or len(multi_split) > 0,
        "n_overlapping_patients": n_overlapping,
        "overlaps": overlaps,
        "patients_in_multiple_splits": multi_split,
        "every_patient_in_exactly_one_split": len(multi_split) == 0,
        "split_sepsis_rates": split_sepsis_rates,
        "n_patients_total": manifest.n_patients_total,
        "n_patients_assigned": sum(
            1 for p in manifest.patients if p.split in VALID_SPLITS
        ),
    }


def save_split_manifests(
    manifest: SplitManifest,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save the full manifest and per-split patient ID lists."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "split_manifest.json"
    manifest.save(manifest_path)

    saved: dict[str, Path] = {"manifest": manifest_path}
    for split in VALID_SPLITS:
        ids = manifest.patient_ids_by_split(split)
        split_path = out / f"{split}_patient_ids.json"
        with open(split_path, "w") as f:
            json.dump(
                {
                    "split": split,
                    "n_patients": len(ids),
                    "patient_ids": ids,
                    "source": "synthetic" if manifest.strategy == "synthetic" else "real",
                },
                f,
                indent=2,
            )
        saved[split] = split_path
    return saved


def iter_processed_patients_from_directory(
    processed_dir: str | Path,
):
    """Yield ProcessedPatient objects from saved parquet files one at a time.

    Each patient is stored as a separate parquet file named by patient_id.
    Streaming avoids loading the full dataset into memory.
    """
    out = Path(processed_dir)
    if not out.exists():
        return
    for path in sorted(out.glob("*.parquet")):
        df = pd.read_parquet(path)
        pid = df["patient_id"].iloc[0]
        yield ProcessedPatient(
            patient_id=str(pid),
            hospital_id=(
                str(df["hospital_id"].iloc[0])
                if "hospital_id" in df.columns and pd.notna(df["hospital_id"].iloc[0])
                else None
            ),
            hour=df["hour"].tolist(),
            clinical={v: df[v].tolist() for v in CLINICAL_VARIABLES if v in df.columns},
            sepsis_label=[int(x) for x in df["SepsisLabel"].tolist()],
            source_file=(
                str(df["source_file"].iloc[0])
                if "source_file" in df.columns and pd.notna(df["source_file"].iloc[0])
                else None
            ),
            source_dataset=(
                str(df["source_dataset"].iloc[0])
                if "source_dataset" in df.columns and pd.notna(df["source_dataset"].iloc[0])
                else None
            ),
            n_rows=len(df),
        )


def load_processed_patients_from_directory(
    processed_dir: str | Path,
) -> list[ProcessedPatient]:
    """Load all ProcessedPatient objects from saved parquet files.

    Prefer ``iter_processed_patients_from_directory`` for full-dataset
    runs to avoid holding every patient in memory.
    """
    return list(iter_processed_patients_from_directory(processed_dir))



