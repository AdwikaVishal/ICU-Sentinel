"""Dataset ingestion for the PhysioNet/CinC Challenge 2019.

Provides a reusable, memory-efficient loader for the official sepsis
prediction dataset. The loader is designed to work against the actual
dataset when access is available, and includes synthetic-fixture support
for unit testing without requiring real patient data.

Dataset facts (from official documentation at physionet.org):
- One pipe-delimited (.psv) file per patient, e.g. training/p00101.psv
- 41 columns: 40 time-dependent variables + SepsisLabel
- Separator is "|" (no spaces in actual files)
- "NaN" indicates a missing measurement
- SepsisLabel is shifted 6 hours ahead: 1 if t >= t_sepsis - 6, else 0
- Access requires a PhysioNet credentialed account.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("icu_sentinel.p1_clinical.ingestion")


# Official column order per the Challenge data description (Table 1).
# Columns 1-8: Vital signs
# Columns 9-34: Laboratory values
# Columns 35-40: Demographics
# Column 41: Outcome (SepsisLabel)
CHALLENGE_COLUMNS: list[str] = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
    "SepsisLabel",
]

REQUIRED_COLUMNS: list[str] = list(CHALLENGE_COLUMNS)

# Regex for official training file names: training/p00101.psv
# (p followed by 5-6 digits, matching the documented example p00101).
PATIENT_FILE_RE = re.compile(r"^(p\d{5,6})\.psv$")


@dataclass
class PatientFileRef:
    """Reference to a single patient data file."""

    patient_id: str
    filepath: Path
    file_size_bytes: int


@dataclass
class LoadResult:
    """Result of loading a single patient file."""

    patient_id: str
    dataframe: pd.DataFrame
    n_rows: int
    n_missing_values: int
    n_sepsis_positive_hours: int
    file_path: Path


@dataclass
class DatasetCatalog:
    """Catalog of patient files discovered in a dataset directory."""

    root: Path
    files: list[PatientFileRef] = field(default_factory=list)
    n_patients: int = 0
    total_bytes: int = 0

    def __post_init__(self) -> None:
        self.n_patients = len(self.files)
        self.total_bytes = sum(f.file_size_bytes for f in self.files)


class PhysioNetChallenge2019Loader:
    """Loader for the PhysioNet/CinC Challenge 2019 sepsis dataset.

    Iterates over patient files without loading the entire dataset into
    memory at once. Preserves patient identifiers derived from filenames.
    """

    def __init__(
        self,
        root: str | Path,
        columns: list[str] | None = None,
        separator: str = "|",
        na_values: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.columns = columns or REQUIRED_COLUMNS
        self.separator = separator
        self.na_values = na_values or ["NaN", "NA", "na", "", None]
        self.logger = logging.getLogger(
            "icu_sentinel.p1_clinical.ingestion.PhysioNetChallenge2019Loader"
        )

    # ------------------------------------------------------------------
    # Discovery / catalog
    # ------------------------------------------------------------------
    def catalog(self) -> DatasetCatalog:
        """Discover all .psv patient files under ``root``.

        Returns a DatasetCatalog with per-file references. Does not load
        any file contents, so it is safe to call on large directories.
        """
        files: list[PatientFileRef] = []
        if not self.root.exists():
            self.logger.warning("Dataset root does not exist: %s", self.root)
            return DatasetCatalog(root=self.root, files=files)

        for path in sorted(self.root.rglob("*.psv")):
            match = PATIENT_FILE_RE.match(path.name)
            if not match:
                self.logger.debug("Skipping non-standard filename: %s", path.name)
                continue
            patient_id = match.group(1)
            try:
                file_size = path.stat().st_size
            except OSError as exc:
                self.logger.warning("Cannot stat %s: %s", path, exc)
                continue
            files.append(
                PatientFileRef(
                    patient_id=patient_id,
                    filepath=path,
                    file_size_bytes=file_size,
                )
            )

        return DatasetCatalog(root=self.root, files=files)

    def patient_ids(self) -> list[str]:
        """Return sorted list of patient IDs discovered in ``root``."""
        return [ref.patient_id for ref in self.catalog().files]

    # ------------------------------------------------------------------
    # Single-file loading
    # ------------------------------------------------------------------
    def load_patient(self, patient_id: str) -> pd.DataFrame | None:
        """Load a single patient file by ID.

        Returns None if the file does not exist or cannot be parsed.
        """
        ref = self._find_file(patient_id)
        if ref is None:
            self.logger.warning("No file found for patient %s", patient_id)
            return None
        return self.load_file(ref.filepath)

    def load_file(self, filepath: str | Path) -> pd.DataFrame | None:
        """Load a single .psv file into a DataFrame.

        Applies the official column schema and NA handling. Does not
        modify the source file.
        """
        path = Path(filepath)
        if not path.exists():
            self.logger.error("File does not exist: %s", path)
            return None

        try:
            df = pd.read_csv(
                path,
                sep=self.separator,
                names=self.columns,
                na_values=self.na_values,
                dtype={c: "float64" for c in self.columns if c not in ("Age", "Gender", "Unit1", "Unit2", "SepsisLabel")},
            )
        except Exception as exc:  # noqa: BLE001 - surface parse errors to caller
            self.logger.error("Failed to parse %s: %s", path, exc)
            return None

        # Enforce integer-like columns where clinically meaningful.
        for int_col in ("Gender", "SepsisLabel"):
            if int_col in df.columns:
                df[int_col] = pd.to_numeric(df[int_col], errors="coerce")

        return df

    def load_patient_result(self, patient_id: str) -> LoadResult | None:
        """Load a patient file and return a LoadResult with summary stats."""
        ref = self._find_file(patient_id)
        if ref is None:
            return None
        df = self.load_file(ref.filepath)
        if df is None:
            return None
        n_missing = int(df.isna().sum().sum())
        n_pos = 0
        if "SepsisLabel" in df.columns:
            n_pos = int((df["SepsisLabel"] == 1).sum())
        return LoadResult(
            patient_id=patient_id,
            dataframe=df,
            n_rows=len(df),
            n_missing_values=n_missing,
            n_sepsis_positive_hours=n_pos,
            file_path=ref.filepath,
        )

    # ------------------------------------------------------------------
    # Memory-efficient iteration
    # ------------------------------------------------------------------
    def iter_patients(
        self,
        limit: int | None = None,
    ) -> Iterator[LoadResult]:
        """Yield LoadResult for each patient file, one at a time.

        Suitable for streaming large datasets without loading everything
        into memory. Optionally cap the number of patients yielded.
        """
        count = 0
        for ref in self.catalog().files:
            result = self.load_patient_result(ref.patient_id)
            if result is None:
                continue
            yield result
            count += 1
            if limit is not None and count >= limit:
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_file(self, patient_id: str) -> PatientFileRef | None:
        for ref in self.catalog().files:
            if ref.patient_id == patient_id:
                return ref
        return None

    def patient_id_from_filename(self, filename: str) -> str | None:
        """Extract the official patient ID from a filename like p00101.psv."""
        match = PATIENT_FILE_RE.match(Path(filename).name)
        if not match:
            return None
        return match.group(1)


def build_synthetic_patient_file(
    patient_id: str = "p00001",
    n_hours: int = 24,
    seed: int = 42,
    sepsis_onset_hour: int | None = None,
) -> pd.DataFrame:
    """Build a synthetic single-patient DataFrame matching the challenge schema.

    Intended for unit tests only. No real patient data is used.
    """
    rng = np.random.default_rng(seed)
    cols = REQUIRED_COLUMNS
    data: dict[str, list] = {}
    for col in cols:
        if col == "SepsisLabel":
            continue
        if col in ("Age", "Gender", "Unit1", "Unit2"):
            data[col] = [np.nan] * n_hours
        else:
            vals = rng.normal(0, 1, size=n_hours)
            missing_mask = rng.random(n_hours) < 0.2
            vals[missing_mask] = np.nan
            data[col] = vals.tolist()

    # Demographics are constant across hours for a patient.
    data["Age"] = [float(rng.integers(18, 90))] * n_hours
    data["Gender"] = [float(rng.integers(0, 2))] * n_hours
    data["Unit1"] = [float(rng.integers(0, 2))] * n_hours
    data["Unit2"] = [float(rng.integers(0, 2))] * n_hours

    # ICULOS: hours since ICU admit, starting at 1.
    data["ICULOS"] = [float(h + 1) for h in range(n_hours)]
    data["HospAdmTime"] = [float(rng.integers(-72, 0))] * n_hours

    # SepsisLabel: shifted 6 hours ahead of onset.
    labels = [0] * n_hours
    if sepsis_onset_hour is not None:
        for t in range(n_hours):
            if t >= sepsis_onset_hour - 6:
                labels[t] = 1
    data["SepsisLabel"] = labels

    return pd.DataFrame(data, columns=REQUIRED_COLUMNS)


def write_synthetic_psv(
    df: pd.DataFrame,
    filepath: str | Path,
    separator: str = "|",
) -> Path:
    """Write a DataFrame to a pipe-delimited .psv file (no header)."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=separator, index=False, header=False, na_rep="NaN")
    return path
