"""Patient repositories for replay.

Two implementations are provided:

- ``SyntheticDemoPatientRepository`` generates deterministic synthetic ICU
  stays for local demonstration. Every patient is explicitly labelled as
  synthetic. The synthetic ground-truth label is used only for retrospective
  display/evaluation, never as a prediction input.
- ``DirectoryPatientRepository`` reads P1-processed per-patient parquet files
  when real de-identified PhysioNet data is available.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ml.p1_clinical.features.feature_config import (
    LABORATORY_VARIABLES,
    STATIC_VARIABLES,
    VITAL_SIGN_VARIABLES,
)
from ml.p1_clinical.preprocessing.patient_splits import (
    load_processed_patients_from_directory,
)
from ml.p1_clinical.preprocessing.schemas import CLINICAL_VARIABLES, ProcessedPatient

from .replay_schemas import PatientSummary


class PatientNotFoundError(LookupError):
    """Raised when a patient ID is not present in the repository."""


class PatientRepository(Protocol):
    data_source: str
    is_synthetic: bool

    def list_patients(self) -> list[PatientSummary]:
        ...

    def get_patient(self, patient_id: str) -> ProcessedPatient:
        ...

    def patient_exists(self, patient_id: str) -> bool:
        ...


@dataclass(frozen=True)
class DemoProfile:
    patient_id: str
    seed: int
    n_hours: int
    onset_hour: int | None
    peak_severity: float
    missing_rate: float
    lab_interval: int
    skip_hours: tuple[int, ...] = ()

    @property
    def is_positive(self) -> bool:
        return self.onset_hour is not None


REPLAY_PROFILES: tuple[DemoProfile, ...] = (
    DemoProfile("DEMO-0001", 101, 36, None, 0.15, 0.08, 4),
    DemoProfile("DEMO-0002", 102, 48, 30, 0.95, 0.12, 4),
    DemoProfile("DEMO-0003", 103, 30, 20, 0.60, 0.15, 6, (12, 13, 22)),
    DemoProfile("DEMO-0004", 104, 24, 16, 1.00, 0.10, 4),
    DemoProfile("DEMO-0005", 105, 18, None, 0.10, 0.06, 6),
    DemoProfile("DEMO-0006", 106, 60, 44, 0.85, 0.20, 4, (7, 8, 31)),
)

TRAINING_PROFILES: tuple[DemoProfile, ...] = tuple(
    DemoProfile(
        patient_id=f"DEMO-TRAIN-{index:04d}",
        seed=500 + index,
        n_hours=30 + (index % 4) * 6,
        onset_hour=(None if index % 3 == 0 else 14 + (index * 5) % 20),
        peak_severity=(0.2 if index % 3 == 0 else 0.5 + (index % 5) * 0.1),
        missing_rate=0.06 + (index % 4) * 0.03,
        lab_interval=4 if index % 2 == 0 else 6,
    )
    for index in range(10)
)


_VITAL_BASE: dict[str, tuple[float, float, float, float]] = {
    # variable: (normal mean, sepsis change, normal noise, lower/upper clip omitted)
    "HR": (76.0, 30.0, 6.0, 0.0),
    "O2Sat": (98.0, -6.0, 1.5, 0.0),
    "Temp": (36.7, 1.4, 0.35, 0.0),
    "SBP": (124.0, -28.0, 10.0, 0.0),
    "MAP": (88.0, -20.0, 7.0, 0.0),
    "DBP": (70.0, -15.0, 6.0, 0.0),
    "Resp": (16.0, 10.0, 3.0, 0.0),
    "EtCO2": (36.0, -4.0, 3.0, 0.0),
}

_LAB_BASE: dict[str, tuple[float, float, float, float]] = {
    "Lactate": (1.0, 3.8, 0.4, 0.0),
    "WBC": (7.5, 6.0, 1.8, 0.0),
    "Creatinine": (0.9, 0.8, 0.25, 0.0),
    "BUN": (14.0, 12.0, 4.0, 0.0),
    "Platelets": (250.0, -90.0, 35.0, 0.0),
    "Hgb": (12.0, -1.5, 1.0, 0.0),
    "Glucose": (110.0, 35.0, 18.0, 0.0),
    "Potassium": (4.0, 0.4, 0.3, 0.0),
    "pH": (7.40, -0.07, 0.025, 0.0),
    "HCO3": (24.0, -3.0, 1.5, 0.0),
    "BaseExcess": (0.0, -3.0, 1.5, 0.0),
    "FiO2": (21.0, 25.0, 3.0, 0.0),
    "PaCO2": (40.0, -4.0, 4.0, 0.0),
    "SaO2": (97.0, -5.0, 1.5, 0.0),
    "AST": (30.0, 20.0, 8.0, 0.0),
    "Alkalinephos": (80.0, 20.0, 12.0, 0.0),
    "Calcium": (9.0, -0.3, 0.4, 0.0),
    "Chloride": (103.0, 1.0, 2.5, 0.0),
    "Bilirubin_direct": (0.2, 0.2, 0.1, 0.0),
    "Bilirubin_total": (0.8, 0.5, 0.25, 0.0),
    "Magnesium": (2.0, 0.1, 0.2, 0.0),
    "Phosphate": (3.2, -0.2, 0.5, 0.0),
    "TroponinI": (0.03, 0.02, 0.02, 0.0),
    "Hct": (37.0, -4.0, 2.5, 0.0),
    "PTT": (30.0, 4.0, 4.0, 0.0),
    "WBC_placeholder": (0.0, 0.0, 0.0, 0.0),
    "Fibrinogen": (300.0, -40.0, 40.0, 0.0),
}

_RARELY_MEASURED = {"EtCO2", "FiO2", "TroponinI", "Bilirubin_direct"}


class SyntheticDemoPatientRepository:
    """Deterministic synthetic patient repository for local demonstration."""

    data_source = "synthetic_demo"
    is_synthetic = True
    source_dataset = "synthetic_demo"

    def __init__(
        self,
        *,
        replay_profiles: tuple[DemoProfile, ...] = REPLAY_PROFILES,
        include_training: bool = True,
    ) -> None:
        self._patients: dict[str, ProcessedPatient] = {}
        for profile in replay_profiles:
            patient = _build_demo_patient(profile)
            self._patients[patient.patient_id] = patient
        if include_training:
            for profile in TRAINING_PROFILES:
                patient = _build_demo_patient(profile)
                self._patients[patient.patient_id] = patient

    def list_patients(self) -> list[PatientSummary]:
        replay_ids = {p.patient_id for p in REPLAY_PROFILES}
        return [
            summarize_patient(patient, is_synthetic=True)
            for patient_id, patient in sorted(self._patients.items())
            if patient_id in replay_ids
        ]

    def get_patient(self, patient_id: str) -> ProcessedPatient:
        if patient_id not in self._patients:
            raise PatientNotFoundError(f"Patient not found: {patient_id}")
        return self._patients[patient_id]

    def patient_exists(self, patient_id: str) -> bool:
        return patient_id in self._patients

    def training_patients(self) -> list[ProcessedPatient]:
        return [
            patient
            for patient_id, patient in sorted(self._patients.items())
            if patient_id.startswith("DEMO-TRAIN-")
        ]


class DirectoryPatientRepository:
    """Repository backed by P1-processed per-patient parquet files.

    Patient listing is powered by the P1 split manifest (a single JSON read)
    so startup stays fast even with tens of thousands of patients. Full
    :class:`ProcessedPatient` objects are loaded lazily from parquet on first
    access and cached in memory.
    """

    data_source = "processed_parquet"
    is_synthetic = False

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.exists():
            raise ValueError(
                f"Processed patient directory not found: {self.directory}"
            )
        # Load split manifest for lightweight patient index.
        manifest_path = self.directory.parent / "splits" / "split_manifest.json"
        self._index: dict[str, dict] = {}
        if manifest_path.exists():
            import json

            with open(manifest_path) as f:
                manifest = json.load(f)
            for entry in manifest.get("patients", []):
                pid = entry["patient_id"]
                self._index[pid] = {
                    "patient_id": pid,
                    "split": entry.get("split"),
                    "n_records": entry.get("n_records", 0),
                    "has_sepsis": entry.get("has_sepsis", False),
                    "source_file": entry.get("source_file"),
                    "source_dataset": entry.get("source_dataset"),
                    "hour_min": None,
                    "hour_max": None,
                    "n_sepsis_positive_hours": entry.get("n_sepsis_positive_hours", 0),
                    "n_sepsis_negative_hours": entry.get("n_sepsis_negative_hours", 0),
                }
        else:
            # Fall back to scanning parquet filenames.
            for path in sorted(self.directory.glob("*.parquet")):
                pid = path.stem
                self._index[pid] = {
                    "patient_id": pid,
                    "split": None,
                    "n_records": 0,
                    "has_sepsis": False,
                    "source_file": path.name,
                    "source_dataset": None,
                    "hour_min": None,
                    "hour_max": None,
                    "n_sepsis_positive_hours": 0,
                    "n_sepsis_negative_hours": 0,
                }
        if not self._index:
            raise ValueError(
                f"No processable patient parquet files found in {self.directory}"
            )
        # Load pre-computed data-quality metrics from cache (generated by
        # compute_patient_metrics.py). Falls back to per-patient parquet
        # reads on first access if the cache is absent.
        self._patients: dict[str, ProcessedPatient] = {}
        self._hour_ranges: dict[str, tuple[int, int]] = {}
        self._load_metrics_cache()

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    def _patient_ids(self) -> list[str]:
        return sorted(self._index.keys())

    def _get_hour_range(self, patient_id: str) -> tuple[int, int]:
        """Return (min_hour, max_hour) for a patient, reading parquet if needed."""
        if patient_id in self._hour_ranges:
            return self._hour_ranges[patient_id]
        path = self.directory / f"{patient_id}.parquet"
        if not path.exists():
            self._hour_ranges[patient_id] = (0, 0)
            return (0, 0)
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        names = set(pf.schema.names)
        if "hour" not in names:
            self._hour_ranges[patient_id] = (0, 0)
            return (0, 0)
        df = pf.read(columns=["hour"]).to_pandas()
        hours = [int(h) for h in df["hour"].tolist() if h >= 0]
        hr = (min(hours), max(hours)) if hours else (0, 0)
        self._hour_ranges[patient_id] = hr
        # Also cache n_rows in the index.
        self._index[patient_id]["hour_min"] = hr[0]
        self._index[patient_id]["hour_max"] = hr[1]
        self._index[patient_id]["n_records"] = len(df)
        return hr

    def _compute_patient_metrics(self) -> None:
        """Compute data-quality metrics for every patient from source parquet.

        Reads only the hour and clinical columns (not the full row) so memory
        usage stays bounded. Results are cached in ``self._index`` so that
        ``list_patients()`` can return them without re-reading parquet files.

        Metric definitions:
          * total_hours   - number of hourly records (row count)
          * recorded_hours- rows with at least one non-null clinical observation
          * missing_hours - total_hours - recorded_hours
          * observation_count - non-null clinical measurements across all
                                supported clinical variables (ICULOS excluded)
        """
        observation_variables = [
            v for v in CLINICAL_VARIABLES if v != "ICULOS"
        ]
        read_columns = ["hour"] + observation_variables

        for pid in self._patient_ids():
            entry = self._index[pid]
            path = self.directory / f"{pid}.parquet"
            if not path.exists():
                entry.update(
                    total_hours=0,
                    recorded_hours=0,
                    missing_hours=0,
                    observation_count=0,
                    n_records=0,
                    hour_min=0,
                    hour_max=0,
                )
                continue
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            schema_names = set(pf.schema.names)
            available = [c for c in read_columns if c in schema_names]
            if "hour" not in schema_names or not available:
                entry.update(
                    total_hours=0,
                    recorded_hours=0,
                    missing_hours=0,
                    observation_count=0,
                    n_records=0,
                    hour_min=0,
                    hour_max=0,
                )
                continue
            df = pf.read(columns=available).to_pandas()
            total_hours = int(len(df))
            hours = [int(h) for h in df["hour"].tolist() if h >= 0]
            if hours:
                hour_min = min(hours)
                hour_max = max(hours)
            else:
                hour_min = 0
                hour_max = 0
            obs_cols = [c for c in observation_variables if c in df.columns]
            if obs_cols:
                obs_frame = df[obs_cols]
                rows_with_obs = int(obs_frame.notna().any(axis=1).sum())
                observation_count = int(obs_frame.notna().sum().sum())
            else:
                rows_with_obs = 0
                observation_count = 0
            entry.update(
                total_hours=total_hours,
                recorded_hours=rows_with_obs,
                missing_hours=max(0, total_hours - rows_with_obs),
                observation_count=observation_count,
                n_records=total_hours,
                hour_min=hour_min,
                hour_max=hour_max,
            )

    def _load_metrics_cache(self) -> None:
        """Load pre-computed data-quality metrics from a JSON cache file.

        The cache is generated by ``compute_patient_metrics.py`` which reads
        every patient parquet once and writes per-patient totals. Loading
        from cache keeps backend startup fast while still serving accurate,
        real-data metrics through the API.
        """
        cache_path = self.directory.parent / "reports" / "patient_metrics_cache.json"
        if not cache_path.exists():
            return
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            loaded = 0
            for pid, metrics in cache.items():
                if pid in self._index:
                    self._index[pid].update(metrics)
                    loaded += 1
            print(
                f"Loaded data-quality metrics cache: {loaded}/{len(self._index)} patients"
            )
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            # If the cache is corrupt or unreadable, metrics remain at their
            # defaults (zeros) and individual patients can still be computed
            # on demand via _load_full_patient.
            pass

    def _load_full_patient(self, patient_id: str) -> ProcessedPatient:
        """Load and cache a full ProcessedPatient from its parquet file."""
        if patient_id in self._patients:
            return self._patients[patient_id]
        entry = self._index.get(patient_id)
        if entry is None:
            raise PatientNotFoundError(f"Patient not found: {patient_id}")
        path = self.directory / f"{patient_id}.parquet"
        if not path.exists():
            raise PatientNotFoundError(f"Patient not found: {patient_id}")
        df = pd.read_parquet(path)
        clinical = {}
        for var in CLINICAL_VARIABLES:
            if var in df.columns:
                clinical[var] = [
                    float(x) if x is not None else float("nan")
                    for x in df[var].tolist()
                ]
            else:
                clinical[var] = [float("nan")] * len(df)
        hours = [int(h) for h in df["hour"].tolist()]
        sepsis_labels = [int(x) for x in df["SepsisLabel"].tolist()] if "SepsisLabel" in df.columns else [0] * len(df)
        patient = ProcessedPatient(
            patient_id=str(df["patient_id"].iloc[0]),
            hospital_id=None,
            hour=hours,
            clinical=clinical,
            sepsis_label=sepsis_labels,
            source_file=str(df["source_file"].iloc[0]) if "source_file" in df.columns else entry.get("source_file"),
            source_dataset=str(df["source_dataset"].iloc[0]) if "source_dataset" in df.columns else entry.get("source_dataset"),
            n_rows=len(df),
        )
        self._patients[patient_id] = patient
        return patient

    # ------------------------------------------------------------------
    # PatientRepository protocol
    # ------------------------------------------------------------------
    def list_patients(self) -> list[PatientSummary]:
        summaries: list[PatientSummary] = []
        for pid in self._patient_ids():
            entry = self._index[pid]
            total_hours = int(entry.get("total_hours", 0))
            recorded_hours = int(entry.get("recorded_hours", 0))
            summaries.append(
                PatientSummary(
                    patient_id=pid,
                    n_hours=int(entry.get("n_records", 0)),
                    total_hours=total_hours,
                    recorded_hours=recorded_hours,
                    missing_hours=int(entry.get("missing_hours", 0)),
                    observation_count=int(entry.get("observation_count", 0)),
                    data_source="processed_parquet",
                    source_dataset=entry["source_dataset"],
                    is_synthetic=False,
                    has_retrospective_label=entry["has_sepsis"],
                )
            )
        return summaries

    def get_patient(self, patient_id: str) -> ProcessedPatient:
        return self._load_full_patient(patient_id)

    def patient_exists(self, patient_id: str) -> bool:
        return patient_id in self._index

    def training_patients(self) -> list[ProcessedPatient]:
        return []


def summarize_patient(patient: ProcessedPatient, *, is_synthetic: bool) -> PatientSummary:
    hours = [int(h) for h in patient.hour if int(h) >= 0]
    if not hours:
        return PatientSummary(
            patient_id=patient.patient_id,
            n_hours=0,
            total_hours=0,
            recorded_hours=0,
            missing_hours=0,
            observation_count=0,
            data_source="synthetic_demo" if is_synthetic else "processed_parquet",
            source_dataset=patient.source_dataset,
            is_synthetic=is_synthetic,
        )
    min_hour = min(hours)
    max_hour = max(hours)
    total = max_hour - min_hour + 1
    recorded = len(set(hours))
    observation_count = 0
    for variable in CLINICAL_VARIABLES:
        if variable == "ICULOS":
            # ICULOS is time-in-ICU metadata, not a clinical observation.
            continue
        values = patient.clinical.get(variable, [])
        observation_count += sum(
            1 for value in values if value is not None and _finite(value)
        )
    has_label = any(label in (0, 1) for label in patient.sepsis_label)
    return PatientSummary(
        patient_id=patient.patient_id,
        n_hours=patient.n_rows,
        total_hours=total,
        recorded_hours=recorded,
        missing_hours=max(0, total - recorded),
        observation_count=observation_count,
        data_source="synthetic_demo" if is_synthetic else "processed_parquet",
        source_dataset=patient.source_dataset,
        is_synthetic=is_synthetic,
        has_retrospective_label=has_label,
    )


def _build_demo_patient(profile: DemoProfile) -> ProcessedPatient:
    rng = np.random.default_rng(profile.seed)
    hours: list[int] = []
    clinical: dict[str, list[float]] = {variable: [] for variable in CLINICAL_VARIABLES}
    labels: list[int] = []

    age = float(int(rng.integers(38, 86)))
    gender = float(int(rng.integers(0, 2)))
    unit1 = float(int(rng.integers(0, 2)))
    unit2 = 1.0 - unit1
    hosp_adm = float(-int(rng.integers(1, 8)))
    skip = set(profile.skip_hours)

    for hour in range(1, profile.n_hours + 1):
        if hour in skip:
            continue
        severity = _severity_at_hour(hour, profile.onset_hour, profile.peak_severity)
        hours.append(hour)
        for variable in CLINICAL_VARIABLES:
            value = _demo_value(
                variable=variable,
                hour=hour,
                severity=severity,
                profile=profile,
                rng=rng,
                age=age,
                gender=gender,
                unit1=unit1,
                unit2=unit2,
                hosp_adm=hosp_adm,
            )
            clinical[variable].append(value)
        labels.append(_demo_label(hour, profile.onset_hour))

    return ProcessedPatient(
        patient_id=profile.patient_id,
        hospital_id="DEMO-HOSP",
        hour=hours,
        clinical=clinical,
        sepsis_label=labels,
        source_file=f"{profile.patient_id}.psv",
        source_dataset="synthetic_demo",
        n_rows=len(hours),
    )


def _severity_at_hour(hour: int, onset_hour: int | None, peak: float) -> float:
    if onset_hour is None:
        return 0.0
    if hour < onset_hour:
        # Mild sub-threshold drift in the 6 hours before the demonstration onset.
        lead = max(0, 6 - (onset_hour - hour)) / 6.0
        return peak * 0.10 * lead
    ramp = min(1.0, (hour - onset_hour + 1) / 12.0)
    return float(peak * ramp)


def _demo_label(hour: int, onset_hour: int | None) -> int:
    if onset_hour is None:
        return 0
    return int(hour >= max(1, onset_hour - 6))


def _demo_value(
    *,
    variable: str,
    hour: int,
    severity: float,
    profile: DemoProfile,
    rng: np.random.Generator,
    age: float,
    gender: float,
    unit1: float,
    unit2: float,
    hosp_adm: float,
) -> float:
    if variable == "Age":
        return age
    if variable == "Gender":
        return gender
    if variable == "Unit1":
        return unit1
    if variable == "Unit2":
        return unit2
    if variable == "HospAdmTime":
        return hosp_adm
    if variable == "ICULOS":
        return float(hour)

    if variable in _VITAL_BASE:
        mean, change, noise, _ = _VITAL_BASE[variable]
        if variable == "EtCO2" and rng.random() > 0.15:
            return math.nan
        if rng.random() < profile.missing_rate:
            return math.nan
        value = mean + change * severity + rng.normal(0, noise)
        return _clip_vital(variable, value)

    if variable in _LAB_BASE:
        mean, change, noise, _ = _LAB_BASE[variable]
        measured = hour % profile.lab_interval == 0
        if variable in _RARELY_MEASURED and rng.random() > 0.20:
            measured = False
        if not measured or rng.random() < profile.missing_rate:
            return math.nan
        value = mean + change * severity + rng.normal(0, noise)
        return _clip_lab(variable, value)

    return math.nan


def _clip_vital(variable: str, value: float) -> float:
    bounds = {
        "HR": (25.0, 220.0),
        "O2Sat": (50.0, 100.0),
        "Temp": (30.0, 43.0),
        "SBP": (50.0, 250.0),
        "MAP": (30.0, 180.0),
        "DBP": (20.0, 140.0),
        "Resp": (4.0, 60.0),
        "EtCO2": (10.0, 80.0),
    }
    low, high = bounds.get(variable, (float("-inf"), float("inf")))
    return float(min(high, max(low, value)))


def _clip_lab(variable: str, value: float) -> float:
    nonnegative = {
        "Lactate",
        "WBC",
        "Creatinine",
        "BUN",
        "Platelets",
        "Hgb",
        "Glucose",
        "Potassium",
        "HCO3",
        "FiO2",
        "PaCO2",
        "SaO2",
        "AST",
        "Alkalinephos",
        "Calcium",
        "Chloride",
        "Bilirubin_direct",
        "Bilirubin_total",
        "Magnesium",
        "Phosphate",
        "TroponinI",
        "Hct",
        "PTT",
        "Fibrinogen",
    }
    if variable in nonnegative:
        value = max(0.0, value)
    if variable == "pH":
        return float(min(7.8, max(6.5, value)))
    return float(value)


def _finite(value: float | None) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "DirectoryPatientRepository",
    "PatientNotFoundError",
    "PatientRepository",
    "REPLAY_PROFILES",
    "SyntheticDemoPatientRepository",
    "summarize_patient",
]
