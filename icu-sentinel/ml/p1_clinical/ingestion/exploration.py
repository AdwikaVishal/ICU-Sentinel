"""Dataset exploration for the PhysioNet/CinC Challenge 2019.

Computes dataset-level summaries: patient counts, record counts,
missingness percentages, target-label distribution, temporal coverage,
and duplicate/invalid-record analysis.

All functions operate on a directory of pipe-delimited .psv patient
files. When no real dataset is present, functions degrade gracefully and
mark findings as unavailable (never fabricate statistics).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset_loader import (
    REQUIRED_COLUMNS,
    PhysioNetChallenge2019Loader,
    build_synthetic_patient_file,
    write_synthetic_psv,
)
from .validation import summarize_validation, validate_directory

logger = logging.getLogger("icu_sentinel.p1_clinical.ingestion.exploration")

# Column groupings per the official documentation (Table 1).
VITAL_SIGN_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
]
LABORATORY_COLUMNS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
DEMOGRAPHIC_COLUMNS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
OUTCOME_COLUMN = "SepsisLabel"


@dataclass
class DatasetOverview:
    """High-level overview of a Challenge dataset directory."""

    root: str
    n_patients: int = 0
    n_records: int = 0
    n_files_valid: int = 0
    n_files_malformed: int = 0
    total_size_bytes: int = 0
    n_sepsis_patients: int = 0
    n_non_sepsis_patients: int = 0
    n_sepsis_positive_hours: int = 0
    n_sepsis_negative_hours: int = 0
    min_iculos: float | None = None
    max_iculos: float | None = None
    mean_records_per_patient: float | None = None
    median_records_per_patient: float | None = None
    data_available: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "n_patients": self.n_patients,
            "n_records": self.n_records,
            "n_files_valid": self.n_files_valid,
            "n_files_malformed": self.n_files_malformed,
            "total_size_bytes": self.total_size_bytes,
            "n_sepsis_patients": self.n_sepsis_patients,
            "n_non_sepsis_patients": self.n_non_sepsis_patients,
            "n_sepsis_positive_hours": self.n_sepsis_positive_hours,
            "n_sepsis_negative_hours": self.n_sepsis_negative_hours,
            "min_iculos": self.min_iculos,
            "max_iculos": self.max_iculos,
            "mean_records_per_patient": self.mean_records_per_patient,
            "median_records_per_patient": self.median_records_per_patient,
            "data_available": self.data_available,
            "notes": list(self.notes),
        }


@dataclass
class MissingnessReportRow:
    column: str
    n_missing: int
    n_total: int
    missing_percent: float
    group: str  # vital_sign / laboratory / demographic / outcome


@dataclass
class LabelDistribution:
    n_patients_total: int = 0
    n_sepsis_patients: int = 0
    n_non_sepsis_patients: int = 0
    n_sepsis_positive_hours: int = 0
    n_sepsis_negative_hours: int = 0
    sepsis_label_positive_rate_hours: float | None = None
    sepsis_label_positive_rate_patients: float | None = None
    data_available: bool = False

    def to_dict(self) -> dict:
        return {
            "n_patients_total": self.n_patients_total,
            "n_sepsis_patients": self.n_sepsis_patients,
            "n_non_sepsis_patients": self.n_non_sepsis_patients,
            "n_sepsis_positive_hours": self.n_sepsis_positive_hours,
            "n_sepsis_negative_hours": self.n_sepsis_negative_hours,
            "sepsis_label_positive_rate_hours": self.sepsis_label_positive_rate_hours,
            "sepsis_label_positive_rate_patients": self.sepsis_label_positive_rate_patients,
            "data_available": self.data_available,
        }


class DatasetExplorer:
    """Explores a Challenge dataset directory and produces reports."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.loader = PhysioNetChallenge2019Loader(self.root)
        self.logger = logging.getLogger(
            "icu_sentinel.p1_clinical.ingestion.exploration.DatasetExplorer"
        )

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------
    def compute_overview(self) -> DatasetOverview:
        """Compute a high-level overview of the dataset directory."""
        catalog = self.loader.catalog()
        overview = DatasetOverview(root=str(self.root))
        overview.n_patients = catalog.n_patients
        overview.total_size_bytes = catalog.total_bytes
        overview.data_available = catalog.n_patients > 0

        if not overview.data_available:
            overview.notes.append(
                "No .psv patient files found; dataset not available locally."
            )
            return overview

        n_sepsis_patients = 0
        n_non_sepsis_patients = 0
        n_sepsis_pos_hours = 0
        n_sepsis_neg_hours = 0
        total_records = 0
        iculos_values: list[float] = []
        records_per_patient: list[int] = []

        for result in self.loader.iter_patients():
            total_records += result.n_rows
            records_per_patient.append(result.n_rows)
            if result.dataframe is not None and OUTCOME_COLUMN in result.dataframe.columns:
                labels = result.dataframe[OUTCOME_COLUMN].dropna()
                n_pos = int((labels == 1).sum())
                n_neg = int((labels == 0).sum())
                if n_pos > 0:
                    n_sepsis_patients += 1
                else:
                    n_non_sepsis_patients += 1
                n_sepsis_pos_hours += n_pos
                n_sepsis_neg_hours += n_neg
            if "ICULOS" in result.dataframe.columns:
                iculos_values.extend(
                    result.dataframe["ICULOS"].dropna().astype(float).tolist()
                )

        overview.n_records = total_records
        overview.n_files_valid = catalog.n_patients
        overview.n_sepsis_patients = n_sepsis_patients
        overview.n_non_sepsis_patients = n_non_sepsis_patients
        overview.n_sepsis_positive_hours = n_sepsis_pos_hours
        overview.n_sepsis_negative_hours = n_sepsis_neg_hours
        if iculos_values:
            overview.min_iculos = float(min(iculos_values))
            overview.max_iculos = float(max(iculos_values))
        if records_per_patient:
            overview.mean_records_per_patient = float(sum(records_per_patient) / len(records_per_patient))
            srt = sorted(records_per_patient)
            mid = len(srt) // 2
            overview.median_records_per_patient = (
                float(srt[mid]) if len(srt) % 2 == 1
                else float((srt[mid - 1] + srt[mid]) / 2)
            )
        overview.notes.append(
            f"Overview computed from actual dataset files in {self.root}."
        )
        return overview

    # ------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------
    def compute_missingness(self) -> list[MissingnessReportRow]:
        """Compute per-column missingness across all patient files."""
        rows: list[MissingnessReportRow] = []
        col_groups = {
            "vital_sign": VITAL_SIGN_COLUMNS,
            "laboratory": LABORATORY_COLUMNS,
            "demographic": DEMOGRAPHIC_COLUMNS,
            "outcome": [OUTCOME_COLUMN],
        }
        accum: dict[str, list[float]] = {c: [] for c in REQUIRED_COLUMNS}
        n_rows_total = 0

        for result in self.loader.iter_patients():
            if result.dataframe is None:
                continue
            n_rows_total += len(result.dataframe)
            for col in REQUIRED_COLUMNS:
                if col in result.dataframe.columns:
                    missing = int(result.dataframe[col].isna().sum())
                    accum[col].append(float(missing))

        if n_rows_total == 0:
            return rows

        for group, cols in col_groups.items():
            for col in cols:
                if col not in accum:
                    continue
                total_missing = int(sum(accum[col]))
                n_total_for_col = n_rows_total
                pct = (total_missing / n_total_for_col * 100.0) if n_total_for_col else 0.0
                rows.append(MissingnessReportRow(
                    column=col,
                    n_missing=total_missing,
                    n_total=n_total_for_col,
                    missing_percent=round(pct, 4),
                    group=group,
                ))
        return rows

    def missingness_dataframe(self) -> pd.DataFrame:
        rows = self.compute_missingness()
        if not rows:
            return pd.DataFrame(
                columns=["column", "n_missing", "n_total", "missing_percent", "group"]
            )
        return pd.DataFrame(
            [
                {
                    "column": r.column,
                    "n_missing": r.n_missing,
                    "n_total": r.n_total,
                    "missing_percent": r.missing_percent,
                    "group": r.group,
                }
                for r in rows
            ]
        )

    # ------------------------------------------------------------------
    # Label distribution
    # ------------------------------------------------------------------
    def compute_label_distribution(self) -> LabelDistribution:
        dist = LabelDistribution()
        n_sepsis_patients = 0
        n_non_sepsis_patients = 0
        n_pos_hours = 0
        n_neg_hours = 0
        n_patients = 0

        for result in self.loader.iter_patients():
            if result.dataframe is None or OUTCOME_COLUMN not in result.dataframe.columns:
                continue
            n_patients += 1
            labels = result.dataframe[OUTCOME_COLUMN].dropna()
            n_pos = int((labels == 1).sum())
            n_neg = int((labels == 0).sum())
            n_pos_hours += n_pos
            n_neg_hours += n_neg
            if n_pos > 0:
                n_sepsis_patients += 1
            else:
                n_non_sepsis_patients += 1

        dist.n_patients_total = n_patients
        dist.n_sepsis_patients = n_sepsis_patients
        dist.n_non_sepsis_patients = n_non_sepsis_patients
        dist.n_sepsis_positive_hours = n_pos_hours
        dist.n_sepsis_negative_hours = n_neg_hours
        dist.data_available = n_patients > 0
        if n_patients > 0:
            dist.sepsis_label_positive_rate_patients = round(
                n_sepsis_patients / n_patients, 4
            )
        total_hours = n_pos_hours + n_neg_hours
        if total_hours > 0:
            dist.sepsis_label_positive_rate_hours = round(
                n_pos_hours / total_hours, 4
            )
        return dist

    # ------------------------------------------------------------------
    # Patient-level record counts
    # ------------------------------------------------------------------
    def patient_record_counts(self) -> pd.DataFrame:
        """Return a DataFrame with patient_id and record count per patient."""
        rows: list[dict] = []
        for result in self.loader.iter_patients():
            rows.append(
                {
                    "patient_id": result.patient_id,
                    "n_records": result.n_rows,
                    "n_sepsis_positive_hours": result.n_sepsis_positive_hours,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=["patient_id", "n_records", "n_sepsis_positive_hours"]
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Duplicates / invalid records
    # ------------------------------------------------------------------
    def detect_issues(self) -> dict:
        """Run validation across the directory and summarize issues."""
        reports = validate_directory(self.root)
        summary = summarize_validation(reports)
        malformed_files = [
            {
                "patient_id": r.patient_id,
                "filepath": r.filepath,
                "issues": r.issues,
            }
            for r in reports if r.malformed
        ]
        return {
            "validation_summary": summary,
            "malformed_files": malformed_files,
            "data_available": len(reports) > 0,
        }

    # ------------------------------------------------------------------
    # All reports
    # ------------------------------------------------------------------
    def generate_all_reports(self, reports_dir: str | Path) -> dict:
        """Generate every exploration report and write to ``reports_dir``."""
        out = Path(reports_dir)
        out.mkdir(parents=True, exist_ok=True)

        overview = self.compute_overview()
        with open(out / "dataset_overview.json", "w") as f:
            json.dump(overview.to_dict(), f, indent=2)

        miss_df = self.missingness_dataframe()
        miss_df.to_csv(out / "missingness_report.csv", index=False)

        label_dist = self.compute_label_distribution()
        with open(out / "label_distribution.json", "w") as f:
            json.dump(label_dist.to_dict(), f, indent=2)

        counts_df = self.patient_record_counts()
        counts_df.to_csv(out / "patient_record_counts.csv", index=False)

        issues = self.detect_issues()
        with open(out / "validation_summary.json", "w") as f:
            json.dump(issues, f, indent=2)

        return {
            "dataset_overview": overview.to_dict(),
            "missingness": miss_df.to_dict(orient="records"),
            "label_distribution": label_dist.to_dict(),
            "n_patient_record_counts": len(counts_df),
            "validation_summary": issues,
        }


def generate_synthetic_exploration_outputs(
    reports_dir: str | Path,
    n_patients: int = 5,
    n_hours: int = 24,
    seed: int = 42,
) -> dict:
    """Generate exploration reports from synthetic data.

    Used only when the real dataset is unavailable. Outputs are clearly
    marked as synthetic so they are never mistaken for real statistics.
    """
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Create a temporary synthetic directory.
    synth_root = out / "_synthetic_input"
    synth_root.mkdir(parents=True, exist_ok=True)
    for i in range(n_patients):
        pid = f"p{i:06d}"
        sepsis_onset = None
        if n_hours > 12 and i % 2 == 1:
            sepsis_onset = int(np.random.default_rng(seed + i).integers(6, n_hours - 6))
        df = build_synthetic_patient_file(
            patient_id=pid, n_hours=n_hours, seed=seed + i,
            sepsis_onset_hour=sepsis_onset,
        )
        write_synthetic_psv(df, synth_root / f"{pid}.psv")

    explorer = DatasetExplorer(synth_root)
    results = explorer.generate_all_reports(out)

    # Mark everything as synthetic.
    for key in ("dataset_overview", "label_distribution"):
        if key in results and isinstance(results[key], dict):
            results[key]["data_available"] = False
            results[key]["notes"] = results[key].get("notes", []) + [
                ("SYNTHETIC DATA: generated for exploration-template purposes only. "
                "Not real PhysioNet/CinC Challenge 2019 statistics.")
            ]

    missingness_path = out / "missingness_report.csv"
    if missingness_path.exists():
        miss_df = pd.read_csv(missingness_path)
        miss_df["source"] = "synthetic"
        miss_df.to_csv(missingness_path, index=False)

    results["synthetic"] = True
    results["note"] = (
        "All statistics below are derived from synthetic fixture data, not the "
        "real PhysioNet/CinC Challenge 2019 dataset, because the real dataset "
        "requires PhysioNet credentialed access."
    )
    return results
