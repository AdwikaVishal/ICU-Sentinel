"""Shared output schema for the P1 preprocessing pipeline.

Defines the stable contract that P2 (boosting) and P3 (temporal models)
consume. The schema is deliberately minimal and uses only column names
that exist in the official PhysioNet/CinC Challenge 2019 data, so no
incompatible P1-only column names are introduced.

Output columns (all present, in order):
    patient_id       - original patient identifier (string, e.g. "p00101")
    hospital_id      - source dataset identity / hospital set (string,
                       e.g. "set_a" or "set_b") when available
    hour             - ICU hour since admission (int), preserved ordering
    <40 clinical variables> - original clinical variables (float, NaN ok)
    SepsisLabel      - original target label (int 0/1), unchanged
    source_file      - source .psv filename (string)
    source_dataset   - source dataset identity (string)

No P1-only derived or engineered columns are added here. Feature
engineering (rolling windows, slopes, etc.) is Part 4 responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The 40 clinical variables from the official Challenge schema (Table 1),
# in canonical order, excluding the outcome column.
CLINICAL_VARIABLES: list[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]

# The full output schema column order.
OUTPUT_COLUMNS: list[str] = [
    "patient_id",
    "hospital_id",
    "hour",
    *CLINICAL_VARIABLES,
    "SepsisLabel",
    "source_file",
    "source_dataset",
]

# Columns that must be present and non-null for a row to be considered
# minimally valid.
REQUIRED_OUTPUT_COLUMNS: list[str] = [
    "patient_id", "hospital_id", "hour", "SepsisLabel",
]

# Columns that are treated as categorical / identifier fields (not imputed
# as continuous numerics).
IDENTIFIER_COLUMNS: list[str] = [
    "patient_id", "hospital_id", "hour", "Gender", "Unit1", "Unit2",
    "SepsisLabel",
]

# Columns that are integer-valued in the source data.
INTEGER_COLUMNS: list[str] = [
    "Gender", "Unit1", "Unit2", "SepsisLabel",
]


@dataclass
class ProcessedPatient:
    """A single fully-processed patient record ready for P2/P3 consumption."""

    patient_id: str
    hospital_id: str | None
    hour: list[int]
    clinical: dict[str, list[float]]
    sepsis_label: list[int]
    source_file: str
    source_dataset: str
    n_rows: int

    def to_dataframe(self):
        """Convert to a pandas DataFrame matching OUTPUT_COLUMNS."""
        import pandas as pd

        data: dict = {
            "patient_id": [self.patient_id] * self.n_rows,
            "hospital_id": [self.hospital_id] * self.n_rows,
            "hour": list(self.hour),
        }
        for var in CLINICAL_VARIABLES:
            data[var] = list(self.clinical[var])
        data["SepsisLabel"] = list(self.sepsis_label)
        data["source_file"] = [self.source_file] * self.n_rows
        data["source_dataset"] = [self.source_dataset] * self.n_rows
        return pd.DataFrame(data, columns=OUTPUT_COLUMNS)


@dataclass
class ValidationIssue:
    """A single structured validation issue."""

    severity: str  # "error" or "warning"
    rule: str
    patient_id: str | None
    hour: int | None
    column: str | None
    message: str
    recovered: bool = False
    recovery_strategy: str | None = None


@dataclass
class ValidationReport:
    """Aggregated validation report for one or more patients."""

    patient_id: str | None
    n_issues: int = 0
    n_errors: int = 0
    n_warnings: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    passed: bool = False

    def add_issue(
        self,
        severity: str,
        rule: str,
        patient_id: str | None = None,
        hour: int | None = None,
        column: str | None = None,
        message: str = "",
        recovered: bool = False,
        recovery_strategy: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                severity=severity,
                rule=rule,
                patient_id=patient_id,
                hour=hour,
                column=column,
                message=message,
                recovered=recovered,
                recovery_strategy=recovery_strategy,
            )
        )
        self.n_issues += 1
        if severity == "error":
            self.n_errors += 1
        elif severity == "warning":
            self.n_warnings += 1
        self.passed = self.n_errors == 0

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "n_issues": self.n_issues,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "passed": self.passed,
            "issues": [
                {
                    "severity": i.severity,
                    "rule": i.rule,
                    "patient_id": i.patient_id,
                    "hour": i.hour,
                    "column": i.column,
                    "message": i.message,
                    "recovered": i.recovered,
                    "recovery_strategy": i.recovery_strategy,
                }
                for i in self.issues
            ],
        }
