"""Centralized configuration for the P1 temporal feature-engineering pipeline.

All feature parameters live here so they are not scattered across modules.
The configuration is serializable (to/from dict and JSON) and saved with
each experiment for reproducibility.

Defaults are experimental engineering parameters requiring clinical
validation. They are not presented as medically established values.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Rolling window sizes in hours. The current hour is included in each
# window (trailing, causal window ending at the current observation).
ROLLING_WINDOW_HOURS: list[int] = [6, 12, 24]

# Rolling aggregation functions.
ROLLING_AGGREGATIONS: list[str] = ["mean", "min", "max", "std", "latest"]

# Minimum number of valid (non-missing) observations required to compute
# a rolling statistic. Below this, the statistic is NaN.
ROLLING_MIN_OBSERVATIONS: int = 1

# Trend/slope windows in hours (trailing, causal).
TREND_WINDOW_HOURS: list[int] = [6, 12, 24]

# Minimum observations required to compute a trend/slope.
TREND_MIN_OBSERVATIONS: int = 2

# Maximum forward-fill validity window in hours. A measurement is carried
# forward only if the gap since the last valid observation is <= this
# value. Configurable by variable or variable group.
FORWARD_FILL_MAX_WINDOW_HOURS: dict[str, int] = {
    "default": 4,
    "vital_signs": 4,
    "laboratory": 6,
}

# Variables that participate in feature engineering (subset of the 40
# clinical variables that are meaningfully numeric and continuous).
# Age, Gender, Unit1, Unit2, HospAdmTime are treated as static/demographic
# and are not rolled or trended.
FEATURE_VARIABLES: list[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets",
]

# Static / demographic variables carried through unchanged.
STATIC_VARIABLES: list[str] = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

# Vital-sign subset (for clinical features like shock index).
VITAL_SIGN_VARIABLES: list[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
]

# Laboratory subset.
LABORATORY_VARIABLES: list[str] = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets",
]

# Columns that must never be used as model input features (target only).
TARGET_COLUMNS: list[str] = ["SepsisLabel"]

# Identifier / metadata columns preserved but not used as features.
IDENTIFIER_COLUMNS: list[str] = [
    "patient_id", "hospital_id", "hour", "source_file", "source_dataset",
]

# SIRS thresholds (documented, not invented).
# Temperature: >38.3 C or <36.0 C
# Heart rate: >90 bpm
# Respiration: >20 breaths/min
# (WBC and PaCO2 are lab components; included when available)
SIRS_TEMP_HIGH: float = 38.3
SIRS_TEMP_LOW: float = 36.0
SIRS_HR_HIGH: float = 90.0
SIRS_RESP_HIGH: float = 20.0
SIRS_WBC_HIGH: float = 12.0  # 12 * 10^3/uL
SIRS_WBC_LOW: float = 4.0    # 4 * 10^3/uL


@dataclass
class FeatureConfig:
    """Serializable feature-engineering configuration."""

    rolling_window_hours: list[int] = field(default_factory=lambda: list(ROLLING_WINDOW_HOURS))
    rolling_aggregations: list[str] = field(default_factory=lambda: list(ROLLING_AGGREGATIONS))
    rolling_min_observations: int = ROLLING_MIN_OBSERVATIONS
    trend_window_hours: list[int] = field(default_factory=lambda: list(TREND_WINDOW_HOURS))
    trend_min_observations: int = TREND_MIN_OBSERVATIONS
    forward_fill_max_window_hours: dict[str, int] = field(
        default_factory=lambda: dict(FORWARD_FILL_MAX_WINDOW_HOURS)
    )
    feature_variables: list[str] = field(default_factory=lambda: list(FEATURE_VARIABLES))
    static_variables: list[str] = field(default_factory=lambda: list(STATIC_VARIABLES))
    vital_sign_variables: list[str] = field(default_factory=lambda: list(VITAL_SIGN_VARIABLES))
    laboratory_variables: list[str] = field(default_factory=lambda: list(LABORATORY_VARIABLES))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    identifier_columns: list[str] = field(default_factory=lambda: list(IDENTIFIER_COLUMNS))
    compute_missingness_features: bool = True
    compute_forward_fill: bool = True
    compute_rolling_features: bool = True
    compute_trend_features: bool = True
    compute_clinical_features: bool = True
    shocks_index_enabled: bool = True
    sirs_count_enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> FeatureConfig:
        return cls(**data)

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> FeatureConfig:
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def forward_fill_window_for(self, variable: str) -> int:
        """Return the forward-fill max window (hours) for a variable."""
        # Check variable group first, then default.
        if variable in self.vital_sign_variables:
            return self.forward_fill_max_window_hours.get(
                "vital_signs",
                self.forward_fill_max_window_hours.get("default", 4),
            )
        if variable in self.laboratory_variables:
            return self.forward_fill_max_window_hours.get(
                "laboratory",
                self.forward_fill_max_window_hours.get("default", 4),
            )
        return self.forward_fill_max_window_hours.get("default", 4)
