"""Centralized configuration for P1 clinical baseline scoring.

All clinical thresholds, variable mappings, and scoring parameters live here.
Serializable to/from JSON for reproducibility. Defaults are documented from
authoritative sources and are not adjusted from memory.

Threshold sources:
- SIRS: Bone RC, et al. "Definitions for sepsis and organ failure and
  guidelines for the use of innovative therapies in sepsis." Chest.
  1992;101(6):1644-1655.
- qSOFA: Singer M, et al. "The Third International Consensus Definitions for
  Sepsis and Septic Shock (Sepsis-3)." JAMA. 2016;315(8):801-810.
- NEWS2: Royal College of Physicians. National Early Warning Score (NEWS) 2.
  https://www.rcplondon.ac.uk/guidelines-assets/16-02-news2-standardising-assessment-acute-illness-severity-nhs
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# SIRS thresholds (Bone et al., Chest 1992)
# ---------------------------------------------------------------------------
SIRS_TEMP_HIGH_C: float = 38.3   # Temperature > 38.3 C
SIRS_TEMP_LOW_C: float = 36.0    # Temperature < 36.0 C
SIRS_HR_HIGH_BPM: float = 90.0   # Heart rate > 90 bpm
SIRS_RESP_HIGH_BPM: float = 20.0 # Respiration > 20 breaths/min
SIRS_WBC_HIGH: float = 12.0      # WBC > 12 (10^3/uL)
SIRS_WBC_LOW: float = 4.0        # WBC < 4 (10^3/uL)
SIRS_POSITIVE_THRESHOLD: int = 2 # >= 2 criteria = SIRS positive
SIRS_COMPONENTS_REQUIRED: int = 4

# ---------------------------------------------------------------------------
# qSOFA thresholds (Sepsis-3, Singer et al., JAMA 2016)
# ---------------------------------------------------------------------------
QSOFA_RESP_HIGH_BPM: float = 22.0   # Respiratory rate >= 22 breaths/min
QSOFA_SBP_LOW_MMHG: float = 100.0   # Systolic BP <= 100 mmHg
# Altered mental status: GCS < 15 (not available in Challenge dataset)
QSOFA_POSITIVE_THRESHOLD: int = 2   # >= 2 criteria = qSOFA positive
QSOFA_COMPONENTS_REQUIRED: int = 3

# ---------------------------------------------------------------------------
# NEWS2 thresholds (Royal College of Physicians, NEWS2)
# Parameter scores: 0, 1, 2, or 3 per parameter
# ---------------------------------------------------------------------------

# Respiratory rate (breaths/min): <=8->3, 9-11->1, 12-20->0, 21-24->1, >=25->3
NEWS2_RR_SCORES = [
    (None, 8, 3),      # <= 8
    (9, 11, 1),        # 9-11
    (12, 20, 0),       # 12-20 (normal)
    (21, 24, 1),       # 21-24
    (25, None, 3),     # >= 25
]

# SpO2 (%): <=91->3, 92-93->2, 94-95->1, 96-97->0, >=98->0
NEWS2_SPO2_SCORES = [
    (None, 91, 3),     # <= 91
    (92, 93, 2),       # 92-93
    (94, 95, 1),       # 94-95
    (96, 97, 0),       # 96-97
    (98, None, 0),     # >= 98
]

# Supplemental oxygen: No = 0, Yes = +2 (NOT available in dataset)
NEWS2_O2_SUPPLEMENTAL_SCORE: int = 2

# Temperature (C): <=35.0->3, 35.1-36.0->1, 36.1-38.0->0, 38.1-39.0->0, >=39.1->2
NEWS2_TEMP_SCORES = [
    (None, 35.0, 3),      # <= 35.0
    (35.1, 36.0, 1),      # 35.1-36.0
    (36.1, 38.0, 0),      # 36.1-38.0 (normal)
    (38.1, 39.0, 0),      # 38.1-39.0
    (39.1, None, 2),      # >= 39.1
]

# Systolic BP (mmHg): <=90->3, 91-100->2, 101-119->1, 120-219->0, >=220->3
NEWS2_SBP_SCORES = [
    (None, 90, 3),       # <= 90
    (91, 100, 2),        # 91-100
    (101, 119, 1),       # 101-119
    (120, 219, 0),       # 120-219 (normal)
    (220, None, 3),      # >= 220
]

# Heart rate (bpm): <=40->3, 41-50->1, 51-90->0, 91-110->0, 111-130->1, 131-140->2, >=141->3
NEWS2_HR_SCORES = [
    (None, 40, 3),       # <= 40
    (41, 50, 1),         # 41-50
    (51, 90, 0),         # 51-90 (normal)
    (91, 110, 0),        # 91-110
    (111, 130, 1),       # 111-130
    (131, 140, 2),       # 131-140
    (141, None, 3),      # >= 141
]

# Level of consciousness: A (alert) = 0, C/V/P/U = 3 (NOT available in dataset)
NEWS2_CONSCIOUSNESS_ALERT_SCORE: int = 0
NEWS2_CONSCIOUSNESS_CHANGED_SCORE: int = 3

# NEWS2 clinical concern / positive threshold
NEWS2_POSITIVE_THRESHOLD: int = 5   # >= 5 = clinical concern
NEWS2_COMPONENTS_REQUIRED: int = 7

# ---------------------------------------------------------------------------
# Variable mapping: clinical component -> dataset variable
# ---------------------------------------------------------------------------

# Dataset clinical variables (from Part 4 config)
FEATURE_VARIABLES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]

# Variable units (from official documentation)
VARIABLE_UNITS = {
    "HR": "beats_per_minute",
    "O2Sat": "percent",
    "Temp": "deg_c",
    "SBP": "mm_hg",
    "MAP": "mm_hg",
    "DBP": "mm_hg",
    "Resp": "breaths_per_minute",
    "EtCO2": "mm_hg",
    "PaCO2": "mm_hg",
    "WBC": "count_times_10_3_per_ul",
}

# Missingness policy
MISSING_NEVER_NORMAL: bool = True  # missing values are never treated as normal


@dataclass
class ClinicalScoreConfig:
    """Serializable clinical baseline scoring configuration."""

    # SIRS
    sirs_temp_high_c: float = SIRS_TEMP_HIGH_C
    sirs_temp_low_c: float = SIRS_TEMP_LOW_C
    sirs_hr_high_bpm: float = SIRS_HR_HIGH_BPM
    sirs_resp_high_bpm: float = SIRS_RESP_HIGH_BPM
    sirs_wbc_high: float = SIRS_WBC_HIGH
    sirs_wbc_low: float = SIRS_WBC_LOW
    sirs_positive_threshold: int = SIRS_POSITIVE_THRESHOLD
    sirs_components_required: int = SIRS_COMPONENTS_REQUIRED

    # qSOFA
    qsofa_resp_high_bpm: float = QSOFA_RESP_HIGH_BPM
    qsofa_sbp_low_mmhg: float = QSOFA_SBP_LOW_MMHG
    qsofa_positive_threshold: int = QSOFA_POSITIVE_THRESHOLD
    qsofa_components_required: int = QSOFA_COMPONENTS_REQUIRED

    # NEWS2
    news2_positive_threshold: int = NEWS2_POSITIVE_THRESHOLD
    news2_components_required: int = NEWS2_COMPONENTS_REQUIRED

    # Variable mappings
    sirs_variable_map: dict = field(default_factory=lambda: {
        "temperature": "Temp",
        "heart_rate": "HR",
        "respiration": "Resp",
        "wbc": "WBC",
    })
    qsofa_variable_map: dict = field(default_factory=lambda: {
        "respiration": "Resp",
        "systolic_bp": "SBP",
        "altered_mental_status": None,  # not available in dataset
    })
    news2_variable_map: dict = field(default_factory=lambda: {
        "respiration": "Resp",
        "spo2": "O2Sat",
        "supplemental_oxygen": None,  # not available in dataset
        "temperature": "Temp",
        "systolic_bp": "SBP",
        "heart_rate": "HR",
        "consciousness": None,  # not available in dataset
    })

    # Flags
    missing_never_normal: bool = MISSING_NEVER_NORMAL
    compute_sirs: bool = True
    compute_qsofa: bool = True
    compute_news2_style: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ClinicalScoreConfig:
        return cls(**data)

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> ClinicalScoreConfig:
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def variable_units(self, variable: str) -> str | None:
        return VARIABLE_UNITS.get(variable)
