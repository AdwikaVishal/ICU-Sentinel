"""Human-readable clinical metadata used by replay and explanations.

Units are taken from P1's documented feature schema. Display names are
neutral descriptions of the PhysioNet variables; they do not introduce
clinical thresholds or new medical claims.
"""

from __future__ import annotations

from functools import lru_cache

from ml.p1_clinical.features.feature_config import (
    LABORATORY_VARIABLES,
    STATIC_VARIABLES,
    VITAL_SIGN_VARIABLES,
)
from ml.p1_clinical.features.feature_schema import FeatureSchema, build_default_schema

VITAL_VARIABLES = list(VITAL_SIGN_VARIABLES)
LAB_VARIABLES = list(LABORATORY_VARIABLES)
STATIC_VARIABLES = list(STATIC_VARIABLES)

DISPLAY_NAMES: dict[str, str] = {
    "HR": "Heart rate",
    "O2Sat": "Oxygen saturation (SpO2)",
    "Temp": "Temperature",
    "SBP": "Systolic blood pressure",
    "MAP": "Mean arterial pressure",
    "DBP": "Diastolic blood pressure",
    "Resp": "Respiratory rate",
    "EtCO2": "End-tidal CO2",
    "BaseExcess": "Base excess",
    "HCO3": "Bicarbonate",
    "FiO2": "Inspired oxygen fraction",
    "pH": "Blood pH",
    "PaCO2": "Partial pressure of carbon dioxide",
    "SaO2": "Arterial oxygen saturation",
    "AST": "Aspartate aminotransferase",
    "BUN": "Blood urea nitrogen",
    "Alkalinephos": "Alkaline phosphatase",
    "Calcium": "Calcium",
    "Chloride": "Chloride",
    "Creatinine": "Creatinine",
    "Bilirubin_direct": "Direct bilirubin",
    "Glucose": "Glucose",
    "Lactate": "Lactate",
    "Magnesium": "Magnesium",
    "Phosphate": "Phosphate",
    "Potassium": "Potassium",
    "Bilirubin_total": "Total bilirubin",
    "TroponinI": "Troponin I",
    "Hct": "Hematocrit",
    "Hgb": "Hemoglobin",
    "PTT": "Partial thromboplastin time",
    "WBC": "White blood cell count",
    "Fibrinogen": "Fibrinogen",
    "Platelets": "Platelet count",
    "Age": "Age",
    "Gender": "Gender code",
    "Unit1": "ICU unit 1 indicator",
    "Unit2": "ICU unit 2 indicator",
    "HospAdmTime": "Hospital admission time offset",
    "ICULOS": "ICU length of stay",
}


@lru_cache(maxsize=1)
def _schema() -> FeatureSchema:
    return build_default_schema()


def display_name(variable: str) -> str:
    return DISPLAY_NAMES.get(variable, variable)


def unit_for(variable: str) -> str | None:
    entry = _schema().get_feature(variable)
    return entry.units if entry is not None else None


def category_for(variable: str) -> str:
    if variable in VITAL_VARIABLES:
        return "vital_signs"
    if variable in LAB_VARIABLES:
        return "laboratory"
    if variable in STATIC_VARIABLES or variable in {"Age", "Gender", "HospAdmTime", "ICULOS"}:
        return "static_demographic"
    return "other_clinical"
