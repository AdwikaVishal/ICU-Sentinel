# Common fixtures for P1 tests.
# Generates small synthetic patient data for unit tests without touching real data.

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticPatient:
    patient_id: str
    hours: int
    sepsis_hour: int | None = None


def make_synthetic_patients(n_patients: int = 5, hours_per_patient: int = 24,
                            seed: int = 42) -> pd.DataFrame:
    """Build a synthetic observations dataframe for unit tests.

    Columns: patient_id, hour, parameter, value, unit.
    Deterministic given seed; no real patient data is used.
    """
    rng = np.random.default_rng(seed)
    rows = []
    params = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp", "Glucose"]
    for p in range(n_patients):
        pid = f"P{p:03d}"
        for h in range(hours_per_patient):
            for param in params:
                base = {"HR": 80, "O2Sat": 97, "Temp": 36.8, "SBP": 120,
                        "DBP": 70, "MAP": 85, "Resp": 18, "Glucose": 90}[param]
                val = base + rng.normal(0, base * 0.1)
                # introduce missingness
                if rng.random() < 0.15:
                    val = np.nan
                rows.append({
                    "patient_id": pid,
                    "hour": h,
                    "parameter": param,
                    "value": float(val) if not np.isnan(val) else None,
                    "unit": None,
                })
    return pd.DataFrame(rows)


def make_synthetic_metadata(n_patients: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        pid = f"P{p:03d}"
        rows.append({
            "patient_id": pid,
            "hospital_id": f"H{p % 3}",
            "icu_id": f"ICU{p % 2}",
            "gender": rng.choice(["M", "F"]),
            "age": int(rng.integers(18, 90)),
            "height": float(rng.normal(170, 10)),
            "weight": float(rng.normal(75, 15)),
            "died_in_hospital": int(rng.integers(0, 2)),
            "sepsis_label": int(rng.integers(0, 2)),
        })
    return pd.DataFrame(rows)
