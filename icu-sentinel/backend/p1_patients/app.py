"""Read-only P1 patient-data API.

The API serves only artefacts that already exist on disk. It never
fabricates patient fields, hospital identifiers, or observations, and it
never falls back to synthetic demo patients. When the required data
directory or feature file is not configured, endpoints return HTTP 503.

Configuration (environment variables):
    ICU_SENTINEL_PROCESSED_DIR  per-patient processed parquet directory
                                (default: data/processed)
    ICU_SENTINEL_FEATURES_FILE  combined feature parquet file
                                (default: data/features/combined_features.parquet)
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="ICU Sentinel P1 Patient API", version="1.0.0")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _processed_dir() -> Path:
    raw = os.getenv("ICU_SENTINEL_PROCESSED_DIR")
    return Path(raw).expanduser() if raw and raw.strip() else _REPO_ROOT / "data" / "processed"


def _features_file() -> Path:
    raw = os.getenv("ICU_SENTINEL_FEATURES_FILE")
    if raw and raw.strip():
        return Path(raw).expanduser()
    return _REPO_ROOT / "data" / "features" / "combined_features.parquet"


def _validate_patient_id(patient_id: str) -> str:
    if not _PATIENT_ID_RE.match(patient_id):
        raise HTTPException(status_code=400, detail="Invalid patient identifier")
    return patient_id


def _patient_file(patient_id: str) -> Path:
    _validate_patient_id(patient_id)
    base = _processed_dir().resolve()
    candidate = (base / f"{patient_id}.parquet").resolve()
    if base not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid patient identifier")
    return candidate


def _load_patient(patient_id: str) -> pd.DataFrame:
    path = _patient_file(patient_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Patient not found")
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - surface corrupt artefacts honestly
        raise HTTPException(
            status_code=500, detail=f"Could not read patient artefact: {type(exc).__name__}"
        ) from exc


def _safe_value(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return None if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [
        {str(k): _safe_value(v) for k, v in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _optional_scalar(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return None
    non_null = frame[column].dropna()
    if len(non_null) == 0:
        return None
    return _safe_value(non_null.iloc[0])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "icu-sentinel-p1-patients"}


@app.get("/patients")
def list_patients() -> dict:
    directory = _processed_dir()
    if not directory.is_dir():
        raise HTTPException(
            status_code=503,
            detail=(
                "Processed patient directory is not configured. Set "
                "ICU_SENTINEL_PROCESSED_DIR or generate data/processed first."
            ),
        )
    patient_ids = sorted(
        path.stem for path in directory.glob("*.parquet") if _PATIENT_ID_RE.match(path.stem)
    )
    return {"n_patients": len(patient_ids), "patient_ids": patient_ids}


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str) -> dict:
    frame = _load_patient(patient_id)
    if len(frame) == 0:
        raise HTTPException(status_code=404, detail="Patient has no records")

    hours = pd.to_numeric(frame["hour"], errors="coerce") if "hour" in frame.columns else None
    labels = (
        pd.to_numeric(frame["SepsisLabel"], errors="coerce")
        if "SepsisLabel" in frame.columns
        else None
    )
    positive_hours = labels[labels == 1] if labels is not None else None
    onset_hour = None
    if positive_hours is not None and len(positive_hours) > 0 and hours is not None:
        onset_hour = _safe_value(hours.loc[positive_hours.index].min())

    return {
        "patient_id": patient_id,
        "n_rows": int(len(frame)),
        "hour_min": _safe_value(hours.min()) if hours is not None else None,
        "hour_max": _safe_value(hours.max()) if hours is not None else None,
        "hospital_id": _optional_scalar(frame, "hospital_id"),
        "source_dataset": _optional_scalar(frame, "source_dataset"),
        "n_sepsis_positive_hours": (
            int((labels == 1).sum()) if labels is not None else None
        ),
        "sepsis_onset_hour": onset_hour,
        "has_derived_target": bool(
            {"sepsis_within_6h", "target_eligible"}.issubset(frame.columns)
        ),
        "columns": [str(c) for c in frame.columns],
    }


@app.get("/patients/{patient_id}/timeline")
def get_patient_timeline(
    patient_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict:
    frame = _load_patient(patient_id)
    if "hour" in frame.columns:
        frame = frame.sort_values("hour", kind="stable")
    page = frame.iloc[offset : offset + limit]
    return {
        "patient_id": patient_id,
        "offset": offset,
        "limit": limit,
        "n_rows_total": int(len(frame)),
        "n_rows_returned": int(len(page)),
        "records": _records(page),
    }


@app.get("/patients/{patient_id}/features")
def get_patient_features(
    patient_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict:
    _validate_patient_id(patient_id)
    path = _features_file()
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Combined feature artefact is not configured. Set "
                "ICU_SENTINEL_FEATURES_FILE or generate "
                "data/features/combined_features.parquet first."
            ),
        )
    try:
        schema = pq.read_schema(path)
        if "patient_id" not in schema.names:
            raise HTTPException(
                status_code=500,
                detail="Feature artefact has no patient_id column",
            )
        table = pq.read_table(path, filters=[("patient_id", "=", patient_id)])
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface corrupt artefacts honestly
        raise HTTPException(
            status_code=500, detail=f"Could not read feature artefact: {type(exc).__name__}"
        ) from exc

    frame = table.to_pandas()
    if len(frame) == 0:
        raise HTTPException(status_code=404, detail="No features found for patient")
    if "hour" in frame.columns:
        frame = frame.sort_values("hour", kind="stable")
    page = frame.iloc[offset : offset + limit]
    return {
        "patient_id": patient_id,
        "offset": offset,
        "limit": limit,
        "n_rows_total": int(len(frame)),
        "n_rows_returned": int(len(page)),
        "records": _records(page),
    }
