"""Tests for the read-only P1 patient API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.p1_patients.app import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    features = tmp_path / "features"
    processed.mkdir()
    features.mkdir()

    frame = pd.DataFrame(
        {
            "patient_id": ["p00001"] * 4,
            "hospital_id": [None] * 4,
            "source_dataset": ["set_a"] * 4,
            "hour": [1, 2, 3, 4],
            "HR": [90.0, np.nan, 95.0, 100.0],
            "SepsisLabel": [0, 0, 1, 1],
            "sepsis_within_6h": [0.0, 0.0, 1.0, 1.0],
            "target_eligible": [1, 1, 1, 1],
        }
    )
    frame.to_parquet(processed / "p00001.parquet", index=False)

    feature_frame = pd.DataFrame(
        {
            "patient_id": ["p00001"] * 2,
            "hour": [1, 2],
            "HR_rolling_mean_6h": [90.0, 90.0],
        }
    )
    feature_frame.to_parquet(features / "combined_features.parquet", index=False)

    monkeypatch.setenv("ICU_SENTINEL_PROCESSED_DIR", str(processed))
    monkeypatch.setenv(
        "ICU_SENTINEL_FEATURES_FILE", str(features / "combined_features.parquet")
    )
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_patients(client):
    response = client.get("/patients")
    assert response.status_code == 200
    body = response.json()
    assert body["n_patients"] == 1
    assert body["patient_ids"] == ["p00001"]


def test_get_patient_does_not_invent_hospital_id(client):
    response = client.get("/patients/p00001")
    assert response.status_code == 200
    body = response.json()
    assert body["patient_id"] == "p00001"
    assert body["hospital_id"] is None
    assert body["source_dataset"] == "set_a"
    assert body["sepsis_onset_hour"] == 3
    assert body["n_sepsis_positive_hours"] == 2
    assert body["has_derived_target"] is True


def test_timeline_is_ordered_and_json_safe(client):
    response = client.get("/patients/p00001/timeline", params={"limit": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["n_rows_total"] == 4
    assert body["n_rows_returned"] == 2
    assert [r["hour"] for r in body["records"]] == [1, 2]
    assert body["records"][1]["HR"] is None


def test_features_endpoint(client):
    response = client.get("/patients/p00001/features")
    assert response.status_code == 200
    body = response.json()
    assert body["n_rows_total"] == 2
    assert body["records"][0]["HR_rolling_mean_6h"] == 90.0


def test_missing_patient_returns_404(client):
    response = client.get("/patients/p99999")
    assert response.status_code == 404


def test_invalid_patient_id_returns_400(client):
    response = client.get("/patients/bad%20id")
    assert response.status_code == 400


def test_unconfigured_processed_dir_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("ICU_SENTINEL_PROCESSED_DIR", str(tmp_path / "missing"))
    client = TestClient(app)
    response = client.get("/patients")
    assert response.status_code == 503


def test_unconfigured_features_returns_503(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ICU_SENTINEL_FEATURES_FILE", str(tmp_path / "missing.parquet"))
    response = client.get("/patients/p00001/features")
    assert response.status_code == 503
