"""API-level tests for the P3 FastAPI application.

The application is exercised through ``TestClient`` with an isolated,
temporary SQLite alert database. Every prediction, explanation and alert
in these tests comes from the explicitly labelled synthetic demo model.
"""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

from fastapi.testclient import TestClient  # noqa: E402

from backend.p3.app import create_app  # noqa: E402
from backend.p3.container import P3Settings  # noqa: E402

PATIENT = "DEMO-0004"
PREDICTION = {
    "patient_id": PATIENT,
    "hour": 20,
    "probability": 0.99,
    "model_name": "demo_logistic_regression",
    "model_version": "demo-1.0.0",
    "prediction_source": "synthetic_demo_model",
    "is_synthetic": True,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    settings = P3Settings(
        alert_db_path=str(tmp_path_factory.mktemp("p3-api") / "alerts.db"),
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


# ----------------------------------------------------------------------
# Meta / provider
# ----------------------------------------------------------------------
def test_health_and_meta(client: TestClient) -> None:
    for path in ("/health", "/api/v3/health"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    meta = client.get("/api/v3/meta")
    assert meta.status_code == 200
    body = meta.json()
    assert body["service"] == "icu-sentinel-p3"
    assert body["is_synthetic"] is True
    assert "not a diagnosis" in body["disclaimer"]
    assert body["provider"]["model"]["source"] == "synthetic_demo_model"


def test_provider_status_is_labelled_synthetic(client: TestClient) -> None:
    response = client.get("/api/v3/provider/status")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["model"]["source"] == "synthetic_demo_model"
    assert body["model"]["is_synthetic"] is True


def test_alert_threshold_is_disclosed_as_not_validated(client: TestClient) -> None:
    response = client.get("/api/v3/alerts/threshold")
    assert response.status_code == 200
    body = response.json()
    assert body["clinically_validated"] is False
    assert "not a clinically validated" in body["disclosure"].lower()
    assert body["dedup_window_hours"] >= 0


# ----------------------------------------------------------------------
# Patients / timeline / replay
# ----------------------------------------------------------------------
def test_patient_list_and_timeline(client: TestClient) -> None:
    response = client.get("/api/v3/patients")
    assert response.status_code == 200
    body = response.json()
    assert body["is_synthetic"] is True
    ids = [patient["patient_id"] for patient in body["patients"]]
    assert PATIENT in ids

    timeline = client.get(f"/api/v3/patients/{PATIENT}/timeline")
    assert timeline.status_code == 200
    timeline_body = timeline.json()
    assert timeline_body["is_synthetic"] is True
    assert timeline_body["count"] == len(timeline_body["hours"])


def test_replay_start_uses_first_hour_without_alerts(client: TestClient) -> None:
    # Regression: the route passes evaluate_alerts through to ReplayEngine.start.
    response = client.get(f"/api/v3/replay/{PATIENT}")
    assert response.status_code == 200
    body = response.json()
    assert body["session"]["at_start"] is True
    assert body["hour"] == body["session"]["min_hour"]
    assert body["prediction_status"] in {"available", "missing_hour", "unavailable"}
    assert body["data_source"]
    assert body["is_synthetic"] is True
    assert "not a diagnosis" in body["disclaimer"]


def test_replay_navigation_and_model_input(client: TestClient) -> None:
    stepped = client.post(
        f"/api/v3/replay/{PATIENT}/step",
        json={"hour": 10, "direction": "forward", "evaluate_alerts": False},
    )
    assert stepped.status_code == 200
    assert stepped.json()["hour"] == 11

    jumped = client.post(
        f"/api/v3/replay/{PATIENT}/jump",
        json={"hour": 20, "evaluate_alerts": False},
    )
    assert jumped.status_code == 200
    assert jumped.json()["hour"] == 20

    reset = client.post(f"/api/v3/replay/{PATIENT}/reset")
    assert reset.status_code == 200
    assert reset.json()["session"]["at_start"] is True

    ended = client.post(f"/api/v3/replay/{PATIENT}/end")
    assert ended.status_code == 200
    assert ended.json()["session"]["at_end"] is True


def test_model_input_exposes_no_retrospective_label_as_feature(client: TestClient) -> None:
    response = client.get(f"/api/v3/replay/{PATIENT}/model-input", params={"hour": 12})
    assert response.status_code == 200
    body = response.json()
    assert body["hour"] == 12
    assert "SepsisLabel" not in body["feature_names"]
    assert "SepsisLabel" not in body["feature_values"]
    assert all(hour <= 12 for hour in body["observed_hours_used"])
    assert body["label_disclosure"]

    # The reported input hash must be stable for the same hour.
    again = client.get(f"/api/v3/replay/{PATIENT}/model-input", params={"hour": 12})
    assert again.json()["input_hash"] == body["input_hash"]


def test_replay_rejects_unknown_patient_and_bad_hour(client: TestClient) -> None:
    missing = client.get("/api/v3/replay/DOES-NOT-EXIST")
    assert missing.status_code == 404
    assert missing.json()["error"] == "patient_not_found"

    bad_hour = client.get(f"/api/v3/replay/{PATIENT}", params={"hour": 10000})
    assert bad_hour.status_code == 400
    assert bad_hour.json()["error"] == "hour_out_of_range"


# ----------------------------------------------------------------------
# Explanations
# ----------------------------------------------------------------------
def test_explanation_catalog_and_global(client: TestClient) -> None:
    catalog = client.get("/api/v3/explanations/models")
    assert catalog.status_code == 200
    entries = catalog.json()["models"]
    assert entries
    assert all(entry["model"]["is_synthetic"] for entry in entries)

    global_response = client.get("/api/v3/explanations/global")
    assert global_response.status_code == 200
    global_body = global_response.json()
    assert global_body["status"] == "available"
    assert global_body["features"]
    assert global_body["model"]["is_synthetic"] is True

    feature = client.get("/api/v3/explanations/features/heart_rate")
    assert feature.status_code == 200
    assert feature.json()["feature_name"] == "heart_rate"


def test_patient_explanation_and_comparison(client: TestClient) -> None:
    response = client.get(f"/api/v3/explanations/{PATIENT}/20")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available"
    assert body["model"]["source"] == "synthetic_demo_model"
    assert body["model"]["is_synthetic"] is True
    assert body["contributions"]
    assert body["validation"]["passed"] is True

    comparison = client.get(
        f"/api/v3/explanations/{PATIENT}/compare",
        params={"hour_a": 10, "hour_b": 20},
    )
    assert comparison.status_code == 200
    comparison_body = comparison.json()
    assert comparison_body["hour_a"] == 10
    assert comparison_body["hour_b"] == 20
    assert comparison_body["changes"]


def test_explanation_for_missing_hour_is_explicitly_unavailable(client: TestClient) -> None:
    # DEMO-0003 has no recorded row at hour 12.
    response = client.get("/api/v3/explanations/DEMO-0003/12")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["reason"]
    assert body["contributions"] == []


# ----------------------------------------------------------------------
# Alerts
# ----------------------------------------------------------------------
def test_alert_lifecycle_create_dedupe_acknowledge_clear(client: TestClient) -> None:
    created = client.post("/api/v3/alerts/evaluate", json=PREDICTION)
    assert created.status_code == 200
    created_body = created.json()
    assert created_body["action"] == "created"
    alert = created_body["alert"]
    assert alert["status"] == "ACTIVE"
    assert alert["is_synthetic"] is True
    assert alert["threshold"] == pytest.approx(0.60)

    # A second high probability within the dedup window updates the episode.
    continued = client.post(
        "/api/v3/alerts/evaluate",
        json={**PREDICTION, "hour": 21, "probability": 0.995},
    )
    assert continued.status_code == 200
    continued_body = continued.json()
    assert continued_body["action"] == "updated"
    # The update is absorbed into the existing episode by design.
    assert continued_body["deduplicated"] is True
    assert continued_body["alert"]["alert_id"] == alert["alert_id"]
    assert continued_body["alert"]["occurrence_count"] == 2

    active = client.get("/api/v3/alerts/active")
    assert active.status_code == 200
    matching = [
        item for item in active.json()["alerts"] if item["alert_id"] == alert["alert_id"]
    ]
    assert len(matching) == 1

    acknowledged = client.post(f"/api/v3/alerts/{alert['alert_id']}/acknowledge")
    assert acknowledged.status_code == 200
    assert acknowledged.json()["status"] == "ACKNOWLEDGED"

    # Acknowledging twice is idempotent, not an error.
    acknowledged_again = client.post(f"/api/v3/alerts/{alert['alert_id']}/acknowledge")
    assert acknowledged_again.status_code == 200
    assert acknowledged_again.json()["status"] == "ACKNOWLEDGED"

    cleared = client.post(
        f"/api/v3/alerts/{alert['alert_id']}/clear",
        json={"reason": "manual_review_complete"},
    )
    assert cleared.status_code == 200
    assert cleared.json()["status"] == "CLEARED"
    assert cleared.json()["cleared_reason"] == "manual_review_complete"

    # A cleared alert is terminal and cannot return to ACKNOWLEDGED.
    inaccessible = client.post(f"/api/v3/alerts/{alert['alert_id']}/acknowledge")
    assert inaccessible.status_code == 409
    assert inaccessible.json()["error"] == "invalid_alert_state"

    history = client.get("/api/v3/alerts/history")
    assert history.status_code == 200
    ids = [item["alert_id"] for item in history.json()["alerts"]]
    assert alert["alert_id"] in ids


def test_alert_filters_and_not_found(client: TestClient) -> None:
    filtered = client.get("/api/v3/alerts", params={"patient_id": PATIENT, "limit": 5})
    assert filtered.status_code == 200
    assert filtered.json()["count"] <= 5

    missing = client.get("/api/v3/alerts/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"] == "alert_not_found"


def test_alert_evaluation_validation(client: TestClient) -> None:
    response = client.post(
        "/api/v3/alerts/evaluate",
        json={**PREDICTION, "probability": 1.5},
    )
    assert response.status_code == 422


def test_below_threshold_does_not_create_alert(client: TestClient) -> None:
    response = client.post(
        "/api/v3/alerts/evaluate",
        json={**PREDICTION, "patient_id": "DEMO-0003", "hour": 5, "probability": 0.05},
    )
    assert response.status_code == 200
    assert response.json()["action"] in {"none", "unchanged"}
    assert response.json()["alert"] is None
