"""P1 patient data API stub.

Planned endpoints (implemented in PART 8):
    GET /patients
    GET /patients/{id}
    GET /patients/{id}/timeline
    GET /patients/{id}/features

This stub exposes a configured FastAPI application with health-check only,
so the service can start and be wired up in PART 8 without real patient data.
"""

from fastapi import FastAPI

app = FastAPI(title="ICU Sentinel P1 Patient API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "icu-sentinel-p1-patients"}
