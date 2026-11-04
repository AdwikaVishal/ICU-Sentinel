"""FastAPI application for the ICU Sentinel P3 modules.

Run locally with either of::

    uvicorn backend.p3.app:app --reload --port 8000

The API is a retrospective research prototype. Synthetic demo data and
predictions are labelled as such; nothing here is a diagnosis.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml.p3_temporal.alerts.alert_store import AlertNotFoundError, AlertStateError
from ml.p3_temporal.replay.demo_data import PatientNotFoundError
from ml.p3_temporal.replay.model_input import ModelInputError
from ml.p3_temporal.replay.timeline import HourOutOfRangeError

from backend.p3.container import P3Container, P3Settings
from backend.p3.routes import alerts, explanations, replay
from backend.p3.schemas import HealthResponse

API_DISCLAIMER = (
    "Retrospective research prototype. Predictions, explanations and alerts "
    "are not a diagnosis, do not confirm sepsis, and must not be used for "
    "clinical decisions. Synthetic demonstration data is explicitly labelled."
)


def _error(status_code: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": code, "detail": detail},
    )


def create_app(settings: P3Settings | None = None) -> FastAPI:
    settings = settings or P3Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = P3Container.build(settings)
        app.state.container = container
        try:
            yield
        finally:
            container.close()
            app.state.container = None

    app = FastAPI(
        title="ICU Sentinel P3 API",
        version="3.0.0",
        description=API_DISCLAIMER,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(replay.router)
    app.include_router(explanations.router)
    app.include_router(alerts.router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/api/v3/health", response_model=HealthResponse, tags=["meta"])
    def api_health() -> HealthResponse:
        return HealthResponse()

    @app.get("/api/v3/meta", tags=["meta"])
    def meta(request: Request) -> dict:
        container = getattr(request.app.state, "container", None)
        provider = (
            container.replay_engine.provider_status().model_dump(mode="json")
            if container is not None
            else None
        )
        return {
            "service": "icu-sentinel-p3",
            "version": "3.0.0",
            "disclaimer": API_DISCLAIMER,
            "provider": provider,
            "data_source": getattr(container, "repository", None).data_source
            if container is not None
            else None,
            "is_synthetic": getattr(container, "repository", None).is_synthetic
            if container is not None
            else None,
        }

    @app.exception_handler(PatientNotFoundError)
    async def _patient_not_found(_: Request, exc: PatientNotFoundError) -> JSONResponse:
        return _error(404, "patient_not_found", str(exc))

    @app.exception_handler(HourOutOfRangeError)
    async def _hour_out_of_range(_: Request, exc: HourOutOfRangeError) -> JSONResponse:
        return _error(400, "hour_out_of_range", str(exc))

    @app.exception_handler(ModelInputError)
    async def _model_input_error(_: Request, exc: ModelInputError) -> JSONResponse:
        return _error(422, "model_input_unavailable", str(exc))

    @app.exception_handler(AlertNotFoundError)
    async def _alert_not_found(_: Request, exc: AlertNotFoundError) -> JSONResponse:
        return _error(404, "alert_not_found", str(exc))

    @app.exception_handler(AlertStateError)
    async def _alert_state(_: Request, exc: AlertStateError) -> JSONResponse:
        return _error(409, "invalid_alert_state", str(exc))

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return _error(400, "invalid_request", str(exc))

    return app


app = create_app()
