"""Dependency container for the P3 API.

The container is the only place that chooses concrete repositories,
prediction providers and persistence. Routes depend on the abstract P3
services, so a P1/P2/P3 model provider can be swapped without changing
the API surface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ml.p3_temporal.alerts.alert_engine import AlertEngine
from ml.p3_temporal.alerts.alert_rules import AlertThresholdConfig
from ml.p3_temporal.alerts.alert_store import AlertStore
from ml.p3_temporal.common.model_bundle import ModelBundle, ModelRegistry, load_p1_model_bundle
from ml.p3_temporal.common.prediction import PredictionProvider
from ml.p3_temporal.explainability.demo_model import build_demo_model_bundle
from ml.p3_temporal.explainability.patient_explainer import ExplanationService
from ml.p3_temporal.replay.demo_data import (
    DirectoryPatientRepository,
    PatientRepository,
    SyntheticDemoPatientRepository,
)
from ml.p3_temporal.replay.prediction_provider import ModelPredictionProvider
from ml.p3_temporal.replay.replay_engine import ReplayEngine

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str) -> Path | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class P3Settings:
    """Runtime configuration for the P3 API."""

    data_dir: Path | None = None
    alert_db_path: str = str(_REPO_ROOT / "data" / "p3_alerts.db")
    p1_model_dir: Path | None = None
    p1_model_name: str | None = None
    p1_model_version: str = "1.0.0"
    demo_model_when_no_artifact: bool = True
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    )
    alert_config: AlertThresholdConfig = field(default_factory=AlertThresholdConfig.from_env)

    @classmethod
    def from_env(cls) -> "P3Settings":
        p1_model_dir = _env_path("ICU_SENTINEL_P1_MODEL_DIR")
        p1_model_name = os.getenv("ICU_SENTINEL_P1_MODEL_NAME") or None
        if p1_model_dir is not None and not p1_model_name:
            p1_model_name = "logistic_regression"
        alert_db_path = os.getenv(
            "ICU_SENTINEL_ALERT_DB", str(_REPO_ROOT / "data" / "p3_alerts.db")
        )
        return cls(
            data_dir=_env_path("ICU_SENTINEL_DATA_DIR"),
            alert_db_path=alert_db_path,
            p1_model_dir=p1_model_dir,
            p1_model_name=p1_model_name,
            p1_model_version=os.getenv("ICU_SENTINEL_P1_MODEL_VERSION", "1.0.0"),
            demo_model_when_no_artifact=os.getenv(
                "ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT", "true"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            cors_origins=_env_list(
                "ICU_SENTINEL_CORS_ORIGINS",
                (
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://localhost:3000",
                ),
            ),
            alert_config=AlertThresholdConfig.from_env(),
        )


class P3Container:
    """Fully wired P3 services."""

    def __init__(
        self,
        *,
        repository: PatientRepository,
        bundle: ModelBundle,
        provider: PredictionProvider,
        alert_store: AlertStore,
        alert_engine: AlertEngine,
        replay_engine: ReplayEngine,
        explanation_service: ExplanationService,
        registry: ModelRegistry,
    ) -> None:
        self.repository = repository
        self.bundle = bundle
        self.provider = provider
        self.alert_store = alert_store
        self.alert_engine = alert_engine
        self.replay_engine = replay_engine
        self.explanation_service = explanation_service
        self.registry = registry

    def close(self) -> None:
        self.alert_store.close()

    @classmethod
    def build(cls, settings: P3Settings | None = None) -> "P3Container":
        settings = settings or P3Settings.from_env()

        repository: PatientRepository
        if settings.data_dir is not None:
            repository = DirectoryPatientRepository(settings.data_dir)
        else:
            repository = SyntheticDemoPatientRepository()

        bundle = _resolve_bundle(settings, repository)
        registry = ModelRegistry()
        registry.register(bundle)
        provider = ModelPredictionProvider(bundle)

        alert_store = AlertStore(settings.alert_db_path)
        alert_engine = AlertEngine(alert_store, settings.alert_config)
        replay_engine = ReplayEngine(
            repository,
            provider,
            alert_engine=alert_engine,
        )
        explanation_service = ExplanationService(registry, replay_engine)

        return cls(
            repository=repository,
            bundle=bundle,
            provider=provider,
            alert_store=alert_store,
            alert_engine=alert_engine,
            replay_engine=replay_engine,
            explanation_service=explanation_service,
            registry=registry,
        )


def _resolve_bundle(settings: P3Settings, repository: PatientRepository) -> ModelBundle:
    if settings.p1_model_dir is not None and settings.p1_model_name:
        return load_p1_model_bundle(
            settings.p1_model_dir,
            settings.p1_model_name,
            settings.p1_model_version,
        )

    if not settings.demo_model_when_no_artifact:
        raise RuntimeError(
            "No P1 model artifact configured and demonstration fallback is disabled. "
            "Set ICU_SENTINEL_P1_MODEL_DIR/NAME or enable the demo model."
        )

    training_patients = getattr(repository, "training_patients", lambda: [])()
    if not training_patients:
        # A real-data repository has no synthetic training patients. The
        # demo model is only a local fallback for the synthetic demo data.
        raise RuntimeError(
            "A real processed-data repository is configured, but no P1 model "
            "artifact was supplied. Set ICU_SENTINEL_P1_MODEL_DIR and "
            "ICU_SENTINEL_P1_MODEL_NAME."
        )
    return build_demo_model_bundle(training_patients)
