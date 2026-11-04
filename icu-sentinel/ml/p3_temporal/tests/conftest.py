"""Shared fixtures for the P3 test suite.

The demo model is fitted once per session on synthetic demonstration
patients. Every fixture is explicitly synthetic; no real patient data is
read.
"""

from __future__ import annotations

import warnings

import pytest

from ml.p3_temporal.alerts.alert_engine import AlertEngine
from ml.p3_temporal.alerts.alert_rules import AlertThresholdConfig
from ml.p3_temporal.alerts.alert_store import AlertStore
from ml.p3_temporal.common.model_bundle import ModelRegistry
from ml.p3_temporal.explainability.demo_model import build_demo_model_bundle
from ml.p3_temporal.explainability.patient_explainer import ExplanationService
from ml.p3_temporal.replay.demo_data import (
    REPLAY_PROFILES,
    SyntheticDemoPatientRepository,
)
from ml.p3_temporal.replay.prediction_provider import ModelPredictionProvider
from ml.p3_temporal.replay.replay_engine import ReplayEngine


@pytest.fixture(autouse=True)
def _quiet_p1_warnings() -> None:
    # P1's feature pipeline emits pandas chained-assignment and
    # fragmentation warnings. They are pre-existing and unrelated to P3.
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="shap")


@pytest.fixture(scope="session")
def demo_repository() -> SyntheticDemoPatientRepository:
    return SyntheticDemoPatientRepository()


@pytest.fixture(scope="session")
def demo_bundle(demo_repository: SyntheticDemoPatientRepository):
    return build_demo_model_bundle(demo_repository.training_patients())


@pytest.fixture()
def registry(demo_bundle) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(demo_bundle)
    return registry


@pytest.fixture()
def alert_store(tmp_path) -> AlertStore:
    store = AlertStore(tmp_path / "alerts.db")
    yield store
    store.close()


@pytest.fixture()
def alert_engine(alert_store) -> AlertEngine:
    return AlertEngine(
        alert_store,
        AlertThresholdConfig(threshold=0.60, dedup_window_hours=6),
    )


@pytest.fixture()
def replay_engine(demo_repository, demo_bundle, alert_engine) -> ReplayEngine:
    return ReplayEngine(
        demo_repository,
        ModelPredictionProvider(demo_bundle),
        alert_engine=alert_engine,
    )


@pytest.fixture()
def explanation_service(registry, replay_engine) -> ExplanationService:
    return ExplanationService(registry, replay_engine)


@pytest.fixture(scope="session")
def replay_profile_ids() -> list[str]:
    return [profile.patient_id for profile in REPLAY_PROFILES]
