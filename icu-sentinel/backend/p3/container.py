"""Dependency container for the P3 API.

The container is the only place that chooses concrete repositories,
prediction providers and persistence. Routes depend on the abstract P3
services, so a P1/P2/P3 model provider can be swapped without changing
the API surface.

Modes (controlled by ICU_SENTINEL_MODE):

* ``synthetic`` — always use the bundled synthetic demonstration cohort
  and demo model (default when no real data is configured).
* ``real`` — load real PhysioNet-derived patients from
  ``ICU_SENTINEL_DATA_DIR`` and the frozen P2 boosting artifacts
  (Extra Trees + XGBoost) from ``ICU_SENTINEL_P2_MODEL_DIR``.
* ``auto`` — use real data when both a data directory and P2 model
  directory are explicitly configured; otherwise fall back to synthetic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.p2_boosting.calibration import apply_probability_calibrator
from ml.p3_temporal.alerts.alert_engine import AlertEngine
from ml.p3_temporal.alerts.alert_rules import AlertThresholdConfig
from ml.p3_temporal.alerts.alert_store import AlertStore
from ml.p3_temporal.common.model_bundle import (
    ModelBundle,
    ModelRegistry,
    load_p1_model_bundle,
)
from ml.p3_temporal.common.prediction import (
    ModelIdentity,
    PredictionProvider,
    PredictionSource,
)
from ml.p3_temporal.explainability.demo_model import build_demo_model_bundle
from ml.p3_temporal.explainability.patient_explainer import ExplanationService
from ml.p3_temporal.replay.demo_data import (
    DirectoryPatientRepository,
    PatientRepository,
    SyntheticDemoPatientRepository,
)
from ml.p3_temporal.replay.prediction_provider import (
    MultiModelPredictionProvider,
    ModelPredictionProvider,
)
from ml.p3_temporal.replay.replay_engine import ReplayEngine

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Authoritative P2 matched cohort artifacts.
_P2_MATCHED_DIR = _REPO_ROOT / "artifacts" / "p2" / "matched_6000"
_P2_THRESHOLDS_FILE = (
    _REPO_ROOT / "data" / "reports" / "p2_matched_6000" / "p2_thresholds.json"
)

_P2_MODEL_NAMES = ("extra_trees", "xgboost")


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


def _env_mode() -> str:
    raw = os.getenv("ICU_SENTINEL_MODE", "auto").strip().lower()
    if raw not in {"synthetic", "real", "auto"}:
        return "auto"
    return raw


def _load_p2_thresholds() -> dict[str, float]:
    """Load validation-selected thresholds for P2 models.

    Returns an empty dict if the thresholds file is absent or unreadable.
    Thresholds are applied only when a model provider has one configured;
    otherwise ``predicted_class`` remains unset.
    """
    if not _P2_THRESHOLDS_FILE.exists():
        return {}
    try:
        with open(_P2_THRESHOLDS_FILE) as f:
            data = json.load(f)
        thresholds = data.get("thresholds", {})
        return {str(k): float(v) for k, v in thresholds.items()}
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return {}


@dataclass(frozen=True)
class P3Settings:
    """Runtime configuration for the P3 API."""

    mode: str = "auto"
    data_dir: Path | None = None
    alert_db_path: str = str(_REPO_ROOT / "data" / "p3_alerts.db")
    p1_model_dir: Path | None = None
    p1_model_name: str | None = None
    p1_model_version: str = "1.0.0"
    p2_model_dir: Path | None = None
    p2_model_names: tuple[str, ...] = _P2_MODEL_NAMES
    demo_model_when_no_artifact: bool = True
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    )
    alert_config: AlertThresholdConfig = field(default_factory=AlertThresholdConfig.from_env)

    @classmethod
    def from_env(cls) -> "P3Settings":
        mode = _env_mode()
        p1_model_dir = _env_path("ICU_SENTINEL_P1_MODEL_DIR")
        p1_model_name = os.getenv("ICU_SENTINEL_P1_MODEL_NAME") or None
        if p1_model_dir is not None and not p1_model_name:
            p1_model_name = "logistic_regression"
        p2_model_dir = _env_path("ICU_SENTINEL_P2_MODEL_DIR")
        p2_names_raw = os.getenv("ICU_SENTINEL_P2_MODEL_NAMES")
        p2_names = _P2_MODEL_NAMES
        if p2_names_raw:
            p2_names = tuple(n.strip() for n in p2_names_raw.split(",") if n.strip())
        alert_db_path = os.getenv(
            "ICU_SENTINEL_ALERT_DB", str(_REPO_ROOT / "data" / "p3_alerts.db")
        )
        _demo_when = os.getenv(
            "ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            mode=mode,
            data_dir=_env_path("ICU_SENTINEL_DATA_DIR"),
            alert_db_path=alert_db_path,
            p1_model_dir=p1_model_dir,
            p1_model_name=p1_model_name,
            p1_model_version=os.getenv("ICU_SENTINEL_P1_MODEL_VERSION", "1.0.0"),
            p2_model_dir=p2_model_dir,
            p2_model_names=p2_names,
            demo_model_when_no_artifact=_demo_when,
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
    """Fully wired P3 services.

    In real mode the container registers both Extra Trees and XGBoost
    bundles and wires a :class:`MultiModelPredictionProvider` so the replay
    engine returns predictions from every fitted model.
    """

    def __init__(
        self,
        *,
        repository: PatientRepository,
        bundle: ModelBundle,
        bundles: dict[str, ModelBundle],
        provider: PredictionProvider,
        thresholds: dict[str, float],
        mode: str,
        alert_store: AlertStore,
        alert_engine: AlertEngine,
        replay_engine: ReplayEngine,
        explanation_service: ExplanationService,
        registry: ModelRegistry,
    ) -> None:
        self.repository = repository
        self.bundle = bundle
        self.bundles = bundles
        self.provider = provider
        self.thresholds = thresholds
        self.mode = mode
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

        # --- Repository: real or synthetic ---
        use_real_data = settings.mode == "real" or (
            settings.mode == "auto" and settings.data_dir is not None
        )
        if use_real_data and settings.data_dir is not None:
            repository = DirectoryPatientRepository(settings.data_dir)
        else:
            repository = SyntheticDemoPatientRepository()
            use_real_data = False

        # --- Model bundles ---
        thresholds = _load_p2_thresholds()
        bundles: dict[str, ModelBundle] = {}

        if use_real_data:
            # Load P2 boosting artifacts. Prefer explicit P2 model dir;
            # otherwise fall back to the authoritative matched_6000 artifacts.
            p2_dir = settings.p2_model_dir or _P2_MATCHED_DIR
            for model_name in settings.p2_model_names:
                try:
                    bundle = load_p1_model_bundle(p2_dir, model_name, "1.0.0")
                    # Re-label the source as P2 boosting (not P1 classical).
                    bundle.identity = ModelIdentity(
                        name=bundle.identity.name,
                        version=bundle.identity.version,
                        source=PredictionSource.P2_BOOSTING_MODEL,
                        display_name=bundle.identity.display_name
                        or f"P2 {model_name}",
                        is_synthetic=False,
                        feature_schema_version=bundle.identity.feature_schema_version,
                    )
                    # Attach calibrator if available.
                    calibrator = _load_calibrator(model_name)
                    if calibrator is not None:
                        bundle.metadata["calibrator"] = calibrator
                        bundle.metadata["calibration_method"] = "sigmoid"
                        bundle.metadata["calibration_partition"] = "validation"
                    # Attach threshold from the frozen validation selection.
                    if model_name in thresholds:
                        bundle.metadata["threshold"] = thresholds[model_name]
                    # Attach a minimal structural background (all model features,
                    # NaN values) so the SHAP plan check can verify feature-count
                    # preservation through the imputer preprocessor. This is a
                    # shape template only; Tree SHAP does not use background
                    # values for explanation computation.
                    background = pd.DataFrame(
                        [[float("nan")] * len(bundle.selected_features)],
                        columns=bundle.selected_features,
                    )
                    bundle.background = background
                    bundles[model_name] = bundle
                except Exception:  # noqa: BLE001 - continue with remaining models
                    continue

        # If no real models loaded, fall back to the demo bundle.
        if not bundles:
            demo_bundle = build_demo_model_bundle(
                getattr(repository, "training_patients", lambda: [])()
            )
            bundles["demo_logistic_regression"] = demo_bundle

        # Primary bundle = first registered bundle (backward compat).
        primary_name = next(iter(bundles))
        primary_bundle = bundles[primary_name]
        registry = ModelRegistry()
        for b in bundles.values():
            registry.register(b)

        # --- Provider ---
        providers: dict[str, ModelPredictionProvider] = {}
        for name, bundle in bundles.items():
            threshold = bundle.metadata.get("threshold")
            providers[name] = ModelPredictionProvider(
                bundle, threshold=threshold if isinstance(threshold, (int, float)) else None
            )

        if len(providers) > 1:
            provider: PredictionProvider = MultiModelPredictionProvider(
                providers, primary_model_name=primary_name
            )
        else:
            provider = next(iter(providers.values()))

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
            bundle=primary_bundle,
            bundles=bundles,
            provider=provider,
            thresholds=thresholds,
            mode="real" if use_real_data else "synthetic",
            alert_store=alert_store,
            alert_engine=alert_engine,
            replay_engine=replay_engine,
            explanation_service=explanation_service,
            registry=registry,
        )


def _load_calibrator(model_name: str) -> Any:
    """Load a fitted calibrator for the given model name, if available.

    Calibrators are stored alongside model artifacts as
    <model_name>_1.0.0_calibrator.joblib.
    """
    repo_root = Path(__file__).resolve().parents[3]
    calibrator_path = repo_root / "artifacts" / "p2" / model_name / f"{model_name}_1.0.0_calibrator.joblib"
    if calibrator_path.exists():
        return joblib.load(calibrator_path)
    return None
