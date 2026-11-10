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

import joblib

from ml.p2_boosting.calibration import apply_probability_calibrator
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
        _demo_when = os.getenv(
            "ICU_SENTINEL_DEMO_MODEL_WHEN_NO_ARTIFACT", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            data_dir=_env_path("ICU_SENTINEL_DATA_DIR"),
            alert_db_path=alert_db_path,
            p1_model_dir=p1_model_dir,
            p1_model_name=p1_model_name,
            p1_model_version=os.getenv("ICU_SENTINEL_P1_MODEL_VERSION", "1.0.0"),
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


def _load_calibrator(model_name: str) -> Any:
    """Load a fitted calibrator for the given model name, if available.

    Calibrators are stored alongside model artifacts as
    <model_name>_1.0.0_calibrator.joblib.
    """
    # _REPO_ROOT is 3 levels up from this file (project root)
    # Artifacts are at /artifacts/p2/<model_name>/
    repo_root = Path(__file__).resolve().parents[3]
    calibrator_path = repo_root / "artifacts" / "p2" / model_name / f"{model_name}_1.0.0_calibrator.joblib"
    if calibrator_path.exists():
        return joblib.load(calibrator_path)
    return None


def _resolve_bundle(settings: P3Settings, repository: PatientRepository) -> ModelBundle:
    """Choose a model bundle with the following priority:

    1. Explicit P1 model artifact (via ICU_SENTINEL_P1_MODEL_DIR/NAME).
    2. Auto-detected model artifact in the default PhysioNet artifacts directory,
       if the data directory points to the full processed dataset.
    3. Synthetic/demo model bundle (only when no real-data configuration is present).
    """

    # --- Priority 1: Explicit configuration ---
    if settings.p1_model_dir is not None and settings.p1_model_name:
        bundle = load_p1_model_bundle(
            settings.p1_model_dir,
            settings.p1_model_name,
            settings.p1_model_version,
        )
        # Load and attach calibrator if available
        calibrator = _load_calibrator(settings.p1_model_name)
        if calibrator is not None:
            bundle.metadata["calibrator"] = calibrator
            bundle.metadata["calibration_method"] = "sigmoid"
            bundle.metadata["calibration_partition"] = "validation"
        return bundle

    # --- Priority 2: Auto-detect model artifact in default location ---
    # If the data directory is the full PhysioNet processed directory, check for
    # model artifacts in the standard location and load the first available model.
    data_dir = settings.data_dir
    if data_dir is not None:
        default_artifact_dir = Path(__file__).resolve().parents[2] / "artifacts" / "physionet2019" / "full_dataset_retrain"
        if data_dir == Path("data/processed_full") or str(data_dir).endswith("processed_full"):
            # Auto-detect: look for any model subdirectory with a .joblib file
            if default_artifact_dir.is_dir():
                for model_name in sorted(default_artifact_dir.iterdir()):
                    if model_name.is_dir():
                        features_path = model_name / f"{model_name}_1.0.0_features.json"
                        joblib_path = model_name / f"{model_name}_1.0.0.joblib"
                        if features_path.is_file() and joblib_path.is_file():
                            try:
                                bundle = load_p1_model_bundle(
                                    str(default_artifact_dir),
                                    model_name.name,
                                    "1.0.0",
                                )
                                # Load and attach calibrator if available
                                calibrator = _load_calibrator(model_name.name)
                                if calibrator is not None:
                                    bundle.metadata["calibrator"] = calibrator
                                    bundle.metadata["calibration_method"] = "sigmoid"
                                    bundle.metadata["calibration_partition"] = "validation"
                                return bundle
                            except Exception:
                                continue

    # --- Priority 3: Synthetic/demo model ---
    # Demo model only used when no real-data configuration is present
    # and no explicit model artifact path was given.
    return build_demo_model_bundle(
        getattr(repository, "training_patients", lambda: [])()
    )
