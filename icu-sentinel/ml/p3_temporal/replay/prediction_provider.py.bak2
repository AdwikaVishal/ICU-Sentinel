"""Prediction provider abstraction and concrete providers.

Providers are replaceable through dependency injection. Replay logic never
imports a particular model type.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ml.p3_temporal.common.model_bundle import ModelBundle
from ml.p3_temporal.common.prediction import (
    ModelIdentity,
    PredictionContext,
    PredictionProvider,
    PredictionResult,
    PredictionSource,
    ProviderStatus,
    make_prediction_id,
    sanitize_feature_values,
    unavailable_prediction,
)


class BasePredictionProvider:
    """Common deterministic prediction-record construction."""

    def __init__(self, identity: ModelIdentity) -> None:
        self._identity = identity

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    @property
    def required_features(self) -> list[str] | None:
        return None

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            available=True,
            provider_name=self._identity.name,
            model=self._identity,
            required_features=self.required_features,
        )

    def _available_result(
        self,
        context: PredictionContext,
        probability: float,
    ) -> PredictionResult:
        probability = float(min(1.0, max(0.0, probability)))
        return PredictionResult(
            patient_id=context.patient_id,
            hour=context.hour,
            probability=probability,
            predicted_class=None,
            threshold=None,
            model=self._identity,
            source=self._identity.source,
            available=True,
            prediction_id=make_prediction_id(
                patient_id=context.patient_id,
                hour=context.hour,
                model_name=self._identity.name,
                model_version=self._identity.version,
                input_hash=context.input_hash,
            ),
            feature_schema_version=context.feature_schema_version,
            model_input_hash=context.input_hash,
            is_synthetic=self._identity.is_synthetic,
        )

    def _unavailable(
        self,
        context: PredictionContext,
        reason: str,
    ) -> PredictionResult:
        return unavailable_prediction(
            patient_id=context.patient_id,
            hour=context.hour,
            identity=self._identity,
            reason=reason,
            feature_schema_version=context.feature_schema_version,
            input_hash=context.input_hash,
        )


class SyntheticHeuristicProvider(BasePredictionProvider):
    """Deterministic hand-written demonstration score.

    This is NOT a trained model. It exists so replay can be exercised
    before model artifacts are available. It is explicitly labelled, it
    does not use the retrospective label, and SHAP explanations are not
    available for it.
    """

    def __init__(self) -> None:
        super().__init__(
            ModelIdentity(
                name="synthetic-heuristic-demo",
                version="1.0.0",
                source=PredictionSource.SYNTHETIC_HEURISTIC,
                display_name="Synthetic demonstration heuristic (not a trained model)",
                is_synthetic=True,
                feature_schema_version="demo-observations-v1",
            )
        )

    @property
    def required_features(self) -> list[str] | None:
        return None

    def predict(self, context: PredictionContext) -> PredictionResult:
        if not context.observations:
            return self._unavailable(context, "No recorded observation at this hour.")
        values = {
            observation.parameter: float(observation.value)
            for observation in context.observations
            if observation.available and observation.value is not None
        }
        if len(values) < 3:
            return self._unavailable(
                context,
                "Too few available observations at this hour to apply the demo score.",
            )

        z = -5.2
        contributions = (
            ("HR", 0.032, 80.0),
            ("Temp", 0.32, 37.0),
            ("Resp", 0.045, 16.0),
            ("SBP", -0.018, 120.0),
            ("O2Sat", -0.070, 97.0),
            ("Lactate", 0.23, 1.0),
            ("WBC", 0.055, 8.0),
        )
        used = 0
        for variable, coefficient, baseline in contributions:
            if variable in values:
                z += coefficient * (values[variable] - baseline)
                used += 1
        if used == 0:
            return self._unavailable(
                context,
                "None of the demo score variables were measured at this hour.",
            )
        probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        return self._available_result(context, probability)


class ModelPredictionProvider(BasePredictionProvider):
    """Prediction provider backed by a fitted scikit-learn-compatible model."""

    def __init__(self, bundle: ModelBundle) -> None:
        super().__init__(bundle.identity)
        self.bundle = bundle

    @property
    def required_features(self) -> list[str] | None:
        return list(self.bundle.selected_features)

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            available=True,
            provider_name=f"model:{self.bundle.name}",
            model=self._identity,
            required_features=self.required_features,
        )

    def predict(self, context: PredictionContext) -> PredictionResult:
        if context.feature_values is None:
            return self._unavailable(
                context,
                "No recorded observation at this hour; no model input was constructed.",
            )
        missing = [
            feature
            for feature in self.bundle.selected_features
            if feature not in context.feature_values
        ]
        if missing:
            return self._unavailable(
                context,
                f"Model input is missing {len(missing)} required feature(s).",
            )

        values = sanitize_feature_values(
            {feature: context.feature_values[feature] for feature in self.bundle.selected_features}
        )
        frame = pd.DataFrame([values], columns=self.bundle.selected_features)
        try:
            probabilities = self.bundle.model.predict_proba(frame)
        except Exception as exc:  # noqa: BLE001 - model errors must not break replay
            return self._unavailable(
                context,
                f"Model prediction failed: {type(exc).__name__}: {exc}",
            )

        array = np.asarray(probabilities)
        if array.ndim != 2 or array.shape[1] < 2:
            return self._unavailable(context, "Model did not return binary probabilities.")
        probability = float(array[0, 1])
        # Apply calibrated sigmoid if a calibrator is available in bundle metadata
        calibrator = self.bundle.metadata.get("calibrator") if hasattr(self.bundle, "metadata") else None
        if calibrator is not None:
            try:
                from ml.p2_boosting.calibration import apply_probability_calibrator
                # Apply calibrator to the raw probability array, then take first element
                calibrated = apply_probability_calibrator(calibrator, array[:, 1])
                probability = float(calibrated[0])
            except Exception:
                # If calibration fails, fall through with uncalibrated probability
                pass
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            return self._unavailable(context, "Model returned a non-finite probability.")
        return self._available_result(context, probability)


class UnavailableProvider(BasePredictionProvider):
    """Explicit provider used when no prediction model is available."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            ModelIdentity(
                name="no-model",
                version="0.0.0",
                source=PredictionSource.UNAVAILABLE,
                display_name="No prediction model available",
                is_synthetic=False,
            )
        )
        self.reason = reason

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            available=False,
            provider_name="unavailable",
            model=self._identity,
            reason=self.reason,
        )

    def predict(self, context: PredictionContext) -> PredictionResult:
        return self._unavailable(context, self.reason)


def provider_from_bundle(bundle: ModelBundle | None) -> PredictionProvider:
    """Choose a model provider when a bundle exists, otherwise unavailable."""

    if bundle is None:
        return UnavailableProvider("No model artifact or demonstration model is available.")
    return ModelPredictionProvider(bundle)


def safe_probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number
