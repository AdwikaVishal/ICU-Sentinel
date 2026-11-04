"""Patient-hour and hourly-comparison explanations plus the service layer."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ml.p3_temporal.common.model_bundle import ModelBundle, ModelRegistry
from ml.p3_temporal.common.prediction import (
    PredictionContext,
    PredictionResult,
    PredictionSource,
    canonical_json,
    safe_number,
)
from ml.p3_temporal.replay.replay_engine import ReplayEngine

from .explainer_factory import (
    ExplainerFactory,
    ExplainerNotAvailable,
    ExplanationOutput,
    FittedExplainer,
    plan_status,
)
from .explanation_schemas import (
    ExplanationMethod,
    ExplanationModelMetadata,
    ExplanationStatus,
    ExplanationValidation,
    FeatureChange,
    FeatureContribution,
    GlobalExplanation,
    HourlyExplanationComparison,
    ModelCatalogEntry,
    ModelCatalogResponse,
    PatientExplanation,
)
from .feature_mapping import FeatureMapper
from .global_explainer import GlobalExplainer

_EPSILON = 1e-12

_ADDITIVITY_TOLERANCE: dict[ExplanationMethod, float] = {
    ExplanationMethod.LINEAR_SHAP: 1e-5,
    ExplanationMethod.TREE_SHAP: 1e-5,
    ExplanationMethod.KERNEL_SHAP: 5e-2,
}


def validate_additivity(output: ExplanationOutput, row: int = 0) -> ExplanationValidation:
    """Check base value + summed SHAP against the model output."""

    tolerance = _ADDITIVITY_TOLERANCE.get(output.method, 1e-4)
    base = float(output.base_values[row])
    values = np.asarray(output.shap_values[row], dtype=float)
    model_output = float(output.model_output[row])
    reconstructed = base + float(np.sum(values))
    error = abs(reconstructed - model_output)
    scale = max(1.0, abs(model_output))
    passed = bool(error <= tolerance * scale)
    return ExplanationValidation(
        additive_error=error,
        tolerance=tolerance,
        passed=passed,
        checked_output_space=output.output_space,
        explanation_output=reconstructed,
        note=(
            "Additivity checked as base_value + sum(SHAP) against the model "
            "output in the stated space."
            + (
                ""
                if passed
                else " The check did not pass; treat the contribution split with caution."
            )
        ),
    )


class PatientExplainer:
    """Build patient-hour explanations from a model bundle and causal input."""

    def __init__(
        self,
        factory: ExplainerFactory | None = None,
        mapper: FeatureMapper | None = None,
    ) -> None:
        self.factory = factory or ExplainerFactory()
        self.mapper = mapper or FeatureMapper()

    def explain_context(
        self,
        bundle: ModelBundle,
        context: PredictionContext,
        prediction: PredictionResult | None = None,
    ) -> PatientExplanation:
        if prediction is not None and not prediction.available:
            return _unavailable(
                context,
                prediction,
                ExplanationStatus.UNAVAILABLE,
                prediction.unavailable_reason or "Prediction is unavailable.",
            )

        if prediction is not None and prediction.source in {
            PredictionSource.SYNTHETIC_HEURISTIC,
            PredictionSource.UNAVAILABLE,
        }:
            return _unavailable(
                context,
                prediction,
                ExplanationStatus.UNSUPPORTED,
                (
                    "SHAP explanations are only available for fitted models. "
                    "This prediction came from a hand-written demonstration score."
                ),
            )

        if prediction is not None and context.input_hash != prediction.model_input_hash:
            return _unavailable(
                context,
                prediction,
                ExplanationStatus.ERROR,
                (
                    "The explanation input does not match the prediction input "
                    "hash. The explanation was withheld rather than attributed "
                    "to a different input."
                ),
            )

        if context.feature_values is None:
            return _unavailable(
                context,
                prediction,
                ExplanationStatus.UNAVAILABLE,
                "No feature vector was constructed for this hour.",
            )

        missing = [
            feature
            for feature in bundle.selected_features
            if feature not in context.feature_values
        ]
        if missing:
            return _unavailable(
                context,
                prediction,
                ExplanationStatus.UNAVAILABLE,
                f"Explanation input is missing {len(missing)} model feature(s).",
            )

        plan = self.factory.plan(bundle)
        if not plan.supported:
            return _unavailable(
                context,
                prediction,
                plan_status(plan, bundle),
                plan.reason or "This model cannot be explained.",
                method=plan.method,
                output_space=plan.output_space,
            )

        frame = pd.DataFrame(
            [
                {
                    feature: _feature_float(context.feature_values[feature])
                    for feature in bundle.selected_features
                }
            ],
            columns=bundle.selected_features,
        )
        try:
            fitted = self.factory.get(bundle)
            output = fitted.explain(frame)
        except ExplainerNotAvailable as exc:
            return _unavailable(
                context,
                prediction,
                exc.status,
                exc.reason,
                method=plan.method,
                output_space=plan.output_space,
            )
        except Exception as exc:  # noqa: BLE001 - never fabricate an explanation
            return _unavailable(
                context,
                prediction,
                ExplanationStatus.ERROR,
                f"Explanation failed: {type(exc).__name__}: {exc}",
                method=plan.method,
                output_space=plan.output_space,
            )

        return self._build_explanation(bundle, context, prediction, fitted, output)

    # ------------------------------------------------------------------
    def _build_explanation(
        self,
        bundle: ModelBundle,
        context: PredictionContext,
        prediction: PredictionResult | None,
        fitted: FittedExplainer,
        output: ExplanationOutput,
    ) -> PatientExplanation:
        raw_values = {
            feature: _feature_float(context.feature_values[feature])  # type: ignore[index]
            for feature in output.feature_names
        }
        shap_row = output.shap_values[0]

        contributions: list[FeatureContribution] = []
        for index, feature in enumerate(output.feature_names):
            metadata = self.mapper.describe(feature)
            shap_value = float(shap_row[index])
            contributions.append(
                FeatureContribution(
                    feature_name=feature,
                    display_name=metadata.display_name,
                    category=metadata.category,
                    description=metadata.description,
                    unit=metadata.unit,
                    is_missing_indicator=metadata.is_missing_indicator,
                    value=safe_number(raw_values.get(feature)),
                    shap_value=shap_value,
                    magnitude=abs(shap_value),
                    direction=_direction(shap_value),
                    rank=1,
                )
            )
        contributions.sort(key=lambda item: (-item.magnitude, item.feature_name))
        for rank, contribution in enumerate(contributions, start=1):
            contribution.rank = rank

        positive = [item for item in contributions if item.shap_value > _EPSILON]
        negative = [item for item in contributions if item.shap_value < -_EPSILON]

        base_value = float(output.base_values[0])
        model_output = float(output.model_output[0])
        validation = validate_additivity(output)
        explanation_id = _explanation_id(
            patient_id=context.patient_id,
            hour=context.hour,
            bundle=bundle,
            input_hash=context.input_hash,
            method=output.method,
        )

        return PatientExplanation(
            status=ExplanationStatus.AVAILABLE,
            patient_id=context.patient_id,
            hour=context.hour,
            prediction=prediction,
            model=bundle.identity,
            explanation_method=output.method,
            output_space=output.output_space,
            base_value=base_value,
            model_output=model_output,
            prediction_probability=prediction.probability if prediction else None,
            contributions=contributions,
            positive_contributions=positive,
            negative_contributions=negative,
            validation=validation,
            input_hash=context.input_hash,
            explanation_id=explanation_id,
            metadata=ExplanationModelMetadata(
                model=bundle.identity,
                explanation_method=output.method,
                output_space=output.output_space,
                feature_schema_version=bundle.feature_schema_version,
                background_rows=(
                    int(len(bundle.background)) if bundle.background is not None else 0
                ),
            ),
        )


class ExplanationService:
    """Resolve predictions, model bundles and explanations for the API."""

    def __init__(
        self,
        registry: ModelRegistry,
        replay_engine: ReplayEngine,
        *,
        factory: ExplainerFactory | None = None,
        mapper: FeatureMapper | None = None,
        global_explainer: GlobalExplainer | None = None,
    ) -> None:
        self.registry = registry
        self.replay = replay_engine
        self.factory = factory or ExplainerFactory()
        self.mapper = mapper or FeatureMapper()
        self.patient_explainer = PatientExplainer(self.factory, self.mapper)
        self.global_explainer = global_explainer or GlobalExplainer(
            self.factory, self.mapper
        )

    # ------------------------------------------------------------------
    def explain(self, patient_id: str, hour: int) -> PatientExplanation:
        state = self.replay.get_state(patient_id, hour, evaluate_alerts=False)
        prediction = state.prediction
        if not prediction.available:
            return PatientExplanation(
                status=ExplanationStatus.UNAVAILABLE,
                patient_id=patient_id,
                hour=state.hour,
                prediction=prediction,
                reason=prediction.unavailable_reason or "Prediction is unavailable.",
            )

        if prediction.source in {
            PredictionSource.SYNTHETIC_HEURISTIC,
            PredictionSource.UNAVAILABLE,
        }:
            return PatientExplanation(
                status=ExplanationStatus.UNSUPPORTED,
                patient_id=patient_id,
                hour=state.hour,
                prediction=prediction,
                reason=(
                    "SHAP explanations require a fitted model. The active "
                    "prediction provider is a hand-written demonstration score."
                ),
            )

        bundle = self.registry.get(prediction.model.name, prediction.model.version)
        if bundle is None:
            return PatientExplanation(
                status=ExplanationStatus.UNSUPPORTED,
                patient_id=patient_id,
                hour=state.hour,
                prediction=prediction,
                reason=(
                    "The model used for this prediction is not registered for "
                    "explanation, so no SHAP values are available."
                ),
            )

        context = self.replay.get_model_input(patient_id, state.hour)
        return self.patient_explainer.explain_context(bundle, context, prediction)

    # ------------------------------------------------------------------
    def compare(
        self,
        patient_id: str,
        hour_a: int,
        hour_b: int,
    ) -> HourlyExplanationComparison:
        explanation_a = self.explain(patient_id, hour_a)
        explanation_b = self.explain(patient_id, hour_b)
        if (
            explanation_a.status != ExplanationStatus.AVAILABLE
            or explanation_b.status != ExplanationStatus.AVAILABLE
        ):
            failing = (
                explanation_a
                if explanation_a.status != ExplanationStatus.AVAILABLE
                else explanation_b
            )
            return HourlyExplanationComparison(
                status=failing.status,
                patient_id=patient_id,
                hour_a=hour_a,
                hour_b=hour_b,
                explanation_a=explanation_a,
                explanation_b=explanation_b,
                reason=failing.reason or "One of the selected hours cannot be explained.",
            )

        changes = _build_changes(explanation_a, explanation_b)
        return HourlyExplanationComparison(
            status=ExplanationStatus.AVAILABLE,
            patient_id=patient_id,
            hour_a=hour_a,
            hour_b=hour_b,
            explanation_a=explanation_a,
            explanation_b=explanation_b,
            changes=changes,
            largest_changes=changes[:8],
        )

    # ------------------------------------------------------------------
    def global_explanation(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> GlobalExplanation:
        bundle = self._resolve_bundle(model_name, model_version)
        if bundle is None:
            return GlobalExplanation(
                status=ExplanationStatus.UNAVAILABLE,
                reason="No registered model bundle matches the request.",
            )
        return self.global_explainer.explain(bundle)

    def catalog(self) -> ModelCatalogResponse:
        entries: list[ModelCatalogEntry] = []
        for bundle in self.registry.list_bundles():
            plan = self.factory.plan(bundle)
            entries.append(
                ModelCatalogEntry(
                    model=bundle.identity,
                    explainable=plan.supported,
                    explanation_method=plan.method,
                    reason=plan.reason,
                    background_rows=(
                        int(len(bundle.background)) if bundle.background is not None else 0
                    ),
                    feature_schema_version=bundle.feature_schema_version,
                    notes=bundle.notes,
                )
            )
        return ModelCatalogResponse(models=entries, count=len(entries))

    def _resolve_bundle(
        self, model_name: str | None, model_version: str | None
    ) -> ModelBundle | None:
        if model_name is None:
            return self.registry.latest()
        if model_version is None:
            return self.registry.latest(model_name)
        return self.registry.get(model_name, model_version)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _feature_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _direction(shap_value: float) -> str:
    if shap_value > _EPSILON:
        return "increases_model_output"
    if shap_value < -_EPSILON:
        return "decreases_model_output"
    return "neutral"


def _explanation_id(
    *,
    patient_id: str,
    hour: int,
    bundle: ModelBundle,
    input_hash: str,
    method: ExplanationMethod,
) -> str:
    payload = {
        "patient_id": patient_id,
        "hour": hour,
        "model": bundle.name,
        "version": bundle.version,
        "input_hash": input_hash,
        "method": method.value,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return "exp_" + digest[:24]


def _unavailable(
    context: PredictionContext,
    prediction: PredictionResult | None,
    status: ExplanationStatus,
    reason: str,
    *,
    method: ExplanationMethod = ExplanationMethod.NOT_APPLICABLE,
    output_space: str | None = None,
) -> PatientExplanation:
    return PatientExplanation(
        status=status,
        patient_id=context.patient_id,
        hour=context.hour,
        prediction=prediction,
        model=prediction.model if prediction is not None else None,
        explanation_method=method,
        output_space=output_space,
        input_hash=context.input_hash,
        reason=reason,
    )


def _build_changes(
    explanation_a: PatientExplanation,
    explanation_b: PatientExplanation,
) -> list[FeatureChange]:
    map_a = {item.feature_name: item for item in explanation_a.contributions}
    map_b = {item.feature_name: item for item in explanation_b.contributions}
    names = sorted(set(map_a) | set(map_b))
    changes: list[FeatureChange] = []
    for name in names:
        item_a = map_a.get(name)
        item_b = map_b.get(name)
        shap_a = item_a.shap_value if item_a is not None else 0.0
        shap_b = item_b.shap_value if item_b is not None else 0.0
        delta = shap_b - shap_a
        reference = item_b or item_a
        assert reference is not None
        changes.append(
            FeatureChange(
                feature_name=name,
                display_name=reference.display_name,
                category=reference.category,
                shap_a=shap_a,
                shap_b=shap_b,
                delta=delta,
                magnitude=abs(delta),
                direction=(
                    "increased"
                    if delta > _EPSILON
                    else ("decreased" if delta < -_EPSILON else "unchanged")
                ),
            )
        )
    changes.sort(key=lambda item: (-item.magnitude, item.feature_name))
    return changes
