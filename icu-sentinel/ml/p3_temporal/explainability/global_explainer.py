"""Global (model-level) SHAP importance.

Importance is the mean absolute SHAP contribution over the model bundle's
background dataset. It describes the fitted model, not clinical causality
and not population-level clinical risk.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from ml.p3_temporal.common.model_bundle import ModelBundle

from .explainer_factory import (
    ExplainerFactory,
    ExplainerNotAvailable,
    ExplanationOutput,
    plan_status,
)
from .explanation_schemas import (
    ExplanationMethod,
    ExplanationModelMetadata,
    ExplanationStatus,
    GlobalExplanation,
    GlobalFeatureGroup,
    GlobalFeatureImportance,
)
from .feature_mapping import FeatureMapper

DEFAULT_MAX_BACKGROUND_ROWS = 100


@dataclass(frozen=True)
class _CacheKey:
    name: str
    version: str
    rows: int


class GlobalExplainer:
    """Compute and cache global SHAP importance for model bundles."""

    def __init__(
        self,
        factory: ExplainerFactory | None = None,
        mapper: FeatureMapper | None = None,
        *,
        max_background_rows: int = DEFAULT_MAX_BACKGROUND_ROWS,
    ) -> None:
        self.factory = factory or ExplainerFactory()
        self.mapper = mapper or FeatureMapper()
        self.max_background_rows = max(1, int(max_background_rows))
        self._cache: dict[_CacheKey, GlobalExplanation] = {}
        self._lock = threading.RLock()

    def explain(self, bundle: ModelBundle) -> GlobalExplanation:
        plan = self.factory.plan(bundle)
        if not plan.supported:
            return GlobalExplanation(
                status=plan_status(plan, bundle),
                model=bundle.identity,
                explanation_method=plan.method,
                output_space=plan.output_space,
                reason=plan.reason,
                metadata=self._metadata(bundle, plan.method, plan.output_space, 0),
            )
        if bundle.background is None or len(bundle.background) == 0:
            return GlobalExplanation(
                status=ExplanationStatus.UNAVAILABLE,
                model=bundle.identity,
                explanation_method=plan.method,
                output_space=plan.output_space,
                reason="Global importance requires a background dataset.",
                metadata=self._metadata(bundle, plan.method, plan.output_space, 0),
            )

        rows = min(self.max_background_rows, len(bundle.background))
        key = _CacheKey(bundle.name, bundle.version, rows)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        background = bundle.background.head(rows)
        try:
            fitted = self.factory.get(bundle)
            output = fitted.explain(background)
        except ExplainerNotAvailable as exc:
            return GlobalExplanation(
                status=exc.status,
                model=bundle.identity,
                explanation_method=plan.method,
                output_space=plan.output_space,
                reason=exc.reason,
                metadata=self._metadata(bundle, plan.method, plan.output_space, 0),
            )

        explanation = self._build(bundle, output, rows)
        with self._lock:
            self._cache[key] = explanation
        return explanation

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------
    def _build(
        self,
        bundle: ModelBundle,
        output: ExplanationOutput,
        rows: int,
    ) -> GlobalExplanation:
        shap_values = output.shap_values
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        mean_signed = np.mean(shap_values, axis=0)

        features: list[GlobalFeatureImportance] = []
        for index in np.argsort(-mean_abs, kind="stable"):
            name = output.feature_names[int(index)]
            metadata = self.mapper.describe(name)
            features.append(
                GlobalFeatureImportance(
                    rank=len(features) + 1,
                    feature_name=name,
                    display_name=metadata.display_name,
                    category=metadata.category,
                    description=metadata.description,
                    unit=metadata.unit,
                    is_missing_indicator=metadata.is_missing_indicator,
                    mean_abs_shap=float(mean_abs[int(index)]),
                    mean_shap=float(mean_signed[int(index)]),
                )
            )

        grouped: dict[str, list[float]] = {}
        for feature in features:
            grouped.setdefault(feature.category, []).append(feature.mean_abs_shap)
        groups = [
            GlobalFeatureGroup(
                category=category,
                mean_abs_shap=float(np.mean(values)),
                n_features=len(values),
            )
            for category, values in grouped.items()
        ]
        groups.sort(key=lambda group: (-group.mean_abs_shap, group.category))

        return GlobalExplanation(
            status=ExplanationStatus.AVAILABLE,
            model=bundle.identity,
            explanation_method=output.method,
            output_space=output.output_space,
            features=features,
            feature_groups=groups,
            background_rows=rows,
            metadata=self._metadata(bundle, output.method, output.output_space, rows),
        )

    def _metadata(
        self,
        bundle: ModelBundle,
        method: ExplanationMethod,
        output_space: str | None,
        rows: int,
    ) -> ExplanationModelMetadata:
        return ExplanationModelMetadata(
            model=bundle.identity,
            explanation_method=method,
            output_space=output_space or "unknown",
            feature_schema_version=bundle.feature_schema_version,
            background_rows=rows,
        )
