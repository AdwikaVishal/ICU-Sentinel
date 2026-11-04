"""Explainability tests: honest availability, ordering and additivity."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml.p3_temporal.common.model_bundle import ModelBundle
from ml.p3_temporal.common.prediction import ModelIdentity, PredictionSource
from ml.p3_temporal.explainability.explainer_factory import (
    ExplainerFactory,
    ExplainerNotAvailable,
    plan_status,
)
from ml.p3_temporal.explainability.explanation_schemas import (
    ExplanationMethod,
    ExplanationStatus,
)
from ml.p3_temporal.explainability.feature_mapping import FeatureMapper
from ml.p3_temporal.explainability.global_explainer import GlobalExplainer
from ml.p3_temporal.explainability.patient_explainer import (
    ExplanationService,
    PatientExplainer,
    validate_additivity,
)
from ml.p3_temporal.replay.prediction_provider import (
    ModelPredictionProvider,
    SyntheticHeuristicProvider,
)
from ml.p3_temporal.replay.replay_engine import ReplayEngine


class TinyKernelModel:
    """Two-feature model with predict_proba for Kernel SHAP coverage."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.asarray(frame, dtype=float)
        logits = 0.9 * values[:, 0] - 0.4 * values[:, 1]
        probability = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probability, probability])


def _tree_bundle(demo_bundle) -> ModelBundle:
    background = demo_bundle.background.copy()
    labels = (background["HR"] > background["HR"].median()).astype(int).to_numpy()
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=12, max_depth=4, random_state=7
                ),
            ),
        ]
    )
    estimator.fit(background[demo_bundle.selected_features], labels)
    return ModelBundle(
        identity=ModelIdentity(
            name="test_random_forest",
            version="test-1.0.0",
            source=PredictionSource.SYNTHETIC_DEMO_MODEL,
            display_name="Test random forest",
            is_synthetic=True,
        ),
        model=estimator,
        selected_features=list(demo_bundle.selected_features),
        background=background,
        feature_schema_version=demo_bundle.feature_schema_version,
    )


def _kernel_bundle() -> ModelBundle:
    rng = np.random.default_rng(3)
    background = pd.DataFrame(
        {
            "x0": rng.normal(size=12),
            "x1": rng.normal(size=12),
        }
    )
    return ModelBundle(
        identity=ModelIdentity(
            name="test_kernel_model",
            version="test-1.0.0",
            source=PredictionSource.SYNTHETIC_DEMO_MODEL,
            display_name="Test kernel model",
            is_synthetic=True,
        ),
        model=TinyKernelModel(),
        selected_features=["x0", "x1"],
        background=background,
        feature_schema_version="test-1.0.0",
    )


def _without_background(bundle: ModelBundle) -> ModelBundle:
    return dataclasses.replace(bundle, background=None)


# ----------------------------------------------------------------------
# Feature mapping
# ----------------------------------------------------------------------
def test_feature_mapping_uses_p1_units_and_readable_names():
    mapper = FeatureMapper()
    hr = mapper.describe("HR")
    assert hr.display_name == "Heart rate"
    assert hr.unit == "beats_per_minute"
    assert hr.category == "current"
    assert hr.metadata_available is True

    rolling = mapper.describe("HR_rolling_mean_6h")
    assert rolling.category == "rolling"
    assert "over 6h" in rolling.display_name
    assert rolling.unit == "beats_per_minute"
    assert rolling.is_engineered is True

    missing = mapper.describe("HR_is_missing")
    assert missing.is_missing_indicator is True
    assert "missing indicator" in missing.display_name

    recency = mapper.describe("HR_hours_since_last")
    assert recency.unit == "hours"
    assert "Hours since last" in recency.display_name


def test_unknown_features_keep_their_name_and_metadata_is_marked_unavailable():
    mapper = FeatureMapper()
    metadata = mapper.describe("mystery_feature_42")
    assert metadata.display_name == "mystery_feature_42"
    assert metadata.category == "unknown"
    assert metadata.description is None
    assert metadata.unit is None
    assert metadata.metadata_available is False


# ----------------------------------------------------------------------
# Explainer factory
# ----------------------------------------------------------------------
def test_linear_explainer_plan_for_demo_model(demo_bundle):
    factory = ExplainerFactory()
    plan = factory.plan(demo_bundle)
    assert plan.supported is True
    assert plan.method == ExplanationMethod.LINEAR_SHAP
    assert plan.output_space == "decision_function_log_odds"


def test_linear_explainer_without_background_is_unavailable(demo_bundle):
    bundle = _without_background(demo_bundle)
    factory = ExplainerFactory()
    plan = factory.plan(bundle)
    assert plan.supported is False
    assert plan_status(plan, bundle) == ExplanationStatus.UNAVAILABLE
    with pytest.raises(ExplainerNotAvailable) as excinfo:
        factory.get(bundle)
    assert excinfo.value.status == ExplanationStatus.UNAVAILABLE


def test_unknown_model_without_background_is_unsupported():
    bundle = ModelBundle(
        identity=ModelIdentity(
            name="custom",
            version="1.0.0",
            source=PredictionSource.SYNTHETIC_DEMO_MODEL,
            is_synthetic=True,
        ),
        model=TinyKernelModel(),
        selected_features=["x0", "x1"],
        background=None,
    )
    factory = ExplainerFactory()
    plan = factory.plan(bundle)
    assert plan.supported is False
    assert plan_status(plan, bundle) == ExplanationStatus.UNAVAILABLE


def test_tree_explainer_is_used_for_tree_models(demo_bundle):
    bundle = _tree_bundle(demo_bundle)
    factory = ExplainerFactory()
    plan = factory.plan(bundle)
    assert plan.supported is True
    assert plan.method == ExplanationMethod.TREE_SHAP

    fitted = factory.get(bundle)
    frame = bundle.background.head(3)
    output = fitted.explain(frame)
    assert output.shap_values.shape == (3, len(bundle.selected_features))
    assert output.base_values.shape == (3,)
    assert validate_additivity(output).passed is True
    assert output.output_space == "probability"


def test_kernel_explainer_is_used_when_no_exact_explainer_exists():
    bundle = _kernel_bundle()
    factory = ExplainerFactory()
    plan = factory.plan(bundle)
    assert plan.supported is True
    assert plan.method == ExplanationMethod.KERNEL_SHAP

    fitted = factory.get(bundle)
    output = fitted.explain(bundle.background.head(2))
    assert output.shap_values.shape == (2, 2)
    assert validate_additivity(output).passed is True


# ----------------------------------------------------------------------
# Patient explanations
# ----------------------------------------------------------------------
def test_patient_explanation_is_real_shap_and_additive(explanation_service):
    explanation = explanation_service.explain("DEMO-0004", 20)
    assert explanation.status == ExplanationStatus.AVAILABLE
    assert explanation.explanation_method == ExplanationMethod.LINEAR_SHAP
    assert explanation.contributions
    assert explanation.validation is not None
    assert explanation.validation.passed is True
    assert explanation.validation.additive_error < 1e-5
    assert explanation.base_value is not None
    assert explanation.model_output is not None
    assert explanation.prediction_probability is not None
    assert explanation.explanation_id


def test_explanation_input_hash_matches_prediction_hash(explanation_service):
    explanation = explanation_service.explain("DEMO-0004", 20)
    assert explanation.prediction is not None
    assert explanation.input_hash == explanation.prediction.model_input_hash


def test_contributions_are_sorted_by_magnitude_with_dense_ranks(explanation_service):
    explanation = explanation_service.explain("DEMO-0002", 30)
    magnitudes = [item.magnitude for item in explanation.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert [item.rank for item in explanation.contributions] == list(
        range(1, len(explanation.contributions) + 1)
    )
    assert all(item.direction == "increases_model_output" for item in explanation.positive_contributions)
    assert all(item.direction == "decreases_model_output" for item in explanation.negative_contributions)


def test_explanation_is_deterministic(explanation_service):
    first = explanation_service.explain("DEMO-0002", 30)
    second = explanation_service.explain("DEMO-0002", 30)
    assert first.explanation_id == second.explanation_id
    assert [c.shap_value for c in first.contributions] == [
        c.shap_value for c in second.contributions
    ]


def test_heuristic_prediction_is_not_given_fake_shap(demo_repository):
    engine = ReplayEngine(demo_repository, SyntheticHeuristicProvider())
    from ml.p3_temporal.common.model_bundle import ModelRegistry

    service = ExplanationService(ModelRegistry(), engine)
    explanation = service.explain("DEMO-0002", 24)
    assert explanation.status == ExplanationStatus.UNSUPPORTED
    assert explanation.contributions == []
    assert explanation.positive_contributions == []
    assert explanation.negative_contributions == []
    assert explanation.base_value is None
    assert "hand-written" in (explanation.reason or "")


def test_missing_hour_explanation_is_unavailable(explanation_service):
    explanation = explanation_service.explain("DEMO-0003", 12)
    assert explanation.status == ExplanationStatus.UNAVAILABLE
    assert explanation.contributions == []
    assert explanation.reason


def test_explanation_is_withheld_when_input_hash_does_not_match(explanation_service, demo_bundle):
    state = explanation_service.replay.get_state("DEMO-0004", 20)
    context = explanation_service.replay.get_model_input("DEMO-0004", 20)
    mismatched = dataclasses.replace(context, input_hash="deliberately-different")
    explainer = PatientExplainer()
    explanation = explainer.explain_context(demo_bundle, mismatched, state.prediction)
    assert explanation.status == ExplanationStatus.ERROR
    assert explanation.contributions == []
    assert "does not match" in (explanation.reason or "")


def test_unregistered_model_is_unsupported(demo_repository, demo_bundle):
    engine = ReplayEngine(demo_repository, ModelPredictionProvider(demo_bundle))
    from ml.p3_temporal.common.model_bundle import ModelRegistry

    service = ExplanationService(ModelRegistry(), engine)
    explanation = service.explain("DEMO-0004", 20)
    assert explanation.status == ExplanationStatus.UNSUPPORTED
    assert "not registered" in (explanation.reason or "")


# ----------------------------------------------------------------------
# Comparison and global explanations
# ----------------------------------------------------------------------
def test_hourly_comparison_reports_signed_changes(explanation_service):
    comparison = explanation_service.compare("DEMO-0004", 16, 20)
    assert comparison.status == ExplanationStatus.AVAILABLE
    assert comparison.changes
    assert comparison.largest_changes == comparison.changes[:8]
    by_name = {item.feature_name: item for item in comparison.changes}
    first = next(iter(by_name.values()))
    assert first.delta == pytest.approx(first.shap_b - first.shap_a)
    assert all(item.magnitude == abs(item.delta) for item in comparison.changes)


def test_hourly_comparison_with_missing_hour_is_unavailable(explanation_service):
    comparison = explanation_service.compare("DEMO-0003", 11, 12)
    assert comparison.status == ExplanationStatus.UNAVAILABLE
    assert comparison.changes == []
    assert comparison.reason


def test_global_explanation_is_ranked_and_grouped(explanation_service):
    global_explanation = explanation_service.global_explanation()
    assert global_explanation.status == ExplanationStatus.AVAILABLE
    assert global_explanation.background_rows > 0
    magnitudes = [feature.mean_abs_shap for feature in global_explanation.features]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert [feature.rank for feature in global_explanation.features] == list(
        range(1, len(global_explanation.features) + 1)
    )
    assert sum(group.n_features for group in global_explanation.feature_groups) == len(
        global_explanation.features
    )
    assert global_explanation.explanation_method == ExplanationMethod.LINEAR_SHAP


def test_global_explanation_without_background_is_unavailable(demo_bundle):
    global_explainer = GlobalExplainer()
    explanation = global_explainer.explain(_without_background(demo_bundle))
    assert explanation.status == ExplanationStatus.UNAVAILABLE
    assert explanation.features == []


def test_model_catalog_reports_explainability(explanation_service):
    catalog = explanation_service.catalog()
    assert catalog.count == 1
    entry = catalog.models[0]
    assert entry.explainable is True
    assert entry.explanation_method == ExplanationMethod.LINEAR_SHAP
    assert entry.background_rows > 0
