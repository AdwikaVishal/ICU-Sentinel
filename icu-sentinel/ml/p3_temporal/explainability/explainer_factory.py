"""SHAP explainer selection and thin, validated wrappers.

The factory never fabricates explanation values. If a model family cannot
be explained, or if the required background dataset is missing, the
factory returns an explicit ``unsupported``/``unavailable`` plan instead
of a placeholder explanation.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.p3_temporal.common.model_bundle import ModelBundle
from ml.p3_temporal.common.prediction import sanitize_feature_values

from .explanation_schemas import ExplanationMethod, ExplanationStatus

OUTPUT_SPACE_DECISION = "decision_function_log_odds"
OUTPUT_SPACE_PROBABILITY = "probability"
OUTPUT_SPACE_RAW = "raw_model_output"

_LINEAR_CLASS_NAMES = {
    "LogisticRegression",
    "LogisticRegressionCV",
    "RidgeClassifier",
    "RidgeClassifierCV",
    "LinearSVC",
    "SGDClassifier",
    "Perceptron",
    "PassiveAggressiveClassifier",
}

_TREE_CLASS_NAMES = {
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
}


_SHAP_MODULE = None


def _load_shap():
    """Import SHAP without letting a broken optional torch install abort it.

    Some SHAP versions probe ``torch`` during import and only catch
    ``ImportError``. A corrupted CPU torch install raises ``OSError``
    (WinError 1114, c10.dll) instead. This repo does not use torch, so a
    broken torch is masked for the import only; SHAP's non-image explainers
    continue to work.
    """

    global _SHAP_MODULE
    if _SHAP_MODULE is not None:
        return _SHAP_MODULE

    # This project never imports torch. Block SHAP's optional probe before the
    # import so a broken torch DLL (which can crash the interpreter rather than
    # raise ImportError) cannot take down the process.
    if (
        os.environ.get("ICU_SENTINEL_ENABLE_TORCH") != "1"
        and "torch" not in sys.modules
    ):
        sys.modules["torch"] = None

    try:
        import shap as shap_module
    except OSError as exc:
        message = str(exc).lower()
        if "torch" not in message and "c10" not in message:
            raise
        sys.modules["torch"] = None
        import shap as shap_module

    _SHAP_MODULE = shap_module
    return _SHAP_MODULE


class ExplainerNotAvailable(Exception):
    """Raised when an explainer cannot be built for a model bundle."""

    def __init__(self, status: ExplanationStatus, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class ExplainerPlan:
    supported: bool
    method: ExplanationMethod
    reason: str | None = None
    output_space: str | None = None
    requires_background: bool = False


def plan_status(plan: ExplainerPlan, bundle: ModelBundle) -> ExplanationStatus:
    """Translate an explainer plan into an explicit public status."""

    if plan.supported:
        return ExplanationStatus.AVAILABLE
    if plan.requires_background and not _has_background(bundle):
        return ExplanationStatus.UNAVAILABLE
    return ExplanationStatus.UNSUPPORTED


@dataclass(frozen=True)
class ExplanationOutput:
    """Validated SHAP output for one or more model inputs."""

    shap_values: np.ndarray  # shape (n_rows, n_features)
    base_values: np.ndarray  # shape (n_rows,)
    model_output: np.ndarray  # shape (n_rows,)
    method: ExplanationMethod
    output_space: str
    feature_names: tuple[str, ...]


class FittedExplainer:
    """A built SHAP explainer bound to one model bundle."""

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        method: ExplanationMethod,
        output_space: str,
        explain: Callable[[pd.DataFrame], np.ndarray],
        predict_output: Callable[[pd.DataFrame], np.ndarray],
        base_values: Callable[[int], np.ndarray],
    ) -> None:
        self.bundle = bundle
        self.method = method
        self.output_space = output_space
        self.feature_names = tuple(bundle.selected_features)
        self._explain = explain
        self._predict_output = predict_output
        self._base_values = base_values

    def explain(self, frame: pd.DataFrame) -> ExplanationOutput:
        prepared = prepare_feature_frame(frame, self.feature_names)
        values = np.asarray(self._explain(prepared), dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ExplainerNotAvailable(
                ExplanationStatus.ERROR,
                "SHAP returned an unexpected contribution shape.",
            )
        if not np.all(np.isfinite(values)):
            raise ExplainerNotAvailable(
                ExplanationStatus.ERROR,
                "SHAP returned non-finite contribution values.",
            )
        output = np.asarray(self._predict_output(prepared), dtype=float).reshape(-1)
        if output.shape[0] != values.shape[0]:
            raise ExplainerNotAvailable(
                ExplanationStatus.ERROR,
                "Model output length does not match the explanation rows.",
            )
        base = np.asarray(self._base_values(values.shape[0]), dtype=float).reshape(-1)
        if base.shape[0] == 1 and values.shape[0] > 1:
            base = np.repeat(base, values.shape[0])
        if base.shape[0] != values.shape[0]:
            raise ExplainerNotAvailable(
                ExplanationStatus.ERROR,
                "SHAP base values do not match the explanation rows.",
            )
        return ExplanationOutput(
            shap_values=values,
            base_values=base,
            model_output=output,
            method=self.method,
            output_space=self.output_space,
            feature_names=self.feature_names,
        )


class ExplainerFactory:
    """Choose and cache a SHAP explainer for a model bundle."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], FittedExplainer] = {}
        self._plans: dict[tuple[str, str], ExplainerPlan] = {}
        self._lock = threading.RLock()

    def plan(self, bundle: ModelBundle) -> ExplainerPlan:
        key = (bundle.name, bundle.version)
        with self._lock:
            if key in self._plans:
                return self._plans[key]
            plan = self._build_plan(bundle)
            self._plans[key] = plan
            return plan

    def get(self, bundle: ModelBundle) -> FittedExplainer:
        key = (bundle.name, bundle.version)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            plan = self.plan(bundle)
            if not plan.supported:
                raise ExplainerNotAvailable(
                    plan_status(plan, bundle),
                    plan.reason or "This model family cannot be explained here.",
                )
            try:
                fitted = self._build(bundle, plan)
            except ExplainerNotAvailable:
                raise
            except Exception as exc:  # noqa: BLE001 - explainer build must be honest
                raise ExplainerNotAvailable(
                    ExplanationStatus.UNAVAILABLE,
                    f"Could not build a SHAP explainer: {type(exc).__name__}: {exc}",
                ) from exc
            self._cache[key] = fitted
            return fitted

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._plans.clear()

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _build_plan(self, bundle: ModelBundle) -> ExplainerPlan:
        if bundle.model is None:
            return ExplainerPlan(
                supported=False,
                method=ExplanationMethod.NOT_APPLICABLE,
                reason="No fitted model object is available.",
            )
        if not bundle.selected_features:
            return ExplainerPlan(
                supported=False,
                method=ExplanationMethod.NOT_APPLICABLE,
                reason="The model bundle defines no ordered feature list.",
            )

        estimator = unwrap_estimator(bundle.model)
        n_classes = _n_classes(estimator)
        if n_classes is not None and n_classes != 2:
            return ExplainerPlan(
                supported=False,
                method=ExplanationMethod.NOT_APPLICABLE,
                reason="Only binary sepsis models are supported for explanation.",
            )

        if is_linear_model(estimator):
            if not hasattr(estimator, "decision_function"):
                return ExplainerPlan(
                    supported=False,
                    method=ExplanationMethod.LINEAR_SHAP,
                    reason="Linear model does not expose decision_function.",
                )
            if not _has_background(bundle):
                return ExplainerPlan(
                    supported=False,
                    method=ExplanationMethod.LINEAR_SHAP,
                    reason=(
                        "Linear SHAP requires a background dataset; this model "
                        "bundle does not include one."
                    ),
                    output_space=OUTPUT_SPACE_DECISION,
                    requires_background=True,
                )
            if not _preprocessing_preserves_features(bundle):
                return ExplainerPlan(
                    supported=False,
                    method=ExplanationMethod.LINEAR_SHAP,
                    reason=(
                        "Model preprocessing changes the feature count, so SHAP "
                        "contributions cannot be attributed to the original features."
                    ),
                )
            return ExplainerPlan(
                supported=True,
                method=ExplanationMethod.LINEAR_SHAP,
                output_space=OUTPUT_SPACE_DECISION,
                requires_background=True,
            )

        if is_tree_model(estimator):
            if not _preprocessing_preserves_features(bundle):
                return ExplainerPlan(
                    supported=False,
                    method=ExplanationMethod.TREE_SHAP,
                    reason=(
                        "Model preprocessing changes the feature count, so SHAP "
                        "contributions cannot be attributed to the original features."
                    ),
                )
            return ExplainerPlan(
                supported=True,
                method=ExplanationMethod.TREE_SHAP,
                output_space=OUTPUT_SPACE_PROBABILITY,
            )

        if not _has_background(bundle):
            return ExplainerPlan(
                supported=False,
                method=ExplanationMethod.KERNEL_SHAP,
                reason=(
                    "This model family has no exact SHAP explainer here, and no "
                    "background dataset is available for Kernel SHAP."
                ),
                requires_background=True,
            )
        if not hasattr(bundle.model, "predict") and not hasattr(bundle.model, "predict_proba"):
            return ExplainerPlan(
                supported=False,
                method=ExplanationMethod.KERNEL_SHAP,
                reason="The model object has no predict or predict_proba method.",
            )
        return ExplainerPlan(
            supported=True,
            method=ExplanationMethod.KERNEL_SHAP,
            output_space=OUTPUT_SPACE_PROBABILITY,
            requires_background=True,
        )

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def _build(self, bundle: ModelBundle, plan: ExplainerPlan) -> FittedExplainer:
        if plan.method == ExplanationMethod.LINEAR_SHAP:
            return self._build_linear(bundle)
        if plan.method == ExplanationMethod.TREE_SHAP:
            return self._build_tree(bundle)
        if plan.method == ExplanationMethod.KERNEL_SHAP:
            return self._build_kernel(bundle)
        raise ExplainerNotAvailable(
            ExplanationStatus.UNSUPPORTED, "No SHAP method applies to this model."
        )

    def _build_linear(self, bundle: ModelBundle) -> FittedExplainer:
        preprocessor = make_preprocessor(bundle.model)
        estimator = unwrap_estimator(bundle.model)
        background = bundle.background
        assert background is not None

        background_frame = prepare_feature_frame(background, bundle.selected_features)
        background_matrix = transform(preprocessor, background_frame)
        masker = _load_shap().maskers.Independent(background_matrix, max_samples=len(background_matrix))
        explainer = _load_shap().LinearExplainer(estimator, masker=masker)

        def _explain(frame: pd.DataFrame) -> np.ndarray:
            matrix = transform(preprocessor, frame)
            values = np.asarray(explainer.shap_values(matrix), dtype=float)
            return values

        def _predict(frame: pd.DataFrame) -> np.ndarray:
            matrix = transform(preprocessor, frame)
            return np.asarray(estimator.decision_function(matrix), dtype=float)

        expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)

        def _base(n_rows: int) -> np.ndarray:
            return _select_binary_base(expected)

        return FittedExplainer(
            bundle,
            method=ExplanationMethod.LINEAR_SHAP,
            output_space=OUTPUT_SPACE_DECISION,
            explain=_explain,
            predict_output=_predict,
            base_values=_base,
        )

    def _build_tree(self, bundle: ModelBundle) -> FittedExplainer:
        preprocessor = make_preprocessor(bundle.model)
        estimator = unwrap_estimator(bundle.model)
        module = type(estimator).__module__
        external_booster = module.startswith(("xgboost", "lightgbm", "catboost"))

        if external_booster:
            try:
                explainer = _load_shap().TreeExplainer(estimator, model_output="probability")
                output_space = OUTPUT_SPACE_PROBABILITY
            except Exception:  # noqa: BLE001 - fall back to the library default
                explainer = _load_shap().TreeExplainer(estimator)
                output_space = OUTPUT_SPACE_RAW
        else:
            # scikit-learn tree ensembles expose probability-like leaf
            # values for classifiers, so the default tree output is the
            # probability used by predict_proba.
            explainer = _load_shap().TreeExplainer(estimator)
            output_space = (
                OUTPUT_SPACE_PROBABILITY
                if hasattr(estimator, "predict_proba")
                else OUTPUT_SPACE_RAW
            )

        def _explain(frame: pd.DataFrame) -> np.ndarray:
            matrix = transform(preprocessor, frame)
            values = explainer.shap_values(matrix)
            return _select_binary_shap(values)

        def _predict(frame: pd.DataFrame) -> np.ndarray:
            matrix = transform(preprocessor, frame)
            if output_space == OUTPUT_SPACE_PROBABILITY and hasattr(estimator, "predict_proba"):
                return np.asarray(estimator.predict_proba(matrix), dtype=float)[:, 1]
            return _raw_tree_output(estimator, matrix)

        expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)

        def _base(n_rows: int) -> np.ndarray:
            return _select_binary_base(expected)

        return FittedExplainer(
            bundle,
            method=ExplanationMethod.TREE_SHAP,
            output_space=output_space,
            explain=_explain,
            predict_output=_predict,
            base_values=_base,
        )

    def _build_kernel(self, bundle: ModelBundle) -> FittedExplainer:
        background = bundle.background
        assert background is not None
        background_frame = prepare_feature_frame(background, bundle.selected_features)
        model = bundle.model

        def _probability(data: Any) -> np.ndarray:
            frame = _as_feature_frame(data, bundle.selected_features)
            if hasattr(model, "predict_proba"):
                probabilities = np.asarray(model.predict_proba(frame), dtype=float)
                if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                    return probabilities[:, 1]
            return np.asarray(model.predict(frame), dtype=float)

        explainer = _load_shap().KernelExplainer(_probability, background_frame)

        def _explain(frame: pd.DataFrame) -> np.ndarray:
            values = explainer.shap_values(frame, nsamples=256, silent=True)
            return _select_binary_shap(values)

        def _predict(frame: pd.DataFrame) -> np.ndarray:
            return np.asarray(_probability(frame), dtype=float)

        expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)

        def _base(n_rows: int) -> np.ndarray:
            return _select_binary_base(expected)

        return FittedExplainer(
            bundle,
            method=ExplanationMethod.KERNEL_SHAP,
            output_space=OUTPUT_SPACE_PROBABILITY,
            explain=_explain,
            predict_output=_predict,
            base_values=_base,
        )


# ----------------------------------------------------------------------
# Model introspection helpers
# ----------------------------------------------------------------------
def unwrap_estimator(model: Any) -> Any:
    if isinstance(model, Pipeline) and model.steps:
        return model.steps[-1][1]
    return model


def make_preprocessor(model: Any) -> Pipeline | None:
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        return Pipeline(model.steps[:-1])
    return None


def transform(preprocessor: Pipeline | None, frame: pd.DataFrame) -> np.ndarray:
    matrix = preprocessor.transform(frame) if preprocessor is not None else frame.to_numpy()
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ExplainerNotAvailable(
            ExplanationStatus.UNSUPPORTED,
            "Model preprocessing did not return a two-dimensional feature matrix.",
        )
    return array


def is_linear_model(estimator: Any) -> bool:
    name = type(estimator).__name__
    module = type(estimator).__module__
    return name in _LINEAR_CLASS_NAMES or (
        module.startswith("sklearn.linear_model") and "Regression" not in name and "Regressor" not in name
    )


def is_tree_model(estimator: Any) -> bool:
    name = type(estimator).__name__
    module = type(estimator).__module__
    if name in _TREE_CLASS_NAMES:
        return True
    return module.startswith(("xgboost", "lightgbm", "catboost"))


def prepare_feature_frame(frame: pd.DataFrame, features: list[str] | tuple[str, ...]) -> pd.DataFrame:
    feature_names = list(features)
    missing = [feature for feature in feature_names if feature not in frame.columns]
    if missing:
        raise ExplainerNotAvailable(
            ExplanationStatus.UNAVAILABLE,
            f"Explanation input is missing {len(missing)} required feature(s).",
        )
    selected = frame[feature_names].copy().replace([np.inf, -np.inf], np.nan)
    if len(selected) == 1:
        values = sanitize_feature_values(selected.iloc[0].to_dict())
        selected = pd.DataFrame([values], columns=feature_names)
    return selected


def _as_feature_frame(data: Any, features: list[str]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data
    else:
        array = np.asarray(data)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        frame = pd.DataFrame(array, columns=features)
    return frame[features].replace([np.inf, -np.inf], np.nan)


def _has_background(bundle: ModelBundle) -> bool:
    return bundle.background is not None and len(bundle.background) > 0


def _n_classes(estimator: Any) -> int | None:
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return None
    try:
        return int(len(classes))
    except TypeError:
        return None


def _preprocessing_preserves_features(bundle: ModelBundle) -> bool:
    preprocessor = make_preprocessor(bundle.model)
    if preprocessor is None:
        return True
    background = bundle.background
    if background is not None and len(background) > 0:
        try:
            frame = prepare_feature_frame(background.head(5), bundle.selected_features)
            matrix = transform(preprocessor, frame)
        except Exception:  # noqa: BLE001 - plan remains conservative
            return False
        return int(matrix.shape[1]) == len(bundle.selected_features)
    # Without a background we cannot verify the transform shape; a
    # pipeline preprocessor is then treated as unsupported rather than
    # risking a misattributed explanation.
    return False


def _select_binary_shap(values: Any) -> np.ndarray:
    if isinstance(values, list):
        if not values:
            raise ExplainerNotAvailable(
                ExplanationStatus.ERROR, "SHAP returned an empty contribution list."
            )
        array = np.asarray(values[-1], dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        # (n_rows, n_features, n_classes): use the positive class.
        array = array[:, :, -1]
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _select_binary_base(expected: np.ndarray) -> np.ndarray:
    flat = np.asarray(expected, dtype=float).reshape(-1)
    if flat.size == 0:
        raise ExplainerNotAvailable(
            ExplanationStatus.ERROR, "SHAP returned no base value."
        )
    return np.asarray([float(flat[-1])], dtype=float)


def _raw_tree_output(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    """Best-effort raw output for tree estimators without predict_proba."""

    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(matrix), dtype=float)
    for keyword in ("output_margin", "raw_score"):
        try:
            return np.asarray(
                estimator.predict(matrix, **{keyword: True}), dtype=float
            )
        except (TypeError, ValueError):
            continue
    return np.asarray(estimator.predict(matrix), dtype=float)
