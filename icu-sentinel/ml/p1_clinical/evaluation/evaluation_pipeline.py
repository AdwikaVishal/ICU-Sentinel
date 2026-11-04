"""End-to-end evaluation pipeline for P1 research evaluation.

Orchestrates all evaluation components:
- Classification metrics
- Calibration
- Lead time
- Alert burden
- Feature ablation
- Subgroup analysis
- Bootstrap confidence intervals
- Cross-hospital evaluation
- Model comparison
"""

from __future__ import annotations

import logging
from datetime import UTC

import pandas as pd

from .alert_burden import compute_alert_burden
from .bootstrap import compute_bootstrap_confidence_intervals
from .calibration import compute_calibration
from .classification_metrics import compute_all_metrics
from .cross_hospital import run_cross_hospital_evaluation
from .evaluation_config import EvaluationConfig
from .evaluation_schema import (
    EvaluationResult,
    ModelComparisonEntry,
    ModelComparisonResult,
)
from .lead_time import compute_lead_time
from .subgroup_analysis import compute_subgroup_analysis

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.evaluation_pipeline")


class EvaluationPipeline:
    """End-to-end evaluation pipeline for P1 research evaluation."""

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        source: str = "synthetic",
    ) -> None:
        self.config = config or EvaluationConfig()
        self.source = source
        self.logger = logging.getLogger(
            "icu_sentinel.p1_clinical.evaluation.evaluation_pipeline"
        )

    def evaluate_model(
        self,
        model_name: str,
        model_version: str,
        predictions_df: pd.DataFrame,
        split: str,
        feature_schema: object | None = None,
        X_train: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> dict:
        """Evaluate a single model on a split.

        Parameters
        ----------
        model_name : str
            Name of the model.
        model_version : str
            Version of the model.
        predictions_df : pd.DataFrame
            Prediction DataFrame with patient_id, hour, true_label,
            predicted_probability, predicted_class.
        split : str
            Split identity.
        feature_schema : object, optional
            Feature schema for feature ablation.
        X_train, y_train, X_test, y_test : optional
            Training and test data for feature ablation.
        source : str
            Data source label.

        Returns
        -------
        dict
            Dictionary with classification_metrics, calibration,
            lead_time, alert_burden results.
        """
        result_dict: dict = {}

        # Classification metrics
        metrics_result = compute_all_metrics(
            predictions_df["true_label"],
            predictions_df["predicted_class"],
            predictions_df["predicted_probability"],
            model_name,
            model_version,
            split,
            source=self.source,
        )
        result_dict["classification_metrics"] = metrics_result

        # Calibration
        cal_result = compute_calibration(
            predictions_df["true_label"],
            predictions_df["predicted_probability"],
            model_name,
            model_version,
            split,
            source=self.source,
        )
        result_dict["calibration"] = cal_result

        # Lead time
        lead_result = compute_lead_time(
            predictions_df,
            model_name,
            model_version,
            split,
            threshold=self.config.alert_threshold,
            source=self.source,
        )
        result_dict["lead_time"] = lead_result

        # Alert burden
        alert_result = compute_alert_burden(
            predictions_df,
            model_name,
            model_version,
            split,
            threshold=self.config.alert_threshold,
            cooldown_hours=self.config.alert_cooldown_hours,
            min_duration_hours=self.config.alert_min_duration_hours,
            min_consecutive=self.config.alert_min_consecutive,
            source=self.source,
        )
        result_dict["alert_burden"] = alert_result

        # Feature ablation (if training data provided)
        if (X_train is not None and y_train is not None
                and X_test is not None and y_test is not None
                and feature_schema is not None):
            # Need model train/predict functions - these would be
            # provided by the caller. For now, skip feature ablation
            # in the basic pipeline and note it.
            logger.info(
                "Feature ablation requires model train/predict functions; "
                "skipping for %s", model_name
            )

        return result_dict

    def evaluate_all_models(
        self,
        predictions_by_model: dict[str, pd.DataFrame],
        split: str,
        feature_schema: object | None = None,
        X_train: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> EvaluationResult:
        """Evaluate all models on a split.

        Parameters
        ----------
        predictions_by_model : dict[str, pd.DataFrame]
            Dictionary mapping model name to predictions DataFrame.
        split : str
            Split identity.
        feature_schema : object, optional
            Feature schema for feature ablation.
        X_train, y_train, X_test, y_test : optional
            Training and test data for feature ablation.

        Returns
        -------
        EvaluationResult
            Complete evaluation result.
        """
        result = EvaluationResult(
            config_version=self.config.__class__.__name__,
            evaluation_timestamp=self._timestamp(),
            source=self.source,
        )

        # Evaluate each model
        model_entries: list = []
        for model_name, pred_df in predictions_by_model.items():
            model_result = self.evaluate_model(
                model_name,
                "1.0.0",
                pred_df,
                split,
                feature_schema=feature_schema,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            result_dict = model_result

            # Classification metrics
            cm = result_dict.get("classification_metrics")
            if cm:
                result.classification_metrics.append(cm)

            # Calibration
            cal = result_dict.get("calibration")
            if cal:
                result.calibration_results.append(cal)

            # Lead time
            lt = result_dict.get("lead_time")
            if lt:
                result.lead_time_results.append(lt)

            # Alert burden
            ab = result_dict.get("alert_burden")
            if ab:
                result.alert_burden_results.append(ab)

            # Build model comparison entry
            metrics = {}
            if cm:
                for me in cm.metrics:
                    metrics[me.metric_name] = me.value

            n_patients = int(pred_df["patient_id"].nunique())
            n_rows = len(pred_df)

            # Determine model type
            model_type = self._classify_model_type(model_name)

            entry = ModelComparisonEntry(
                model_name=model_name,
                model_version="1.0.0",
                model_type=model_type,
                split=split,
                metrics=metrics,
                n_patients=n_patients,
                n_rows=n_rows,
                source=self.source,
            )
            model_entries.append(entry)

        result.model_comparison_results.append(
            ModelComparisonResult(
                split=split,
                models=model_entries,
                source=self.source,
            )
        )

        # Subgroup analysis (if subgroup columns exist)
        for model_name, pred_df in predictions_by_model.items():
            for subgroup_col in ["gender", "age_group", "source_dataset"]:
                if subgroup_col in pred_df.columns:
                    sub_result = compute_subgroup_analysis(
                        pred_df,
                        model_name,
                        "1.0.0",
                        split,
                        subgroup_dimension=subgroup_col,
                        subgroup_column=subgroup_col,
                        threshold=self.config.alert_threshold,
                        source=self.source,
                    )
                    result.subgroup_results.append(sub_result)

        # Bootstrap confidence intervals
        for model_name, pred_df in predictions_by_model.items():
            bs_result = compute_bootstrap_confidence_intervals(
                pred_df,
                model_name,
                "1.0.0",
                split,
                metric_names=["auroc", "auprc", "sensitivity",
                              "specificity", "precision", "f1"],
                n_iterations=self.config.bootstrap_n_iterations,
                confidence_level=self.config.bootstrap_confidence_level,
                random_seed=self.config.random_seed,
                source=self.source,
            )
            result.bootstrap_results.append(bs_result)

        # Cross-hospital evaluation (if source_dataset column exists)
        for model_name, pred_df in predictions_by_model.items():
            if "source_dataset" in pred_df.columns:
                hospitals = pred_df["source_dataset"].unique()
                if len(hospitals) >= 2:
                    predictions_by_hospital = {
                        h: pred_df[pred_df["source_dataset"] == h]
                        for h in hospitals
                    }
                    ch_results = run_cross_hospital_evaluation(
                        predictions_by_hospital,
                        model_name,
                        "1.0.0",
                        threshold=self.config.alert_threshold,
                        source=self.source,
                    )
                    result.cross_hospital_results.extend(ch_results)

        return result

    def _classify_model_type(self, model_name: str) -> str:
        """Classify model type based on model name."""
        name_lower = model_name.lower()
        if name_lower in ("logistic_regression", "random_forest", "svm",
                          "extra_trees", "xgboost", "lightgbm", "catboost"):
            return "classical_ml"
        elif name_lower in ("sirs", "qsofa", "news2_style"):
            return "clinical_baseline"
        elif name_lower in ("gru", "lstm", "transformer"):
            return "temporal"
        else:
            return "unknown"

    @staticmethod
    def _timestamp() -> str:
        """Get current timestamp as ISO 8601 string."""
        from datetime import datetime
        return datetime.now(UTC).isoformat()


def run_evaluation_pipeline(
    predictions_by_model: dict[str, pd.DataFrame],
    split: str,
    config: EvaluationConfig | None = None,
    feature_schema: object | None = None,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    source: str = "synthetic",
) -> EvaluationResult:
    """Convenience function to run the complete evaluation pipeline.

    Parameters
    ----------
    predictions_by_model : dict[str, pd.DataFrame]
        Dictionary mapping model name to predictions DataFrame.
    split : str
        Split identity.
    config : EvaluationConfig, optional
        Evaluation configuration.
    feature_schema : object, optional
        Feature schema for feature ablation.
    X_train, y_train, X_test, y_test : optional
        Training and test data for feature ablation.
    source : str
        Data source label.

    Returns
    -------
    EvaluationResult
        Complete evaluation result.
    """
    pipeline = EvaluationPipeline(config=config, source=source)
    return pipeline.evaluate_all_models(
        predictions_by_model,
        split,
        feature_schema=feature_schema,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )