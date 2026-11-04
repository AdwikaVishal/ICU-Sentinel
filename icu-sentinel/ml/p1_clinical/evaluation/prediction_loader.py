"""Prediction loader for P1 research evaluation.

Loads prediction outputs from trained models (Part 6 classical ML) and
clinical baselines (Part 5) into a standardized prediction format for
evaluation. Provides a unified interface so P2 and P3 can submit
predictions to the same evaluation framework.

The prediction input contract requires:
- patient_id
- hour (timestamp)
- true SepsisLabel
- predicted probability or score
- model_name
- model_version
- split identity

Optional fields:
- hospital_id (only when genuinely available)
- source_dataset
- predicted_class
- alert threshold
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..classical_ml.prediction_schema import (
    PredictionBatch,
    PredictionRecord,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.evaluation.prediction_loader")


@dataclass
class PredictionLoaderResult:
    """Result of loading predictions."""

    model_name: str
    model_version: str
    split: str
    n_predictions: int
    predictions: list[PredictionRecord]
    source: str = "synthetic"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame."""
        return pd.DataFrame([p.to_dict() for p in self.predictions])

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "split": self.split,
            "n_predictions": self.n_predictions,
            "predictions": [p.to_dict() for p in self.predictions],
            "source": self.source,
        }


class PredictionLoader:
    """Loads prediction outputs into a standardized format.

    Supports loading from:
    - InferenceResult objects (from Part 6 classical ML inference)
    - PredictionBatch objects
    - DataFrames with the required prediction columns
    - JSON files containing prediction records
    """

    def __init__(self, source: str = "synthetic") -> None:
        self.source = source
        self.logger = logging.getLogger(
            "icu_sentinel.p1_clinical.evaluation.prediction_loader.PredictionLoader"
        )

    def load_from_inference_result(
        self,
        inference_result,
    ) -> PredictionLoaderResult:
        """Load predictions from an InferenceResult object.

        Parameters
        ----------
        inference_result : InferenceResult
            Result from ModelInference.predict().

        Returns
        -------
        PredictionLoaderResult
            Standardized prediction batch.
        """
        records = []
        for pr in inference_result.predictions.predictions:
            records.append(
                PredictionRecord(
                    patient_id=pr.patient_id,
                    hour=pr.hour,
                    true_label=pr.true_label,
                    predicted_probability=pr.predicted_probability,
                    predicted_class=pr.predicted_class,
                    model_name=pr.model_name,
                    model_version=pr.model_version,
                    split=pr.split,
                    source_dataset=pr.source_dataset,
                    threshold=pr.threshold,
                    features_used=pr.features_used,
                )
            )

        return PredictionLoaderResult(
            model_name=inference_result.model_name,
            model_version=inference_result.model_version,
            split=inference_result.split,
            n_predictions=len(records),
            predictions=records,
            source=self.source,
        )

    def load_from_prediction_batch(
        self,
        batch: PredictionBatch,
    ) -> PredictionLoaderResult:
        """Load predictions from a PredictionBatch object.

        Parameters
        ----------
        batch : PredictionBatch
            Prediction batch from Part 6.

        Returns
        -------
        PredictionLoaderResult
            Standardized prediction batch.
        """
        records = []
        for pr in batch.predictions:
            records.append(
                PredictionRecord(
                    patient_id=pr.patient_id,
                    hour=pr.hour,
                    true_label=pr.true_label,
                    predicted_probability=pr.predicted_probability,
                    predicted_class=pr.predicted_class,
                    model_name=pr.model_name,
                    model_version=pr.model_version,
                    split=pr.split,
                    source_dataset=pr.source_dataset,
                    threshold=pr.threshold,
                    features_used=pr.features_used,
                )
            )

        return PredictionLoaderResult(
            model_name=batch.model_name,
            model_version=batch.model_version,
            split=batch.split,
            n_predictions=len(records),
            predictions=records,
            source=self.source,
        )

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        model_name: str,
        model_version: str,
        split: str,
    ) -> PredictionLoaderResult:
        """Load predictions from a pandas DataFrame.

        The DataFrame must contain columns:
        patient_id, hour, true_label, predicted_probability,
        predicted_class, model_name, model_version, split

        Optional columns:
        source_dataset, threshold, features_used
        """
        required = [
            "patient_id",
            "hour",
            "true_label",
            "predicted_probability",
            "predicted_class",
            "model_name",
            "model_version",
            "split",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        records = []
        for _, row in df.iterrows():
            records.append(
                PredictionRecord(
                    patient_id=str(row["patient_id"]),
                    hour=int(row["hour"]),
                    true_label=int(row["true_label"]),
                    predicted_probability=(
                        float(row["predicted_probability"])
                        if pd.notna(row["predicted_probability"])
                        else None
                    ),
                    predicted_class=int(row["predicted_class"]),
                    model_name=str(row["model_name"]),
                    model_version=str(row["model_version"]),
                    split=str(row["split"]),
                    source_dataset=(
                        str(row["source_dataset"])
                        if "source_dataset" in row.index
                        and pd.notna(row["source_dataset"])
                        else None
                    ),
                    threshold=(
                        float(row["threshold"])
                        if "threshold" in row.index
                        and pd.notna(row["threshold"])
                        else None
                    ),
                )
            )

        return PredictionLoaderResult(
            model_name=model_name,
            model_version=model_version,
            split=split,
            n_predictions=len(records),
            predictions=records,
            source=self.source,
        )

    def load_from_json(
        self,
        filepath: str | Path,
    ) -> PredictionLoaderResult:
        """Load predictions from a JSON file.

        The JSON file should contain a list of prediction records with
        the fields defined in PredictionRecord.
        """
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Could be a single PredictionBatch or PredictionLoaderResult
            if "predictions" in data:
                records = []
                for pr in data["predictions"]:
                    records.append(
                        PredictionRecord(
                            patient_id=pr["patient_id"],
                            hour=pr["hour"],
                            true_label=pr["true_label"],
                            predicted_probability=pr.get("predicted_probability"),
                            predicted_class=pr["predicted_class"],
                            model_name=pr["model_name"],
                            model_version=pr["model_version"],
                            split=pr["split"],
                            source_dataset=pr.get("source_dataset"),
                            threshold=pr.get("threshold"),
                        )
                    )
                model_name = data.get("model_name", "unknown")
                model_version = data.get("model_version", "1.0.0")
                split = data.get("split", "test")
            else:
                # Single record
                pr = data
                records = [
                    PredictionRecord(
                        patient_id=pr["patient_id"],
                        hour=pr["hour"],
                        true_label=pr["true_label"],
                        predicted_probability=pr.get("predicted_probability"),
                        predicted_class=pr["predicted_class"],
                        model_name=pr["model_name"],
                        model_version=pr["model_version"],
                        split=pr["split"],
                        source_dataset=pr.get("source_dataset"),
                        threshold=pr.get("threshold"),
                    )
                ]
                model_name = pr["model_name"]
                model_version = pr["model_version"]
                split = pr["split"]
        else:
            # List of records
            records = []
            for pr in data:
                records.append(
                    PredictionRecord(
                        patient_id=pr["patient_id"],
                        hour=pr["hour"],
                        true_label=pr["true_label"],
                        predicted_probability=pr.get("predicted_probability"),
                        predicted_class=pr["predicted_class"],
                        model_name=pr["model_name"],
                        model_version=pr["model_version"],
                        split=pr["split"],
                        source_dataset=pr.get("source_dataset"),
                        threshold=pr.get("threshold"),
                    )
                )
                model_name = records[0].model_name if records else "unknown"
                model_version = records[0].model_version if records else "1.0.0"
                split = records[0].split if records else "test"

        return PredictionLoaderResult(
            model_name=model_name,
            model_version=model_version,
            split=split,
            n_predictions=len(records),
            predictions=records,
            source=self.source,
        )

    def save_to_json(
        self,
        result: PredictionLoaderResult,
        filepath: str | Path,
    ) -> None:
        """Save prediction loader result to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def validate_predictions(
        self,
        result: PredictionLoaderResult,
    ) -> dict:
        """Validate prediction loader result.

        Returns a dict with validation results.
        """
        issues: list[str] = []

        if result.n_predictions == 0:
            issues.append("No predictions in result")

        # Check probability values are in [0, 1]
        for pr in result.predictions:
            if pr.predicted_probability is not None and (
                pr.predicted_probability < 0 or pr.predicted_probability > 1
            ):
                issues.append(
                    f"Patient {pr.patient_id} hour {pr.hour}: "
                        f"probability {pr.predicted_probability} out of [0,1]"
                    )
            if pr.predicted_class not in (0, 1):
                issues.append(
                    f"Patient {pr.patient_id} hour {pr.hour}: "
                    f"invalid class {pr.predicted_class}"
                )
            if pr.true_label not in (0, 1):
                issues.append(
                    f"Patient {pr.patient_id} hour {pr.hour}: "
                    f"invalid true label {pr.true_label}"
                )

        # Check all predictions have patient_id and hour
        for pr in result.predictions:
            if not pr.patient_id:
                issues.append("Missing patient_id in prediction")
            if pr.hour is None:
                issues.append(f"Missing hour for patient {pr.patient_id}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "n_predictions": result.n_predictions,
        }


def load_predictions_from_artifacts(
    artifact_dir: str | Path,
    model_names: list[str],
    model_version: str = "1.0.0",
    split: str = "test",
    source: str = "synthetic",
) -> dict[str, PredictionLoaderResult]:
    """Load predictions for multiple models from saved artifacts.

    Parameters
    ----------
    artifact_dir : str | Path
        Directory containing model artifacts.
    model_names : list[str]
        List of model names to load.
    model_version : str
        Model version string.
    split : str
        Split identity (validation or test).
    source : str
        Data source label.

    Returns
    -------
    dict[str, PredictionLoaderResult]
        Dictionary mapping model name to PredictionLoaderResult.
    """
    from .model_serialization import load_feature_list, load_model_artifact

    loader = PredictionLoader(source=source)  # noqa: F841
    artifact_path = Path(artifact_dir)
    results: dict[str, PredictionLoaderResult] = {}

    for model_name in model_names:
        artifact_file = (
            artifact_path / model_name / f"{model_name}_{model_version}.joblib"
        )
        if not artifact_file.exists():
            logger.warning(
                "Model artifact not found: %s", artifact_file
            )
            continue

        artifact = load_model_artifact(artifact_file)
        artifact["model"]

        # Load selected features
        feature_file = (
            artifact_path
            / model_name
            / f"{model_name}_{model_version}_features.json"
        )
        try:
            selected_features = load_feature_list(feature_file)
        except FileNotFoundError:
            logger.warning(
                "Feature list not found for %s", model_name
            )
            selected_features = []  # noqa: F841

        # We need X data to generate predictions. This method loads the
        # model but requires X data to be provided separately.
        # The actual prediction generation happens in ModelInference.
        # This method is a placeholder for loading model metadata.
        results[model_name] = PredictionLoaderResult(
            model_name=model_name,
            model_version=model_version,
            split=split,
            n_predictions=0,
            predictions=[],
            source=source,
        )

    return results