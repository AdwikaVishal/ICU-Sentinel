"""Prediction output schema for P1 classical ML models.

Defines the contract for model predictions. Each prediction record
contains patient_id, hour, true SepsisLabel, predicted probability,
predicted class, model name, model version, and split identity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class PredictionRecord:
    """A single model prediction for one patient-hour."""

    patient_id: str
    hour: int
    true_label: int
    predicted_probability: float | None
    predicted_class: int
    model_name: str
    model_version: str
    split: str  # train / validation / test
    source_dataset: str | None = None
    threshold: float | None = None
    features_used: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PredictionBatch:
    """A batch of prediction records for one model on one split."""

    model_name: str
    model_version: str
    split: str
    n_predictions: int
    n_positive_true: int
    n_positive_pred: int
    predictions: list[PredictionRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "split": self.split,
            "n_predictions": self.n_predictions,
            "n_positive_true": self.n_positive_true,
            "n_positive_pred": self.n_positive_pred,
            "predictions": [p.to_dict() for p in self.predictions],
        }

    def to_dataframe(self):
        """Convert to a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([p.to_dict() for p in self.predictions])

    @classmethod
    def from_dataframe(
        cls,
        df,
        model_name: str,
        model_version: str,
        split: str,
    ) -> PredictionBatch:
        """Create a PredictionBatch from a DataFrame."""
        records = []
        for _, row in df.iterrows():
            records.append(
                PredictionRecord(
                    patient_id=str(row["patient_id"]),
                    hour=int(row["hour"]),
                    true_label=int(row["true_label"]),
                    predicted_probability=(
                        float(row["predicted_probability"])
                        if row["predicted_probability"] is not None
                        and pd.notna(row["predicted_probability"])
                        else None
                    ),
                    predicted_class=int(row["predicted_class"]),
                    model_name=model_name,
                    model_version=model_version,
                    split=split,
                    source_dataset=(
                        str(row["source_dataset"])
                        if "source_dataset" in row.index
                        and pd.notna(row["source_dataset"])
                        else None
                    ),
                    threshold=(
                        float(row["threshold"])
                        if row["threshold"] is not None
                        and pd.notna(row["threshold"])
                        else None
                    ),
                )
            )
        return cls(
            model_name=model_name,
            model_version=model_version,
            split=split,
            n_predictions=len(records),
            n_positive_true=int(sum(r.true_label for r in records)),
            n_positive_pred=int(sum(r.predicted_class for r in records)),
            predictions=records,
        )


@dataclass
class PredictionSchema:
    """Schema documentation for prediction outputs."""

    version: str = "1.0.0"
    description: str = (
        "P1 classical ML prediction schema. Each row is one patient-hour "
        "prediction with true label, predicted probability and class."
    )
    columns: list[str] = field(
        default_factory=lambda: [
            "patient_id",
            "hour",
            "true_label",
            "predicted_probability",
            "predicted_class",
            "model_name",
            "model_version",
            "split",
            "source_dataset",
            "threshold",
        ]
    )
    identifier_columns: list[str] = field(
        default_factory=lambda: ["patient_id", "hour", "split"]
    )
    target_column: str = "true_label"
    probability_column: str = "predicted_probability"
    class_column: str = "predicted_class"
    model_columns: list[str] = field(
        default_factory=lambda: ["model_name", "model_version"]
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> PredictionSchema:
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> PredictionSchema:
        return cls(**data)


MODEL_VERSION: str = "1.0.0"
