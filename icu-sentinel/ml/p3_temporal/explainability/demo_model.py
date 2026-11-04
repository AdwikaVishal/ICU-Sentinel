"""A genuinely fitted demonstration model for local P3 integration.

The model is trained on synthetic demonstration patients only. It is a
real scikit-learn LogisticRegression pipeline, not a fake set of SHAP
values, but it is explicitly labelled as synthetic demonstration data
and must never be presented as a clinically validated sepsis model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.p1_clinical.features.feature_pipeline import FeaturePipeline
from ml.p1_clinical.preprocessing.schemas import ProcessedPatient

from ml.p3_temporal.common.model_bundle import ModelBundle
from ml.p3_temporal.common.prediction import ModelIdentity, PredictionSource

DEMO_FEATURE_SCHEMA_VERSION = "p1-feature-subset-demo-v1"

DEMO_FEATURE_COLUMNS: list[str] = [
    # Current observations.
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "Lactate",
    "WBC",
    "Creatinine",
    "BUN",
    "Platelets",
    "Hgb",
    "Glucose",
    "Potassium",
    "pH",
    "Age",
    "Gender",
    # Missingness indicators and recency.
    "HR_is_missing",
    "O2Sat_is_missing",
    "Temp_is_missing",
    "SBP_is_missing",
    "Resp_is_missing",
    "Lactate_is_missing",
    "WBC_is_missing",
    "HR_hours_since_last",
    "Temp_hours_since_last",
    "Lactate_hours_since_last",
    "WBC_hours_since_last",
    # Causal rolling features.
    "HR_rolling_mean_6h",
    "HR_rolling_max_6h",
    "SBP_rolling_mean_6h",
    "SBP_rolling_min_12h",
    "Resp_rolling_mean_6h",
    "Temp_rolling_max_6h",
    "O2Sat_rolling_min_6h",
    "Lactate_rolling_mean_12h",
    "WBC_rolling_mean_12h",
    # Causal trend features.
    "HR_abs_change_6h",
    "SBP_abs_change_6h",
    "Resp_slope_6h",
    "Temp_abs_change_6h",
    "Lactate_abs_change_12h",
    "O2Sat_slope_6h",
    # P1 clinical derived features.
    "shock_index",
    "sirs_count",
    "sirs_components_available",
]


def build_demo_model_bundle(
    training_patients: list[ProcessedPatient],
    *,
    random_state: int = 42,
) -> ModelBundle:
    """Fit a real LogisticRegression on synthetic demonstration patients."""

    if not training_patients:
        raise ValueError("No synthetic training patients were supplied.")

    pipeline = FeaturePipeline()
    frames: list[pd.DataFrame] = []
    for patient in training_patients:
        frame = pipeline.transform_patient(patient)
        if frame is None or len(frame) == 0:
            continue
        frame = frame.copy()
        frame["SepsisLabel"] = pd.to_numeric(frame["SepsisLabel"], errors="coerce")
        frame = frame[frame["SepsisLabel"].isin([0, 1])]
        frames.append(frame)
    if not frames:
        raise ValueError("Synthetic training patients produced no labelled feature rows.")

    combined = pd.concat(frames, ignore_index=True)
    missing = [feature for feature in DEMO_FEATURE_COLUMNS if feature not in combined.columns]
    if missing:
        raise ValueError(f"Demo feature schema is not produced by P1 pipeline: {missing}")

    X = combined[DEMO_FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    y = combined["SepsisLabel"].astype(int).to_numpy()
    if len(set(y.tolist())) < 2:
        raise ValueError("Synthetic training labels need both classes to fit a demo model.")

    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    estimator.fit(X, y)

    background = X.sample(n=min(120, len(X)), random_state=random_state).reset_index(drop=True)
    identity = ModelIdentity(
        name="demo_logistic_regression",
        version="demo-1.0.0",
        source=PredictionSource.SYNTHETIC_DEMO_MODEL,
        display_name="Synthetic demonstration logistic regression",
        is_synthetic=True,
        feature_schema_version=DEMO_FEATURE_SCHEMA_VERSION,
    )
    return ModelBundle(
        identity=identity,
        model=estimator,
        selected_features=list(DEMO_FEATURE_COLUMNS),
        background=background,
        feature_schema_version=DEMO_FEATURE_SCHEMA_VERSION,
        metadata={
            "training_source": "synthetic_demo_patients",
            "n_training_patients": len(training_patients),
            "n_training_rows": int(len(X)),
            "n_positive_rows": int(y.sum()),
            "n_negative_rows": int(len(y) - y.sum()),
            "label_source": "synthetic_demonstration_labels",
            "model_family": "logistic_regression",
            "clinically_validated": False,
        },
        notes=(
            "Real scikit-learn model fitted on synthetic demonstration data. "
            "Predictions and SHAP values explain this fitted model only; they "
            "do not represent clinical performance or causal medical effects."
        ),
    )
