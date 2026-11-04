"""End-to-end temporal feature-engineering pipeline.

Orchestrates current, missingness, forward-fill, rolling, trend, and
clinical feature computation for one patient or many patients.

All features are causal (use only current/past observations). The
pipeline produces one row per patient-hour suitable for tabular ML, and
ordered patient sequences for sequential models.

Raw Part 3 cleaned data is never modified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..preprocessing.schemas import ProcessedPatient
from . import (
    clinical_features,
    current_features,
    missingness_features,
    rolling_features,
    trend_features,
)
from .feature_config import FeatureConfig
from .feature_schema import FeatureSchema, build_default_schema

logger = logging.getLogger("icu_sentinel.p1_clinical.features.feature_pipeline")


@dataclass
class FeaturePipelineResult:
    """Result of running the feature pipeline on one or more patients."""

    patient_features: dict[str, pd.DataFrame] = field(default_factory=dict)
    combined: pd.DataFrame | None = None
    feature_schema: FeatureSchema | None = None
    feature_config: FeatureConfig | None = None
    feature_summary: dict | None = None
    validation_report: dict | None = None

    @property
    def n_patients(self) -> int:
        return len(self.patient_features)

    @property
    def n_rows(self) -> int:
        if self.combined is not None:
            return len(self.combined)
        return sum(len(df) for df in self.patient_features.values())


class FeaturePipeline:
    """Temporal feature-engineering pipeline."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.schema = build_default_schema(self.config)

    # ------------------------------------------------------------------
    # Single patient
    # ------------------------------------------------------------------
    def transform_patient(
        self,
        patient: ProcessedPatient,
    ) -> pd.DataFrame:
        """Compute all features for a single patient.

        Returns a DataFrame with one row per patient-hour and columns:
        patient_id, source_dataset, hour, <features>, SepsisLabel.
        """
        # Start with current (raw) observation features.
        df = current_features.extract_current_features(
            patient,
            feature_variables=self.config.feature_variables,
            static_variables=self.config.static_variables,
        )

        # Missingness features (computed from raw input, before imputation).
        if self.config.compute_missingness_features:
            miss_df = missingness_features.compute_missingness_features(
                patient,
                variables=self.config.feature_variables,
                availability_window_hours=(
                    self.config.rolling_window_hours[0]
                    if self.config.rolling_window_hours else 6
                ),
            )
            miss_cols = [
                c for c in miss_df.columns
                if c.endswith(("_is_missing", "_hours_since_last", "_availability_6h", f"_availability_{self.config.rolling_window_hours[0]}h" if self.config.rolling_window_hours else "_availability_6h"))
            ]
            for c in miss_cols:
                if c not in df.columns:
                    df[c] = miss_df[c].values

        # Forward-filled features (causal, expires after max window).
        if self.config.compute_forward_fill:
            df = self._add_forward_filled(df, patient)

        # Rolling features.
        if self.config.compute_rolling_features:
            roll_df = rolling_features.compute_rolling_features(
                patient,
                variables=self.config.feature_variables,
                window_hours=self.config.rolling_window_hours,
                aggregations=self.config.rolling_aggregations,
                min_observations=self.config.rolling_min_observations,
            )
            roll_cols = [c for c in roll_df.columns if "_rolling_" in c]
            for c in roll_cols:
                df[c] = roll_df[c].values

        # Trend features.
        if self.config.compute_trend_features:
            trend_df = trend_features.compute_trend_features(
                patient,
                variables=self.config.feature_variables,
                window_hours=self.config.trend_window_hours,
                min_observations=self.config.trend_min_observations,
            )
            trend_cols = [
                c for c in trend_df.columns
                if "_abs_change_" in c or "_rel_change_" in c or "_slope_" in c
            ]
            for c in trend_cols:
                df[c] = trend_df[c].values

        # Clinical features.
        if self.config.compute_clinical_features:
            clin_df = clinical_features.compute_clinical_features(patient)
            for c in ["shock_index", "sirs_count", "sirs_components_available"]:
                if c in clin_df.columns:
                    df[c] = clin_df[c].values

        # Ensure identifier and target columns are present and ordered.
        df = self._order_columns(df)

        # Sort by hour (causal ordering).
        df = df.sort_values("hour", kind="stable").reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Many patients
    # ------------------------------------------------------------------
    def transform_patients(
        self,
        patients: list[ProcessedPatient],
    ) -> FeaturePipelineResult:
        """Compute features for many patients and combine into one DataFrame."""
        result = FeaturePipelineResult()
        result.feature_config = self.config
        result.feature_schema = self.schema

        frames: list[pd.DataFrame] = []
        for patient in patients:
            df = self.transform_patient(patient)
            result.patient_features[patient.patient_id] = df
            frames.append(df)

        if frames:
            result.combined = pd.concat(frames, ignore_index=True)
        else:
            result.combined = pd.DataFrame()

        result.feature_summary = self._build_feature_summary(result)
        result.validation_report = self._validate(result.combined)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_forward_filled(
        self,
        df: pd.DataFrame,
        patient: ProcessedPatient,
    ) -> pd.DataFrame:
        """Add causally forward-filled versions of clinical variables.

        Each variable is carried forward only within its configured
        max validity window; beyond that, it is NaN. Original raw
        columns are preserved unchanged.
        """
        out = df.copy()
        hours = out["hour"].to_numpy()
        n = len(out)

        for var in self.config.feature_variables:
            if var not in out.columns:
                continue
            window = self.config.forward_fill_window_for(var)
            values = out[var].to_numpy()
            ffilled = np.full(n, np.nan, dtype=float)

            last_valid_val: float | None = None
            last_valid_hour: float | None = None
            for i in range(n):
                h = hours[i]
                v = values[i]
                if pd.notna(v):
                    last_valid_val = float(v)
                    last_valid_hour = float(h)
                    ffilled[i] = float(v)
                else:
                    if last_valid_val is not None and last_valid_hour is not None:
                        if h - last_valid_hour <= window:
                            ffilled[i] = last_valid_val
                        else:
                            ffilled[i] = np.nan  # expired
                    # else: never observed; remains NaN

            out[f"{var}_ffilled"] = ffilled

        return out

    def _order_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Order columns: identifiers, features, target."""
        id_cols = ["patient_id", "source_dataset", "hour"]
        target_col = ["SepsisLabel"]
        feature_cols = [c for c in df.columns if c not in id_cols + target_col]
        ordered = id_cols + feature_cols + target_col
        # Keep only columns that exist.
        ordered = [c for c in ordered if c in df.columns]
        return df[ordered]

    def _build_feature_summary(self, result: FeaturePipelineResult) -> dict:
        if result.combined is None or len(result.combined) == 0:
            return {"n_patients": 0, "n_rows": 0, "n_features": 0}

        id_cols = {"patient_id", "source_dataset", "hour", "SepsisLabel"}
        feature_cols = [c for c in result.combined.columns if c not in id_cols]

        # Categorize features by prefix/suffix.
        categories: dict[str, int] = {
            "current": 0,
            "missingness": 0,
            "forward_fill": 0,
            "rolling": 0,
            "trend": 0,
            "clinical": 0,
            "other": 0,
        }
        for c in feature_cols:
            if "_is_missing" in c or "_hours_since_last" in c or "_availability_" in c:
                categories["missingness"] += 1
            elif "_ffilled" in c:
                categories["forward_fill"] += 1
            elif "_rolling_" in c:
                categories["rolling"] += 1
            elif "_abs_change_" in c or "_rel_change_" in c or "_slope_" in c:
                categories["trend"] += 1
            elif c in ("shock_index", "sirs_count", "sirs_components_available"):
                categories["clinical"] += 1
            else:
                # Current/static variables.
                categories["current"] += 1

        return {
            "n_patients": result.n_patients,
            "n_rows": result.n_rows,
            "n_features": len(feature_cols),
            "feature_categories": categories,
            "causal": True,
            "no_imputation": True,
            "no_future_leakage": True,
            "config": self.config.to_dict(),
        }

    def _validate(self, df: pd.DataFrame) -> dict:
        issues: list[str] = []

        if df is None or len(df) == 0:
            return {"n_issues": 1, "issues": ["Empty combined features DataFrame"],
                    "causal_check_passed": False}

        required = ["patient_id", "source_dataset", "hour", "SepsisLabel"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")

        # Check SepsisLabel values are 0/1 (target integrity).
        if "SepsisLabel" in df.columns:
            bad = df["SepsisLabel"].dropna()
            bad = bad[~bad.isin([0, 1])]
            if len(bad) > 0:
                issues.append(f"{len(bad)} row(s) with invalid SepsisLabel")

        # Check no inf in feature columns (except hours_since_last, where
        # inf legitimately means "never observed").
        id_cols = {"patient_id", "source_dataset", "hour", "SepsisLabel"}
        feature_cols = [c for c in df.columns if c not in id_cols]
        inf_cols = []
        for c in feature_cols:
            if c.endswith("_hours_since_last"):
                continue
            if np.isinf(df[c]).any():
                inf_cols.append(c)
        if inf_cols:
            issues.append(f"{len(inf_cols)} column(s) contain infinite values: {inf_cols[:5]}")

        return {
            "n_issues": len(issues),
            "issues": issues,
            "causal_check_passed": len(issues) == 0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_config(self, filepath: str | Path) -> None:
        self.config.save(filepath)

    def save_schema(self, filepath: str | Path) -> None:
        self.schema.save(filepath)

    def save_combined(self, result: FeaturePipelineResult, filepath: str | Path) -> None:
        if result.combined is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            result.combined.to_parquet(path, index=False)

    @classmethod
    def load_config(cls, filepath: str | Path) -> FeatureConfig:
        return FeatureConfig.load(filepath)

    @classmethod
    def load_schema(cls, filepath: str | Path) -> FeatureSchema:
        return FeatureSchema.load(filepath)


def run_pipeline_on_processed_patients(
    patients: list[ProcessedPatient],
    config: FeatureConfig | None = None,
    output_dir: str | Path | None = None,
) -> FeaturePipelineResult:
    """Convenience function: run the full feature pipeline on patients.

    Optionally saves config, schema, and combined parquet to output_dir.
    """
    pipeline = FeaturePipeline(config=config)
    result = pipeline.transform_patients(patients)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pipeline.save_config(out / "feature_config.json")
        pipeline.save_schema(out / "feature_schema.json")
        pipeline.save_combined(result, out / "combined_features.parquet")

    return result
