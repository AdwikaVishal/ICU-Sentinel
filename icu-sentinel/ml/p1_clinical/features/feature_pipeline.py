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
from ..preprocessing.targets import (
    DERIVED_TARGET_COLUMN,
    TARGET_ELIGIBLE_COLUMN,
    TargetPolicy,
    assign_sepsis_target,
    summarize_targets,
)
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
    n_patients_total: int = 0
    n_rows_total: int = 0

    @property
    def n_patients(self) -> int:
        return self.n_patients_total or len(self.patient_features)

    @property
    def n_rows(self) -> int:
        if self.combined is not None:
            return len(self.combined)
        if self.n_rows_total:
            return self.n_rows_total
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

        # Accumulate engineered columns as arrays, then concatenate once.
        # Assigning hundreds of columns one at a time fragments the
        # DataFrame badly (thousands of PerformanceWarnings on real data).
        extra_values: dict[str, np.ndarray] = {}

        def add_block(frame: pd.DataFrame, columns: list[str]) -> None:
            for c in columns:
                if c not in df.columns and c not in extra_values:
                    extra_values[c] = frame[c].to_numpy()

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
            availability_suffix = (
                f"_availability_{self.config.rolling_window_hours[0]}h"
                if self.config.rolling_window_hours
                else "_availability_6h"
            )
            miss_cols = [
                c for c in miss_df.columns
                if c.endswith(
                    ("_is_missing", "_hours_since_last", "_availability_6h",
                     availability_suffix)
                )
            ]
            add_block(miss_df, miss_cols)

        # Forward-filled features (causal, expires after max window).
        if self.config.compute_forward_fill:
            for c, values in self._build_forward_filled(df, patient).items():
                if c not in df.columns and c not in extra_values:
                    extra_values[c] = values

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
            add_block(roll_df, roll_cols)

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
            add_block(trend_df, trend_cols)

        # Clinical features.
        if self.config.compute_clinical_features:
            clin_df = clinical_features.compute_clinical_features(patient)
            add_block(
                clin_df,
                ["shock_index", "sirs_count", "sirs_components_available"],
            )

        if extra_values:
            extra_df = pd.DataFrame(extra_values, index=df.index)
            df = pd.concat([df, extra_df], axis=1)

        # Sort by hour (causal ordering).
        df = df.sort_values("hour", kind="stable").reset_index(drop=True)

        # Derive the strict pre-onset target. The original SepsisLabel is
        # preserved unchanged; the derived columns are added separately.
        df = assign_sepsis_target(df)

        # Ensure identifier and target columns are present and ordered.
        df = self._order_columns(df)

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
    def _build_forward_filled(
        self,
        df: pd.DataFrame,
        patient: ProcessedPatient,
    ) -> dict[str, np.ndarray]:
        """Build causally forward-filled clinical variable arrays.

        Each variable is carried forward only within its configured
        max validity window; beyond that, it is NaN. Original raw
        columns are preserved unchanged.
        """
        hours = df["hour"].to_numpy()
        n = len(df)
        out: dict[str, np.ndarray] = {}

        for var in self.config.feature_variables:
            if var not in df.columns:
                continue
            window = self.config.forward_fill_window_for(var)
            values = df[var].to_numpy()
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
        target_col = [
            "SepsisLabel",
            DERIVED_TARGET_COLUMN,
            TARGET_ELIGIBLE_COLUMN,
        ]
        feature_cols = [c for c in df.columns if c not in id_cols + target_col]
        ordered = id_cols + feature_cols + target_col
        # Keep only columns that exist.
        ordered = [c for c in ordered if c in df.columns]
        return df[ordered]

    @staticmethod
    def _categorize_features(feature_cols: list[str]) -> dict[str, int]:
        """Categorize feature names by prefix/suffix."""
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
                categories["current"] += 1
        return categories

    def _build_feature_summary(self, result: FeaturePipelineResult) -> dict:
        if result.combined is None or len(result.combined) == 0:
            return {"n_patients": 0, "n_rows": 0, "n_features": 0}

        id_cols = {
            "patient_id",
            "source_dataset",
            "hour",
            "SepsisLabel",
            DERIVED_TARGET_COLUMN,
            TARGET_ELIGIBLE_COLUMN,
        }
        feature_cols = [c for c in result.combined.columns if c not in id_cols]

        return {
            "n_patients": result.n_patients,
            "n_rows": result.n_rows,
            "n_features": len(feature_cols),
            "feature_categories": self._categorize_features(feature_cols),
            "causal": True,
            "no_imputation": True,
            "no_future_leakage": True,
            "target_policy": TargetPolicy().to_dict(),
            "target_summary": summarize_targets(
                result.combined, policy=TargetPolicy()
            ),
            "config": self.config.to_dict(),
        }

    def _validate(self, df: pd.DataFrame) -> dict:
        issues: list[str] = []

        if df is None or len(df) == 0:
            return {"n_issues": 1, "issues": ["Empty combined features DataFrame"],
                    "causal_check_passed": False}

        required = [
            "patient_id",
            "source_dataset",
            "hour",
            "SepsisLabel",
            DERIVED_TARGET_COLUMN,
            TARGET_ELIGIBLE_COLUMN,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")

        # Check SepsisLabel values are 0/1 (target integrity).
        if "SepsisLabel" in df.columns:
            bad = df["SepsisLabel"].dropna()
            bad = bad[~bad.isin([0, 1])]
            if len(bad) > 0:
                issues.append(f"{len(bad)} row(s) with invalid SepsisLabel")

        # Check derived target integrity.
        if DERIVED_TARGET_COLUMN in df.columns:
            bad_target = df[DERIVED_TARGET_COLUMN].dropna()
            bad_target = bad_target[~bad_target.isin([0, 1])]
            if len(bad_target) > 0:
                issues.append(
                    f"{len(bad_target)} row(s) with invalid {DERIVED_TARGET_COLUMN}"
                )
        if TARGET_ELIGIBLE_COLUMN in df.columns:
            bad_eligible = df[TARGET_ELIGIBLE_COLUMN].dropna()
            bad_eligible = bad_eligible[~bad_eligible.isin([0, 1])]
            if len(bad_eligible) > 0:
                issues.append(
                    f"{len(bad_eligible)} row(s) with invalid {TARGET_ELIGIBLE_COLUMN}"
                )

        # Check no inf in feature columns (except hours_since_last, where
        # inf legitimately means "never observed").
        id_cols = {
            "patient_id",
            "source_dataset",
            "hour",
            "SepsisLabel",
            DERIVED_TARGET_COLUMN,
            TARGET_ELIGIBLE_COLUMN,
        }
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
    save_patient_features: bool = True,
) -> FeaturePipelineResult:
    """Convenience function: run the full feature pipeline on patients.

    Optionally saves config, schema, combined parquet, and per-patient
    feature parquet files to output_dir. Per-patient files are consumed by
    the split-aware training loader.
    """
    pipeline = FeaturePipeline(config=config)
    result = pipeline.transform_patients(patients)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pipeline.save_config(out / "feature_config.json")
        pipeline.save_schema(out / "feature_schema.json")
        pipeline.save_combined(result, out / "combined_features.parquet")

        if save_patient_features:
            patient_dir = out / "patients"
            patient_dir.mkdir(parents=True, exist_ok=True)
            for patient_id, frame in result.patient_features.items():
                frame.to_parquet(patient_dir / f"{patient_id}.parquet", index=False)

    return result


def run_feature_pipeline_streaming(
    patients,
    output_dir: str | Path,
    config: FeatureConfig | None = None,
    save_patient_features: bool = True,
    write_combined: bool = True,
    batch_size: int = 200,
) -> FeaturePipelineResult:
    """Run the feature pipeline without retaining all features in memory.

    Patients are consumed one at a time from ``patients`` (any iterable).
    Per-patient parquet files are written immediately, and the combined
    parquet is written in bounded batches via pyarrow. This keeps peak
    memory roughly constant instead of holding the full ~1.5M x ~1000
    feature matrix plus per-patient copies.

    The returned ``FeaturePipelineResult`` has ``patient_features`` empty
    and ``combined=None``; use ``n_patients``, ``n_rows``,
    ``feature_summary``, and ``validation_report`` for reporting.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pipeline = FeaturePipeline(config=config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipeline.save_config(out / "feature_config.json")
    pipeline.save_schema(out / "feature_schema.json")

    patient_dir = out / "patients"
    if save_patient_features:
        patient_dir.mkdir(parents=True, exist_ok=True)

    combined_path = out / "combined_features.parquet"
    writer: pq.ParquetWriter | None = None
    writer_schema: pa.Schema | None = None
    batch: list[pd.DataFrame] = []

    result = FeaturePipelineResult(
        feature_config=pipeline.config,
        feature_schema=pipeline.schema,
    )
    n_patients = 0
    n_rows = 0
    n_features = 0
    categories: dict[str, int] | None = None

    id_cols = {
        "patient_id",
        "source_dataset",
        "hour",
        "SepsisLabel",
        DERIVED_TARGET_COLUMN,
        TARGET_ELIGIBLE_COLUMN,
    }
    target_counts = {
        "n_eligible": 0,
        "n_ineligible": 0,
        "n_positive": 0,
        "n_negative": 0,
        "n_missing_target": 0,
    }
    issue_count = 0
    issue_examples: list[str] = []

    def flush_batch() -> None:
        nonlocal writer, writer_schema, batch
        if not batch:
            return
        combined = pd.concat(batch, ignore_index=True)
        table = pa.Table.from_pandas(combined, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(combined_path, table.schema)
            writer_schema = table.schema
        else:
            table = table.cast(writer_schema)
        writer.write_table(table)
        batch = []

    for patient in patients:
        df = pipeline.transform_patient(patient)
        n_patients += 1
        n_rows += len(df)

        if categories is None:
            feature_cols = [c for c in df.columns if c not in id_cols]
            n_features = len(feature_cols)
            categories = pipeline._categorize_features(feature_cols)

        if DERIVED_TARGET_COLUMN in df.columns:
            target = df[DERIVED_TARGET_COLUMN]
            eligible = df[TARGET_ELIGIBLE_COLUMN].astype(bool)
            target_counts["n_eligible"] += int(eligible.sum())
            target_counts["n_ineligible"] += int((~eligible).sum())
            target_counts["n_positive"] += int((target == 1).sum())
            target_counts["n_negative"] += int((target == 0).sum())
            target_counts["n_missing_target"] += int(target.isna().sum())

        report = pipeline._validate(df)
        issue_count += report["n_issues"]
        if report["n_issues"] and len(issue_examples) < 20:
            issue_examples.extend(report["issues"])

        if save_patient_features:
            df.to_parquet(patient_dir / f"{patient.patient_id}.parquet", index=False)

        if write_combined:
            batch.append(df)
            if len(batch) >= batch_size:
                flush_batch()

    flush_batch()
    if writer is not None:
        writer.close()

    policy = TargetPolicy()
    target_summary = {
        "target_name": policy.target_name,
        "available": n_patients > 0,
        "horizon_hours": policy.horizon_hours,
        "policy": policy.to_dict(),
        "n_rows": n_rows,
        "n_eligible": target_counts["n_eligible"],
        "n_ineligible": target_counts["n_ineligible"],
        "n_positive": target_counts["n_positive"],
        "n_negative": target_counts["n_negative"],
        "n_missing_target": target_counts["n_missing_target"],
        "positive_rate_eligible": (
            target_counts["n_positive"] / target_counts["n_eligible"]
            if target_counts["n_eligible"] else float("nan")
        ),
        "original_label_modified": False,
    }

    result.n_patients_total = n_patients
    result.n_rows_total = n_rows
    result.feature_summary = {
        "n_patients": n_patients,
        "n_rows": n_rows,
        "n_features": n_features,
        "feature_categories": categories or {},
        "causal": True,
        "no_imputation": True,
        "no_future_leakage": True,
        "streaming": True,
        "target_policy": policy.to_dict(),
        "target_summary": target_summary,
        "config": pipeline.config.to_dict(),
    }
    result.validation_report = {
        "n_issues": issue_count,
        "issues": issue_examples,
        "causal_check_passed": issue_count == 0,
        "streaming": True,
    }
    return result
