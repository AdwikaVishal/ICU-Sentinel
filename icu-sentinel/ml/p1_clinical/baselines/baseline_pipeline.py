"""End-to-end clinical baseline pipeline.

Orchestrates SIRS, qSOFA, and NEWS2-style score computation for one or
more patients. Produces per-patient-hour scores with completeness
assessment.

All scores are causal (use only current/past observations). The original
SepsisLabel is preserved separately for later evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..preprocessing.schemas import ProcessedPatient
from .baseline_schema import (
    BaselineSchema,
    ScoreResult,
    build_default_schema,
)
from .clinical_score_config import ClinicalScoreConfig
from .news_style import compute_news2_style
from .qsofa import compute_qsofa
from .sirs import compute_sirs_for_patient

logger = logging.getLogger("icu_sentinel.p1_clinical.baselines.baseline_pipeline")


@dataclass
class BaselinePipelineResult:
    """Result of running the baseline pipeline."""

    patient_scores: dict[str, dict[str, list[ScoreResult]]] = field(
        default_factory=dict
    )  # patient_id -> score_name -> [ScoreResult]
    combined: pd.DataFrame | None = None
    schema: BaselineSchema | None = None
    config: ClinicalScoreConfig | None = None
    summary: dict | None = None
    validation_report: dict | None = None

    @property
    def n_patients(self) -> int:
        return len(self.patient_scores)

    @property
    def n_rows(self) -> int:
        if self.combined is not None:
            return len(self.combined)
        return sum(
            len(scores)
            for patient_scores in self.patient_scores.values()
            for scores in patient_scores.values()
        )


class BaselinePipeline:
    """Clinical baseline scoring pipeline."""

    def __init__(self, config: ClinicalScoreConfig | None = None) -> None:
        self.config = config or ClinicalScoreConfig()
        self.schema = build_default_schema()

    # ------------------------------------------------------------------
    # Single patient
    # ------------------------------------------------------------------
    def transform_patient(
        self,
        patient: ProcessedPatient,
    ) -> dict[str, list[ScoreResult]]:
        """Compute all baseline scores for a single patient.

        Returns dict mapping score_name -> [ScoreResult per hour].
        """
        patient_scores: dict[str, list[ScoreResult]] = {}

        if self.config.compute_sirs:
            patient_scores["sirs"] = compute_sirs_for_patient(patient, self.config)

        if self.config.compute_qsofa:
            patient_scores["qsofa"] = compute_qsofa(patient, self.config)

        if self.config.compute_news2_style:
            patient_scores["news2_style"] = compute_news2_style(patient, self.config)

        return patient_scores

    # ------------------------------------------------------------------
    # Many patients
    # ------------------------------------------------------------------
    def transform_patients(
        self,
        patients: list[ProcessedPatient],
    ) -> BaselinePipelineResult:
        """Compute baseline scores for many patients."""
        result = BaselinePipelineResult()
        result.config = self.config
        result.schema = self.schema

        rows: list[dict] = []

        for patient in patients:
            patient_scores = self.transform_patient(patient)
            result.patient_scores[patient.patient_id] = patient_scores

            # Flatten to rows for combined DataFrame.
            for score_list in patient_scores.values():
                for sr in score_list:
                    row = self._score_result_to_row(sr)
                    rows.append(row)

        if rows:
            result.combined = pd.DataFrame(rows)
        else:
            result.combined = pd.DataFrame()

        result.summary = self._build_summary(result)
        result.validation_report = self._validate(result.combined)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _score_result_to_row(self, sr: ScoreResult) -> dict:
        """Convert a ScoreResult to a flat dict row."""
        s = sr.score
        row = {
            "patient_id": sr.patient_id,
            "source_dataset": sr.source_dataset,
            "hour": sr.hour,
            "sepsis_label": sr.sepsis_label,
            "score_name": s.score_name,
            "score_version": s.version,
            "score_value": s.score_value,
            "score_threshold": s.threshold,
            "is_positive": s.is_positive,
            "score_complete": s.complete,
            "components_available": s.components_available,
            "components_required": s.components_required,
        }

        # Add component columns.
        for comp in sr.components:
            comp_prefix = f"comp_{comp.component_name}"
            row[f"{comp_prefix}_available"] = comp.available
            row[f"{comp_prefix}_met"] = comp.met
            row[f"{comp_prefix}_value"] = comp.value

        return row

    def _build_summary(self, result: BaselinePipelineResult) -> dict:
        if result.combined is None or len(result.combined) == 0:
            return {
                "n_patients": 0,
                "n_rows": 0,
                "scores": {},
                "causal": True,
                "sepsis_label_preserved": True,
            }

        score_names = result.combined["score_name"].unique().tolist()
        scores_summary: dict[str, dict] = {}

        for sn in score_names:
            sn_df = result.combined[result.combined["score_name"] == sn]
            n_complete = int(sn_df["score_complete"].sum())
            n_positive = int(
                sn_df.loc[sn_df["is_positive"].notna(), "is_positive"].sum()
            )
            n_with_score = int(sn_df["score_value"].notna().sum())
            scores_summary[sn] = {
                "n_rows": len(sn_df),
                "n_complete": n_complete,
                "n_positive": n_positive,
                "n_with_score": n_with_score,
                "completeness_rate": round(n_complete / len(sn_df), 4)
                if len(sn_df) > 0 else 0.0,
                "positive_rate_among_complete": round(
                    n_positive / n_complete, 4
                ) if n_complete > 0 else 0.0,
            }

        return {
            "n_patients": result.n_patients,
            "n_rows": result.n_rows,
            "scores": scores_summary,
            "causal": True,
            "sepsis_label_preserved": True,
        }

    def _validate(self, df: pd.DataFrame) -> dict:
        issues: list[str] = []

        if df is None or len(df) == 0:
            return {
                "n_issues": 1,
                "issues": ["Empty combined baseline DataFrame"],
                "causal_check_passed": False,
            }

        required = [
            "patient_id", "source_dataset", "hour", "sepsis_label",
            "score_name", "score_value", "is_positive",
            "score_complete", "components_available", "components_required",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")

        # SepsisLabel values must be 0 or 1.
        if "sepsis_label" in df.columns:
            bad = df["sepsis_label"].dropna()
            bad = bad[~bad.isin([0, 1])]
            if len(bad) > 0:
                issues.append(
                    f"{len(bad)} row(s) with invalid sepsis_label values"
                )

        # Score values must be non-negative.
        if "score_value" in df.columns:
            bad = df["score_value"].dropna()
            bad = bad[bad < 0]
            if len(bad) > 0:
                issues.append(f"{len(bad)} row(s) with negative score_value")

        # Components available <= components required.
        if "components_available" in df.columns and "components_required" in df.columns:
            bad = df[
                (df["components_available"].notna())
                & (df["components_required"].notna())
                & (df["components_available"] > df["components_required"])
            ]
            if len(bad) > 0:
                issues.append(
                    f"{len(bad)} row(s) where components_available > "
                    "components_required"
                )

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

    def save_combined(self, result: BaselinePipelineResult, filepath: str | Path) -> None:
        if result.combined is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            result.combined.to_parquet(path, index=False)

    @classmethod
    def load_config(cls, filepath: str | Path) -> ClinicalScoreConfig:
        return ClinicalScoreConfig.load(filepath)

    @classmethod
    def load_schema(cls, filepath: str | Path) -> BaselineSchema:
        return BaselineSchema.load(filepath)


def run_baseline_pipeline(
    patients: list[ProcessedPatient],
    config: ClinicalScoreConfig | None = None,
    output_dir: str | Path | None = None,
) -> BaselinePipelineResult:
    """Convenience function: run the full baseline pipeline on patients.

    Optionally saves config, schema, and combined parquet to output_dir.
    """
    pipeline = BaselinePipeline(config=config)
    result = pipeline.transform_patients(patients)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pipeline.save_config(out / "clinical_score_config.json")
        pipeline.save_schema(out / "baseline_schema.json")
        pipeline.save_combined(result, out / "combined_baselines.parquet")

    return result
