"""qSOFA (quick Sequential Organ Failure Assessment) clinical baseline.

Implements qSOFA per Sepsis-3 (Singer M, et al. JAMA. 2016;315(8):801-810).

qSOFA criteria (>= 2 of 3 for positive):
1. Respiratory rate >= 22 breaths/min
2. Systolic blood pressure <= 100 mmHg
3. Altered mental status (GCS < 15)

Dataset variable mapping:
- Respiratory rate -> Resp (breaths/min)
- Systolic BP -> SBP (mmHg)
- Altered mental status -> NOT available in Challenge dataset

CRITICAL: Altered mental status cannot be inferred from any dataset
variable. Do not substitute MAP for SBP. Do not invent a proxy.
If altered mental status is unavailable, qSOFA is partial at best.

Missing components are never counted as negative. A complete qSOFA
score requires all 3 components. Partial scores are exposed but NOT
labeled as complete qSOFA.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..preprocessing.schemas import ProcessedPatient
from .baseline_schema import ComponentResult, ScoreEntry, ScoreResult
from .clinical_score_config import (
    ClinicalScoreConfig,
)
from .score_completeness import (
    CompletenessLevel,
    ComponentAvailability,
    assess_completeness,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.baselines.qsofa")

QSOFA_VERSION = "sepsis_3_2016"


def _check_resp(value: float | None, config: ClinicalScoreConfig) -> bool | None:
    """Check respiration criterion: >= 22 breaths/min."""
    if value is None or pd.isna(value):
        return None
    return bool(value >= config.qsofa_resp_high_bpm)


def _check_sbp(value: float | None, config: ClinicalScoreConfig) -> bool | None:
    """Check systolic BP criterion: <= 100 mmHg."""
    if value is None or pd.isna(value):
        return None
    return bool(value <= config.qsofa_sbp_low_mmhg)


def _check_ams() -> None:
    """Altered mental status check.

    Returns None always because AMS is not available in the Challenge
    dataset and cannot be inferred from other variables.
    """
    return


def compute_qsofa(
    patient: ProcessedPatient,
    config: ClinicalScoreConfig | None = None,
) -> list[ScoreResult]:
    """Compute qSOFA for all hours of a patient.

    Returns a list of ScoreResult, one per patient-hour.

    The score is positive only if >= 2 criteria are met AND all 3
    components are available (complete score). Since altered mental
    status is not available in the dataset, qSOFA will be partial
    unless AMS data is provided externally.

    Missing components are never counted as negative.
    """
    cfg = config or ClinicalScoreConfig()
    n = patient.n_rows
    results: list[ScoreResult] = []

    # Component definitions: (name, variable, check_fn, available_fn)
    components_def = [
        ("respiration", "Resp", _check_resp),
        ("systolic_bp", "SBP", _check_sbp),
        ("altered_mental_status", None, _check_ams),
    ]

    for i in range(n):
        pid = patient.patient_id
        source_ds = patient.source_dataset
        hour = patient.hour[i]
        sepsis_label = patient.sepsis_label[i]

        score_components: list[ComponentResult] = []
        availabilities: list[ComponentAvailability] = []
        n_met = 0

        for comp_name, var, check_fn in components_def:
            if var is None:
                # Variable not available in dataset at all (e.g., AMS).
                value = None
                available = False
                met = None
                reason = "variable_not_in_dataset"
            else:
                raw_values = patient.clinical.get(var)
                if raw_values is None:
                    value = None
                    available = False
                    met = None
                    reason = "variable_not_in_patient_data"
                else:
                    value = raw_values[i]
                    available = value is not None and not pd.isna(value)
                    met = check_fn(value, cfg) if available else None
                    reason = None

            if met is True:
                n_met += 1

            unit = cfg.variable_units(var) if var else None
            threshold = _qsofa_threshold_str(comp_name, cfg)

            score_components.append(ComponentResult(
                component_name=comp_name,
                variable=var,
                value=value,
                unit=unit,
                threshold=threshold,
                met=met,
                available=available,
            ))

            availabilities.append(ComponentAvailability(
                component_name=comp_name,
                variable=var,
                available=available,
                value=value,
                reason=reason if not available else None,
            ))

        completeness = assess_completeness(
            score_name="qsofa",
            components=availabilities,
            components_required=cfg.qsofa_components_required,
        )

        score_value = n_met if completeness.can_compute_score else None

        if completeness.level == CompletenessLevel.COMPLETE:
            is_positive = bool(score_value >= cfg.qsofa_positive_threshold)
        else:
            is_positive = None

        score_entry = ScoreEntry(
            score_name="qsofa",
            version=QSOFA_VERSION,
            category="clinical_baseline",
            data_type="int",
            formula=(
                "Count of qSOFA criteria met among available components "
                "(RR>=22 breaths/min, SBP<=100 mmHg, altered mental status). "
                "Positive if score >= 2 AND all 3 components available."
            ),
            score_value=score_value,
            threshold=cfg.qsofa_positive_threshold,
            is_positive=is_positive,
            complete=completeness.level == CompletenessLevel.COMPLETE,
            components_available=completeness.components_available,
            components_required=completeness.components_required,
            units="count",
            causal=True,
            missing_behavior="unavailable_components_not_counted",
            notes=completeness.reason,
        )

        results.append(ScoreResult(
            patient_id=pid,
            source_dataset=source_ds,
            hour=hour,
            sepsis_label=sepsis_label,
            score=score_entry,
            components=score_components,
        ))

    return results


def _qsofa_threshold_str(component: str, config: ClinicalScoreConfig) -> str | None:
    """Return human-readable threshold string for a qSOFA component."""
    thresholds = {
        "respiration": f">={config.qsofa_resp_high_bpm} breaths/min",
        "systolic_bp": f"<={config.qsofa_sbp_low_mmhg} mmHg",
        "altered_mental_status": "GCS < 15 (not available in dataset)",
    }
    return thresholds.get(component)


def validate_qsofa_results(results: list[ScoreResult]) -> list[str]:
    """Validate qSOFA results.

    Returns a list of issue messages (empty if valid).
    """
    issues: list[str] = []
    for r in results:
        s = r.score
        if s.score_value is not None and (s.score_value < 0 or s.score_value > s.components_required):
            issues.append(
                f"Patient {r.patient_id} hour {r.hour}: "
                f"qsofa_score {s.score_value} out of range"
            )
        if s.complete and s.is_positive is None:
            issues.append(
                f"Patient {r.patient_id} hour {r.hour}: "
                "complete score but is_positive is None"
            )
        if s.components_available > s.components_required:
            issues.append(
                f"Patient {r.patient_id} hour {r.hour}: "
                "components_available exceeds components_required"
            )
    return issues
