"""SIRS (Systemic Inflammatory Response Syndrome) clinical baseline.

Implements SIRS per Bone et al., Chest 1992;101(6):1644-1655.

SIRS criteria (>= 2 of 4 for positive):
1. Temperature > 38.3 C or < 36.0 C
2. Heart rate > 90 bpm
3. Respiratory rate > 20 breaths/min (or PaCO2 < 32 mmHg)
4. WBC > 12 or < 4 (10^3/uL) (or > 10% bands)

Dataset variable mapping:
- Temperature -> Temp (deg C)
- Heart rate -> HR (beats/min)
- Respiration -> Resp (breaths/min)
- WBC -> WBC (count*10^3/uL)

Missing components are never counted as negative. A complete SIRS score
requires all 4 components. Partial scores are exposed but NOT labeled
as complete SIRS.

PaCO2 and bands are not available in the Challenge dataset, so the
respiration criterion uses only Resp (>20), and the WBC criterion uses
only WBC count (no bands).
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

logger = logging.getLogger("icu_sentinel.p1_clinical.baselines.sirs")

SIRS_VERSION = "1992_consensus"


def _check_temp(value: float | None, config: ClinicalScoreConfig) -> bool | None:
    """Check temperature criterion: > 38.3 or < 36.0 C."""
    if value is None or pd.isna(value):
        return None
    return bool(value > config.sirs_temp_high_c or value < config.sirs_temp_low_c)


def _check_hr(value: float | None, config: ClinicalScoreConfig) -> bool | None:
    """Check heart rate criterion: > 90 bpm."""
    if value is None or pd.isna(value):
        return None
    return bool(value > config.sirs_hr_high_bpm)


def _check_resp(value: float | None, config: ClinicalScoreConfig) -> bool | None:
    """Check respiration criterion: > 20 breaths/min."""
    if value is None or pd.isna(value):
        return None
    return bool(value > config.sirs_resp_high_bpm)


def _check_wbc(value: float | None, config: ClinicalScoreConfig) -> bool | None:
    """Check WBC criterion: > 12 or < 4 (10^3/uL)."""
    if value is None or pd.isna(value):
        return None
    return bool(value > config.sirs_wbc_high or value < config.sirs_wbc_low)


def compute_sirs(
    patient: ProcessedPatient,
    config: ClinicalScoreConfig | None = None,
) -> ScoreResult:
    """Compute SIRS score for a single patient-hour.

    Returns a ScoreResult with the SIRS score, components, and
    completeness assessment. The score is positive only if >= 2 criteria
    are met AND all 4 components are available (complete score).

    Missing components are never counted as negative.
    """
    cfg = config or ClinicalScoreConfig()
    n = patient.n_rows
    hour_vals = patient.hour

    # Component check functions and their variable mappings.
    checks = [
        ("temperature", "Temp", _check_temp),
        ("heart_rate", "HR", _check_hr),
        ("respiration", "Resp", _check_resp),
        ("wbc", "WBC", _check_wbc),
    ]

    results: list[ScoreResult] = []

    for i in range(n):
        pid = patient.patient_id
        source_ds = patient.source_dataset
        hour = hour_vals[i]
        sepsis_label = patient.sepsis_label[i]

        # Gather component results.
        components: list[ComponentResult] = []
        availabilities: list[ComponentAvailability] = []
        n_met = 0

        for comp_name, var, check_fn in checks:
            raw_values = patient.clinical.get(var)
            if raw_values is None:
                # Variable not in patient data at all.
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

            components.append(ComponentResult(
                component_name=comp_name,
                variable=var if available else None,
                value=value,
                unit=cfg.variable_units(var) if var else None,
                threshold=_sirs_threshold_str(comp_name, cfg),
                met=met,
                available=available,
            ))

            availabilities.append(ComponentAvailability(
                component_name=comp_name,
                variable=var if available else None,
                available=available,
                value=value,
                reason=reason if not available else None,
            ))

        # Completeness assessment.
        completeness = assess_completeness(
            score_name="sirs",
            components=availabilities,
            components_required=cfg.sirs_components_required,
        )

        # Score value: count of met criteria among available components.
        score_value = n_met if completeness.can_compute_score else None

        # Positive determination: score >= threshold AND complete.
        if completeness.level == CompletenessLevel.COMPLETE:
            is_positive = bool(score_value >= cfg.sirs_positive_threshold)
        else:
            # Partial score: not labeled as complete SIRS.
            is_positive = None  # cannot determine positive/negative reliably

        score_entry = ScoreEntry(
            score_name="sirs",
            version=SIRS_VERSION,
            category="clinical_baseline",
            data_type="int",
            formula=(
                "Count of SIRS criteria met among available components "
                "(Temp>38.3 or <36.0 C, HR>90 bpm, Resp>20 breaths/min, "
                "WBC>12 or <4 x10^3/uL). "
                "Positive if score >= 2 AND all 4 components available."
            ),
            score_value=score_value,
            threshold=cfg.sirs_positive_threshold,
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
            components=components,
        ))

    return results[0] if n == 1 else results  # type: ignore[return-value]


def compute_sirs_for_patient(
    patient: ProcessedPatient,
    config: ClinicalScoreConfig | None = None,
) -> list[ScoreResult]:
    """Compute SIRS for all hours of a patient.

    Returns a list of ScoreResult, one per patient-hour.
    """
    cfg = config or ClinicalScoreConfig()
    n = patient.n_rows
    results: list[ScoreResult] = []

    checks = [
        ("temperature", "Temp", _check_temp),
        ("heart_rate", "HR", _check_hr),
        ("respiration", "Resp", _check_resp),
        ("wbc", "WBC", _check_wbc),
    ]

    for i in range(n):
        pid = patient.patient_id
        source_ds = patient.source_dataset
        hour = patient.hour[i]
        sepsis_label = patient.sepsis_label[i]

        components: list[ComponentResult] = []
        availabilities: list[ComponentAvailability] = []
        n_met = 0

        for comp_name, var, check_fn in checks:
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

            components.append(ComponentResult(
                component_name=comp_name,
                variable=var if available else None,
                value=value,
                unit=cfg.variable_units(var) if var else None,
                threshold=_sirs_threshold_str(comp_name, cfg),
                met=met,
                available=available,
            ))

            availabilities.append(ComponentAvailability(
                component_name=comp_name,
                variable=var if available else None,
                available=available,
                value=value,
                reason=reason if not available else None,
            ))

        completeness = assess_completeness(
            score_name="sirs",
            components=availabilities,
            components_required=cfg.sirs_components_required,
        )

        score_value = n_met if completeness.can_compute_score else None

        if completeness.level == CompletenessLevel.COMPLETE:
            is_positive = bool(score_value >= cfg.sirs_positive_threshold)
        else:
            is_positive = None

        score_entry = ScoreEntry(
            score_name="sirs",
            version=SIRS_VERSION,
            category="clinical_baseline",
            data_type="int",
            formula=(
                "Count of SIRS criteria met among available components "
                "(Temp>38.3 or <36.0 C, HR>90 bpm, Resp>20 breaths/min, "
                "WBC>12 or <4 x10^3/uL). "
                "Positive if score >= 2 AND all 4 components available."
            ),
            score_value=score_value,
            threshold=cfg.sirs_positive_threshold,
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
            components=components,
        ))

    return results


def _sirs_threshold_str(component: str, config: ClinicalScoreConfig) -> str | None:
    """Return human-readable threshold string for a SIRS component."""
    thresholds = {
        "temperature": (
            f">{config.sirs_temp_high_c} C or <{config.sirs_temp_low_c} C"
        ),
        "heart_rate": f">{config.sirs_hr_high_bpm} bpm",
        "respiration": f">{config.sirs_resp_high_bpm} breaths/min",
        "wbc": (
            f">{config.sirs_wbc_high} or <{config.sirs_wbc_low} "
            "(10^3/uL)"
        ),
    }
    return thresholds.get(component)


def validate_sirs_results(results: list[ScoreResult]) -> list[str]:
    """Validate SIRS results.

    Returns a list of issue messages (empty if valid).
    """
    issues: list[str] = []
    for r in results:
        s = r.score
        if s.score_value is not None and (s.score_value < 0 or s.score_value > s.components_required):
            issues.append(
                f"Patient {r.patient_id} hour {r.hour}: "
                f"sirs_score {s.score_value} out of range"
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
