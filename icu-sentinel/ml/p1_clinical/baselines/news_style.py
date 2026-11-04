"""NEWS-style clinical baseline scoring.

Implements a NEWS2-style score per Royal College of Physicians NEWS2,
using only the dataset variables that are available.

NEWS2 parameters (scores 0, 1, 2, or 3 per parameter):
1. Respiratory rate (breaths/min): <=8->3, 9-11->1, 12-20->0, 21-24->1, >=25->3
2. SpO2 (%): <=91->3, 92-93->2, 94-95->1, 96-97->0, >=98->0
3. Supplemental oxygen: No->0, Yes->+2 (NOT available in dataset)
4. Temperature (C): <=35.0->3, 35.1-36.0->1, 36.1-38.0->0, 38.1-39.0->0, >=39.1->2
5. Systolic BP (mmHg): <=90->3, 91-100->2, 101-119->1, 120-219->0, >=220->3
6. Heart rate (bpm): <=40->3, 41-50->1, 51-90->0, 91-110->0, 111-130->1, 131-140->2, >=141->3
7. Level of consciousness: A->0, C/V/P/U->3 (NOT available in dataset)

NEWS2 clinical concern threshold: >= 5 points

Dataset variable mapping:
- Respiration -> Resp
- SpO2 -> O2Sat
- Supplemental oxygen -> NOT available
- Temperature -> Temp
- Systolic BP -> SBP
- Heart rate -> HR
- Consciousness -> NOT available

This is a MODIFIED NEWS-style score, not full NEWS2, because supplemental
oxygen administration and consciousness level are not available in the
Challenge dataset. Do not claim this is validated NEWS2.

Missing components are never counted as normal. Score completeness is
tracked separately.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..preprocessing.schemas import ProcessedPatient
from .baseline_schema import ComponentResult, ScoreEntry, ScoreResult
from .clinical_score_config import (
    NEWS2_HR_SCORES,
    NEWS2_RR_SCORES,
    NEWS2_SBP_SCORES,
    NEWS2_SPO2_SCORES,
    NEWS2_TEMP_SCORES,
    ClinicalScoreConfig,
)
from .score_completeness import (
    CompletenessLevel,
    ComponentAvailability,
    assess_completeness,
)

logger = logging.getLogger("icu_sentinel.p1_clinical.baselines.news_style")

NEWS2_VERSION = "news2_rcp_modified"


def _score_from_ranges(value: float, scores: list[tuple]) -> int | None:
    """Map a value to a NEWS2 parameter score using (low, high, score) tuples.

    Each tuple is (inclusive_low, inclusive_high, score). None means
    unbounded on that side.
    """
    for low, high, score in scores:
        if low is not None and value < low:
            continue
        if high is not None and value > high:
            continue
        return score
    return None


def _check_rr(value: float | None) -> int | None:
    """Score respiratory rate per NEWS2."""
    if value is None or pd.isna(value):
        return None
    return _score_from_ranges(float(value), NEWS2_RR_SCORES)


def _check_spo2(value: float | None) -> int | None:
    """Score SpO2 per NEWS2."""
    if value is None or pd.isna(value):
        return None
    return _score_from_ranges(float(value), NEWS2_SPO2_SCORES)


def _check_temp(value: float | None) -> int | None:
    """Score temperature per NEWS2."""
    if value is None or pd.isna(value):
        return None
    return _score_from_ranges(float(value), NEWS2_TEMP_SCORES)


def _check_sbp(value: float | None) -> int | None:
    """Score systolic BP per NEWS2."""
    if value is None or pd.isna(value):
        return None
    return _score_from_ranges(float(value), NEWS2_SBP_SCORES)


def _check_hr(value: float | None) -> int | None:
    """Score heart rate per NEWS2."""
    if value is None or pd.isna(value):
        return None
    return _score_from_ranges(float(value), NEWS2_HR_SCORES)


def compute_news2_style(
    patient: ProcessedPatient,
    config: ClinicalScoreConfig | None = None,
) -> list[ScoreResult]:
    """Compute NEWS2-style score for all hours of a patient.

    Returns a list of ScoreResult, one per patient-hour.

    This is a modified NEWS-style score because supplemental oxygen and
    consciousness level are not available in the dataset. The score is
    the sum of available parameter scores. Clinical concern threshold is
    >= 5, but only applies when the score is complete.

    Missing components are never counted as normal (score 0).
    """
    cfg = config or ClinicalScoreConfig()
    n = patient.n_rows
    results: list[ScoreResult] = []

    # Component definitions: (name, variable, check_fn, unit)
    components_def = [
        ("respiration", "Resp", _check_rr, "breaths_per_minute"),
        ("spo2", "O2Sat", _check_spo2, "percent"),
        ("supplemental_oxygen", None, None, None),  # not available
        ("temperature", "Temp", _check_temp, "deg_c"),
        ("systolic_bp", "SBP", _check_sbp, "mm_hg"),
        ("heart_rate", "HR", _check_hr, "beats_per_minute"),
        ("consciousness", None, None, None),  # not available
    ]

    for i in range(n):
        pid = patient.patient_id
        source_ds = patient.source_dataset
        hour = patient.hour[i]
        sepsis_label = patient.sepsis_label[i]

        score_components: list[ComponentResult] = []
        availabilities: list[ComponentAvailability] = []
        total_score = 0

        for comp_name, var, check_fn, unit in components_def:
            if var is None or check_fn is None:
                # Component not available in dataset.
                value = None
                available = False
                score = None
                reason = "variable_not_in_dataset"
            else:
                raw_values = patient.clinical.get(var)
                if raw_values is None:
                    value = None
                    available = False
                    score = None
                    reason = "variable_not_in_patient_data"
                else:
                    value = raw_values[i]
                    available = value is not None and not pd.isna(value)
                    score = check_fn(value) if available else None
                    reason = None

            if score is not None:
                total_score += score

            threshold = _news2_threshold_str(comp_name)

            score_components.append(ComponentResult(
                component_name=comp_name,
                variable=var,
                value=float(value) if value is not None else None,
                unit=unit,
                threshold=threshold,
                met=(score > 0 if score is not None else None),
                available=available,
            ))

            availabilities.append(ComponentAvailability(
                component_name=comp_name,
                variable=var,
                available=available,
                value=float(value) if value is not None else None,
                reason=reason if not available else None,
            ))

        completeness = assess_completeness(
            score_name="news2_style",
            components=availabilities,
            components_required=cfg.news2_components_required,
        )

        score_value = total_score if completeness.can_compute_score else None

        if completeness.level == CompletenessLevel.COMPLETE:
            is_positive = bool(score_value >= cfg.news2_positive_threshold)
        else:
            is_positive = None

        score_entry = ScoreEntry(
            score_name="news2_style",
            version=NEWS2_VERSION,
            category="clinical_baseline",
            data_type="int",
            formula=(
                "Sum of NEWS2 parameter scores (RR, SpO2, supplemental O2, "
                "Temperature, SBP, HR, consciousness). "
                "Modified NEWS-style score; full NEWS2 not available because "
                "supplemental O2 and consciousness are not in the dataset. "
                "Clinical concern if score >= 5 AND all components available."
            ),
            score_value=score_value,
            threshold=cfg.news2_positive_threshold,
            is_positive=is_positive,
            complete=completeness.level == CompletenessLevel.COMPLETE,
            components_available=completeness.components_available,
            components_required=completeness.components_required,
            units="points",
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


def _news2_threshold_str(component: str) -> str | None:
    """Return human-readable threshold string for a NEWS2 component."""
    thresholds = {
        "respiration": "<=8->3, 9-11->1, 12-20->0, 21-24->1, >=25->3",
        "spo2": "<=91->3, 92-93->2, 94-95->1, 96-97->0, >=98->0",
        "supplemental_oxygen": "No->0, Yes->+2 (not available)",
        "temperature": "<=35.0->3, 35.1-36.0->1, 36.1-38.0->0, 38.1-39.0->0, >=39.1->2",
        "systolic_bp": "<=90->3, 91-100->2, 101-119->1, 120-219->0, >=220->3",
        "heart_rate": "<=40->3, 41-50->1, 51-90->0, 91-110->0, 111-130->1, 131-140->2, >=141->3",
        "consciousness": "A->0, C/V/P/U->3 (not available)",
    }
    return thresholds.get(component)


def validate_news2_results(results: list[ScoreResult]) -> list[str]:
    """Validate NEWS2-style results.

    Returns a list of issue messages (empty if valid).
    """
    issues: list[str] = []
    for r in results:
        s = r.score
        if s.score_value is not None and (s.score_value < 0 or s.score_value > (3 * s.components_required)):
            issues.append(
                f"Patient {r.patient_id} hour {r.hour}: "
                f"news2_score {s.score_value} out of range"
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
