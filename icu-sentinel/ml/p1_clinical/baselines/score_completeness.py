"""Score completeness tracking for P1 clinical baselines.

Tracks whether each clinical score is complete (all required components
available) or partial (some components missing). Missing components are
never treated as normal values.

Completeness levels:
- COMPLETE: all required components have valid measurements
- PARTIAL: some but not all components have valid measurements
- NONE: no components have valid measurements
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("icu_sentinel.p1_clinical.baselines.score_completeness")


class CompletenessLevel:
    COMPLETE = "complete"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class ComponentAvailability:
    """Availability status for a single score component."""

    component_name: str
    variable: str | None
    available: bool
    value: float | None
    reason: str | None = None  # e.g. "missing_value", "variable_not_in_dataset"


@dataclass
class ScoreCompleteness:
    """Completeness result for a single score at one patient-hour."""

    score_name: str
    level: str  # COMPLETE / PARTIAL / NONE
    components_available: int
    components_required: int
    components: list[ComponentAvailability] = field(default_factory=list)
    can_compute_score: bool = False
    reason: str | None = None

    @property
    def availability_ratio(self) -> float:
        if self.components_required == 0:
            return 0.0
        return self.components_available / self.components_required

    def to_dict(self) -> dict:
        return {
            "score_name": self.score_name,
            "level": self.level,
            "components_available": self.components_available,
            "components_required": self.components_required,
            "availability_ratio": round(self.availability_ratio, 4),
            "can_compute_score": self.can_compute_score,
            "reason": self.reason,
            "components": [
                {
                    "component_name": c.component_name,
                    "variable": c.variable,
                    "available": c.available,
                    "value": c.value,
                    "reason": c.reason,
                }
                for c in self.components
            ],
        }


def assess_completeness(
    score_name: str,
    components: list[ComponentAvailability],
    components_required: int,
) -> ScoreCompleteness:
    """Assess completeness of a score based on component availability.

    A score is COMPLETE only if ALL required components are available.
    PARTIAL if at least one but not all are available.
    NONE if none are available.

    Missing components are never treated as normal values.
    """
    n_available = sum(1 for c in components if c.available)

    if n_available == components_required:
        level = CompletenessLevel.COMPLETE
        can_compute = True
        reason = None
    elif n_available > 0:
        level = CompletenessLevel.PARTIAL
        can_compute = True  # partial score can be computed but labeled partial
        reason = (
            f"{components_required - n_available} of {components_required} "
            "component(s) unavailable; score is partial, not complete"
        )
    else:
        level = CompletenessLevel.NONE
        can_compute = False
        reason = "No components available; score cannot be computed"

    return ScoreCompleteness(
        score_name=score_name,
        level=level,
        components_available=n_available,
        components_required=components_required,
        components=list(components),
        can_compute_score=can_compute,
        reason=reason,
    )


def is_score_complete(completeness: ScoreCompleteness) -> bool:
    """Return True if the score is complete (all components available)."""
    return completeness.level == CompletenessLevel.COMPLETE


def is_score_partial(completeness: ScoreCompleteness) -> bool:
    """Return True if the score is partial (some components missing)."""
    return completeness.level == CompletenessLevel.PARTIAL


def is_score_none(completeness: ScoreCompleteness) -> bool:
    """Return True if no components are available."""
    return completeness.level == CompletenessLevel.NONE
