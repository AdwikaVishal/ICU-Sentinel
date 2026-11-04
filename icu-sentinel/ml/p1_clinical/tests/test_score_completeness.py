"""Tests for score_completeness module.

Tests completeness level determination (COMPLETE/PARTIAL/NONE),
availability ratio, and component tracking.
"""

from __future__ import annotations

import pytest

from ml.p1_clinical.baselines.score_completeness import (
    CompletenessLevel,
    ComponentAvailability,
    ScoreCompleteness,
    assess_completeness,
    is_score_complete,
    is_score_none,
    is_score_partial,
)

# ------------------------------------------------------------------
# CompletenessLevel constants
# ------------------------------------------------------------------


def test_completeness_level_constants():
    assert CompletenessLevel.COMPLETE == "complete"
    assert CompletenessLevel.PARTIAL == "partial"
    assert CompletenessLevel.NONE == "none"


# ------------------------------------------------------------------
# assess_completeness tests
# ------------------------------------------------------------------


def test_assess_complete_all_available():
    comps = [
        ComponentAvailability("a", "var_a", True, 1.0),
        ComponentAvailability("b", "var_b", True, 2.0),
        ComponentAvailability("c", "var_c", True, 3.0),
    ]
    c = assess_completeness("test_score", comps, 3)
    assert c.level == CompletenessLevel.COMPLETE
    assert c.components_available == 3
    assert c.components_required == 3
    assert c.availability_ratio == 1.0
    assert c.can_compute_score is True
    assert is_score_complete(c)
    assert not is_score_partial(c)
    assert not is_score_none(c)


def test_assess_partial_some_available():
    comps = [
        ComponentAvailability("a", "var_a", True, 1.0),
        ComponentAvailability("b", "var_b", False, None),
        ComponentAvailability("c", "var_c", True, 3.0),
    ]
    c = assess_completeness("test_score", comps, 3)
    assert c.level == CompletenessLevel.PARTIAL
    assert c.components_available == 2
    assert c.availability_ratio == pytest.approx(2 / 3)
    assert c.can_compute_score is True
    assert is_score_partial(c)
    assert not is_score_complete(c)


def test_assess_none_none_available():
    comps = [
        ComponentAvailability("a", "var_a", False, None),
        ComponentAvailability("b", "var_b", False, None),
    ]
    c = assess_completeness("test_score", comps, 2)
    assert c.level == CompletenessLevel.NONE
    assert c.components_available == 0
    assert c.availability_ratio == 0.0
    assert c.can_compute_score is False
    assert is_score_none(c)


def test_assess_reason_messages():
    comps_all = [
        ComponentAvailability("a", "var_a", True, 1.0),
        ComponentAvailability("b", "var_b", True, 2.0),
    ]
    c = assess_completeness("test_score", comps_all, 2)
    assert c.reason is None

    comps_partial = [
        ComponentAvailability("a", "var_a", True, 1.0),
        ComponentAvailability("b", "var_b", False, None),
    ]
    c2 = assess_completeness("test_score", comps_partial, 2)
    assert c2.reason is not None
    assert "unavailable" in c2.reason

    comps_none = [
        ComponentAvailability("a", "var_a", False, None),
        ComponentAvailability("b", "var_b", False, None),
    ]
    c3 = assess_completeness("test_score", comps_none, 2)
    assert c3.reason is not None
    assert "No components available" in c3.reason


# ------------------------------------------------------------------
# ScoreCompleteness dataclass tests
# ------------------------------------------------------------------


def test_score_completeness_to_dict():
    comps = [
        ComponentAvailability("temp", "Temp", True, 37.0),
    ]
    c = ScoreCompleteness(
        score_name="sirs",
        level=CompletenessLevel.PARTIAL,
        components_available=1,
        components_required=4,
        components=comps,
        can_compute_score=True,
        reason="3 of 4 components unavailable",
    )
    d = c.to_dict()
    assert d["score_name"] == "sirs"
    assert d["level"] == "partial"
    assert d["components_available"] == 1
    assert d["components_required"] == 4
    assert d["availability_ratio"] == 0.25
    assert d["can_compute_score"] is True
    assert len(d["components"]) == 1


# ------------------------------------------------------------------
# Availability ratio edge cases
# ------------------------------------------------------------------


def test_availability_ratio_zero_required():
    comps = []
    c = assess_completeness("test_score", comps, 0)
    assert c.availability_ratio == 0.0
