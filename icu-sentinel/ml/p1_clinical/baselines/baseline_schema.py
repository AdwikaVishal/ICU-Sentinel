"""Shared score schema for P1 clinical baselines.

Defines the contract for clinical baseline score outputs (SIRS, qSOFA,
NEWS-style). Each score entry documents:

- score_name: name of the score (e.g. "sirs", "qsofa", "news2_style")
- version: score version
- category: baseline category
- data_type: int / float / str
- formula: human-readable calculation
- threshold: positive threshold for the score
- units: unit of measure, where applicable
- causal: whether the score uses only current/past observations
- complete: whether all required components were available
- components_available: number of components with valid measurements
- components_required: total number of components required
- missing_behavior: how missing input is handled
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ScoreEntry:
    """A single clinical baseline score result for one patient-hour."""

    score_name: str
    version: str
    category: str
    data_type: str
    formula: str
    score_value: int | None
    threshold: int | None
    is_positive: bool | None
    complete: bool
    components_available: int
    components_required: int
    units: str | None = None
    causal: bool = True
    missing_behavior: str = "components_without_data_not_counted"
    notes: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComponentResult:
    """Result for a single component of a clinical score."""

    component_name: str
    variable: str | None
    value: float | None
    unit: str | None
    threshold: str | None
    met: bool | None  # True if criterion met, False if not, None if missing
    available: bool


@dataclass
class ScoreResult:
    """Complete score result including all components."""

    patient_id: str
    source_dataset: str | None
    hour: int
    sepsis_label: int
    score: ScoreEntry
    components: list[ComponentResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "source_dataset": self.source_dataset,
            "hour": self.hour,
            "sepsis_label": self.sepsis_label,
            "score": self.score.to_dict(),
            "components": [c.__dict__ for c in self.components],
        }


@dataclass
class BaselineSchema:
    """Schema documentation for all clinical baseline scores."""

    version: str = "1.0.0"
    description: str = (
        "P1 clinical baseline scoring schema for ICU Sentinel. "
        "Scores are computed per patient-hour using only current/past "
        "observations (causal)."
    )
    scores: list[ScoreEntry] = field(default_factory=list)

    def add_score(self, entry: ScoreEntry) -> None:
        self.scores.append(entry)

    def get_score(self, score_name: str) -> ScoreEntry | None:
        for s in self.scores:
            if s.score_name == score_name:
                return s
        return None

    def score_names(self) -> list[str]:
        return [s.score_name for s in self.scores]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "description": self.description,
            "scores": [s.to_dict() for s in self.scores],
        }

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> BaselineSchema:
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)
        scores = [ScoreEntry(**sd) for sd in data.get("scores", [])]
        return cls(version=data.get("version", "1.0.0"),
                   description=data.get("description", ""),
                   scores=scores)


def build_default_schema() -> BaselineSchema:
    """Build the default baseline schema documenting all scores."""

    schema = BaselineSchema()

    # SIRS
    schema.add_score(ScoreEntry(
        score_name="sirs",
        version="1992_consensus",
        category="clinical_baseline",
        data_type="int",
        formula="Count of SIRS criteria met among available components "
                "(Temp>38.3 or <36.0, HR>90, Resp>20, WBC>12 or <4). "
                "Positive if score >= 2 and all 4 components available.",
        score_value=None,
        threshold=2,
        is_positive=None,
        complete=False,
        components_available=0,
        components_required=4,
        units="count",
        causal=True,
        missing_behavior="unavailable_components_not_counted",
        notes="SIRS per Bone et al., Chest 1992. Partial scores exposed but "
              "not labeled as complete SIRS.",
    ))

    # qSOFA
    schema.add_score(ScoreEntry(
        score_name="qsofa",
        version="sepsis_3_2016",
        category="clinical_baseline",
        data_type="int",
        formula="Count of qSOFA criteria met among available components "
                "(RR>=22, SBP<=100, altered mental status). "
                "Positive if score >= 2 and all 3 components available.",
        score_value=None,
        threshold=2,
        is_positive=None,
        complete=False,
        components_available=0,
        components_required=3,
        units="count",
        causal=True,
        missing_behavior="unavailable_components_not_counted",
        notes="qSOFA per Sepsis-3 (Singer et al., JAMA 2016). Altered mental "
              "status is not available in the Challenge dataset; qSOFA will "
              "be partial unless AMS data is provided externally.",
    ))

    # NEWS2-style
    schema.add_score(ScoreEntry(
        score_name="news2_style",
        version="news2_rcp",
        category="clinical_baseline",
        data_type="int",
        formula="Sum of NEWS2 parameter scores (RR, SpO2, supplemental O2, "
                "Temperature, SBP, HR, consciousness). "
                "Modified NEWS-style score; full NEWS2 not available because "
                "supplemental O2 and consciousness are not in the dataset.",
        score_value=None,
        threshold=5,
        is_positive=None,
        complete=False,
        components_available=0,
        components_required=7,
        units="points",
        causal=True,
        missing_behavior="unavailable_components_not_counted",
        notes="NEWS2 per Royal College of Physicians. Modified NEWS-style "
              "score; supplemental oxygen and consciousness level are not "
              "available in the Challenge dataset.",
    ))

    return schema
