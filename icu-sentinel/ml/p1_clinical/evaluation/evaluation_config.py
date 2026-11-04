"""Centralized configuration for P1 research evaluation.

All evaluation parameters, thresholds, alert settings, bootstrap settings,
and subgroup definitions live here. Serializable to/from JSON for
reproducibility.

Defaults are reasonable starting points and are NOT tuned on any real
dataset (real data is unavailable).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Random seed for reproducibility.
RANDOM_SEED: int = 42

# Alert threshold defaults.
ALERT_THRESHOLD: float = 0.5
ALERT_COOLDOWN_HOURS: int = 6
ALERT_MIN_DURATION_HOURS: int = 1
ALERT_MIN_CONSECUTIVE: int = 1

# Bootstrap defaults.
BOOTSTRAP_N_ITERATIONS: int = 1000
BOOTSTRAP_CONFIDENCE_LEVEL: float = 0.95

# Metric thresholds.
AUROC_HIGH_RISK: float = 0.80
AUPRC_BASELINE_RATE: float | None = None

# Lead-time defaults.
LEAD_TIME_WINDOW_HOURS: int = 6

# Alert burden defaults.
FALSE_ALERTS_PER_PATIENT_DAY_WINDOW_HOURS: int = 24

# Feature ablation defaults.
FEATURE_ABLATION_GROUPS: list[str] = [
    "current",
    "missingness",
    "forward_fill",
    "rolling",
    "trend",
    "clinical",
]


@dataclass
class EvaluationConfig:
    """Serializable evaluation configuration."""

    # Reproducibility.
    random_seed: int = RANDOM_SEED

    # Alerting.
    alert_threshold: float = ALERT_THRESHOLD
    alert_cooldown_hours: int = ALERT_COOLDOWN_HOURS
    alert_min_duration_hours: int = ALERT_MIN_DURATION_HOURS
    alert_min_consecutive: int = ALERT_MIN_CONSECUTIVE

    # Bootstrap.
    bootstrap_n_iterations: int = BOOTSTRAP_N_ITERATIONS
    bootstrap_confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL

    # Lead time.
    lead_time_window_hours: int = LEAD_TIME_WINDOW_HOURS

    # Alert burden.
    false_alerts_per_patient_day_window_hours: int = (
        FALSE_ALERTS_PER_PATIENT_DAY_WINDOW_HOURS
    )

    # Feature ablation.
    feature_ablation_groups: list[str] = field(
        default_factory=lambda: [
            "current",
            "missingness",
            "forward_fill",
            "rolling",
            "trend",
            "clinical",
        ]
    )

    # Subgroups.
    subgroup_definitions: dict[str, list[str]] = field(default_factory=dict)

    # Metrics.
    auroc_high_risk: float = AUROC_HIGH_RISK
    auprc_baseline_rate: float | None = AUPRC_BASELINE_RATE

    # Models to evaluate.
    evaluate_logistic_regression: bool = True
    evaluate_random_forest: bool = True
    evaluate_svm: bool = True
    evaluate_sirs: bool = True
    evaluate_qsofa: bool = True
    evaluate_news2_style: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EvaluationConfig:
        return cls(**data)

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> EvaluationConfig:
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def enabled_models(self) -> list[str]:
        """Return list of ML model names enabled for evaluation."""
        models: list[str] = []
        if self.evaluate_logistic_regression:
            models.append("logistic_regression")
        if self.evaluate_random_forest:
            models.append("random_forest")
        if self.evaluate_svm:
            models.append("svm")
        return models

    def enabled_baselines(self) -> list[str]:
        """Return list of clinical baseline names enabled for evaluation."""
        models: list[str] = []
        if self.evaluate_sirs:
            models.append("sirs")
        if self.evaluate_qsofa:
            models.append("qsofa")
        if self.evaluate_news2_style:
            models.append("news2_style")
        return models

    def all_model_names(self) -> list[str]:
        """Return all model names (ML + baselines) enabled for evaluation."""
        return self.enabled_models() + self.enabled_baselines()
