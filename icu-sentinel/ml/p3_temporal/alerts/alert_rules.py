"""Threshold configuration and pure alert rules.

All thresholds are configuration values. The defaults are demonstration
values only and are not clinically validated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .alert_schemas import AlertThresholdInfo

DEFAULT_RULE_ID = "sepsis_risk_threshold"
DEFAULT_THRESHOLD = 0.60
DEFAULT_THRESHOLD_SOURCE = "demo_configuration"
DEFAULT_THRESHOLD_VERSION = "demo-threshold-v1"
DEFAULT_DEDUP_WINDOW_HOURS = 6


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AlertThresholdConfig:
    """Configurable alert threshold and episode lifecycle policy.

    The threshold is not derived from test-set labels. It is supplied by
    configuration and is marked as demonstration/configuration-only
    unless a validated threshold is explicitly configured later.
    """

    threshold: float = DEFAULT_THRESHOLD
    source: str = DEFAULT_THRESHOLD_SOURCE
    version: str = DEFAULT_THRESHOLD_VERSION
    rule_id: str = DEFAULT_RULE_ID
    dedup_window_hours: int = DEFAULT_DEDUP_WINDOW_HOURS
    clear_on_below_threshold: bool = True
    clinically_validated: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("alert threshold must be between 0.0 and 1.0")
        if self.dedup_window_hours < 0:
            raise ValueError("dedup_window_hours must be >= 0")
        if not self.source.strip() or not self.version.strip():
            raise ValueError("threshold source and version must be non-empty")

    def to_public(self) -> AlertThresholdInfo:
        return AlertThresholdInfo(
            rule_id=self.rule_id,
            threshold=float(self.threshold),
            source=self.source,
            version=self.version,
            dedup_window_hours=int(self.dedup_window_hours),
            clear_on_below_threshold=bool(self.clear_on_below_threshold),
            clinically_validated=bool(self.clinically_validated),
        )

    @classmethod
    def from_env(cls) -> AlertThresholdConfig:
        return cls(
            threshold=_env_float("ICU_SENTINEL_ALERT_THRESHOLD", DEFAULT_THRESHOLD),
            source=os.getenv("ICU_SENTINEL_ALERT_THRESHOLD_SOURCE", DEFAULT_THRESHOLD_SOURCE),
            version=os.getenv("ICU_SENTINEL_ALERT_THRESHOLD_VERSION", DEFAULT_THRESHOLD_VERSION),
            rule_id=os.getenv("ICU_SENTINEL_ALERT_RULE_ID", DEFAULT_RULE_ID),
            dedup_window_hours=_env_int(
                "ICU_SENTINEL_ALERT_DEDUP_WINDOW_HOURS", DEFAULT_DEDUP_WINDOW_HOURS
            ),
            clear_on_below_threshold=_env_bool(
                "ICU_SENTINEL_ALERT_CLEAR_ON_BELOW", True
            ),
            clinically_validated=_env_bool(
                "ICU_SENTINEL_ALERT_THRESHOLD_VALIDATED", False
            ),
        )


def is_triggered(probability: float, threshold: float) -> bool:
    """Apply the single configured rule: probability >= threshold."""

    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must be between 0.0 and 1.0")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")
    return float(probability) >= float(threshold)


def is_same_episode(
    *,
    existing_last_hour: int,
    current_hour: int,
    dedup_window_hours: int,
) -> bool:
    """Whether an above-threshold prediction continues an existing episode.

    Predictions arriving out of order (current_hour < last_hour) are
    treated as the same episode and must not roll the episode backwards.
    A gap larger than ``dedup_window_hours`` starts a new episode.
    """

    if current_hour <= existing_last_hour:
        return True
    return (current_hour - existing_last_hour) <= dedup_window_hours
