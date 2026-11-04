"""Map raw P1 feature names to human-readable, honest metadata.

The P1 feature schema is the source of truth for known features. Unknown
feature names are never given an invented clinical meaning: they keep
their original name, are placed in an ``unknown`` category, and are
marked ``metadata_available=False``.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ml.p1_clinical.features.feature_schema import (
    CATEGORY_CLINICAL,
    CATEGORY_CURRENT,
    CATEGORY_FORWARD_FILL,
    CATEGORY_MISSINGNESS,
    CATEGORY_ROLLING,
    CATEGORY_TREND,
    FeatureEntry,
    FeatureSchema,
    build_default_schema,
)

from ml.p3_temporal.common.clinical_catalog import display_name as base_display_name

from .explanation_schemas import FeatureMetadata

CATEGORY_UNKNOWN = "unknown"

_CATEGORY_ORDER: dict[str, int] = {
    CATEGORY_CURRENT: 0,
    CATEGORY_MISSINGNESS: 1,
    CATEGORY_FORWARD_FILL: 2,
    CATEGORY_ROLLING: 3,
    CATEGORY_TREND: 4,
    CATEGORY_CLINICAL: 5,
    CATEGORY_UNKNOWN: 6,
}


@lru_cache(maxsize=1)
def _default_schema() -> FeatureSchema:
    return build_default_schema()


_ROLLING_RE = re.compile(r"^(?P<var>.+)_rolling_(?P<agg>[a-z]+)_(?P<window>\d+)h$")
_TREND_RE = re.compile(
    r"^(?P<var>.+)_(?P<kind>abs_change|rel_change|slope)_(?P<window>\d+)h$"
)
_MISSING_RE = re.compile(r"^(?P<var>.+)_(?P<kind>is_missing|hours_since_last)$")
_AVAILABILITY_RE = re.compile(r"^(?P<var>.+)_availability_(?P<window>\d+)h$")
_FFILL_RE = re.compile(r"^(?P<var>.+)_ffilled$")

_CLINICAL_LABELS: dict[str, str] = {
    "shock_index": "Shock index (heart rate / systolic blood pressure)",
    "sirs_count": "SIRS component count",
    "sirs_components_available": "SIRS components available",
}


class FeatureMapper:
    """Deterministic feature-name to metadata mapper."""

    def __init__(self, schema: FeatureSchema | None = None) -> None:
        self._schema = schema or _default_schema()

    def describe(self, feature_name: str) -> FeatureMetadata:
        entry = self._schema.get_feature(feature_name)
        if entry is not None:
            return self._from_entry(feature_name, entry)
        return self._from_unknown_name(feature_name)

    def describe_many(self, feature_names: list[str]) -> dict[str, FeatureMetadata]:
        return {name: self.describe(name) for name in feature_names}

    def sort_key(self, feature_name: str) -> tuple[int, str]:
        metadata = self.describe(feature_name)
        return (_CATEGORY_ORDER.get(metadata.category, 99), metadata.display_name)

    # ------------------------------------------------------------------
    # Known features (present in the P1 feature schema)
    # ------------------------------------------------------------------
    def _from_entry(self, feature_name: str, entry: FeatureEntry) -> FeatureMetadata:
        source = entry.source_variable
        source_base = _first_variable(source) if source else feature_name
        readable = _readable_feature_name(feature_name, entry, source_base)

        return FeatureMetadata(
            feature_name=feature_name,
            display_name=readable,
            category=entry.category,
            description=entry.formula,
            unit=entry.units,
            feature_type=entry.data_type,
            is_missing_indicator=entry.category == CATEGORY_MISSINGNESS,
            is_engineered=entry.category
            in {CATEGORY_MISSINGNESS, CATEGORY_FORWARD_FILL, CATEGORY_ROLLING, CATEGORY_TREND},
            metadata_available=True,
        )

    # ------------------------------------------------------------------
    # Unknown features: preserve the raw name, invent nothing
    # ------------------------------------------------------------------
    def _from_unknown_name(self, feature_name: str) -> FeatureMetadata:
        is_missing = feature_name.endswith(("_is_missing", "_hours_since_last"))
        is_engineered = bool(
            is_missing
            or "_rolling_" in feature_name
            or "_abs_change_" in feature_name
            or "_rel_change_" in feature_name
            or "_slope_" in feature_name
            or feature_name.endswith("_ffilled")
        )
        return FeatureMetadata(
            feature_name=feature_name,
            display_name=feature_name,
            category=CATEGORY_UNKNOWN,
            description=None,
            unit=None,
            feature_type=None,
            is_missing_indicator=is_missing,
            is_engineered=is_engineered,
            metadata_available=False,
        )


def _first_variable(source_variable: str) -> str:
    return source_variable.split(",")[0].strip()


def _readable_feature_name(
    feature_name: str,
    entry: FeatureEntry,
    source_base: str,
) -> str:
    base = base_display_name(source_base)

    if entry.category == CATEGORY_CURRENT:
        return base

    if entry.category == CATEGORY_CLINICAL and feature_name in _CLINICAL_LABELS:
        return _CLINICAL_LABELS[feature_name]

    match = _MISSING_RE.match(feature_name)
    if match:
        if match.group("kind") == "is_missing":
            return f"{base} missing indicator"
        return f"Hours since last {base.lower()} measurement"

    match = _AVAILABILITY_RE.match(feature_name)
    if match:
        return f"{base} availability over {match.group('window')}h"

    match = _FFILL_RE.match(feature_name)
    if match:
        return f"{base} (forward-filled)"

    match = _ROLLING_RE.match(feature_name)
    if match:
        agg = match.group("agg")
        window = match.group("window")
        return f"{base} {agg.replace('_', ' ')} over {window}h"

    match = _TREND_RE.match(feature_name)
    if match:
        kind = match.group("kind")
        window = match.group("window")
        if kind == "abs_change":
            return f"{base} absolute change over {window}h"
        if kind == "rel_change":
            return f"{base} relative change over {window}h"
        return f"{base} slope over {window}h"

    # Known schema entry with an unexpected naming pattern: use the
    # documented source variable, but keep the raw name visible.
    return f"{base} ({feature_name})"
