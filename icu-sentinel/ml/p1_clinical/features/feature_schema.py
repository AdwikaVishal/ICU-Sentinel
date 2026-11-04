"""Stable feature schema for the P1 temporal feature-engineering pipeline.

Defines the contract that P2 (boosting) and P3 (temporal models) consume.
Each feature entry documents:

- feature_name: column name in the output DataFrame
- source_variable: original clinical variable it derives from (or None)
- category: current / missingness / forward_fill / rolling / trend / clinical
- data_type: float / int / bool
- formula: human-readable calculation
- window_hours: rolling/trend window, where applicable
- units: unit of measure, where applicable
- missing_behavior: how NaN input is handled
- causal: whether the feature uses only current/past observations
- tabular_appropriate: suitable for Logistic Regression / RF / SVM
- sequential_appropriate: suitable for GRU / LSTM / Transformer sequences

The schema also preserves the identifier and target columns:
    patient_id, source_dataset, hour, SepsisLabel
(hospital_id only when genuinely available from source).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Feature categories.
CATEGORY_CURRENT = "current"
CATEGORY_MISSINGNESS = "missingness"
CATEGORY_FORWARD_FILL = "forward_fill"
CATEGORY_ROLLING = "rolling"
CATEGORY_TREND = "trend"
CATEGORY_CLINICAL = "clinical"
CATEGORY_IDENTIFIER = "identifier"
CATEGORY_TARGET = "target"


@dataclass
class FeatureEntry:
    """A single feature definition."""

    feature_name: str
    source_variable: str | None
    category: str
    data_type: str
    formula: str
    window_hours: int | None = None
    units: str | None = None
    missing_behavior: str = "preserved_as_nan"
    causal: bool = True
    tabular_appropriate: bool = True
    sequential_appropriate: bool = True
    notes: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureSchema:
    """Complete feature schema with all feature entries."""

    version: str = "1.0.0"
    description: str = (
        "P1 temporal feature-engineering schema for ICU Sentinel. "
        "All features are causal (use only current/past observations)."
    )
    features: list[FeatureEntry] = field(default_factory=list)
    identifier_columns: list[str] = field(
        default_factory=lambda: ["patient_id", "source_dataset", "hour"]
    )
    target_columns: list[str] = field(default_factory=lambda: ["SepsisLabel"])
    optional_identifier_columns: list[str] = field(
        default_factory=lambda: ["hospital_id"]
    )

    def add_feature(self, entry: FeatureEntry) -> None:
        self.features.append(entry)

    def get_feature(self, feature_name: str) -> FeatureEntry | None:
        for f in self.features:
            if f.feature_name == feature_name:
                return f
        return None

    def feature_names(self) -> list[str]:
        return [f.feature_name for f in self.features]

    def by_category(self, category: str) -> list[FeatureEntry]:
        return [f for f in self.features if f.category == category]

    def causal_features(self) -> list[FeatureEntry]:
        return [f for f in self.features if f.causal]

    def tabular_features(self) -> list[FeatureEntry]:
        return [f for f in self.features if f.tabular_appropriate]

    def sequential_features(self) -> list[FeatureEntry]:
        return [f for f in self.features if f.sequential_appropriate]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "description": self.description,
            "identifier_columns": list(self.identifier_columns),
            "target_columns": list(self.target_columns),
            "optional_identifier_columns": list(self.optional_identifier_columns),
            "features": [f.to_dict() for f in self.features],
        }

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> FeatureSchema:
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)
        features = [FeatureEntry(**fd) for fd in data.get("features", [])]
        return cls(
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            features=features,
            identifier_columns=data.get("identifier_columns", ["patient_id", "source_dataset", "hour"]),
            target_columns=data.get("target_columns", ["SepsisLabel"]),
            optional_identifier_columns=data.get("optional_identifier_columns", ["hospital_id"]),
        )


def build_default_schema(config=None) -> FeatureSchema:
    """Build the default feature schema from the feature configuration.

    This enumerates every feature the pipeline can generate, so P2/P3
    know the complete output space without running the pipeline.
    """

    if config is None:
        from .feature_config import FeatureConfig
        config = FeatureConfig()

    schema = FeatureSchema()

    # --- Identifier columns ---
    for col in ["patient_id", "source_dataset", "hour"]:
        schema.add_feature(FeatureEntry(
            feature_name=col,
            source_variable=None,
            category=CATEGORY_IDENTIFIER,
            data_type="str" if col != "hour" else "int",
            formula="Preserved from Part 3 cleaned output",
            missing_behavior="required",
            causal=True,
            tabular_appropriate=False,
            sequential_appropriate=False,
            notes="Identifier/metadata; not a model input feature",
        ))
    schema.add_feature(FeatureEntry(
        feature_name="hospital_id",
        source_variable=None,
        category=CATEGORY_IDENTIFIER,
        data_type="str",
        formula="Preserved from Part 3 cleaned output (when available)",
        missing_behavior="preserved_as_nan",
        causal=True,
        tabular_appropriate=False,
        sequential_appropriate=False,
        notes="Optional; only when genuinely available from source data",
    ))

    # --- Target column ---
    schema.add_feature(FeatureEntry(
        feature_name="SepsisLabel",
        source_variable="SepsisLabel",
        category=CATEGORY_TARGET,
        data_type="int",
        formula="Original Part 3 SepsisLabel; preserved unchanged (target only)",
        missing_behavior="preserved_as_nan",
        causal=True,
        tabular_appropriate=False,
        sequential_appropriate=False,
        notes="Target field only; never used as a model input feature",
    ))

    # --- Current observation features ---
    for var in config.feature_variables:
        schema.add_feature(FeatureEntry(
            feature_name=var,
            source_variable=var,
            category=CATEGORY_CURRENT,
            data_type="float",
            formula=f"Original {var} measurement at current hour",
            units=_units_for(var),
            missing_behavior="preserved_as_nan",
            causal=True,
            tabular_appropriate=True,
            sequential_appropriate=True,
        ))
    for var in config.static_variables:
        schema.add_feature(FeatureEntry(
            feature_name=var,
            source_variable=var,
            category=CATEGORY_CURRENT,
            data_type="float",
            formula=f"Static {var} value (repeated for each hour)",
            units=_units_for(var),
            missing_behavior="preserved_as_nan",
            causal=True,
            tabular_appropriate=True,
            sequential_appropriate=True,
        ))

    # --- Missingness features ---
    if config.compute_missingness_features:
        for var in config.feature_variables:
            schema.add_feature(FeatureEntry(
                feature_name=f"{var}_is_missing",
                source_variable=var,
                category=CATEGORY_MISSINGNESS,
                data_type="bool",
                formula=f"1 if {var} is missing at current hour, else 0",
                missing_behavior="computed_from_input",
                causal=True,
                tabular_appropriate=True,
                sequential_appropriate=True,
            ))
            schema.add_feature(FeatureEntry(
                feature_name=f"{var}_hours_since_last",
                source_variable=var,
                category=CATEGORY_MISSINGNESS,
                data_type="float",
                formula=f"Hours since last valid {var} measurement (current/past only)",
                units="hours",
                missing_behavior="inf_or_nan_if_never_observed",
                causal=True,
                tabular_appropriate=True,
                sequential_appropriate=True,
                notes="inf if the variable has never been observed up to this hour",
            ))
            schema.add_feature(FeatureEntry(
                feature_name=f"{var}_availability_6h",
                source_variable=var,
                category=CATEGORY_MISSINGNESS,
                data_type="float",
                formula=f"Fraction of hours with valid {var} in trailing 6h window",
                missing_behavior="0_if_no_valid_obs",
                causal=True,
                tabular_appropriate=True,
                sequential_appropriate=True,
                notes="Window is trailing and causal (includes current hour)",
            ))

    # --- Forward-filled features ---
    if config.compute_forward_fill:
        for var in config.feature_variables:
            window = config.forward_fill_window_for(var)
            schema.add_feature(FeatureEntry(
                feature_name=f"{var}_ffilled",
                source_variable=var,
                category=CATEGORY_FORWARD_FILL,
                data_type="float",
                formula=f"Causally forward-filled {var}, expires after {window}h gap",
                units=_units_for(var),
                missing_behavior="na_if_gap_exceeds_window",
                causal=True,
                tabular_appropriate=True,
                sequential_appropriate=True,
                notes=f"Original {var} preserved unchanged; ffilled is a separate column",
            ))

    # --- Rolling features ---
    if config.compute_rolling_features:
        for var in config.feature_variables:
            for window in config.rolling_window_hours:
                for agg in config.rolling_aggregations:
                    schema.add_feature(FeatureEntry(
                        feature_name=f"{var}_rolling_{agg}_{window}h",
                        source_variable=var,
                        category=CATEGORY_ROLLING,
                        data_type="float",
                        formula=f"Causal trailing {agg} of {var} over {window}h window",
                        window_hours=window,
                        units=_units_for(var),
                        missing_behavior="nan_if_insufficient_obs",
                        causal=True,
                        tabular_appropriate=True,
                        sequential_appropriate=True,
                        notes=(
                            f"Window includes current hour; requires >= "
                            f"{config.rolling_min_observations} valid obs"
                        ),
                    ))

    # --- Trend / slope features ---
    if config.compute_trend_features:
        for var in config.feature_variables:
            for window in config.trend_window_hours:
                schema.add_feature(FeatureEntry(
                    feature_name=f"{var}_abs_change_{window}h",
                    source_variable=var,
                    category=CATEGORY_TREND,
                    data_type="float",
                    formula=f"{var}[t] - {var}[t-{window}h] (causal, last valid obs)",
                    units=_units_for(var),
                    missing_behavior="nan_if_insufficient_obs",
                    causal=True,
                    tabular_appropriate=True,
                    sequential_appropriate=True,
                ))
                schema.add_feature(FeatureEntry(
                    feature_name=f"{var}_rel_change_{window}h",
                    source_variable=var,
                    category=CATEGORY_TREND,
                    data_type="float",
                    formula=f"({var}[t] - {var}[t-{window}h]) / |{var}[t-{window}h]|",
                    missing_behavior="nan_if_insufficient_or_zero_denominator",
                    causal=True,
                    tabular_appropriate=True,
                    sequential_appropriate=True,
                    notes="NaN if denominator is zero or missing",
                ))
                schema.add_feature(FeatureEntry(
                    feature_name=f"{var}_slope_{window}h",
                    source_variable=var,
                    category=CATEGORY_TREND,
                    data_type="float",
                    formula=f"Slope of {var} over trailing {window}h (change per hour)",
                    units=f"{_units_for(var)}/hour" if _units_for(var) else "per_hour",
                    missing_behavior="nan_if_insufficient_obs",
                    causal=True,
                    tabular_appropriate=True,
                    sequential_appropriate=True,
                ))

    # --- Clinical features ---
    if config.compute_clinical_features:
        # Shock index = HR / SBP
        schema.add_feature(FeatureEntry(
            feature_name="shock_index",
            source_variable="HR, SBP",
            category=CATEGORY_CLINICAL,
            data_type="float",
            formula="HR / SBP (NaN if SBP is zero or missing)",
            units="unitless",
            missing_behavior="nan_if_denominator_zero_or_missing",
            causal=True,
            tabular_appropriate=True,
            sequential_appropriate=True,
            notes="Experimental derived feature; not a replacement for clinical assessment",
        ))

        # SIRS component count
        schema.add_feature(FeatureEntry(
            feature_name="sirs_count",
            source_variable="Temp, HR, Resp, WBC",
            category=CATEGORY_CLINICAL,
            data_type="int",
            formula="Count of SIRS criteria met among available components",
            missing_behavior="count_over_available_components_only",
            causal=True,
            tabular_appropriate=True,
            sequential_appropriate=True,
            notes=(
                "Components: Temp>38.3 or <36.0, HR>90, Resp>20, WBC>12 or <4. "
                "Missing components are not counted as normal. "
                "sirs_components_available reports how many were evaluable."
            ),
        ))
        schema.add_feature(FeatureEntry(
            feature_name="sirs_components_available",
            source_variable="Temp, HR, Resp, WBC",
            category=CATEGORY_CLINICAL,
            data_type="int",
            formula="Number of SIRS components with valid measurements at current hour",
            missing_behavior="count_of_non_missing_components",
            causal=True,
            tabular_appropriate=True,
            sequential_appropriate=True,
        ))

    return schema


def _units_for(variable: str) -> str | None:
    """Return the documented unit for a clinical variable."""
    unit_map = {
        "HR": "beats_per_minute",
        "O2Sat": "percent",
        "Temp": "deg_c",
        "SBP": "mm_hg",
        "MAP": "mm_hg",
        "DBP": "mm_hg",
        "Resp": "breaths_per_minute",
        "EtCO2": "mm_hg",
        "BaseExcess": "mmol_per_l",
        "HCO3": "mmol_per_l",
        "FiO2": "percent",
        "pH": "unitless",
        "PaCO2": "mm_hg",
        "SaO2": "percent",
        "AST": "iu_per_l",
        "BUN": "mg_per_dl",
        "Alkalinephos": "iu_per_l",
        "Calcium": "mg_per_dl",
        "Chloride": "mmol_per_l",
        "Creatinine": "mg_per_dl",
        "Bilirubin_direct": "mg_per_dl",
        "Glucose": "mg_per_dl",
        "Lactate": "mg_per_dl",
        "Magnesium": "mmol_per_dl",
        "Phosphate": "mg_per_dl",
        "Potassium": "mmol_per_l",
        "Bilirubin_total": "mg_per_dl",
        "TroponinI": "ng_per_ml",
        "Hct": "percent",
        "Hgb": "g_per_dl",
        "PTT": "seconds",
        "WBC": "count_times_10_3_per_ul",
        "Fibrinogen": "mg_per_dl",
        "Platelets": "count_times_10_3_per_ul",
        "Age": "years",
        "Gender": "code",
        "Unit1": "code",
        "Unit2": "code",
        "HospAdmTime": "hours",
        "ICULOS": "hours",
    }
    return unit_map.get(variable)
