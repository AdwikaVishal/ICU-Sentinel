# Feature Engineering Documentation — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 4 — Temporal feature engineering  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Overview

The P1 temporal feature-engineering pipeline transforms cleaned hourly
patient observations (Part 3 output) into a feature-rich dataset suitable
for:

- **P1:** Logistic Regression, Random Forest, SVM
- **P2:** Extra Trees, LightGBM, XGBoost, optional CatBoost
- **P3:** GRU, LSTM, optional Transformer

The same feature definitions are usable across all three workstreams.
The pipeline is implemented in `ml/p1_clinical/features/`.

All features are **causal**: for any patient and prediction hour `t`, no
observation after `t` contributes to any feature at `t`.

---

## 2. Module Structure

| Module | Responsibility |
|--------|---------------|
| `feature_config.py` | Centralized serializable configuration |
| `feature_schema.py` | Stable feature schema (P2/P3 contract) |
| `current_features.py` | Raw observation features |
| `missingness_features.py` | Missingness indicators, time-since-last, availability |
| `rolling_features.py` | Causal rolling statistics (6/12/24h) |
| `trend_features.py` | Absolute/relative change, slopes |
| `clinical_features.py` | Shock index, SIRS component count |
| `feature_pipeline.py` | End-to-end orchestration |

---

## 3. Feature Configuration

All parameters are centralized in `FeatureConfig` (`feature_config.py`)
and serializable to/from JSON. Defaults are **experimental engineering
parameters requiring clinical validation** — they are not medically
established values.

| Parameter | Default | Description |
|-----------|---------|-------------|
| rolling_window_hours | [6, 12, 24] | Trailing causal window sizes |
| rolling_aggregations | [mean, min, max, std, latest] | Rolling aggregation functions |
| rolling_min_observations | 1 | Minimum valid obs for rolling stat |
| trend_window_hours | [6, 12, 24] | Trend/slope window sizes |
| trend_min_observations | 2 | Minimum valid obs for trend |
| forward_fill_max_window_hours | {"default": 4, "vital_signs": 4, "laboratory": 6} | Max FFILL gap (hours) |
| compute_missingness_features | true | Generate missingness features |
| compute_forward_fill | true | Generate forward-filled features |
| compute_rolling_features | true | Generate rolling features |
| compute_trend_features | true | Generate trend features |
| compute_clinical_features | true | Generate clinical features |

Configuration is saved with each experiment (`data/features/feature_config.json`).

---

## 4. Feature Categories

### 4.1 Current Observation Features

Raw clinical variables preserved exactly as measured at each hour. No
imputation or transformation. Original `NaN` values are preserved.

### 4.2 Missingness Features

Generated **before** any imputation or forward filling:

| Feature | Description |
|---------|-------------|
| `{var}_is_missing` | 1 if variable missing at current hour, else 0 |
| `{var}_hours_since_last` | Hours since last valid measurement (`inf` if never observed) |
| `{var}_availability_6h` | Fraction of valid hours in trailing 6h window |

Time-since-last-measurement uses only current/past observations.
`inf` distinguishes never-observed from previously-observed.

### 4.3 Forward-Filled Features

Causal forward filling with configurable max validity window. A
measurement is carried forward only if the gap since the last valid
observation is within the window. Beyond the window, the value is `NaN`.
Original raw columns are preserved unchanged.

### 4.4 Rolling Features

Causal trailing rolling statistics over 6-, 12-, and 24-hour windows:

| Aggregation | Description |
|-------------|-------------|
| mean | Mean of valid observations in window |
| min | Minimum |
| max | Maximum |
| std | Sample standard deviation |
| latest | Most recent valid observation |

Windows include the current hour and exclude future hours. Insufficient
observations yield `NaN` (never fabricated).

### 4.5 Trend / Slope Features

| Feature | Formula |
|---------|---------|
| `{var}_abs_change_{W}h` | value[t] − reference value (last valid at or before t−W) |
| `{var}_rel_change_{W}h` | (value[t] − ref) / \|ref\| (NaN if ref = 0) |
| `{var}_slope_{W}h` | (value[t] − ref) / (t − ref_hour) |

Reference is the last valid observation strictly before `t`, within the
trailing window. Zero denominators yield `NaN`.

### 4.6 Clinical Features

**Shock index** = HR / SBP (NaN if SBP is zero or missing).

**SIRS component count**: Count of SIRS criteria met among available
components:
- Temperature > 38.3°C or < 36.0°C
- Heart rate > 90 bpm
- Respiration > 20 breaths/min
- WBC > 12 or < 4 (10³/µL)

Missing components are not treated as normal. `sirs_components_available`
reports how many components were evaluable. Does not claim a complete
SIRS score when components are unavailable.

---

## 5. Output Schema

The output DataFrame has one row per patient-hour with columns:

```
patient_id, source_dataset, hour, <40 clinical variables>,
<static variables>, <missingness features>, <forward-filled features>,
<rolling features>, <trend features>, <clinical features>, SepsisLabel
```

The stable feature schema is documented in
`data/features/feature_schema.json` and
`ml/p1_clinical/features/feature_schema.py`. Each feature entry documents:
name, source variable, category, data type, formula, window, units,
missing behavior, causality, and tabular/sequential appropriateness.

**Preserved columns:** `patient_id`, `source_dataset`, `hour`,
`SepsisLabel`. `hospital_id` is preserved when genuinely available from
the source.

**SepsisLabel** is preserved unchanged from Part 3 (target only, never
shifted, never used as an input feature).

---

## 6. Causality and Leakage Prevention

For every patient and prediction hour `t`:

- No observation after `t` contributes to any feature at `t`
- No future rolling statistics
- No backward filling
- No centered rolling windows
- No future-derived slopes
- No future measurements for imputation
- No target-label information as input feature
- No patient/source identifiers as predictive features

Automated leakage tests (`test_temporal_leakage.py`) deliberately modify
observations after hour `t` and verify features at hour `t` remain
unchanged.

---

## 7. Output Files

| Path | Description |
|------|-------------|
| `data/features/feature_schema.json` | Stable feature schema |
| `data/features/feature_config.json` | Feature configuration |
| `data/features/combined_features.parquet` | Combined feature data (synthetic) |
| `data/reports/feature_summary.json` | Pipeline execution summary |
| `data/reports/feature_missingness_report.csv` | Per-feature missingness |
| `data/reports/feature_validation_report.json` | Validation results |

> Reports and example features are generated from synthetic fixture data
> and are marked `"source": "synthetic"`. Real-data validation is pending
> PhysioNet credentialed access.

---

## 8. Limitations

- Real PhysioNet data not yet accessed; all validation on synthetic
  fixtures
- Forward-fill validity windows are experimental defaults, not
  clinically established
- No imputation is performed; downstream models must handle NaN
- Hospital ID inferred from source naming convention
- Cross-hospital evaluation architecture prepared but not performed
