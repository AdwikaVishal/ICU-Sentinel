# Data Dictionary — ICU Sentinel

This document defines all features, targets, and metadata used in the ICU Sentinel sepsis early-warning pipeline.

## Target Definition

| Field | Type | Description |
|---|---|---|
| `sepsis_within_6h` | binary (0/1) | Whether sepsis will occur within 6 hours of the observation window start. This is the primary prediction target. |
| `onset` | integer hour | The first hour index where `SepsisLabel == 1`. |
| `target_eligible` | binary | Flag indicating whether the patient is within the eligible observation window (before onset+6). |

## Feature Categories (994 total features)

### Vital Signs
- `HR` — Heart rate (beats/min)
- `SBP` — Systolic blood pressure (mmHg)
- `DBP` — Diastolic blood pressure (mmHg)
- `MAP` — Mean arterial pressure (mmHg)
- `RR` — Respiratory rate
- `Temp` — Body temperature (°C)
- `O2Sat` — Oxygen saturation %
- `SaO2` — Arterial oxygen saturation %
- `EtCO2` — End-tidal CO2
- `FiO2` — Fraction of inspired oxygen
- `MAP`, `SBP`, `DBP` — Absolute change, min, max, mean, std, slope over rolling windows (6h, 12h, 24h)
- `HR`, `Temp`, `RR`, `O2Sat`, `FiO2`, `MAP` — Rolling latest, mean, min, max, std over windows (6h, 12h, 24h)
- `HR_slope_6h`, `SBP_slope_6h`, etc. — Time-slope features over 6h/12h/24h windows

### Laboratory Values
- `WBC` — White blood cell count
- `RBC` — Red blood cell count
- `Hgb` — Hemoglobin
- `Hct` — Hematocrit
- `Platelets` — Platelet count
- `INR` — International normalized ratio
- `PTT` — Partial thromboplastin time
- `PT` — Prothrombin time
- `Fibrinogen` — Fibrinogen level
- `D_dimer` — D-dimer level
- `BUN` — Blood urea nitrogen
- `Creatinine` — Serum creatinine
- `Glucose` — Blood glucose
- `Lactate` — Serum lactate
- `Bilirubin_total` — Total bilirubin
- `Bilirubin_direct` — Direct bilirubin
- `Calcium` — Ionized calcium
- `Magnesium` — Magnesium level
- `Sodium` — Serum sodium (if present)
- `Potassium` — Serum potassium
- `Chloride` — Serum chloride
- `Phosphate` — Serum phosphate
- `TroponinI` — Cardiac troponin I
- `AST` — Aspartate aminotransferase
- `Alkalinephos` — Alkaline phosphatase
- `GGT` — Gamma-glutamyl transferase (if present)
- `LDH` — Lactate dehydrogenase (if present)

### Clinical Scores
- `shock_index` — Calculated as `HR / SBP`
- `sirs_count` — Systemic inflammatory response syndrome score
- `sirs_components_available` — Count of SIRS criteria met

### Demographics
- `Age` — Patient age in years
- `Gender` — Sex (male/female)

### Availability & Missingness Indicators
For each of the above, `_availability_6h`, `_ffilled`, `_is_missing`, `_abs_change_6h`, `_rel_change_6h` suffixes indicate:
- Whether the value was available in the last 6 hours
- Forward-filled value (if original was missing)
- Whether the value is missing
- Absolute change from baseline
- Relative change from baseline

### Rolling Statistics (suffix: `_rolling_latest_`, `_rolling_mean_`, `_rolling_min_`, `_rolling_max_`, `_rolling_std_`)
Applied to most numeric features over window sizes of 6h, 12h, and 24h, computing:
- Latest observed value in window
- Rolling mean
- Rolling minimum
- Rolling maximum
- Rolling standard deviation

### Slope Features (suffix: `_slope_`)
Linear regression slope over window sizes of 6h, 12h, and 24h for selected features.

## Feature Schema

All features are stored in `data/features/feature_schema.json` and `data/features/feature_config.json`. The feature inventory contains:
- `n_features_input`: 994 (raw + derived features after sanitization)
- `n_features_output`: 994 (features selected for model input)
- `selected_feature_names`: Array of all 994 feature names

## Target Policy

- **Horizon**: 6 hours (`horizon_hours=6`)
- **Eligibility**: A patient is eligible for labeling if `onset + 6 <= total_hours_in_stay`. Patients ineligible (at/after onset+6) have `target_eligible = 0` and their target is NaN.
- **Label modification**: The original `SepsisLabel` from PhysioNet is modified to `sepsis_within_6h`:
  - `1` if sepsis onset occurs within the next 6 hours from the feature window start
  - `0` otherwise
- **Original label unchanged**: The raw `SepsisLabel` column is preserved alongside the modified target.

## Data Splits

Patient-level stratified splits are defined in `data/splits/split_manifest.json` (default 500-patient subset) and `data/splits_full/split_manifest.json` (full 39,810-patient cohort). Splits are stratified by the sepsis label and use `random_state=42`:
- **Train**: 350 patients (default) / 27,867 patients (full)
- **Validation**: 50 patients (default) / 3,981 patients (full)
- **Test**: 100 patients (default) / 7,962 patients (full)

No patient appears across multiple partitions (patient-level integrity guaranteed).

## Class Imbalance

The positive class rate in the eligible population is approximately 1.14% (17,551 positive out of 1,541,845 eligible rows). This imbalance is reflected in the low precision observed in model outputs. AUPRC is recommended over AUROC for evaluating model performance in this setting.

## Data Sources

- **Primary**: PhysioNet CinC 2019 Challenge data
- **Features generated**: 2024-2025, 994 features combining vital signs, labs, clinical scores, and temporal abstractions
- **Processed data format**: Parquet files in `data/processed_full/` and `data/processed/`