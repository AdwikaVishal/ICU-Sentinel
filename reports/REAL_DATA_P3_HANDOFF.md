# Real Data P3 Handoff Report

**Generated:** 2026-11-09  
**Purpose:** Document the interface between P2 (real data models) and P3 (temporal/replay/explainability)

---

## 1. Feature Schema (Real Data)

### 1.1 Input Feature Specification
- **Total features:** 994 per patient-hour
- **Feature categories:**
  - Current (raw): 39
  - Missingness: 102
  - Forward-fill: 34
  - Rolling (6/12/24h × 5 aggs): 510
  - Trend (6/12/24h × 3 types): 306
  - Clinical: 3 (shock_index, sirs_count, sirs_components_available)

### 1.2 Feature Naming Convention
```
<variable>                           # Current/raw value
<variable>_is_missing                # Binary missing indicator
<variable>_hours_since_last          # Hours since last observation (inf=never)
<variable>_availability_6h           # Fraction observed in last 6h
<variable>_availability_12h          # Fraction observed in last 12h
<variable>_availability_24h          # Fraction observed in last 24h
<variable>_ffilled                   # Forward-filled (causal, max 4h vitals/6h labs)
<variable>_rolling_<agg>_<window>h   # Rolling agg (mean/min/max/std/latest) over window
<variable>_abs_change_<window>h      # Absolute change over window
<variable>_rel_change_<window>h      # Relative change over window
<variable>_slope_<window>h           # Linear slope over window
```

### 1.3 Key Variables (40 clinical)
**Vital Signs (8):** HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2  
**Laboratory (26):** BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets  
**Demographics (6):** Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS  

**Note:** EtCO2 and TroponinI features have 100% missingness in this dataset - excluded from modeling.

### 1.4 Static Variables (per patient)
- Age, Gender, Unit1, Unit2, HospAdmTime (repeated per hour)

---

## 2. Model Input Schema

### 2.1 Expected Input Format (per prediction request)
```json
{
  "patient_id": "p000001",
  "source_dataset": "set_a",
  "hour": 10,
  "features": {
    "HR": 85.2,
    "HR_is_missing": 0,
    "HR_hours_since_last": 1.0,
    "HR_availability_6h": 0.83,
    "HR_ffilled": 85.2,
    "HR_rolling_mean_6h": 87.1,
    "HR_rolling_max_6h": 92.0,
    "HR_abs_change_6h": -3.2,
    "HR_rel_change_6h": -0.035,
    "HR_slope_6h": -0.5,
    ... (994 total features)
  }
}
```

### 2.2 Model Artifact Paths
| Model | Path |
|-------|------|
| Extra Trees (recommended) | `artifacts/physionet2019/p2_974/extra_trees/extra_trees_1.0.0.joblib` |
| LightGBM | `artifacts/physionet2019/p2_974/lightgbm/lightgbm_1.0.0.joblib` |
| XGBoost | `artifacts/physionet2019/p2_974/xgboost/xgboost_1.0.0.joblib` |

### 2.3 Model Configuration
```json
{
  "model_name": "extra_trees",
  "model_version": "1.0.0",
  "n_features": 994,
  "threshold": 0.11,
  "calibration": "sigmoid",
  "random_seed": 42,
  "source": "physionet2019_features974"
}
```

---

## 3. Model Output Schema

### 3.1 Prediction Response
```json
{
  "patient_id": "p000001",
  "timestamp": "2026-11-09T10:00:00Z",
  "hour": 10,
  "prediction": {
    "probability": 0.15,
    "class": 1,
    "threshold": 0.11,
    "model": "extra_trees",
    "calibrated": true,
    "calibration_method": "sigmoid"
  }
}
```

### 3.2 Batch Prediction Format
```json
{
  "predictions": [
    {"patient_id": "p000001", "hour": 10, "probability": 0.15, "class": 1},
    {"patient_id": "p000001", "hour": 11, "probability": 0.18, "class": 1},
    {"patient_id": "p000002", "hour": 5, "probability": 0.02, "class": 0}
  ],
  "metadata": {
    "model": "extra_trees",
    "version": "1.0.0",
    "threshold": 0.11,
    "calibration": "sigmoid",
    "generated_at": "2026-11-09T10:00:00Z"
  }
}
```

---

## 4. Patient ID Conventions

- **Format:** `p` + 6 digits (e.g., `p000001`, `p100001`)
- **Source datasets:** `set_a` (training_setA), `set_b` (training_setB)
- **No hospital identifiers** - Set A/B are data partitions, not hospitals
- **Consistent across P1, P2, P3**

---

## 5. Timestamp Conventions

- **Hour index:** Integer hours since ICU admission (ICULOS, 1-indexed)
- **P3 internal:** 0-indexed `hour` field (ICULOS - 1)
- **Prediction horizon:** 6 hours (sepsis_within_6h target)
- **Lead time:** Alert hour - sepsis onset hour (positive = early)

---

## 6. Prediction Horizon

- **Target:** `sepsis_within_6h` (1 if sepsis onset within 6 hours, else 0)
- **Official definition:** SepsisLabel = 1 if t ≥ t_sepsis - 6
- **Prediction task:** Predict SepsisLabel (6-hour early warning)
- **Clinical target:** `sepsis_within_6h` (1 = onset window, 0 = pre-onset, -1 = ineligible)

---

## 7. Prediction Probability Format

- **Range:** [0, 1] float
- **Calibration:** Sigmoid (Platt) calibrated on validation set
- **Interpretation:** P(sepsis within 6 hours | current observations)
- **Threshold:** 0.11 for Extra Trees (F1-optimal on validation)

---

## 8. Threshold Configuration

```json
{
  "extra_trees": {
    "threshold": 0.11,
    "selection_metric": "f1",
    "selection_split": "validation",
    "calibration": "sigmoid"
  },
  "lightgbm": {
    "threshold": 0.05,
    "selection_metric": "f1",
    "selection_split": "validation",
    "calibration": "sigmoid"
  },
  "xgboost": {
    "threshold": 0.49,
    "selection_metric": "f1",
    "selection_split": "validation",
    "calibration": "sigmoid"
  }
}
```

---

## 9. Model Artifact Format

### 9.1 Joblib Structure
```python
{
    'model': <sklearn/LightGBM/XGBoost model object>,
    'preprocessing_pipeline': <ColumnTransformer with imputer>,
    'selected_features': ['feature1', 'feature2', ...],  # 994 features
    'hyperparameters': {...},
    'model_name': 'extra_trees',
    'model_version': '1.0.0',
    'threshold': 0.11,
    'random_seed': 42,
    'source': 'physionet2019_features974'
}
```

### 9.2 Loading Example
```python
import joblib
artifact = joblib.load('artifacts/physionet2019/p2_974/extra_trees/extra_trees_1.0.0.joblib')
model = artifact['model']
preprocessor = artifact['preprocessing_pipeline']
features = artifact['selected_features']

# For new patient data X (DataFrame with 994 features)
X_processed = preprocessor.transform(X[features])
probs = model.predict_proba(X_processed)[:, 1]
preds = (probs >= 0.11).astype(int)
```

---

## 10. Explanation Artifact Format (SHAP)

### 10.1 SHAP Values Structure
```json
{
  "patient_id": "p000001",
  "hour": 10,
  "shap_values": {
    "HR": 0.023,
    "HR_rolling_mean_6h": -0.012,
    "Lactate": 0.045,
    ...
  },
  "base_value": 0.015,
  "prediction": 0.15,
  "model": "extra_trees"
}
```

### 10.2 Global Feature Importance
Available in `artifacts/physionet2019/p2_974/extra_trees/feature_importance.json` (to be generated by P3)

---

## 11. Alert Output Format

### 11.1 Alert Schema
```json
{
  "alert_id": "alert_001",
  "patient_id": "p000001",
  "triggered_at": "2026-11-09T10:00:00Z",
  "hour": 10,
  "probability": 0.15,
  "threshold": 0.11,
  "model": "extra_trees",
  "lead_time_estimate": 3.5,
  "severity": "medium"
}
```

### 11.2 Alert Rules (P3 responsibility)
- **Cooldown:** 6 hours between alerts for same patient
- **Minimum lead time:** 1 hour (alerts within 1h of onset not "early")
- **Maximum alerts/patient:** 3 per 24h window
- **Escalation:** Persistent alerts > 2h trigger escalation

---

## 12. Data Access for P3

### 12.1 Required Directories
| Data | Path |
|------|------|
| Model artifacts | `artifacts/physionet2019/p2_974/` |
| Test predictions | `artifacts/physionet2019/p2_974/*/predictions/` |
| Feature schema | `data/processed/physionet2019_features/feature_schema.json` |
| Split manifest | `data/splits/physionet2019_features974/split_manifest.json` |
| Patient features | `data/processed/physionet2019_features/patients/` |

### 12.2 P3 Integration Points
1. **Replay Engine:** Load test patient features → generate predictions → replay timeline
2. **SHAP Explainer:** Load model + feature schema → compute SHAP for individual predictions
3. **Alert Engine:** Stream predictions → apply cooldown/escalation rules → emit alerts

---

## 13. Limitations for P3

1. **Subset only:** 974 patients (2.4% of full dataset)
2. **Missing EtCO2/TroponinI:** Features present but all NaN
3. **No test set:** PhysioNet test set requires credentials
4. **Single split:** No cross-validation results
5. **Calibration on small validation:** 98 patients, 54 positive

---

## 14. Recommended P3 Next Steps

1. **Load Extra Trees model** as default for clinical demo
2. **Generate SHAP explanations** for test set predictions
3. **Build replay timeline** for septic patients showing lead time
4. **Implement alert engine** with 6-hour cooldown
5. **Validate on full dataset** when available

---

*Report generated by ICU Sentinel Phase 2 Real Data Migration Pipeline*