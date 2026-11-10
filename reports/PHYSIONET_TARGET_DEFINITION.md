# PhysioNet Target Definition Report

**Generated:** 2026-11-09  
**Purpose:** Document the official PhysioNet/CinC Challenge 2019 sepsis label definition and its mapping to the ICU Sentinel project target `sepsis_within_6h`

---

## 1. Official Challenge Definition

### 1.1 Source Documentation
From the PhysioNet/CinC Challenge 2019 official documentation:

> **SepsisLabel**: "The label is 1 if the patient is septic at that time or will become septic within 6 hours, and 0 otherwise."
> 
> **Sepsis onset**: Defined as the time of suspected infection AND organ dysfunction (SOFA score increase ≥ 2) per Sepsis-3 criteria.
>
> **Label construction**: For each hourly record, SepsisLabel = 1 if t ≥ t_sepsis - 6, else 0.

### 1.2 Key Points
- **SepsisLabel is a 6-hour early warning label** — positive 6 hours before clinical sepsis onset
- **Not a "current sepsis" indicator** — it's a prediction target for early warning
- **Onset definition**: Sepsis-3 (suspected infection + SOFA ≥ 2)
- **Hourly granularity**: One label per hour of ICU stay

---

## 2. Existing Project Target: `sepsis_within_6h`

### 2.1 Current Implementation (from `ml/p1_clinical/preprocessing/targets.py`)

```python
def build_sepsis_within_6h_target(
    sepsis_label: list[int],
    icu_los: list[float],
) -> tuple[list[int], list[bool]]:
    """
    Build sepsis_within_6h target from raw SepsisLabel.
    
    Positive window: [onset, onset + 5] hours (6 hours total)
    Negative window: [0, onset - 1] hours (before onset)
    Ineligible: [onset + 6, end] hours (after positive window)
    """
    # Find first positive SepsisLabel hour = onset - 6
    # sepsis_within_6h = 1 for hours [onset, onset+5]
    # sepsis_within_6h = 0 for hours before onset
    # sepsis_within_6h = -1 (ineligible) for hours >= onset+6
```

### 2.2 Target Semantics

| Hour relative to onset | SepsisLabel (official) | sepsis_within_6h (project) | Interpretation |
|------------------------|------------------------|----------------------------|----------------|
| t < onset - 6 | 0 | 0 (negative) | No sepsis risk |
| onset - 6 ≤ t < onset | 1 | 0 (negative) | **Early warning window** — label says "will be septic within 6h" |
| onset ≤ t < onset + 6 | 1 | 1 (positive) | **Sepsis onset window** — sepsis has begun |
| t ≥ onset + 6 | 1 | -1 (ineligible) | Post-onset, excluded from prediction |

---

## 3. Critical Mapping Analysis

### 3.1 The Mismatch

**Official SepsisLabel = 1** means: "Patient IS septic OR WILL BE septic within 6 hours"

**Project `sepsis_within_6h` = 1** means: "Patient IS in the 6-hour window starting at sepsis onset"

These are **NOT the same thing**:

| Time Period | Official SepsisLabel | Project Target | Agreement? |
|-------------|---------------------|----------------|------------|
| Pre-onset (early warning) | 1 | 0 | ❌ **DISAGREE** |
| Onset window (0-5h post-onset) | 1 | 1 | ✅ Agree |
| Post-onset (6h+) | 1 | -1 (excluded) | ⚠️ Different handling |

### 3.2 Implication

**The project target `sepsis_within_6h` is a REFINED version** that:
- Excludes the early warning period (SepsisLabel=1 but before onset)
- Only considers the confirmed sepsis onset window as positive
- Marks post-onset hours as ineligible

This is **actually more clinically meaningful** for an early warning system because:
- It defines the "true" onset period as the prediction target
- The early warning window (SepsisLabel=1, pre-onset) is what the MODEL should predict
- Post-onset hours are not useful for "early" warning

---

## 4. Recommended Mapping

### 4.1 For Model Training

**Use the official SepsisLabel directly as the binary target:**

```python
# Target for model: predict SepsisLabel == 1 (6-hour early warning)
# This means: "Will this patient be septic within 6 hours?"
y = (df['SepsisLabel'] == 1).astype(int)
```

**Rationale:**
- Matches official challenge definition exactly
- Model learns to predict the 6-hour early warning
- Consistent with challenge evaluation metrics

### 4.2 For Clinical Interpretation

**Use `sepsis_within_6h` for post-hoc analysis:**

```python
# For lead time analysis, alert evaluation
# Positive = confirmed sepsis onset window
# Negative = pre-onset
# Ineligible = post-onset (excluded)
```

---

## 5. Verification from Data

### 5.1 Empirical Validation

From the dataset validation:
- **training_setA**: 17,136 positive hours out of 790,215 (2.17%)
- **training_setB**: 10,437 positive hours out of 740,517 (1.41%)
- **Combined**: 27,573 positive hours out of 1,530,732 (1.80%)

### 5.2 Label Pattern Analysis

For a septic patient (example from data):
```
ICULOS:  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
SepsisL: 0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1
```

- First positive at ICULOS=6 → **onset = ICULOS=12** (6 hours later)
- Positive window: ICULOS 6-11 (6 hours early warning)
- Onset window: ICULOS 12-17 (6 hours post-onset)

This matches the official definition exactly.

---

## 6. Decision: Target Configuration

### 6.1 Primary Target (Model Training)
**Use official `SepsisLabel` as binary target:**
- `y = (SepsisLabel == 1).astype(int)`
- Predicts: "Sepsis within 6 hours" (official challenge task)
- Compatible with challenge evaluation

### 6.2 Secondary Target (Clinical Analysis)
**Use `sepsis_within_6h` for:**
- Lead time calculation
- Alert burden analysis
- Event-level evaluation
- Onset window identification

### 6.3 Configuration

```python
TARGET_CONFIG = {
    "primary": "SepsisLabel",           # Binary: 0/1 (official challenge target)
    "secondary": "sepsis_within_6h",    # Ternary: -1/0/1 (clinical analysis)
    "prediction_horizon_hours": 6,
    "onset_definition": "Sepsis-3 (infection + SOFA>=2)",
    "label_shift_hours": 6,             # SepsisLabel shifted 6h before onset
}
```

---

## 7. Compatibility with Existing Pipeline

### 7.1 Current Code Status
- `ml/p1_clinical/preprocessing/targets.py` implements `sepsis_within_6h`
- `ml/p1_clinical/preprocessing/pipeline.py` uses this for classical ML
- P2 boosting uses `DERIVED_TARGET_COLUMN = "sepsis_within_6h"`

### 7.2 Required Changes

| Component | Change Needed |
|-----------|---------------|
| `targets.py` | Add `build_official_target()` function returning SepsisLabel |
| `pipeline.py` | Add config option `target_type: "official" | "sepsis_within_6h"` |
| `P2Config` | Add `target_column` parameter |
| Training scripts | Pass target configuration |

### 7.3 Backward Compatibility
- Keep `sepsis_within_6h` as default for clinical analysis
- Add `official` target for challenge-compliant model training
- Both targets derived from same raw SepsisLabel — no data duplication

---

## 8. Summary

| Aspect | Official SepsisLabel | Project sepsis_within_6h |
|--------|---------------------|--------------------------|
| **Definition** | 1 if septic or septic within 6h | 1 only during onset window (6h post-onset) |
| **Early warning period** | Labeled POSITIVE (1) | Labeled NEGATIVE (0) |
| **Onset window** | Labeled POSITIVE (1) | Labeled POSITIVE (1) |
| **Post-onset** | Labeled POSITIVE (1) | Labeled INELIGIBLE (-1) |
| **Use case** | Model training (challenge task) | Clinical analysis (lead time, alerts) |
| **Hourly positive rate** | 1.80% | ~0.9% (half, since early window excluded) |

**Recommendation:** Train models on **official SepsisLabel** for challenge compatibility; use `sepsis_within_6h` for clinical evaluation of early warning performance.

---

*Report generated by ICU Sentinel Phase 2 Real Data Migration Pipeline*