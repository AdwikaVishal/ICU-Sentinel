# PHASE 4: CALIBRATION ANALYSIS

**Date:** 2026-11-09  
**Analyst:** ICU Sentinel Validation System

---

## EXECUTIVE SUMMARY

**Status: NOT COMPLETED**

Calibration analysis has **not been performed** for the current models (12,979-patient training run). Configuration specifies sigmoid calibration method, but no calibration metrics, calibrated models, or reliability diagrams exist in artifacts.

**Action Required:** Perform comprehensive calibration analysis before test set evaluation.

---

## 1. CALIBRATION IMPORTANCE

### Why Calibration Matters

**Definition:** A calibrated model's predicted probabilities match observed frequencies.
- If model predicts 20% risk, approximately 20% of such cases should have the outcome
- Critical for clinical decision-making based on risk scores
- Uncalibrated probabilities can mislead clinicians

### Clinical Use Cases Requiring Calibration

1. **Risk Stratification** - Assigning patients to high/medium/low risk categories
2. **Resource Allocation** - Prioritizing ICU beds based on predicted risk
3. **Treatment Decisions** - Interventions triggered at specific risk thresholds
4. **Patient Communication** - Explaining risk to patients and families

**Without calibration:** Predicted probabilities may be systematically too high or too low, leading to poor clinical decisions even if ranking (AUROC) is good.

---

## 2. EXPECTED CALIBRATION ANALYSES

### 2.1 Calibration Metrics

**Expected Calibration Error (ECE):**
- Mean absolute difference between predicted probabilities and observed frequencies
- Calculated across probability bins (typically 10-20 bins)
- **Interpretation:** Lower is better, 0 = perfect calibration
- **Threshold:** ECE < 0.05 generally considered well-calibrated

**Maximum Calibration Error (MCE):**
- Maximum absolute difference across any bin
- More sensitive to localized miscalibration
- **Interpretation:** Lower is better, < 0.1 reasonable

**Brier Score:**
- Mean squared error between predictions and outcomes
- **Interpretation:** Lower is better, combines calibration and discrimination
- **Decomposition:** Brier = Reliability + Resolution - Uncertainty

### 2.2 Calibration Methods

**Configured Method:** Sigmoid (Platt scaling)

**Should Compare:**
1. **Uncalibrated** - Original model probabilities
2. **Sigmoid Calibration** - Logistic regression on validation predictions
3. **Isotonic Calibration** - Non-parametric monotonic mapping

### 2.3 Visual Diagnostics

**Reliability Diagram (Calibration Plot):**
- X-axis: Predicted probability (binned)
- Y-axis: Observed frequency
- Perfect calibration: diagonal line (y = x)
- Shows over/under-confidence regions

**Calibration Histogram:**
- Distribution of predicted probabilities
- Identifies if model produces extreme (0/1) or moderate (0.4-0.6) predictions

---

## 3. CURRENT STATUS

### What Exists

✓ **Configuration:** Calibration method specified as "sigmoid"  
✓ **Validation Data:** Available for calibration analysis (71,609 observations)  
✓ **Previous Example:** Calibration results exist for 974-patient run  

### What's Missing

❌ **Calibration Metrics:** ECE and MCE not calculated  
❌ **Calibrated Models:** No calibrated model artifacts saved  
❌ **Calibration Plots:** No reliability diagrams generated  
❌ **Brier Decomposition:** Reliability/resolution/uncertainty not calculated  
❌ **Comparison:** Uncalibrated vs calibrated performance not compared  

---

## 4. PREVIOUS RUN CALIBRATION (974-Patient Reference)

### Historical Calibration Results

From the 974-patient training run:

| Model | Uncalibrated ECE | Sigmoid ECE | Isotonic ECE | Improvement |
|-------|-----------------|-------------|--------------|-------------|
| **Extra Trees** | 0.0029 | 0.0001 | 0.0020 | 96.7% (sigmoid) |
| **LightGBM** | 0.0132 | 0.0017 | 0.0017 | 87.1% (both) |
| **XGBoost** | 0.0471 | 0.0021 | 0.0031 | 95.6% (sigmoid) |

**Key Findings from Previous Run:**
- XGBoost showed worst uncalibrated performance (ECE = 0.0471)
- Sigmoid calibration effective for all models (ECE < 0.003)
- Extra Trees had best native calibration (ECE = 0.0029)
- All models well-calibrated after sigmoid transformation

**Note:** These results from 974-patient run, current 12,979-patient models may differ.

---

## 5. EXPECTED CALIBRATION NEEDS FOR CURRENT MODELS

### Based on Model Types

**LightGBM (Recommended Model):**
- Tree-based models often poorly calibrated
- Tend to predict extreme probabilities (near 0 or 1)
- **Expected:** Requires sigmoid calibration
- **Reference:** Previous run showed ECE=0.0132 uncalibrated

**XGBoost (Alternative):**
- Known for poor calibration out-of-box
- Aggressive boosting can overfit probability estimates
- **Expected:** Likely needs calibration (previous ECE=0.0471)

**Extra Trees:**
- Generally better calibrated than boosted models
- Averaging reduces extreme predictions
- **Expected:** May need minimal calibration

---

## 6. CALIBRATION ANALYSIS PROCEDURE

### Step 1: Calculate Uncalibrated Metrics

For each model on validation set:
1. Obtain predicted probabilities (already available)
2. Calculate ECE (10-20 bins)
3. Calculate MCE
4. Calculate Brier score
5. Generate reliability diagram

### Step 2: Fit Calibrators

For each model:
1. **Sigmoid:** Fit logistic regression on validation predictions
2. **Isotonic:** Fit isotonic regression on validation predictions
3. Save calibrator objects

**Critical:** Calibrators fit on validation set only (same data used for threshold optimization - acceptable practice)

### Step 3: Evaluate Calibrated Performance

For each model and calibration method:
1. Apply calibrator to validation predictions
2. Calculate calibrated ECE, MCE, Brier
3. Verify discrimination (AUROC/AUPRC) unchanged
4. Generate calibrated reliability diagrams

### Step 4: Select Calibration Method

For each model:
- Compare ECE across uncalibrated, sigmoid, isotonic
- Choose method with lowest ECE
- Verify no degradation in discrimination
- Document selection rationale

### Step 5: Save Artifacts

For deployment:
- Save calibrated model pipeline (model + calibrator)
- Save calibration performance metrics
- Save reliability plots
- Document calibration method used

---

## 7. CALIBRATION PITFALLS TO AVOID

### Overfitting Calibration

⚠️  **Risk:** If validation set is small, isotonic calibration can overfit  
**Mitigation:** Use sigmoid for small validation sets, isotonic for large  
**Current:** 71,609 observations - sufficient for isotonic

### Data Leakage

⚠️  **Risk:** Calibrating on test set leaks information  
**Correct:** Fit calibrator on validation, apply to test  
**Current:** Must verify calibration uses validation only

### Threshold Recalibration

⚠️  **Note:** After calibration, optimal thresholds may change  
**Action:** Re-optimize thresholds on calibrated probabilities  
**Current:** F1-optimal thresholds (0.73 LightGBM, 0.69 XGBoost) are on uncalibrated probabilities

### Discrimination vs Calibration Tradeoff

⚠️  **Rare Case:** Sometimes calibration slightly reduces AUROC  
**Acceptable:** < 0.01 AUROC drop for improved calibration  
**Action:** Monitor and document any discrimination changes

---

## 8. CALIBRATION FOR CLINICAL DEPLOYMENT

### Regulatory Considerations

**FDA/CE Mark Requirements:**
- Documented calibration analysis
- Reliability diagrams in submission
- Calibration performance metrics
- Explanation of calibration method

**Clinical Validation:**
- Calibration on deployment population
- Prospective calibration monitoring
- Recalibration triggers defined

### Operational Monitoring

**Post-Deployment:**
1. **Calibration Drift Monitoring**
   - Track ECE over time
   - Alert if ECE > threshold (e.g., 0.1)
   - Trigger recalibration if needed

2. **Population Shifts**
   - Monitor if patient demographics change
   - Recalibrate for new populations
   - Maintain separate calibrators if needed

---

## 9. INTEGRATION WITH EXISTING PIPELINE

### Current Pipeline Status

**Training:** ✓ Completed (3 models)  
**Validation Evaluation:** ✓ Completed (metrics calculated)  
**Threshold Optimization:** ✓ Completed (F1-based)  
**Calibration:** ❌ **NOT COMPLETED**  
**Test Evaluation:** ⚠️  Pending (should wait for calibration)  

### Required Pipeline Addition

**Add Calibration Step:**
```
1. Train model on training set ✓
2. Evaluate on validation set ✓
3. Optimize threshold on validation set ✓
4. → FIT CALIBRATOR on validation set ← NEW
5. → EVALUATE CALIBRATED model on validation ← NEW
6. → SAVE CALIBRATED model artifact ← NEW
7. Apply calibrated model to test set
```

---

## 10. PRACTICAL IMPLICATIONS

### Clinical Decision Support

**Without Calibration:**
- Model says "80% risk" → Actual risk could be 40% or 95%
- Clinicians cannot trust probability estimates
- Only ranking information available (high vs low risk)

**With Calibration:**
- Model says "80% risk" → Actual risk ~80% ± calibration error
- Clinicians can use probabilities for decisions
- Risk communication to patients meaningful

### Example Scenarios

**Scenario 1: Treatment Decision**
- Protocol: Give antibiotic if risk > 60%
- Uncalibrated model: Predicts 65%, actual risk 30% → Overtreatment
- Calibrated model: Predicts 65%, actual risk ~65% → Appropriate

**Scenario 2: Resource Allocation**
- ICU bed for patient with highest risk
- Uncalibrated: Rankings correct but magnitudes wrong
- Calibrated: Can estimate absolute benefit of ICU admission

---

## 11. VALIDATION SET CONSIDERATIONS

### Calibration Data

**Available:**
- 71,609 observations
- 966 positive cases (1.35%)
- Same data used for threshold optimization

**Adequacy:**
- Sufficient for sigmoid calibration (parametric)
- Sufficient for isotonic calibration (non-parametric)
- Bins can have ~7,000 observations each (robust)

**Limitation:**
- Same data used for threshold and calibration optimization
- Acceptable practice but reduces statistical independence
- Ideally would have separate calibration holdout

---

## 12. COMPARISON TO TEST SET EVALUATION

### Why Calibration Before Test

**Reason 1: Independence**
- Test set should evaluate final model
- Calibration is part of final model
- Uncalibrated test evaluation doesn't reflect deployment performance

**Reason 2: Clinical Realism**
- Deployed model will be calibrated
- Test evaluation should match deployment scenario
- Uncalibrated test metrics not clinically meaningful

**Reason 3: Fair Comparison**
- All models should be calibrated
- Comparing uncalibrated vs calibrated is unfair
- Standard practice in ML for healthcare

---

## 13. RECOMMENDATIONS

### IMMEDIATE ACTIONS (HIGH PRIORITY)

1. **Perform Calibration Analysis**
   ```
   For each model (Extra Trees, LightGBM, XGBoost):
   a. Calculate uncalibrated ECE, MCE, Brier on validation
   b. Fit sigmoid calibrator on validation
   c. Fit isotonic calibrator on validation
   d. Evaluate both calibrators on validation
   e. Select best calibration method per model
   f. Generate reliability diagrams (before/after)
   g. Save calibrated model artifacts
   ```

2. **Document Calibration Results**
   - Create calibration_results.json matching previous format
   - Include ECE, MCE for uncalibrated and calibrated
   - Document selected calibration method per model
   - Generate calibration report

3. **Update Model Selection**
   - Re-evaluate model selection with calibration considered
   - Verify LightGBM still best after calibration
   - Document if calibration changes recommendation

### BEFORE TEST EVALUATION

1. **Verify Calibration Quality**
   - All models must have ECE < 0.05
   - If any model ECE > 0.05, investigate and iterate
   - Do not proceed to test if calibration poor

2. **Re-optimize Thresholds (Optional)**
   - Consider re-optimizing thresholds on calibrated probabilities
   - Document if thresholds change
   - May improve F1 on calibrated predictions

3. **Create Deployment Artifacts**
   - Package model + calibrator for deployment
   - Document complete prediction pipeline
   - Test end-to-end inference with calibration

### FOR DEPLOYMENT

1. **Monitor Calibration**
   - Track ECE on deployment population
   - Set alert threshold (e.g., ECE > 0.1)
   - Plan recalibration procedure

2. **Clinical Validation**
   - Validate calibration prospectively
   - Compare predicted vs observed risk in practice
   - Recalibrate if population differs

---

## 14. PHASE 4 CONCLUSION

**Status: NOT COMPLETED**

### Current State

❌ Calibration analysis not performed  
❌ Calibration metrics not calculated  
❌ Calibrated models not generated  
❌ Reliability diagrams not created  
❌ Calibration method not selected  

### Required Actions

**Before proceeding to Phase 5 (Test Evaluation):**

1. Generate uncalibrated calibration metrics (ECE, MCE, Brier)
2. Fit sigmoid and isotonic calibrators
3. Evaluate calibrated performance
4. Select calibration method per model
5. Save calibrated model artifacts
6. Generate reliability diagrams
7. Document calibration results

**Estimated Effort:** 2-4 hours (scripting + analysis)

**Blocker:** Cannot proceed to test evaluation without calibration  
**Rationale:** Test evaluation should use calibrated models (deployment-ready)

### Impact on Timeline

**Current Progress:** 3/10 phases complete  
**Calibration Status:** Required but missing  
**Recommendation:** Pause evaluation, complete calibration, then resume

### Alternative Path (Not Recommended)

**If calibration cannot be completed:**
- Clearly document models are uncalibrated
- Label all probability predictions as "uncalibrated scores"
- Restrict use to ranking/triage only (no absolute risk estimates)
- Require calibration before any deployment consideration

---

## 15. COMPARISON TO INDUSTRY STANDARDS

### Healthcare ML Best Practices

**TRIPOD Guidelines (Transparent Reporting of Prediction Models):**
- Calibration analysis required for publication
- Calibration plots expected
- ECE/Brier score standard metrics

**FDA Guidance (Software as Medical Device):**
- Calibration analysis expected in submissions
- Reliability diagrams typical
- Post-market calibration monitoring recommended

**Clinical Journals:**
- NEJM, JAMA, Lancet expect calibration analysis
- Uncalibrated models rarely accepted
- Calibration often more important than discrimination for clinical use

**Current Status:** Does not meet industry standards without calibration

---

## APPENDIX A: Calibration Metrics Formulas

### Expected Calibration Error (ECE)

```
ECE = (1/N) × Σ |B_i| × |acc(B_i) - conf(B_i)|

Where:
- N = total samples
- B_i = bin i
- |B_i| = number of samples in bin i
- acc(B_i) = accuracy in bin i (observed frequency)
- conf(B_i) = average confidence in bin i (predicted probability)
```

### Maximum Calibration Error (MCE)

```
MCE = max_i |acc(B_i) - conf(B_i)|
```

### Brier Score

```
Brier = (1/N) × Σ(p_i - y_i)²

Where:
- p_i = predicted probability for sample i
- y_i = actual outcome (0 or 1)
```

---

## APPENDIX B: Calibration Code Skeleton

```python
from sklearn.calibration import CalibratedClassifierCV

# For each model
model = load_model('lightgbm_1.0.0.joblib')

# Get validation predictions (uncalibrated)
X_val, y_val = load_validation_data()
probs_uncal = model.predict_proba(X_val)[:, 1]

# Calculate uncalibrated metrics
ece_uncal = calculate_ece(y_val, probs_uncal)
mce_uncal = calculate_mce(y_val, probs_uncal)

# Fit calibrators
cal_sigmoid = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
cal_sigmoid.fit(X_val, y_val)

cal_isotonic = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
cal_isotonic.fit(X_val, y_val)

# Evaluate calibrated models
probs_sigmoid = cal_sigmoid.predict_proba(X_val)[:, 1]
ece_sigmoid = calculate_ece(y_val, probs_sigmoid)

probs_isotonic = cal_isotonic.predict_proba(X_val)[:, 1]
ece_isotonic = calculate_ece(y_val, probs_isotonic)

# Save results
results = {
    'uncalibrated': {'ece': ece_uncal, 'mce': mce_uncal},
    'sigmoid': {'ece': ece_sigmoid, 'mce': mce_sigmoid},
    'isotonic': {'ece': ece_isotonic, 'mce': mce_isotonic}
}
```

---

**Report Generated:** 2026-11-09  
**Next Phase:** BLOCKED - Complete calibration before Phase 5 (Test Evaluation)
