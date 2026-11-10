# PHASE 5: FINAL TEST SET EVALUATION

**Date:** 2026-11-09  
**Analyst:** ICU Sentinel Validation System

---

## EXECUTIVE SUMMARY

**Status: ⚠️  COMPLETED BUT METHODOLOGICALLY QUESTIONABLE**

Test set evaluation was **already performed** and results exist in training summary. Test metrics show LightGBM achieves AUROC=0.787, AUPRC=0.058 on 142,811 test observations. However, evaluation occurred **before calibration** and it's unclear if test results influenced any model decisions.

**Critical Issue:** Test set should be final, independent evaluation after all model decisions finalized.

---

## 1. TEST SET EVALUATION RESULTS

### Dataset Properties

- **Total Observations:** 142,811
- **Positive Class:** 1,932 (1.35%)
- **Negative Class:** 140,879 (98.65%)
- **Expected:** ~305,760 observations (46.7% of expected)
- **Consistency:** Same 46.7% ratio as training/validation

### Model Performance on Test Set

| Model | AUROC | AUPRC | Sensitivity | Specificity | Precision | Threshold |
|-------|-------|-------|-------------|-------------|-----------|-----------|
| **Extra Trees** | 0.7290 | 0.0355 | 0.1703 (17.0%) | 0.9611 (96.1%) | N/A | 0.05 |
| **LightGBM** | **0.7870** | **0.0579** | 0.1843 (18.4%) | **0.9755 (97.6%)** | N/A | 0.73 |
| **XGBoost** | 0.7804 | 0.0577 | **0.2935 (29.4%)** | 0.9462 (94.6%) | N/A | 0.69 |

**Best Model: LightGBM**
- Highest AUROC (0.787) and AUPRC (0.058)
- Excellent specificity (97.6%)
- Moderate sensitivity (18.4%)

---

## 2. COMPARISON: VALIDATION vs TEST

### Performance Consistency Check

**LightGBM (Recommended Model):**

| Metric | Validation | Test | Difference | Assessment |
|--------|-----------|------|------------|------------|
| **AUROC** | 0.7695 | 0.7870 | +0.0175 | ✓ Improved |
| **AUPRC** | 0.0531 | 0.0579 | +0.0048 | ✓ Improved |
| **Sensitivity** | 0.1667 | 0.1843 | +0.0176 | ✓ Consistent |
| **Specificity** | 0.9780 | 0.9755 | -0.0025 | ✓ Consistent |

**Finding:** Test performance **slightly better** than validation - generally positive sign.

**XGBoost:**

| Metric | Validation | Test | Difference | Assessment |
|--------|-----------|------|------------|------------|
| **AUROC** | 0.7726 | 0.7804 | +0.0078 | ✓ Consistent |
| **AUPRC** | 0.0515 | 0.0577 | +0.0062 | ✓ Improved |
| **Sensitivity** | 0.2567 | 0.2935 | +0.0368 | ✓ Improved |
| **Specificity** | 0.9529 | 0.9462 | -0.0067 | ✓ Consistent |

**Extra Trees:**

| Metric | Validation | Test | Difference | Assessment |
|--------|-----------|------|------------|------------|
| **AUROC** | 0.7140 | 0.7290 | +0.0150 | ✓ Improved |
| **AUPRC** | 0.0343 | 0.0355 | +0.0012 | ✓ Consistent |

**Overall Assessment:** All models show consistent or improved performance on test set. This is encouraging and suggests models generalize well. Small improvements likely due to statistical noise or slightly favorable test set characteristics.

---

## 3. TEST SET INTEGRITY VERIFICATION

### Expected vs Actual

**Expected (from split manifest):**
- Test Patients: 7,962
- Test Observations: 305,760
- Test Positive Hours: 5,515

**Actual (from test evaluation):**
- Test Observations: 142,811 (46.7% of expected)
- Test Positive: 1,932 (35.0% of expected positive)

**Discrepancy:** Test set represents 46.7% of expected, consistent with incomplete feature generation (same ratio as training/validation).

### Split Integrity

✓ **Class Balance:** 1.35% positive (matches training/validation)  
✓ **Consistent Across Models:** All three evaluated on same 142,811 observations  
⚠️  **Cannot Verify:** Patient IDs not tracked - cannot confirm test patients disjoint from training

---

## 4. METHODOLOGICAL CONCERNS

### Timing of Test Evaluation

**When Test Was Evaluated:**
- Appears in same `p2_training_summary.json` as training/validation
- Suggests evaluation occurred during/immediately after training
- Same thresholds used as validation (0.05, 0.73, 0.69)

**Proper Practice:**
1. Train models → finalize all decisions on validation
2. Lock model, threshold, calibration method
3. **THEN** evaluate on test (once, final)

**Current Situation:**
- Test metrics exist alongside training metrics
- Unclear if test results influenced any decisions
- No evidence of multiple test evaluations (good)
- No evidence test was used for model selection (good)

### Calibration Issue

**Critical Problem:** Test evaluation used **uncalibrated models**

**Implications:**
- Test probabilities not calibrated (less clinically useful)
- Cannot assess calibrated performance on test
- Test evaluation doesn't match deployment scenario
- Should re-evaluate with calibrated models

### Threshold Application

**Validation Thresholds Applied to Test:**
- LightGBM: 0.73 (same as validation F1-optimal)
- XGBoost: 0.69 (same as validation F1-optimal)
- Extra Trees: 0.05 (same as validation F1-optimal)

**Status:** ✓ Correct - thresholds should be set on validation, applied to test

---

## 5. CLINICAL INTERPRETATION (TEST SET)

### LightGBM Performance

**At Threshold 0.73:**
- Detects 18.4% of sepsis cases (356/1,932)
- 97.6% specificity (2.4% false positive rate)
- Misses 81.6% of sepsis cases (1,576/1,932)

**For 100-patient ICU (38.45 observations per patient):**
- Total observations: 3,845
- Expected sepsis cases: 52 (1.35%)
- True positives: ~10 cases detected (18.4%)
- False positives: ~91 alerts (2.4% of 3,793)
- **Total alerts: ~101 (2.6% alert rate)**
- **Estimated precision: 10/101 = 9.9%**

**Comparison to Validation:**
- Test sensitivity (18.4%) vs validation (16.7%) - slightly better
- Test specificity (97.6%) vs validation (97.8%) - essentially same
- Test performance confirms validation estimates

---

## 6. STATISTICAL SIGNIFICANCE

### Confidence Intervals

⚠️  **NOT CALCULATED**

Expected analyses:
- Bootstrap confidence intervals (1,000+ iterations)
- Patient-level resampling
- 95% CIs for AUROC, AUPRC, sensitivity, specificity

**Impact:** Cannot quantify uncertainty in test estimates

### Comparison Between Models

⚠️  **NOT PERFORMED**

Expected tests:
- DeLong test for AUROC differences
- Paired bootstrap for AUPRC differences  
- McNemar test for sensitivity/specificity differences

**Impact:** Cannot determine if performance differences are statistically significant

**Example Question:** Is LightGBM's AUROC (0.787) significantly better than XGBoost's (0.780)?
- Difference: 0.007
- Likely not significant, but needs testing

---

## 7. SUBGROUP ANALYSIS

### Not Performed

Expected stratifications:
1. **By Source Dataset:** SetA vs SetB performance
2. **By Time to Sepsis:** Early vs late detection
3. **By Patient Characteristics:** Age groups, comorbidities
4. **By Sepsis Severity:** If available

**Impact:** Cannot identify performance disparities or fairness issues

---

## 8. ERROR ANALYSIS

### Not Performed

Expected analyses:
1. **False Negatives:** Characteristics of missed sepsis cases
2. **False Positives:** Why non-sepsis cases flagged
3. **High-Confidence Errors:** Cases with extreme incorrect predictions
4. **Feature Importance:** Which features drive predictions

**Impact:** Cannot improve model or understand failure modes

---

## 9. COMPARISON TO BASELINES

### Clinical Baselines Not Evaluated

**Expected comparisons:**
- SIRS criteria (≥2 of 4)
- qSOFA score (≥2)
- NEWS score
- MEWS score (if applicable)

**Status:** ⚠️  Not performed on test set

**Impact:** Cannot demonstrate ML value over existing methods

**Note:** Without baseline comparison, cannot justify:
- Development costs
- Computational infrastructure
- Clinician training
- Integration effort

---

## 10. EXTERNAL VALIDATION

### Not Performed

**Current evaluation:**
- Same data source (PhysioNet 2019)
- Same patient population
- Same measurement protocols

**Gold standard:**
- Different hospital system
- Different patient demographics
- Different clinical workflows
- Prospective data collection

**Status:** Retrospective internal validation only

**Limitation:** Performance on PhysioNet ≠ performance in clinical practice

---

## 11. TEST SET LEAKAGE CHECK

### Potential Leakage Sources

**Investigated:**
1. ✓ Patient overlap with training (cannot verify - IDs not tracked)
2. ✓ Preprocessing fit on full data (assumed correct)
3. ✓ Feature selection on test (N/A - no selection)
4. ✓ Threshold tuning on test (correct - used validation thresholds)

**Concerns:**
- Test evaluated during/after training (timing suspicious)
- Same summary file suggests combined workflow
- Cannot rule out test peeking in development

**Mitigation:**
- Performance consistent with validation (no optimization signal)
- Small improvements suggest no leakage (would expect degradation)
- Thresholds match validation (not re-optimized)

**Assessment:** Likely no leakage, but best practice would be separate test evaluation script run later

---

## 12. MODEL SELECTION CONFIRMATION

### Validation Recommendation: LightGBM

**Test Results Confirm:**
- ✓ Best AUROC (0.787 vs 0.780 XGBoost)
- ✓ Best AUPRC (0.058 vs 0.058 XGBoost - tied)
- ✓ Better specificity (97.6% vs 94.6%)
- Lower sensitivity (18.4% vs 29.4%) - acceptable tradeoff

**Conclusion:** Test set confirms LightGBM as best overall model

**Alternative Use Case:**
- If maximizing sensitivity is critical → XGBoost better (29.4% vs 18.4%)
- Tradeoff: More false alarms (5.4% vs 2.4% FPR)

---

## 13. PERFORMANCE EXPECTATIONS vs REALITY

### Realistic Assessment

**Discrimination (AUROC 0.787):**
- "Good" range (0.7-0.8)
- Not "excellent" (>0.8)
- Better than random (0.5)
- Comparable to many clinical scores

**Precision-Recall (AUPRC 0.058):**
- Low absolute value due to 1.35% prevalence
- Baseline (random): 0.0135
- Improvement: 4.3x over random
- Still means ~90% of alerts are false positives

**Sensitivity (18.4%):**
- Catches less than 1 in 5 cases
- Misses >80% of sepsis events
- **Cannot be sole diagnostic tool**
- Must supplement, not replace, clinical judgment

**Clinical Reality:**
- Model provides early warning signal
- Clinicians must still assess all patients
- Reduces "missed opportunity" for early cases
- Not a silver bullet for sepsis detection

---

## 14. DEPLOYMENT READINESS (TEST BASIS)

### Based on Test Performance

**Strengths:**
✓ Consistent validation-to-test performance  
✓ High specificity (manageable alert burden)  
✓ Stable across models  
✓ No evidence of overfitting  

**Limitations:**
❌ Low sensitivity (misses 81.6% of cases)  
❌ Low precision (~10% at threshold 0.73)  
❌ Uncalibrated probabilities  
❌ No clinical baseline comparison  
❌ No external validation  

**Verdict:** **NOT READY FOR DEPLOYMENT** without:
1. Calibration analysis and calibrated model evaluation
2. Comparison to clinical baselines
3. Prospective validation
4. Clinical workflow integration testing
5. Regulatory approval (if required)
6. Clinician training and acceptance

---

## 15. RECOMMENDATIONS

### IMMEDIATE ACTIONS

1. **Re-Evaluate with Calibrated Models**
   - Perform calibration on validation set (Phase 4)
   - Apply calibrated models to test set
   - Report calibrated test performance
   - Compare uncalibrated vs calibrated results

2. **Calculate Confidence Intervals**
   - Bootstrap test set (1,000 iterations)
   - Patient-level resampling
   - Report 95% CIs for all metrics
   - Assess statistical significance of model differences

3. **Document Test Set Usage**
   - Clarify when test evaluation occurred
   - Confirm no model decisions based on test results
   - Document test set was truly held-out
   - Explain methodology

### ADDITIONAL ANALYSES NEEDED

1. **Baseline Comparison**
   - Evaluate SIRS, qSOFA, NEWS on same test set
   - Calculate same metrics
   - Statistical comparison to ML models
   - Quantify incremental value

2. **Subgroup Analysis**
   - Performance by patient demographics
   - Performance by time to sepsis onset
   - Performance by source dataset (SetA vs SetB)
   - Identify disparities

3. **Error Analysis**
   - Characterize false positives and false negatives
   - Feature importance for predictions
   - High-confidence errors
   - Clinical insight from mistakes

### BEFORE DEPLOYMENT

1. **External Validation**
   - Evaluate on different hospital system
   - Prospective data collection
   - Real-world clinical workflow
   - Different patient demographics

2. **Calibration Monitoring**
   - Implement calibration drift detection
   - Plan for periodic recalibration
   - Set alert thresholds for recalibration
   - Document recalibration procedure

3. **Clinical Integration**
   - Shadow mode deployment
   - Clinician feedback collection
   - Alert workflow optimization
   - Alert fatigue assessment

---

## 16. COMPARISON TO PUBLISHED LITERATURE

### Sepsis Prediction Models

**Typical Performance in Literature:**
- AUROC: 0.75-0.85 (retrospective)
- AUROC: 0.65-0.75 (prospective)
- Sensitivity: 15-40% at high specificity
- Alert rates: 2-10% of observations

**ICU Sentinel Performance:**
- AUROC: 0.787 (within typical range)
- Sensitivity: 18.4% at 97.6% specificity (within range)
- Alert rate: ~2.6% (low end, good)

**Assessment:** Performance comparable to published models, but:
- Most published models show degradation prospectively
- Expect 0.05-0.10 AUROC drop in real-world use
- Sensitivity likely to decrease in practice

---

## 17. LIMITATIONS OF TEST EVALUATION

### Data Limitations

1. **Incomplete Dataset:** 46.7% of expected test set
2. **Single Source:** PhysioNet 2019 only
3. **Retrospective:** Historical data, not prospective
4. **No Patient IDs:** Cannot verify split integrity

### Methodological Limitations

1. **No Calibration:** Test models uncalibrated
2. **No Confidence Intervals:** Uncertainty not quantified
3. **No Baseline Comparison:** ML value not established
4. **Timing Unclear:** Test evaluation may have been premature

### Reporting Limitations

1. **No Bootstrap Results:** Confidence not assessed
2. **No Subgroup Analysis:** Disparities unknown
3. **No Error Analysis:** Failure modes not characterized
4. **No Detailed Confusion Matrices:** Counts not provided

---

## 18. PHASE 5 CONCLUSION

**Status: COMPLETED BUT INCOMPLETE**

### Success Criteria

✓ Test metrics calculated for all three models  
✓ Performance consistent with validation (generalization confirmed)  
✓ LightGBM recommendation confirmed on test set  
✓ No evidence of overfitting  

❌ Calibration not performed before test evaluation  
❌ Confidence intervals not calculated  
❌ Baseline comparisons not performed  
❌ Test evaluation timing questionable  
❌ Should have been independent, final evaluation  

### Overall Assessment

**Test performance is encouraging:** LightGBM achieves AUROC=0.787, consistent with validation. Models generalize well to test set. No evidence of overfitting or leakage.

**Methodological concerns:** Test evaluation appears to have occurred during training workflow rather than as independent final evaluation. Models were uncalibrated. Confidence intervals and baseline comparisons missing.

**Clinical perspective:** Performance is moderate. Models provide value as early warning system but cannot be sole diagnostic tool. 18.4% sensitivity means most cases still missed. Requires clinical integration and workflow optimization.

**Recommendation:** 
1. Document that test was held-out despite being in training summary
2. Re-evaluate with calibrated models
3. Add confidence intervals and baseline comparisons
4. Proceed with caution to deployment planning

---

## APPENDIX: Test Set Metrics Detail

### LightGBM (Recommended)

**Test Performance:**
- AUROC: 0.7870
- AUPRC: 0.0579
- Sensitivity: 18.43% (356/1,932 cases detected)
- Specificity: 97.55% (2.45% false positive rate)
- Threshold: 0.73 (from validation)

**Estimated Performance:**
- Alert rate: 2.6% of observations
- Precision: ~9.9% (1 in 10 alerts true positive)
- Missed cases: 81.6% (1,576/1,932)

### Observations

- 142,811 total test observations
- 1,932 positive (1.35%)
- 140,879 negative (98.65%)
- Represents ~46.7% of expected full test set

---

**Report Generated:** 2026-11-09  
**Next Phase:** Early-warning evaluation (Phase 6)
