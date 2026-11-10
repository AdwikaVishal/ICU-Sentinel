# P2 Research Interpretation

## Executive Summary

This report interprets the results of the P2 advanced tabular ML experiments (Extra Trees, LightGBM, XGBoost) for sepsis early warning on the ICU Sentinel synthetic dataset.

## Critical Finding: No Model Outperforms Random Guessing

**All three models achieve AUROC ≈ 0.5 on the test set**, indicating no discriminative ability beyond chance on this synthetic dataset. This is the most important finding and must guide all downstream interpretation.

## Model-by-Model Analysis

### Extra Trees
- **Strengths**: Highest sensitivity (0.522), catches over half of sepsis cases
- **Weaknesses**: Very low specificity (0.516), generates 1,305 false alerts per 2,789 test hours
- **Clinical implication**: Would overwhelm clinicians with false alarms
- **Calibration**: Already well-calibrated (ECE=0.0036), minimal improvement from calibration

### LightGBM
- **Strengths**: Best validation AUROC (0.605), highest specificity (0.942)
- **Weaknesses**: Very low sensitivity (0.067), misses 93% of sepsis cases
- **Clinical implication**: Too conservative for early warning; unsafe as standalone
- **Calibration**: Poorly calibrated initially (ECE=0.0298), sigmoid calibration helps (ECE=0.0026)

### XGBoost
- **Strengths**: Highest specificity (0.945), fewest false positives
- **Weaknesses**: Lowest sensitivity (0.033), misses 97% of sepsis cases
- **Clinical implication**: Essentially useless for early warning
- **Calibration**: Poorly calibrated (ECE=0.0268), sigmoid helps (ECE=0.0018)

## Why Do All Models Fail?

### 1. Synthetic Data Limitations
- Random normal distributions for clinical variables
- No realistic pathophysiology patterns
- Missingness completely at random (MCAR), not MNAR like real ICU data
- Sepsis onset timing artificial

### 2. Feature Engineering Mismatch
- 994 features, many highly correlated
- Forward-fill with 24h window may not capture clinical relevance
- Rolling windows (6h, 12h, 24h) may miss critical trajectories

### 3. Class Imbalance
- Only ~3% positive hours in test set (90/2,789)
- 15% patient-level but most hours are negative
- Models optimize for majority class without proper calibration

### 4. No True Signal
- Synthetic sepsis labels based on arbitrary onset hour
- No causal relationship between clinical variables and sepsis in synthetic data
- Models memorize training set (AUROC=1.0) but don't generalize

## Clinical Implications

### False Negative Risk (Missed Sepsis)
- **LightGBM misses 93%** of sepsis cases (FN=84/90)
- **XGBoost misses 97%** of sepsis cases (FN=87/90)
- **Extra Trees misses 48%** (FN=43/90) - "best" but still unacceptable

### False Positive Burden (Alert Fatigue)
- **Extra Trees**: 1,305 FP / 2,789 hours = **0.47 alerts/hour** (11/day)
- **LightGBM**: 156 FP / 2,789 hours = **0.056 alerts/hour** (1.3/day)
- **XGBoost**: 148 FP / 2,789 hours = **0.053 alerts/hour** (1.3/day)

**Clinical reality**: Even 1.3 false alerts/day per patient is unsustainable in ICU. With 20 patients, that's 26 false alerts/day.

## Calibration Insights

- **Extra Trees** is naturally well-calibrated (tree ensemble property)
- **LightGBM/XGBoost** overconfident without calibration (common for boosting)
- **Sigmoid (Platt) calibration** consistently improves ECE for boosting methods
- **Isotonic calibration** less reliable with small validation sets

## Bootstrap Confidence Intervals

- All AUROC CIs include 0.5 → **no statistically significant discrimination**
- Wide CIs reflect small test set (100 patients, 2,789 hours)
- Overlapping CIs between all models → **no significant differences**

## Recommendations for Real Data

### If Applying to Real PhysioNet Data:
1. **Expect better performance**: Real data has true pathophysiological signals
2. **Tune hyperparameters**: Use Bayesian optimization on validation
3. **Consider ensembling**: Combine Extra Trees + LightGBM for sensitivity/specificity balance
4. **Calibrate all models**: Mandatory for clinical deployment
5. **Threshold selection**: Target sensitivity ≥ 0.8 for early warning, accept lower specificity

### Architecture Decisions:
1. **Extra Trees** → Best base for sensitivity-critical applications
2. **LightGBM** → Best for specificity-critical applications  
3. **Ensemble** → Weighted average of calibrated probabilities
4. **Never deploy XGBoost alone** for early warning (too conservative)

## Limitations of This Analysis

1. **Synthetic data only** - No clinical validity claims possible
2. **No temporal validation** - Single split, no cross-validation
3. **No subgroup analysis** - Performance may vary by age, comorbidity, ICU type
4. **No lead time analysis** - When do alerts fire relative to onset?
5. **No alert episode analysis** - Sustained vs transient alerts not evaluated
6. **Fixed feature set** - Feature selection on train only, but may miss interactions

## Conclusion

**On this synthetic dataset, no P2 model is clinically viable.** The exercise demonstrates the pipeline works correctly (no leakage, proper splits, calibration, bootstrap) but the data lacks signal.

**Next steps require real PhysioNet/CinC Challenge 2019 data.** The pipeline is ready; the signal is not.