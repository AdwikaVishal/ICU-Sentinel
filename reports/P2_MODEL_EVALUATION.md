# P2 Model Evaluation Report

## Test Set Performance Comparison (Synthetic Data)

| Model | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | Brier | Threshold |
|-------|-------|-------|-------------|-------------|-----------|-----|-------|-----------|
| Extra Trees | 0.490 | 0.031 | 0.522 | 0.516 | 0.035 | 0.065 | 0.032 | 0.030 |
| LightGBM | 0.503 | 0.035 | 0.067 | 0.942 | 0.037 | 0.048 | 0.032 | 0.010 |
| XGBoost | 0.501 | 0.033 | 0.033 | 0.945 | 0.020 | 0.025 | 0.037 | 0.190 |

## Validation Set Performance (Threshold Selection)

| Model | AUROC | AUPRC | Sensitivity | Specificity | F1 | Threshold |
|-------|-------|-------|-------------|-------------|-----|-----------|
| Extra Trees | 0.575 | 0.043 | 0.571 | 0.547 | 0.072 | 0.030 |
| LightGBM | 0.605 | 0.045 | 0.143 | 0.924 | 0.081 | 0.010 |
| XGBoost | 0.532 | 0.041 | 0.167 | 0.945 | 0.115 | 0.190 |

## Key Observations

### Extra Trees
- **Best validation AUROC**: 0.575
- **Highest test sensitivity**: 0.522 (catches more sepsis cases)
- **Lowest test specificity**: 0.516 (many false alerts)
- **Balanced approach**: Moderate sensitivity/specificity trade-off

### LightGBM
- **Best validation AUROC**: 0.605
- **Best validation F1**: 0.081
- **Low test sensitivity**: 0.067 (misses most sepsis cases)
- **High test specificity**: 0.942 (few false alerts)
- **Conservative threshold**: 0.010

### XGBoost
- **Lowest validation AUROC**: 0.532
- **Lowest test sensitivity**: 0.033
- **Highest test specificity**: 0.945
- **Highest threshold**: 0.190
- **Most conservative**: Very few positive predictions

## Confusion Matrices (Test Set)

| Model | TP | TN | FP | FN |
|-------|----|----|----|----|
| Extra Trees | 47 | 1394 | 1305 | 43 |
| LightGBM | 6 | 2543 | 156 | 84 |
| XGBoost | 3 | 2551 | 148 | 87 |

## ROC/PR Curves

ROC and PR curve data saved to:
- `artifacts/p2/roc_data/{model}.json`
- `artifacts/p2/pr_data/{model}.json`

## Calibration

Calibration analysis saved to:
- `artifacts/p2/calibration_results.json`
- `reports/P2_CALIBRATION.md`

### Summary
- Extra Trees: Already well-calibrated (ECE=0.0036), sigmoid improves to 0.0011
- LightGBM: Poorly calibrated (ECE=0.0298), sigmoid improves to 0.0026
- XGBoost: Poorly calibrated (ECE=0.0268), sigmoid improves to 0.0018

## Bootstrap Confidence Intervals (Test Set, n=100)

| Model | AUROC 95% CI | AUPRC 95% CI | Sensitivity 95% CI | Specificity 95% CI |
|-------|--------------|--------------|-------------------|-------------------|
| Extra Trees | [0.420, 0.558] | [0.016, 0.045] | [0.409, 0.622] | [0.487, 0.553] |
| LightGBM | [0.418, 0.654] | [0.021, 0.082] | [0.000, 0.197] | [0.920, 0.963] |
| XGBoost | [0.398, 0.649] | [0.018, 0.059] | [0.000, 0.092] | [0.918, 0.969] |

## Statistical Significance

- All models have AUROC CIs that include 0.5 (no better than random)
- Extra Trees has highest point estimate AUROC but widest CI
- LightGBM and XGBoost have overlapping CIs with Extra Trees
- No statistically significant difference between models on this synthetic data

## Computational Efficiency

| Model | Training Time | Model Size | Inference Speed |
|-------|--------------|------------|-----------------|
| Extra Trees | ~30s | 15.8 MB | Fast |
| LightGBM | ~15s | ~5 MB | Very Fast |
| XGBoost | ~45s | ~8 MB | Fast |

## Limitations

1. **Synthetic Data**: All results from synthetic fixture data, not scientifically valid
2. **Low Prevalence**: 15% patient-level, ~3% hour-level positive rate
3. **Small Test Set**: 100 patients, 2,789 eligible hours
4. **No Hyperparameter Tuning**: Used default configurations
5. **Single Random Seed**: Results may vary with different splits

## Conclusion

On synthetic data, all three models perform near random (AUROC ≈ 0.5). Extra Trees achieves the highest sensitivity but at the cost of many false positives. LightGBM and XGBoost are more conservative with higher specificity but miss most sepsis cases. Calibration significantly improves Brier scores for LightGBM and XGBoost. No model demonstrates clinically useful performance on this synthetic dataset.