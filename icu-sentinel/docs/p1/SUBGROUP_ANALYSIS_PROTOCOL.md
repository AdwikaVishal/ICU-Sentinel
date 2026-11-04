# Subgroup Analysis Protocol — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 7 — Research Evaluation  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Overview

Subgroup analysis evaluates model performance across different patient
subgroups to identify potential disparities in model performance. This is
important for ensuring that the model performs equitably across different
patient populations.

---

## 2. Subgroup Dimensions

The following subgroup dimensions are supported:

| Subgroup Dimension | Description | Source |
|--------------------|-------------|--------|
| Gender | Male vs Female | Patient metadata |
| Age Group | Age brackets (e.g., 18-44, 45-64, 65+) | Patient metadata |
| Source Dataset | Set A vs Set B | Dataset source identifier |

---

## 3. Subgroup Analysis Methodology

For each subgroup dimension:

1. **Group patients** by the subgroup dimension (e.g., gender, age group)
2. **Compute metrics** for each subgroup separately
3. **Compare metrics** across subgroups
4. **Report patient counts** for each subgroup
5. **Handle small subgroups** safely (report but flag as small)

### Subgroup Dimensions

| Dimension | Source | Values |
|-----------|--------|--------|
| Gender | Patient metadata | M, F |
| Age Group | Derived from Age | <18, 18-44, 45-64, 65-74, 75+ |
| Source Dataset | Dataset identity | set_a, set_b |

---

## 4. Subgroup Analysis Output

Subgroup analysis results are saved to
`data/reports/classical_ml/subgroup_analysis_report.json`:

```json
{
  "model_name": "random_forest",
  "model_version": "1.0.0",
  "split": "test",
  "subgroup_dimension": "gender",
  "subgroups": [
    {
      "subgroup_name": "gender",
      "subgroup_value": "M",
      "n_patients": 500,
      "n_positive_patients": 150,
      "n_rows": 2500,
      "n_positive_rows": 500,
      "positive_rate": 0.2,
      "metrics": {
        "auroc": 0.85,
        "auprc": 0.52,
        "sensitivity": 0.75,
        "specificity": 0.82,
        "precision": 0.27,
        "f1": 0.39
      },
      "ci_lower": {
        "auroc": null,
        "auprc": null,
        ...
      },
      "ci_upper": {
        "auroc": null,
        "auprc": null,
        ...
      },
      "bootstrap": false
    }
  ],
  "source": "synthetic"
}
```

---

## 5. Small Subgroup Handling

Subgroups with fewer than a configurable number of patients (default: 5)
are still reported but flagged as small. Conclusions should not be drawn
from subgroups with very few patients.

---

## 6. Limitations

- Subgroup analysis requires demographic data, which may not be available
  for all patients
- Small subgroups may have unreliable performance estimates
- Subgroup analysis does not establish causality
- Multiple comparisons across subgroups increase the risk of false
  discoveries
