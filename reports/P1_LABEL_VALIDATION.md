# P1 Label Validation Report

## Sepsis Label Definition

The PhysioNet/CinC Challenge 2019 dataset uses a **shifted SepsisLabel** convention:
- `SepsisLabel = 1` for hours at or after `t_sepsis - 6` (i.e., 6 hours before sepsis onset and onwards)
- `SepsisLabel = 0` for hours before `t_sepsis - 6`
- For patients who never develop sepsis, `SepsisLabel = 0` for all hours

This means the original `SepsisLabel` **cannot be used directly** as a 6-hour pre-onset prediction target because it is already shifted 6 hours forward.

## Derived Target: `sepsis_within_6h`

To enable strict 6-hour pre-onset prediction, we derive a new target column:

### Definition
- **Positive (1)**: Hours in the window `[onset, onset + 5]` (6 hours starting at sepsis onset)
- **Negative (0)**: Hours before `onset` for septic patients; all hours for non-septic patients
- **Ineligible (NaN)**: Hours at or after `onset + 6` for septic patients

### Formal Definition
Let `onset` be the first hour where `SepsisLabel == 1` for a patient.
- If patient has sepsis (`onset` exists):
  - `sepsis_within_6h = 1` for `hour ∈ [onset, onset + 5]`
  - `sepsis_within_6h = 0` for `hour < onset`
  - `sepsis_within_6h = NaN` (ineligible) for `hour ≥ onset + 6`
- If patient has no sepsis (all `SepsisLabel == 0`):
  - `sepsis_within_6h = 0` for all hours (eligible)
- `target_eligible = 1` where `sepsis_within_6h` is not NaN, else 0

### Validation Results (Synthetic Dataset)

| Split | N Patients | N Eligible Hours | N Positive | N Negative | Positive Rate |
|-------|-----------|-----------------|------------|------------|---------------|
| Train | 350 | 9,438 | 1,063 | 8,375 | 11.3% |
| Validation | 50 | 1,322 | 154 | 1,168 | 11.6% |
| Test | 100 | 2,699 | 302 | 2,397 | 11.2% |

### Key Properties Verified

1. **Original SepsisLabel preserved**: The original `SepsisLabel` column is never modified.
2. **No future information in target**: The derived target only uses information available at prediction time.
3. **Strict pre-onset window**: Positive labels only for the 6 hours starting at true onset.
4. **Post-onset exclusion**: Hours at/after `onset + 6` are marked ineligible to prevent label leakage.
5. **Non-septic patients**: All hours are eligible negatives.
6. **Patient independence**: Targets computed per-patient, no cross-patient contamination.

### Test Set Target Summary (100 patients)

- Total test rows: 3,001
- Eligible rows: 2,699 (89.9%)
- Ineligible rows: 302 (10.1%) - these are post-onset+6 hours
- Positive rows: 302 (11.2% of eligible)
- Negative rows: 2,397 (88.8% of eligible)

### Conclusion

The derived target `sepsis_within_6h` correctly implements a strict 6-hour pre-onset prediction target suitable for early warning system evaluation. The original `SepsisLabel` is preserved for reference and clinical baseline computation.