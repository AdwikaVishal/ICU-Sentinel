# ICU Sentinel — Research Integrity Remediation

Purpose: correct research-integrity issues identified in `docs/execution/MASTER_PROMPT_COMPLETION_AUDIT.md` after the initial real-data fast P1/P2 experiments. Existing artifacts and split manifests are preserved; nothing is overwritten and no model is retrained.

## Phase 1 — Fast split manifest reconciliation

### Finding
P1 and P2 did **not** use identical training cohorts.

- P1 (`scripts/train_p1_fast.py`, log `data/reports/logs/p1_train_fast_stdout.log`): used **8,000** stratified training patients (the script default), 306,558 train records. The run log prints `train_patients_requested: 8000, train_patients_used: 8000`.
- P2 (`scripts/generate_p2_artifacts.py`, log `data/reports/logs/p2_fast_stdout.log`): loaded `data/splits/split_manifest_fast.json` but `data/reports/p2_fast/p2_training_summary.json` reports train `n_rows=228,703`, `n_positive=2,607` — consistent with approximately **6,000** training patients after eligible-hour filtering.
- Timeline (file mtimes): P2 ran at 19:42 (~6,000 patients); `split_manifest_fast.json` was later rewritten to 8,000 patients at 20:03; P1 ran at 21:24 using the 8,000-patient manifest. The exact 6,000-patient manifest that P2 consumed at runtime was **overwritten** and cannot be recovered from disk.

### Conclusion
The premise in the original audit that "P1 used 6,000" is incorrect: P1 used 8,000. P1 (8,000) and P2 (~6,000) used different training cohorts, so matched P1/P2 comparison is currently invalid.

### Reconstruction
The 6,000-patient cohort is deterministically reconstructible because `train_p1_fast.py._make_fast_manifest` applies `sklearn.model_selection.train_test_split(train_size=6000, random_state=42, stratify=has_sepsis)` to the full manifest train split. A validation check: reconstructed 6,000 patients = 230,232 raw records, which after the loader's eligible-hour filtering (`target_eligible`, drop NaN target) yields approximately P2's reported 228,703 eligible train rows. This strongly supports that P2 consumed a 6,000-patient manifest generated with seed 42 stratified sampling.

### Artifact produced (existing manifests NOT overwritten)
- `data/splits/split_manifest_fast_6000.json` — new canonical 6,000-patient fast manifest (train 6,000 / validation 4,034 / test 8,068; no patient overlap; total 20,102 patients). This is a reconstruction and should be treated as the canonical matched cohort for any future matched P1/P2 experiment, not as proof of P2's exact runtime patient IDs.

Reproducibility command (generates the canonical 6,000 manifest without touching existing files):
```
python - <<'PY'
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
src = Path('data/splits/split_manifest.json')
out = Path('data/splits/split_manifest_fast_6000.json')
m = json.load(open(src))
train = [p for p in m['patients'] if p['split'] == 'train']
other = [p for p in m['patients'] if p['split'] != 'train']
labels = [bool(p['has_sepsis']) for p in train]
sel, _ = train_test_split(train, train_size=6000, random_state=42, stratify=labels)
selected = sel + other
selected.sort(key=lambda p: p['patient_id'])
m2 = dict(m); m2['patients'] = selected
m2['n_patients_total'] = len(selected)
m2['n_patients_train'] = len(sel)
m2['n_records_train'] = sum(p['n_records'] for p in sel)
m2['n_sepsis_patients_train'] = sum(1 for p in sel if p['has_sepsis'])
m2['n_sepsis_positive_hours_train'] = sum(p['n_sepsis_positive_hours'] for p in sel)
m2['fast_protocol'] = {'enabled': True, 'random_state': 42, 'stratified_by': 'has_sepsis',
    'train_patients_requested': 6000, 'train_patients_used': len(sel),
    'validation_and_test': 'full',
    'note': 'canonical reconstruction; exact P2 runtime manifest was overwritten'}
with open(out, 'w') as f: json.dump(m2, f, indent=2)
print('train', len(sel), 'val', len([p for p in selected if p["split"]=="validation"]),
      'test', len([p for p in selected if p["split"]=="test"]))
PY
```

## Phase 2 — P1 metric verification

Source of truth: `data/reports/classical_ml_fast/training_summary.json` (models dict) + `data/reports/logs/p1_train_fast_stdout.log`.

### P1 test metrics (test set — one-shot at validation-selected threshold)
Only AUROC, AUPRC, F1 and threshold were persisted in the run log. Test sensitivity, specificity, precision, Brier score, evaluated rows and positive samples are **NOT persisted** (`training_summary.json` `test_metrics` is empty for all three models; model metadata files contain no test metrics).

| Model | Test AUROC | Test AUPRC | Test F1 | Test threshold | Test sens/spec/prec/brier | Test rows | Test positives |
|---|---:|---:|---:|---:|---|---:|---:|
| Logistic regression | 0.7025 | 0.0296 | 0.0694 | 0.83 | unavailable | unavailable | unavailable |
| Random forest | 0.7041 | 0.0274 | 0.0660 | 0.56 | unavailable | unavailable | unavailable |
| SVM (4,000-row cap) | 0.6300 | 0.0209 | 0.0481 | 0.02 | unavailable | unavailable | unavailable |

Threshold-selection methodology: `threshold_selection_metric=f1`, `threshold_grid_step=0.01`, selected on validation split only; no test-data tuning.

### P1 train / validation metrics (persisted in `training_summary.json` `models` dict)
Rows/positives computed from the stored confusion matrices (tp+tn+fp+fn / tp+fn).

| Model | Split | Rows | Positives | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | Brier | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | train | 304,511 | 3,481 | 0.8209 | 0.0464 | 0.7443 | 0.7530 | 0.0337 | 0.0644 | 0.1784 | not stored |
| Logistic regression | validation | 156,036 | 1,753 | 0.6934 | 0.0293 | 0.2031 | 0.9517 | 0.0456 | 0.0745 | 0.1836 | not stored |
| Random forest | train | 304,511 | 3,481 | 0.9255 | 0.4143 | 0.7771 | 0.9115 | 0.0922 | 0.1649 | 0.1399 | not stored |
| Random forest | validation | 156,036 | 1,753 | 0.7311 | 0.0329 | 0.1637 | 0.9616 | 0.0463 | 0.0721 | 0.1451 | not stored |
| SVM | train | 4,000 | 46 | 0.9988 | 0.9815 | 0.9783 | 0.9977 | 0.8333 | 0.9000 | 0.0102 | not stored |
| SVM | validation | 156,036 | 1,753 | 0.6412 | 0.0220 | 0.1306 | 0.9610 | 0.0367 | 0.0573 | 0.0111 | not stored |

SVM train is subsampled to 4,000 rows (`svm_max_train_rows=4000`, `subsampled=True`), explaining the near-perfect train metrics and poor validation/test generalization.

### Corrections needed in `docs/execution/FINAL_RESEARCH_REPORT.md`
The P1 test table in that report is incorrect (it lists LR 0.6794 / RF 0.6894 / SVM 0.6053 AUROC; the run log shows 0.7025 / 0.7041 / 0.6300). It also states P1 used 6,000 training patients; P1 used 8,000. These must be corrected (see Phase 5).

## Phase 3 — Feature consistency investigation

### Finding
- P1: 994 input features → **960 selected** (34 removed). All 34 removed are `*_hours_since_last` missingness features, removed because they are infinite for patients with no prior observation (`removed_infinite`, `fit_on_train_only=True`). P1 models trained on 960 features.
- P2: **994 features** used, including `*_hours_since_last`; infinities are handled by `sanitize_infinities()` in `ml/p2_boosting/pipelines.py` at fit/predict time. P2 models trained on 994 features.

### Interpretation
P1 drops the `*_hours_since_last` features entirely (they carry no information for never-observed variables and would break distance/tree splits). P2 retains them and sanitizes infinities, which changes their scientific meaning (an imputed finite "hours since last" is not the same as "never observed"). This is a deliberate policy difference, not a bug, but it means P1 and P2 are not feature-matched.

### Recommendation (no policy change, no retraining)
For the next matched experiment, adopt one consistent policy across P1 and P2:
- Option A: drop `*_hours_since_last` everywhere (P1's current policy), or
- Option B: keep and sanitize them everywhere (P2's current policy).
Do not change existing artifacts or retrain models; document the chosen policy in the next experiment config.

## Phase 4 — Reproducibility verification

Verified against existing artifacts (no inference from code):

| Requirement | Status | Evidence |
|---|---|---|
| Original target policy | CONFIRMED | `data/features/feature_schema.json` + `data/features/feature_summary.json`: target `sepsis_within_6h`; onset = first hour `SepsisLabel==1`; 1 for onset..onset+5, 0 before onset and non-septic, ineligible at/after onset+6; original `SepsisLabel` unchanged (`original_label_modified=false`). |
| Eligible-hour filtering | CONFIRMED | `ml/p1_clinical/classical_ml/data_loader.py:_build_split_frame` filters `target_eligible` and drops NaN target; `feature_summary.json` target_summary: 1,541,845 eligible / 10,365 ineligible / 17,551 positive / 1,524,294 negative. |
| Patient-level split integrity | CONFIRMED | `training_summary.json` `split_overlap_check`: `leakage_detected=False`, `every_patient_in_exactly_one_split=True`, `n_overlapping_patients=0`. Current `split_manifest_fast.json`: train/val/test overlap = 0. |
| Training-only preprocessing | CONFIRMED | `training_summary.json` `feature_selection_report`: `fit_on_train_only=True`; 34 infinite features removed from train only. |
| Validation-only threshold selection | CONFIRMED | Config `threshold_selection_metric=f1`, `threshold_grid_step=0.01`; thresholds stored per model and applied to test. |
| Frozen test evaluation | CONFIRMED | P2 `p2_training_summary.json` test block: one-shot test metrics at validation threshold; report states no test-data tuning. P1 log prints test metrics at validation-selected threshold. |

### Data-quality notes (metadata unreliability)
P1 model metadata files (`artifacts/p1/classical_ml_fast/*_1.0.0_metadata.json`) are unreliable: `source` reads `"synthetic"` (data is real), `random_seed` reads `0` (training used 42), and `n_rows_train`/`n_patients_train` read `0`. Do not rely on metadata for provenance; rely on `training_summary.json` and run logs.

## Phase 5 — Documentation

### Corrected P1 results table (replaces the incorrect table in FINAL_RESEARCH_REPORT.md)

Test set (one-shot, validation-selected threshold; detailed test metrics unavailable):

| Model | Test AUROC | Test AUPRC | Test F1 | Test threshold |
|---|---:|---:|---:|---:|
| Logistic regression | 0.7025 | 0.0296 | 0.0694 | 0.83 |
| Random forest | 0.7041 | 0.0274 | 0.0660 | 0.56 |
| SVM (4,000-row cap) | 0.6300 | 0.0209 | 0.0481 | 0.02 |

Train cohort: **8,000** stratified patients (304,511 eligible train rows), not 6,000.

### P2 results (for reference, unchanged)
`data/reports/p2_fast/p2_training_summary.json`, test at validation threshold:
- Extra Trees: AUROC 0.719, AUPRC 0.031, F1 0.062, threshold 0.56
- XGBoost: AUROC 0.762, AUPRC 0.035, F1 0.079, threshold 0.69
LightGBM disabled (Windows DLL). P2 used ~6,000 train patients — **not matched to P1's 8,000**.

## GO / NO-GO recommendation

**GO to proceed with matched P1/P2 experiments, but only after cohort reconciliation.**

Do not claim matched P1/P2 results from the current artifacts: P1 used 8,000 train patients and P2 used ~6,000, and P1/P2 used different feature sets (960 vs 994). Before any matched comparison:

1. Regenerate both P1 and P2 on the canonical matched cohort `data/splits/split_manifest_fast_6000.json` (6,000 train patients) — or on a single agreed manifest — using the reproducibility command in Phase 1.
2. Align the feature policy (960 vs 994) per the Phase 3 recommendation.
3. Persist full test-set metrics (sensitivity, specificity, precision, Brier, rows, positives) for P1 so test results are not limited to AUROC/AUPRC/F1/threshold.
4. Correct `docs/execution/FINAL_RESEARCH_REPORT.md` P1 table and patient count.

No clinical/deployment claims are supported by these fast-protocol results.
