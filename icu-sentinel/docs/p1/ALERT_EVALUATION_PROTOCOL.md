# Alert Evaluation Protocol — P1

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 7 — Research Evaluation  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Overview

This document describes the alert evaluation protocol for ICU Sentinel P1.
Alert evaluation measures how often a model generates alerts (positive
predictions), how many of those alerts are false alarms, and how burdensome
those false alerts are to clinical workflow.

Alert evaluation is critical because even a highly accurate model can be
clinically useless if it generates too many false alarms, alerting clinicians
too frequently and causing alert fatigue.

---

## 2. Alert Definition

An **alert** is generated when a model's predicted probability of sepsis
exceeds a configurable threshold (default: 0.5). An **alert episode** is a
consecutive sequence of hours where the model predicts sepsis, subject to
a cooldown period that merges nearby predictions into a single episode.

---

## 3. Alert Episode Definition

An alert episode is a contiguous period where the model predicts sepsis
(positive prediction). The episode starts at the first positive prediction
and ends at the last consecutive positive prediction, subject to:

### Cooldown Period

A configurable cooldown period (default: 6 hours) determines how close two
alerts can be before they're merged into a single episode. If the gap between
two positive predictions is less than or equal to the cooldown window, they
are part of the same episode.

### Minimum Duration

A configurable minimum duration (default: 1 hour) filters out very short
alert episodes that are likely spurious.

### Minimum Consecutive Predictions

A configurable minimum (default: 1) requires a minimum number of consecutive
positive predictions to constitute an alert episode.

---

## 4. Alert Burden Metrics

### False Alerts Per Patient-Day

**False alerts per patient-day** is the primary alert burden metric. It
measures how many false alert episodes occur per patient-day of monitoring.

Formula:

```
False Alerts per Patient-Day = (Number of False Alert Episodes) / (Patient-Days)
```

Where:
- **False alert episode**: An alert episode for a non-sepsis patient
- **Patient-day**: 24 hours of patient monitoring time (summed across all patients)

### Alerts Per Patient

**Alerts per patient** measures the average number of alert episodes per
patient over the evaluation period.

```
Alerts per Patient = (Total Alert Episodes) / (Number of Patients)
```

### Patients with False Alerts

**Patients with false alerts** measures the percentage of patients who
experience at least one false alert episode.

```
Patients with False Alerts = (Patients with ≥1 False Alert) / (Total Patients)
```

---

## 5. Alert Threshold Selection

Alert thresholds are selected using **validation data only**. The test set
is never used for threshold selection.

The default threshold is 0.5, but this can be tuned using validation data
to optimize for a specific metric (e.g., F1-score, sensitivity, specificity).

**Important**: Threshold selection must never use test data. Using test data
for threshold selection would invalidate the test set evaluation.

---

## 6. Alert Aggregation Policy

Alert episodes are aggregated as follows:

1. **Consecutive positive predictions** within the cooldown window form a single episode
2. **Episodes shorter than the minimum duration** are discarded
3. **Episodes with fewer than the minimum consecutive predictions** are discarded

This aggregation prevents alert fatigue by ensuring that brief, isolated
predictions don't generate separate alerts.

---

## 7. Output Format

Alert burden results are saved to `data/reports/classical_ml/alert_burden_report.json`:

```json
{
  "model_name": "logistic_regression",
  "model_version": "1.0.0",
  "split": "test",
  "n_patients": 1000,
  "n_patient_days": 5000.0,
  "n_alert_episodes": 150,
  "n_true_alert_episodes": 100,
  "n_false_alert_episodes": 50,
  "n_patients_with_false_alert": 40,
  "false_alerts_per_patient_day": 0.01,
  "alerts_per_patient": 0.15,
  "mean_alert_duration_hours": 2.5,
  "source": "synthetic"
}
```

---

## 8. Data Integrity

- All alert burden reports generated from synthetic fixtures are marked
  `"source": "synthetic"`
- No clinical validation claims are made from synthetic experiments
- Alert thresholds are tuned on validation data only, never on test data
- Alert episodes are aggregated using a documented causal policy
