#!/usr/bin/env python3
"""Generate P1 clinical baseline evaluation artifacts.

Runs SIRS, qSOFA, and NEWS-style baselines on the processed patient data
and generates evaluation metrics and comparison artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ICU_SENTINEL_ROOT = _REPO_ROOT / "icu-sentinel"
if str(_ICU_SENTINEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ICU_SENTINEL_ROOT))

from ml.p1_clinical.baselines import (
    compute_sirs_for_patient,
    compute_qsofa,
    compute_news2_style,
)
from ml.p1_clinical.baselines.clinical_score_config import ClinicalScoreConfig
from ml.p1_clinical.preprocessing.patient_splits import (
    SplitManifest,
    iter_processed_patients_from_directory,
)
from ml.p1_clinical.evaluation.classification_metrics import (
    compute_binary_metrics,
)


def run_baseline_evaluation(
    processed_dir: Path,
    splits_dir: Path,
    artifact_dir: Path,
    reports_dir: Path,
) -> dict:
    """Run clinical baselines on test split and generate artifacts."""

    # Load split manifest
    manifest = SplitManifest.load(splits_dir / "split_manifest.json")
    test_patient_ids = set(manifest.patient_ids_by_split("test"))

    # Load test patients
    test_patients = [
        p for p in iter_processed_patients_from_directory(processed_dir)
        if p.patient_id in test_patient_ids
    ]

    print(f"Loaded {len(test_patients)} test patients")

    config = ClinicalScoreConfig()

    # Compute baselines for each patient
    all_sirs_results = []
    all_qsofa_results = []
    all_news_results = []

    for patient in test_patients:
        all_sirs_results.extend(compute_sirs_for_patient(patient, config))
        all_qsofa_results.extend(compute_qsofa(patient, config))
        all_news_results.extend(compute_news2_style(patient, config))

    print(f"SIRS results: {len(all_sirs_results)} rows")
    print(f"qSOFA results: {len(all_qsofa_results)} rows")
    print(f"NEWS results: {len(all_news_results)} rows")

    # Extract predictions and true labels for eligible rows
    def extract_eligible_predictions(results, score_name):
        """Extract predictions for complete scores only."""
        y_true = []
        y_pred = []
        y_prob = []
        for r in results:
            if r.score.complete and r.score.is_positive is not None:
                y_true.append(r.sepsis_label)
                y_pred.append(int(r.score.is_positive))
                # Use score_value as pseudo-probability (normalized)
                if r.score.score_value is not None:
                    max_score = r.score.components_required
                    prob = r.score.score_value / max_score if max_score > 0 else 0.0
                else:
                    prob = 0.0
                y_prob.append(prob)
        return np.array(y_true), np.array(y_pred), np.array(y_prob)

    # SIRS evaluation
    sirs_y_true, sirs_y_pred, sirs_y_prob = extract_eligible_predictions(all_sirs_results, "sirs")
    sirs_metrics = compute_binary_metrics(sirs_y_true, sirs_y_pred, sirs_y_prob) if len(sirs_y_true) > 0 else None

    # qSOFA evaluation
    qsofa_y_true, qsofa_y_pred, qsofa_y_prob = extract_eligible_predictions(all_qsofa_results, "qsofa")
    qsofa_metrics = compute_binary_metrics(qsofa_y_true, qsofa_y_pred, qsofa_y_prob) if len(qsofa_y_true) > 0 else None

    # NEWS evaluation
    news_y_true, news_y_pred, news_y_prob = extract_eligible_predictions(all_news_results, "news2_style")
    news_metrics = compute_binary_metrics(news_y_true, news_y_pred, news_y_prob) if len(news_y_true) > 0 else None

    # Save results
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    def save_results(results, name):
        data = []
        for r in results:
            data.append({
                "patient_id": r.patient_id,
                "source_dataset": r.source_dataset,
                "hour": r.hour,
                "sepsis_label": r.sepsis_label,
                "score_value": r.score.score_value,
                "threshold": r.score.threshold,
                "is_positive": r.score.is_positive,
                "complete": r.score.complete,
                "components_available": r.score.components_available,
                "components_required": r.score.components_required,
            })
        df = pd.DataFrame(data)
        df.to_csv(artifact_dir / f"{name}_results.csv", index=False)
        return df

    sirs_df = save_results(all_sirs_results, "sirs")
    qsofa_df = save_results(all_qsofa_results, "qsofa")
    news_df = save_results(all_news_results, "news")

    # Save metrics
    def metrics_to_dict(m: dict | None, n_eligible: int) -> dict:
        if m is None:
            return {"available": False}
        return {
            "available": True,
            "accuracy": m.get("accuracy", float("nan")),
            "precision": m.get("precision", float("nan")),
            "recall": m.get("recall", float("nan")),
            "specificity": m.get("specificity", float("nan")),
            "f1": m.get("f1", float("nan")),
            "auroc": m.get("auroc", float("nan")),
            "auprc": m.get("auprc", float("nan")),
            "brier": m.get("brier", float("nan")),
            "tp": m.get("tp", 0),
            "tn": m.get("tn", 0),
            "fp": m.get("fp", 0),
            "fn": m.get("fn", 0),
            "n_eligible": n_eligible,
        }

    metrics_dict = {
        "sirs": metrics_to_dict(sirs_metrics, len(sirs_y_true)),
        "qsofa": metrics_to_dict(qsofa_metrics, len(qsofa_y_true)),
        "news": metrics_to_dict(news_metrics, len(news_y_true)),
    }

    with open(artifact_dir / "clinical_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save comparison CSV
    comparison_rows = []
    for name, m in metrics_dict.items():
        if m["available"]:
            comparison_rows.append({
                "model": name,
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "sensitivity": m["recall"],
                "specificity": m["specificity"],
                "precision": m["precision"],
                "f1": m["f1"],
                "brier": m["brier"],
                "n_eligible": m["n_eligible"],
            })
    pd.DataFrame(comparison_rows).to_csv(artifact_dir / "clinical_baseline_comparison.csv", index=False)

    # Save confusion matrices
    confusion_matrices = {}
    for name, m in metrics_dict.items():
        if m["available"]:
            confusion_matrices[name] = {
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
    with open(artifact_dir / "clinical_confusion_matrices.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    # Generate report
    with open(reports_dir / "P1_CLINICAL_EVALUATION.md", "w") as f:
        f.write("# P1 Clinical Baseline Evaluation\n\n")
        f.write("## Test Set Results\n\n")
        for name, m in metrics_dict.items():
            f.write(f"### {name.upper()}\n\n")
            if m["available"]:
                f.write(f"- **Eligible predictions**: {m['n_eligible']}\n")
                f.write(f"- **AUROC**: {m['auroc']:.4f}\n")
                f.write(f"- **AUPRC**: {m['auprc']:.4f}\n")
                f.write(f"- **Sensitivity (Recall)**: {m['recall']:.4f}\n")
                f.write(f"- **Specificity**: {m['specificity']:.4f}\n")
                f.write(f"- **Precision**: {m['precision']:.4f}\n")
                f.write(f"- **F1-score**: {m['f1']:.4f}\n")
                f.write(f"- **Brier score**: {m['brier']:.4f}\n")
                f.write(f"- **Confusion matrix**: TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}\n\n")
            else:
                f.write("- No eligible predictions (no complete scores)\n\n")

    print("Clinical baseline evaluation complete!")
    return metrics_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=_ICU_SENTINEL_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=_ICU_SENTINEL_ROOT / "data" / "splits",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "p1",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_REPO_ROOT / "reports",
    )
    args = parser.parse_args()

    run_baseline_evaluation(
        args.processed_dir,
        args.splits_dir,
        args.artifact_dir,
        args.reports_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())