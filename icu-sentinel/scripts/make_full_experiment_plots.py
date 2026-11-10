# ruff: noqa: E402
"""Render plots and comparison tables for a full-dataset experiment run.

Reads the artifacts produced by ``scripts/run_full_experiment.py`` and writes
ROC, precision-recall, calibration, confusion-matrix and feature-importance
figures plus markdown comparison tables.

Usage:
    python scripts/make_full_experiment_plots.py --experiment-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_LABELS = {
    "extra_trees": "Extra Trees",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=REPO_ROOT.parent
        / "artifacts"
        / "physionet2019"
        / "full_dataset_v2",
    )
    args = parser.parse_args()
    exp = args.experiment_dir
    eval_dir, plots = exp / "evaluation", exp / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    results = json.loads((eval_dir / "model_results.json").read_text())

    # ---------------------------------------------------------------- ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in results:
        data = np.load(eval_dir / f"curves_{name}.npz")
        auroc = results[name]["test_metrics"]["metrics"]["auroc"]
        ax.plot(data["fpr"], data["tpr"], label=f"{MODEL_LABELS[name]} (AUC {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — independent test split")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "roc_curves.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------ Precision-recall
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in results:
        data = np.load(eval_dir / f"curves_{name}.npz")
        auprc = results[name]["test_metrics"]["metrics"]["auprc"]
        ax.plot(data["recall"], data["precision"], label=f"{MODEL_LABELS[name]} (AP {auprc:.3f})")
    prevalence = (
        results[next(iter(results))]["test_metrics"]["n_positive"]
        / results[next(iter(results))]["test_metrics"]["n_rows"]
    )
    ax.axhline(prevalence, color="k", linestyle="--", linewidth=0.8,
               label=f"Prevalence {prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves — independent test split")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "precision_recall_curves.png", dpi=160)
    plt.close(fig)

    # ---------------------------------------------------------- Calibration
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in results:
        data = np.load(eval_dir / f"curves_{name}.npz")
        ax.plot(
            data["calibration_bins"],
            data["calibration_obs"],
            marker="o",
            markersize=3,
            label=MODEL_LABELS[name],
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Reliability diagram — test split (15 equal-count bins)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "calibration_curves.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------ Confusion matrix
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = results[name]["test_metrics"]["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        thr = results[name]["threshold"]
        ax.set_title(f"{MODEL_LABELS[name]}\nthr={thr:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    # --------------------------------------------------- Feature importance
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 5))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        path = exp / "models" / name / "feature_importance.csv"
        if not path.exists():
            ax.axis("off")
            continue
        imp = pd.read_csv(path, index_col=0).iloc[:, 0]
        top = imp.head(15)[::-1]
        ax.barh(range(len(top)), top.to_numpy())
        ax.set_yticks(range(len(top)), [str(i) for i in top.index], fontsize=7)
        ax.set_title(MODEL_LABELS[name], fontsize=9)
        ax.set_xlabel("Importance", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "feature_importance.png", dpi=160)
    plt.close(fig)

    # --------------------------------------------------------- Summary table
    rows = []
    for name, res in results.items():
        m = res["test_metrics"]["metrics"]
        cal = res["calibration"]["calibrated_test"]
        ci = {x["metric"]: x for x in res["bootstrap"]["metrics"]}
        rows.append(
            {
                "model": MODEL_LABELS[name],
                "threshold": res["threshold"],
                "auroc": m["auroc"],
                "auroc_ci": f"[{ci['auroc']['ci_lower']:.3f}, {ci['auroc']['ci_upper']:.3f}]",
                "auprc": m["auprc"],
                "auprc_ci": f"[{ci['auprc']['ci_lower']:.3f}, {ci['auprc']['ci_upper']:.3f}]",
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "precision": m["precision"],
                "f1": m["f1"],
                "brier": cal["brier"],
                "ece": cal["ece"],
                "calibration_method": res["calibration"]["selected_method"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(plots.parent / "evaluation" / "test_metrics_table.csv", index=False)
    print(table.to_string(index=False))
    print(f"Plots written to {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
