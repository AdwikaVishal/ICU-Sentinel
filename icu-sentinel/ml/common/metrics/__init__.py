# Common metrics utilities shared between P1 and P2/P3.

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def compute_binary_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute standard binary classification metrics.

    Returns dict with AUROC, AUPRC, sensitivity, specificity,
    precision, F1, Brier score, and confusion-matrix counts.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else float("nan")

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = float("nan")

    try:
        brier = brier_score_loss(y_true, y_prob)
    except ValueError:
        brier = float("nan")

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "brier": float(brier),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
