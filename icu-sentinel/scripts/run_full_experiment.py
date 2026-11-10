# ruff: noqa: E402
"""Full eligible-cohort P2 experiment.

Trains Extra Trees, LightGBM, XGBoost plus Logistic Regression and Random
Forest on the complete eligible patient-level training split, tunes
hyperparameters on validation only, fits probability calibrators on
validation only, selects an alert threshold from validation only, and
evaluates once on the untouched test split.

Every artifact is written under ``--artifact-dir`` and never overwrites a
previous experiment directory.

Usage:
    python scripts/run_full_experiment.py --experiment-id full_dataset_v2
    python scripts/run_full_experiment.py --models extra_trees lightgbm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ml.p1_clinical.classical_ml.data_loader import load_from_manifest_and_features
from ml.p1_clinical.evaluation.alert_episodes import compute_alert_burden
from ml.p1_clinical.evaluation.bootstrap import (
    compute_bootstrap_confidence_intervals,
)
from ml.p1_clinical.evaluation.lead_time import compute_lead_time
from ml.p2_boosting.base import P2ModelResult
from ml.p2_boosting.calibration import (
    apply_probability_calibrator,
    fit_probability_calibrator,
)
from ml.p2_boosting.config import P2Config
from ml.p2_boosting.evaluation import evaluate_predictions
from ml.p2_boosting.pipelines import build_pipeline, hyperparameters_for
from ml.p2_boosting.serialization import save_p2_model

warnings.filterwarnings("ignore", category=UserWarning)

ALL_MODELS = ("extra_trees", "lightgbm", "xgboost", "logistic_regression", "random_forest")

# Hyperparameter candidates. Selection criterion (fixed before running):
# highest validation AUPRC, ties broken by validation AUROC. AUPRC leads
# because sepsis prediction is a rare-event, early-warning task where
# precision at useful recall matters more than global rank quality.
TUNING_GRID: dict[str, list[dict]] = {
    "extra_trees": [
        {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 10},
        {"n_estimators": 200, "max_depth": 30, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 20},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 50},
    ],
    "lightgbm": [
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
        {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 40},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 100},
        {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 50},
    ],
    "xgboost": [
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 1.0},
        {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 5.0},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 10.0},
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 5, "min_child_weight": 2.0},
    ],
    "logistic_regression": [
        {"C": 0.1},
        {"C": 1.0},
    ],
    "random_forest": [
        {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 10},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 50},
    ],
}


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def _sanitize_inplace(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN in place, keeping float32 (no full copy)."""
    values = frame.to_numpy(dtype=np.float32, copy=False)
    np.putmask(values, ~np.isfinite(values), np.nan)
    return frame


def _calibration_stats(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict:
    """Brier, ECE, MCE, calibration slope/intercept (15 equal-count bins)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    brier = float(np.mean((p - y) ** 2))

    order = np.argsort(p, kind="mergesort")
    y_sorted, p_sorted = y[order], p[order]
    bins = np.array_split(np.arange(len(p)), min(n_bins, max(1, len(p) // 50)))
    ece = 0.0
    mce = 0.0
    rows = []
    for idx in bins:
        if len(idx) == 0:
            continue
        conf = float(p_sorted[idx].mean())
        acc = float(y_sorted[idx].mean())
        gap = abs(acc - conf)
        ece += gap * len(idx) / len(p)
        mce = max(mce, gap)
        rows.append({"mean_predicted": conf, "observed": acc, "n": int(len(idx))})

    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(logit.reshape(-1, 1), y.astype(int))
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception:  # noqa: BLE001
        slope, intercept = float("nan"), float("nan")

    return {
        "brier": brier,
        "ece": float(ece),
        "mce": float(mce),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "n_bins": len(rows),
        "reliability": rows,
    }


def _threshold_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Vectorised sensitivity/specificity/precision/F-scores over thresholds."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    pos = y.sum()
    neg = len(y) - pos
    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    counts = np.arange(1, len(y) + 1)
    # number of rows with prob >= threshold
    idx = np.searchsorted(-p_sorted, -thresholds, side="right")
    rows = []
    for thr, i in zip(thresholds, idx):
        if i == 0:
            tp_i = fp_i = 0
        else:
            tp_i, fp_i = int(tp[i - 1]), int(fp[i - 1])
        fn_i = int(pos - tp_i)
        tn_i = int(neg - fp_i)
        sens = tp_i / pos if pos else 0.0
        spec = tn_i / neg if neg else 0.0
        prec = tp_i / (tp_i + fp_i) if (tp_i + fp_i) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        f2 = 5 * prec * sens / (4 * prec + sens) if (4 * prec + sens) else 0.0
        rows.append(
            {
                "threshold": float(thr),
                "sensitivity": sens,
                "specificity": spec,
                "precision": prec,
                "f1": f1,
                "f2": f2,
                "tp": tp_i,
                "fp": fp_i,
                "fn": fn_i,
                "tn": tn_i,
            }
        )
    return pd.DataFrame(rows)


def _burden_summary(
    patient_id: np.ndarray,
    hour: np.ndarray,
    label: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    cooldown: int = 6,
) -> dict:
    """Fast alert-episode burden with the repo's patient-day denominator."""
    frame = pd.DataFrame(
        {
            "patient_id": patient_id,
            "hour": np.asarray(hour, dtype=int),
            "label": np.asarray(label, dtype=int),
            "prob": np.asarray(prob, dtype=float),
        }
    )
    patient_days = float(
        frame.groupby("patient_id")["hour"]
        .agg(lambda s: (s.max() - s.min() + 1) / 24.0)
        .sum()
    )
    septic_patients = set(frame.loc[frame["label"] == 1, "patient_id"].unique())
    alerted = frame[frame["prob"] >= threshold]
    episodes = 0
    false_episodes = 0
    patients_alerted = 0
    for pid, group in alerted.groupby("patient_id"):
        patients_alerted += 1
        hours = np.sort(group["hour"].to_numpy())
        n_ep = 1 + int(np.sum(np.diff(hours) > cooldown))
        episodes += n_ep
        if pid not in septic_patients:
            false_episodes += n_ep
    return {
        "threshold": float(threshold),
        "n_patients": int(frame["patient_id"].nunique()),
        "n_patient_days": patient_days,
        "n_alert_episodes": episodes,
        "n_false_alert_episodes": false_episodes,
        "n_patients_alerted": patients_alerted,
        "alerts_per_patient": episodes / max(1, frame["patient_id"].nunique()),
        "false_alerts_per_patient_day": (
            false_episodes / patient_days if patient_days else float("nan")
        ),
        "pct_patients_alerted": 100.0
        * patients_alerted
        / max(1, frame["patient_id"].nunique()),
    }


def _cross_fitted_brier(
    y: np.ndarray,
    p: np.ndarray,
    patient_ids: np.ndarray,
    method: str,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Patient-grouped cross-fitted Brier score inside the validation split."""
    unique = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    folds = np.array_split(unique, n_splits)
    preds = np.full(len(y), np.nan)
    for fold in folds:
        mask = np.isin(patient_ids, fold)
        if mask.all() or mask.sum() == 0:
            continue
        calibrator = fit_probability_calibrator(
            y[~mask], p[~mask], method=method, partition="validation"
        )
        preds[mask] = apply_probability_calibrator(calibrator, p[mask])
    valid = np.isfinite(preds)
    if valid.sum() == 0:
        return {"brier": float("nan"), "n": 0}
    from sklearn.metrics import brier_score_loss

    return {
        "brier": float(brier_score_loss(y[valid], preds[valid])),
        "n": int(valid.sum()),
    }


# --------------------------------------------------------------------------
# Model builders
# --------------------------------------------------------------------------
def _apply_config(name: str, params: dict, base: P2Config) -> tuple[P2Config, dict]:
    config = deepcopy(base)
    if name == "extra_trees":
        for key, value in params.items():
            setattr(config.extra_trees, key, value)
    elif name == "lightgbm":
        for key, value in params.items():
            setattr(config.lightgbm, key, value)
    elif name == "xgboost":
        for key, value in params.items():
            setattr(config.xgboost, key, value)
    return config, params


def _make_pipeline(name: str, params: dict, config: P2Config, y_train: pd.Series):
    if name in {"extra_trees", "lightgbm", "xgboost"}:
        return build_pipeline(name, config, y_train)
    if name == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(params.get("C", 1.0)),
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=int(params.get("n_estimators", 200)),
                        max_depth=params.get("max_depth", 20),
                        min_samples_leaf=int(params.get("min_samples_leaf", 10)),
                        class_weight="balanced",
                        n_jobs=config.extra_trees.n_jobs,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model '{name}'")


def _feature_importance(pipeline) -> pd.Series | None:
    clf = pipeline.named_steps.get("classifier")
    if clf is None:
        return None
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_)
    if hasattr(clf, "coef_"):
        return pd.Series(np.abs(clf.coef_).ravel())
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=_REPO_ROOT / "data" / "features_full"
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=_REPO_ROOT / "data" / "splits_full_v2"
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=_REPO_ROOT.parent
        / "artifacts"
        / "physionet2019"
        / "full_dataset_v2",
    )
    parser.add_argument("--experiment-id", type=str, default="full_dataset_v2")
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifact_dir = args.artifact_root
    models_dir = artifact_dir / "models"
    eval_dir = artifact_dir / "evaluation"
    plots_dir = artifact_dir / "plots"
    reports_dir = artifact_dir / "reports"
    for path in (models_dir, eval_dir, plots_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print("Loading split-aware dataset (all splits) ...", flush=True)
    dataset = load_from_manifest_and_features(
        manifest_path=args.splits_dir / "split_manifest.json",
        processed_dir=args.features_dir / "patients",
        schema_path=args.features_dir / "feature_schema.json",
        source="real",
    )
    train, validation, test = dataset.train, dataset.validation, dataset.test
    for split in (train, validation, test):
        if split is None or len(split.X) == 0:
            raise RuntimeError("Missing a required split")

    # Drop features that are entirely missing in the training split only.
    # Rationale: a column with no observed training value carries zero
    # information and would be imputed to a constant. This uses training
    # data only, so no validation/test information leaks into the choice.
    feature_names = list(dataset.feature_names)
    train_null = train.X[feature_names].isna().mean()
    informative = [f for f in feature_names if train_null[f] < 1.0]
    dropped = [f for f in feature_names if f not in informative]
    feature_names = informative

    for split in (train, validation, test):
        split.X = _sanitize_inplace(split.X[feature_names])
    print(
        f"Features: {len(feature_names)} (dropped {len(dropped)} all-missing-in-train)",
        flush=True,
    )

    X_train, y_train = train.X, train.y.to_numpy().astype(int)
    X_val, y_val = validation.X, validation.y.to_numpy().astype(int)
    X_test, y_test = test.X, test.y.to_numpy().astype(int)
    val_patients = validation.patient_ids.to_numpy()
    test_patients = test.patient_ids.to_numpy()
    print(
        f"Rows: train={len(X_train)} val={len(X_val)} test={len(X_test)}; "
        f"patients: {train.n_patients}/{validation.n_patients}/{test.n_patients}",
        flush=True,
    )

    base_config = P2Config(random_seed=args.seed)
    base_config.extra_trees.n_jobs = args.n_jobs
    base_config.xgboost.n_jobs = args.n_jobs

    tuning_report: dict[str, list] = {}
    results: dict[str, dict] = {}

    for name in args.models:
        model_started = time.perf_counter()
        candidates = [{}] if args.no_tune else TUNING_GRID[name]
        trials = []
        best = None
        for params in candidates:
            config, params = _apply_config(name, params, base_config)
            t0 = time.perf_counter()
            pipeline = _make_pipeline(name, params, config, y_train)
            pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - t0
            val_prob = pipeline.predict_proba(X_val)[:, 1]
            from sklearn.metrics import average_precision_score, roc_auc_score

            auroc = float(roc_auc_score(y_val, val_prob))
            auprc = float(average_precision_score(y_val, val_prob))
            trial = {
                "params": params,
                "validation_auroc": auroc,
                "validation_auprc": auprc,
                "fit_seconds": round(fit_seconds, 2),
            }
            trials.append(trial)
            print(
                f"[{name}] {params} -> val AUROC={auroc:.4f} AUPRC={auprc:.4f} "
                f"({fit_seconds:.1f}s)",
                flush=True,
            )
            if best is None or (auprc, auroc) > (
                best["validation_auprc"],
                best["validation_auroc"],
            ):
                best = trial
                best_pipeline = pipeline
                best_config = config

        tuning_report[name] = trials
        del pipeline  # free the last (possibly non-selected) candidate
        pipeline = best_pipeline
        config = best_config
        params = best["params"]
        print(
            f"[{name}] selected {params} "
            f"(val AUPRC={best['validation_auprc']:.4f})",
            flush=True,
        )

        train_prob = pipeline.predict_proba(X_train)[:, 1]
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        test_prob = pipeline.predict_proba(X_test)[:, 1]

        # ---- Calibration: method chosen by patient-grouped cross-fitted
        # Brier inside validation only; refit on the whole validation split.
        cv_scores = {}
        for method in ("sigmoid", "isotonic"):
            cv_scores[method] = _cross_fitted_brier(
                y_val, val_prob, val_patients, method, seed=args.seed
            )
        calib_method = min(
            cv_scores, key=lambda m: cv_scores[m]["brier"]
        )
        calibrator = fit_probability_calibrator(
            y_val, val_prob, method=calib_method, partition="validation"
        )
        val_cal = apply_probability_calibrator(calibrator, val_prob)
        test_cal = apply_probability_calibrator(calibrator, test_prob)

        # ---- Threshold: maximise validation F2 (recall-weighted; missed
        # sepsis is costlier than a false alert), on validation only.
        curve = _threshold_curve(y_val, val_cal)
        best_row = curve.loc[curve["f2"].idxmax()]
        threshold = float(best_row["threshold"])
        f1_row = curve.loc[curve["f1"].idxmax()]

        val_metrics = evaluate_predictions(
            y_val, val_cal, threshold=threshold, model_name=name, split="validation"
        )
        test_metrics = evaluate_predictions(
            y_test, test_cal, threshold=threshold, model_name=name, split="test"
        )

        cal_report = {
            "selected_method": calib_method,
            "method_selection": "patient-grouped 5-fold cross-fitted Brier on validation",
            "cross_fitted_brier": cv_scores,
            "uncalibrated_validation": _calibration_stats(y_val, val_prob),
            "uncalibrated_test": _calibration_stats(y_test, test_prob),
            "calibrated_validation": _calibration_stats(y_val, val_cal),
            "calibrated_test": _calibration_stats(y_test, test_cal),
        }

        # ---- Alert / lead-time metrics on test (calibrated probabilities).
        pred_frame = pd.DataFrame(
            {
                "patient_id": test_patients,
                "hour": test.hours.to_numpy(),
                "true_label": y_test,
                "predicted_probability": test_cal,
            }
        )
        if test.source_datasets is not None:
            pred_frame["source_dataset"] = test.source_datasets.to_numpy()
        burden = compute_alert_burden(
            pred_frame,
            model_name=name,
            model_version="1.0.0",
            split="test",
            threshold=threshold,
            cooldown_hours=6,
            source="real",
        )
        lead = compute_lead_time(
            pred_frame,
            model_name=name,
            model_version="1.0.0",
            split="test",
            threshold=threshold,
            source="real",
        )
        burden_dict = burden.to_dict()
        burden_dict.pop("episodes", None)
        lead_dict = lead.to_dict()
        lead_dict.pop("patient_details", None)

        operating_points = {
            f"{row['threshold']:.2f}": {
                "sensitivity": round(float(row["sensitivity"]), 4),
                "specificity": round(float(row["specificity"]), 4),
                "precision": round(float(row["precision"]), 4),
                "f1": round(float(row["f1"]), 4),
                "f2": round(float(row["f2"]), 4),
            }
            for _, row in curve[curve["threshold"].isin(
                [round(x, 2) for x in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
            )].iterrows()
        }
        burden_points = {
            str(sel): _burden_summary(
                test_patients,
                test.hours.to_numpy(),
                y_test,
                test_cal,
                sel,
            )
            for sel in sorted({threshold, float(f1_row["threshold"]), 0.5})
        }

        # ---- Bootstrap confidence intervals on the test split.
        boot_frame = pred_frame.rename(columns={"true_label": "true_label"})
        boot_frame["predicted_class"] = (test_cal >= threshold).astype(int)
        boot = compute_bootstrap_confidence_intervals(
            predictions_df=boot_frame[
                ["patient_id", "true_label", "predicted_probability", "predicted_class"]
            ],
            model_name=name,
            model_version="1.0.0",
            split="test",
            metric_names=[
                "auroc",
                "auprc",
                "sensitivity",
                "specificity",
                "precision",
                "f1",
                "brier",
            ],
            n_iterations=args.bootstrap_iterations,
            confidence_level=0.95,
            random_seed=args.seed,
            source="real",
        )

        # ---- Persist the model + calibration artifacts.
        model_result = P2ModelResult(
            model_name=name,
            model=pipeline,
            preprocessing_pipeline=pipeline.named_steps.get("imputer"),
            selected_features=feature_names,
            hyperparameters=params,
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            threshold=threshold,
            random_seed=args.seed,
            training_timestamp=datetime.now(UTC).isoformat(),
            n_patients_train=train.n_patients,
            n_rows_train=len(X_train),
            n_positive_train=int(y_train.sum()),
            n_negative_train=int((y_train == 0).sum()),
            class_weight="balanced",
            calibration_method=calib_method,
            calibrator=calibrator,
            source="real",
        )
        save_p2_model(model_result, models_dir / name, overwrite=args.overwrite)
        calibrator.save(models_dir / name / "calibrator.joblib", overwrite=True)

        # Augment metadata with the full provenance contract.
        meta_path = models_dir / name / f"{name}_1.0.0_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta.update(
            {
                "experiment_id": args.experiment_id,
                "dataset_version": "physionet2019-1.0.0-training-open",
                "feature_schema_version": "feature_schema.json@full_v2",
                "preprocessing_version": "median-imputer / native-NaN v1",
                "calibration_method": calib_method,
                "selected_threshold": threshold,
                "threshold_selection": {
                    "split": "validation",
                    "criterion": "maximise F2 (recall-weighted); test set never used",
                    "grid_step": 0.01,
                },
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "calibration_metrics": {
                    k: v for k, v in cal_report.items() if k != "method_selection"
                },
                "alert_metrics": burden_dict,
                "lead_time_metrics": lead_dict,
                "n_patients_validation": validation.n_patients,
                "n_rows_validation": len(X_val),
                "n_patients_test": test.n_patients,
                "n_rows_test": len(X_test),
                "fit_seconds_selected_config": best["fit_seconds"],
                "tuning_candidates": trials,
                "dropped_all_missing_features": dropped,
                "artifact_version": "1.0.0",
                "preprocessing_config": {
                    "imputer": (
                        "median"
                        if name in {"extra_trees", "logistic_regression", "random_forest"}
                        else None
                    ),
                    "scaler": "standard" if name == "logistic_regression" else None,
                    "native_nan_handling": name in {"lightgbm", "xgboost"},
                    "fit_on": "training split only",
                },
                "training_duration_seconds": round(
                    time.perf_counter() - model_started, 2
                ),
            }
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        results[name] = {
            "hyperparameters": params,
            "calibration_method": calib_method,
            "threshold": threshold,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "calibration": cal_report,
            "bootstrap": boot.to_dict(),
            "alert_metrics": burden_dict,
            "lead_time_metrics": lead_dict,
            "operating_points": operating_points,
            "burden_at_operating_points": burden_points,
            "tuning_trials": trials,
            "test_probabilities_summary": {
                "mean": float(np.mean(test_cal)),
                "p50": float(np.quantile(test_cal, 0.5)),
                "p99": float(np.quantile(test_cal, 0.99)),
                "max": float(np.max(test_cal)),
            },
        }

        # Persist curve data for plotting / audit.
        fpr, tpr, _ = roc_curve(y_test, test_cal)
        precision, recall, _ = precision_recall_curve(y_test, test_cal)
        np.savez(
            eval_dir / f"curves_{name}.npz",
            fpr=fpr,
            tpr=tpr,
            precision=precision,
            recall=recall,
            calibration_bins=np.array(
                [b["mean_predicted"] for b in cal_report["calibrated_test"]["reliability"]]
            ),
            calibration_obs=np.array(
                [b["observed"] for b in cal_report["calibrated_test"]["reliability"]]
            ),
        )
        importance = _feature_importance(pipeline)
        if importance is not None:
            imp = pd.Series(importance.to_numpy(), index=feature_names).sort_values(
                ascending=False
            )
            imp.head(50).to_csv(models_dir / name / "feature_importance.csv")
        curve.to_csv(models_dir / name / "validation_threshold_curve.csv", index=False)

        print(
            f"[{name}] test AUROC={test_metrics['metrics']['auroc']:.4f} "
            f"AUPRC={test_metrics['metrics']['auprc']:.4f} thr={threshold:.2f} "
            f"({time.perf_counter() - model_started:.0f}s total)",
            flush=True,
        )

    with open(eval_dir / "model_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(eval_dir / "tuning_results.json", "w") as f:
        json.dump(tuning_report, f, indent=2, default=str)
    with open(eval_dir / "validation_metrics.json", "w") as f:
        json.dump(
            {k: v["validation_metrics"] for k, v in results.items()}, f, indent=2
        )
    with open(eval_dir / "test_metrics.json", "w") as f:
        json.dump({k: v["test_metrics"] for k, v in results.items()}, f, indent=2)
    with open(eval_dir / "calibration_metrics.json", "w") as f:
        json.dump({k: v["calibration"] for k, v in results.items()}, f, indent=2)
    with open(eval_dir / "bootstrap_confidence_intervals.json", "w") as f:
        json.dump({k: v["bootstrap"] for k, v in results.items()}, f, indent=2)
    with open(eval_dir / "alert_metrics.json", "w") as f:
        json.dump(
            {
                k: {
                    "alert_burden": v["alert_metrics"],
                    "lead_time": v["lead_time_metrics"],
                    "operating_points": v["operating_points"],
                    "burden_at_operating_points": v["burden_at_operating_points"],
                    "threshold": v["threshold"],
                }
                for k, v in results.items()
            },
            f,
            indent=2,
            default=str,
        )

    summary = {
        "experiment_id": args.experiment_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "models": list(results),
        "n_features": len(feature_names),
        "dropped_all_missing_features": dropped,
        "splits": {
            "train": {
                "n_patients": train.n_patients,
                "n_rows": len(X_train),
                "n_positive": int(y_train.sum()),
            },
            "validation": {
                "n_patients": validation.n_patients,
                "n_rows": len(X_val),
                "n_positive": int(y_val.sum()),
            },
            "test": {
                "n_patients": test.n_patients,
                "n_rows": len(X_test),
                "n_positive": int(y_test.sum()),
            },
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "random_seed": args.seed,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    with open(eval_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
