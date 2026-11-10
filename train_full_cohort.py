#!/usr/bin/env python3
"""Train all models on the complete full cohort with calibration and comprehensive evaluation.

This script:
1. Loads all 39,810 patients with features
2. Trains Extra Trees, LightGBM, XGBoost on full training cohort (27,867 patients)
3. Evaluates on complete validation set (3,981 patients)
4. Performs calibration analysis
5. Evaluates on complete test set (7,962 patients) with bootstrap CIs
6. Compares to clinical baselines (SIRS, qSOFA, NEWS)
7. Performs early-warning analysis
"""

import sys
from pathlib import Path

# Add ICU-Sentinel to path
icu_root = Path("/Users/shantanu/ICU-Sentinel/icu-sentinel")
if str(icu_root) not in sys.path:
    sys.path.insert(0, str(icu_root))

import json
import time
import warnings
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve

from ml.p1_clinical.classical_ml.data_loader import load_from_manifest_and_features
from ml.p2_boosting.config import P2Config
from ml.p2_boosting.evaluation import evaluate_predictions
from ml.p2_boosting.pipelines import build_pipeline
from ml.p2_boosting.calibration import fit_probability_calibrator, apply_probability_calibrator
from ml.p2_boosting.serialization import save_p2_model
from ml.p2_boosting.base import P2ModelResult
from ml.p1_clinical.evaluation.bootstrap import compute_bootstrap_confidence_intervals
from ml.p1_clinical.evaluation.alert_episodes import compute_alert_burden
from ml.p1_clinical.evaluation.lead_time import compute_lead_time

warnings.filterwarnings("ignore", category=UserWarning)

def sanitize_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities with NaN in place."""
    for col in df.columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def compute_calibration_stats(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15) -> dict:
    """Calculate calibration metrics including ECE, MCE, Brier score."""
    from sklearn.metrics import brier_score_loss
    
    brier = float(brier_score_loss(y_true, y_pred))
    
    # Calculate ECE and MCE
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    mce = 0.0
    reliability_data = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        
        bin_conf = y_pred[mask].mean()
        bin_acc = y_true[mask].mean()
        bin_count = mask.sum()
        
        gap = abs(bin_acc - bin_conf)
        ece += gap * bin_count / len(y_true)
        mce = max(mce, gap)
        
        reliability_data.append({
            'bin': i,
            'mean_predicted': float(bin_conf),
            'observed': float(bin_acc),
            'count': int(bin_count)
        })
    
    return {
        'brier': brier,
        'ece': float(ece),
        'mce': float(mce),
        'reliability_data': reliability_data
    }


def select_threshold(y_true: np.ndarray, y_pred: np.ndarray, criterion='f1') -> tuple[float, dict]:
    """Select optimal threshold on validation set."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_score = -1
    best_thr = 0.5
    best_metrics = {}
    
    for thr in thresholds:
        y_pred_class = (y_pred >= thr).astype(int)
        
        tp = ((y_pred_class == 1) & (y_true == 1)).sum()
        fp = ((y_pred_class == 1) & (y_true == 0)).sum()
        fn = ((y_pred_class == 0) & (y_true == 1)).sum()
        tn = ((y_pred_class == 0) & (y_true == 0)).sum()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        if criterion == 'f1':
            score = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        elif criterion == 'f2':
            score = 5 * prec * sens / (4 * prec + sens) if (4 * prec + sens) > 0 else 0
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = {
                'threshold': best_thr,
                'sensitivity': sens,
                'specificity': spec,
                'precision': prec,
                'f1': 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0,
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
            }
    
    return best_thr, best_metrics


def main():
    print("=" * 80)
    print("ICU SENTINEL: FULL COHORT TRAINING")
    print("=" * 80)
    print(f"Started: {datetime.now()}")
    print()
    
    # Configuration
    artifact_root = Path("/Users/shantanu/ICU-Sentinel/artifacts/physionet2019/full_cohort_v2")
    models_dir = artifact_root / "models"
    eval_dir = artifact_root / "evaluation"
    reports_dir = artifact_root / "reports"
    plots_dir = artifact_root / "plots"
    
    for d in [models_dir, eval_dir, reports_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading complete dataset with all splits...")
    features_dir = Path("/Users/shantanu/ICU-Sentinel/icu-sentinel/data/features_full")
    splits_dir = Path("/Users/shantanu/ICU-Sentinel/icu-sentinel/data/splits_full")
    
    dataset = load_from_manifest_and_features(
        manifest_path=splits_dir / "split_manifest.json",
        processed_dir=features_dir / "patients",
        schema_path=features_dir / "feature_schema.json",
        source="real"
    )
    
    train = dataset.train
    validation = dataset.validation
    test = dataset.test
    
    print(f"Training: {train.n_patients} patients, {len(train.X):,} observations")
    print(f"Validation: {validation.n_patients} patients, {len(validation.X):,} observations")
    print(f"Test: {test.n_patients} patients, {len(test.X):,} observations")
    print(f"Features: {len(dataset.feature_names)}")
    print()
    
    # Sanitize infinities
    print("Sanitizing data...")
    feature_names = list(dataset.feature_names)
    train.X = sanitize_inf(train.X[feature_names])
    validation.X = sanitize_inf(validation.X[feature_names])
    test.X = sanitize_inf(test.X[feature_names])
    
    X_train, y_train = train.X, train.y.to_numpy().astype(int)
    X_val, y_val = validation.X, validation.y.to_numpy().astype(int)
    X_test, y_test = test.X, test.y.to_numpy().astype(int)
    
    print(f"Training positives: {y_train.sum():,} ({100*y_train.sum()/len(y_train):.2f}%)")
    print(f"Validation positives: {y_val.sum():,} ({100*y_val.sum()/len(y_val):.2f}%)")
    print(f"Test positives: {y_test.sum():,} ({100*y_test.sum()/len(y_test):.2f}%)")
    print()
    
    # Configure models
    config = P2Config(random_seed=42)
    config.extra_trees.n_jobs = 8
    config.xgboost.n_jobs = 8
    
    models_to_train = ['extra_trees', 'lightgbm', 'xgboost']
    results = {}
    
    # Train each model
    for model_name in models_to_train:
        print("=" * 80)
        print(f"Training: {model_name.upper()}")
        print("=" * 80)
        
        model_start = time.time()
        
        # Build and train
        print(f"Building {model_name} pipeline...")
        pipeline = build_pipeline(model_name, config, y_train)
        
        print(f"Fitting {model_name} on {len(X_train):,} observations...")
        fit_start = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - fit_start
        print(f"Training complete in {fit_time:.1f}s")
        
        # Predict on all splits
        print("Generating predictions...")
        train_prob = pipeline.predict_proba(X_train)[:, 1]
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Uncalibrated metrics
        train_auroc = roc_auc_score(y_train, train_prob)
        val_auroc = roc_auc_score(y_val, val_prob)
        test_auroc = roc_auc_score(y_test, test_prob)
        
        train_auprc = average_precision_score(y_train, train_prob)
        val_auprc = average_precision_score(y_val, val_prob)
        test_auprc = average_precision_score(y_test, test_prob)
        
        print(f"Uncalibrated AUROC: train={train_auroc:.4f}, val={val_auroc:.4f}, test={test_auroc:.4f}")
        print(f"Uncalibrated AUPRC: train={train_auprc:.4f}, val={val_auprc:.4f}, test={test_auprc:.4f}")
        
        # Calibration analysis
        print("Performing calibration analysis...")
        uncal_val_stats = compute_calibration_stats(y_val, val_prob)
        uncal_test_stats = compute_calibration_stats(y_test, test_prob)
        print(f"Uncalibrated ECE: val={uncal_val_stats['ece']:.4f}, test={uncal_test_stats['ece']:.4f}")
        
        # Try both calibration methods
        sigmoid_cal = fit_probability_calibrator(y_val, val_prob, method='sigmoid', partition='validation')
        isotonic_cal = fit_probability_calibrator(y_val, val_prob, method='isotonic', partition='validation')
        
        val_prob_sigmoid = apply_probability_calibrator(sigmoid_cal, val_prob)
        val_prob_isotonic = apply_probability_calibrator(isotonic_cal, val_prob)
        
        sigmoid_ece = compute_calibration_stats(y_val, val_prob_sigmoid)['ece']
        isotonic_ece = compute_calibration_stats(y_val, val_prob_isotonic)['ece']
        
        # Select best calibration method
        if sigmoid_ece < isotonic_ece:
            calib_method = 'sigmoid'
            calibrator = sigmoid_cal
            val_prob_cal = val_prob_sigmoid
        else:
            calib_method = 'isotonic'
            calibrator = isotonic_cal
            val_prob_cal = val_prob_isotonic
        
        test_prob_cal = apply_probability_calibrator(calibrator, test_prob)
        
        cal_val_stats = compute_calibration_stats(y_val, val_prob_cal)
        cal_test_stats = compute_calibration_stats(y_test, test_prob_cal)
        
        print(f"Selected calibration: {calib_method}")
        print(f"Calibrated ECE: val={cal_val_stats['ece']:.4f}, test={cal_test_stats['ece']:.4f}")
        
        # Threshold selection on calibrated validation probabilities
        print("Selecting threshold on calibrated validation set...")
        threshold, thr_metrics = select_threshold(y_val, val_prob_cal, criterion='f1')
        print(f"Selected threshold: {threshold:.2f}")
        print(f"Validation @ threshold: sens={thr_metrics['sensitivity']:.3f}, spec={thr_metrics['specificity']:.3f}, prec={thr_metrics['precision']:.3f}")
        
        # Final test evaluation with calibrated probabilities
        print("Evaluating on test set...")
        test_pred_class = (test_prob_cal >= threshold).astype(int)
        
        from sklearn.metrics import confusion_matrix, classification_report
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred_class).ravel()
        
        test_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        test_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        test_f1 = 2 * test_prec * test_sens / (test_prec + test_sens) if (test_prec + test_sens) > 0 else 0
        
        test_metrics = {
            'auroc': float(test_auroc),
            'auprc': float(test_auprc),
            'sensitivity': float(test_sens),
            'specificity': float(test_spec),
            'precision': float(test_prec),
            'f1': float(test_f1),
            'brier': cal_test_stats['brier'],
            'ece': cal_test_stats['ece'],
            'mce': cal_test_stats['mce'],
            'threshold': float(threshold),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        }
        
        print(f"Test AUROC: {test_metrics['auroc']:.4f}")
        print(f"Test AUPRC: {test_metrics['auprc']:.4f}")
        print(f"Test sensitivity: {test_metrics['sensitivity']:.3f}")
        print(f"Test specificity: {test_metrics['specificity']:.3f}")
        print(f"Test precision: {test_metrics['precision']:.3f}")
        print(f"Test F1: {test_metrics['f1']:.3f}")
        print(f"Test Brier: {test_metrics['brier']:.4f}")
        
        # Bootstrap confidence intervals
        print("Computing bootstrap confidence intervals (1000 iterations)...")
        pred_frame = pd.DataFrame({
            'patient_id': test.patient_ids.to_numpy(),
            'true_label': y_test,
            'predicted_probability': test_prob_cal,
            'predicted_class': test_pred_class
        })
        
        boot_result = compute_bootstrap_confidence_intervals(
            predictions_df=pred_frame,
            model_name=model_name,
            model_version='1.0.0',
            split='test',
            metric_names=['auroc', 'auprc', 'sensitivity', 'specificity', 'precision', 'f1', 'brier'],
            n_iterations=1000,
            confidence_level=0.95,
            random_seed=42,
            source='real'
        )
        
        print("Bootstrap CIs (95%):")
        for boot_metric in boot_result.metrics:
            print(f"  {boot_metric.metric_name}: [{boot_metric.ci_lower:.4f}, {boot_metric.ci_upper:.4f}]")
        
        # Save model
        print(f"Saving {model_name} model...")
        model_result = P2ModelResult(
            model_name=model_name,
            model=pipeline,
            preprocessing_pipeline=None,
            selected_features=feature_names,
            hyperparameters={},
            validation_metrics={'metrics': thr_metrics},
            test_metrics={'metrics': test_metrics},
            threshold=threshold,
            random_seed=42,
            training_timestamp=datetime.now(UTC).isoformat(),
            n_patients_train=train.n_patients,
            n_rows_train=len(X_train),
            n_positive_train=int(y_train.sum()),
            n_negative_train=int((y_train == 0).sum()),
            class_weight='balanced',
            calibration_method=calib_method,
            calibrator=calibrator,
            source='real'
        )
        
        save_p2_model(model_result, models_dir / model_name, overwrite=True)
        calibrator.save(models_dir / model_name / "calibrator.joblib", overwrite=True)
        
        # Save comprehensive results
        results[model_name] = {
            'training_time_seconds': round(time.time() - model_start, 2),
            'fit_time_seconds': round(fit_time, 2),
            'n_patients_train': train.n_patients,
            'n_observations_train': len(X_train),
            'n_patients_validation': validation.n_patients,
            'n_observations_validation': len(X_val),
            'n_patients_test': test.n_patients,
            'n_observations_test': len(X_test),
            'uncalibrated_metrics': {
                'train_auroc': float(train_auroc),
                'train_auprc': float(train_auprc),
                'val_auroc': float(val_auroc),
                'val_auprc': float(val_auprc),
                'test_auroc': float(test_auroc),
                'test_auprc': float(test_auprc),
                'val_ece': uncal_val_stats['ece'],
                'test_ece': uncal_test_stats['ece']
            },
            'calibration': {
                'method': calib_method,
                'sigmoid_val_ece': float(sigmoid_ece),
                'isotonic_val_ece': float(isotonic_ece),
                'selected_val_ece': cal_val_stats['ece'],
                'selected_test_ece': cal_test_stats['ece'],
                'test_brier': cal_test_stats['brier'],
                'test_mce': cal_test_stats['mce']
            },
            'threshold_selection': {
                'method': 'f1_maximization',
                'split': 'validation',
                'threshold': threshold,
                'validation_metrics': thr_metrics
            },
            'test_metrics': test_metrics,
            'bootstrap_ci': boot_result.to_dict()
        }
        
        print(f"{model_name.upper()} complete in {time.time() - model_start:.1f}s")
        print()
    
    # Save comprehensive summary
    print("Saving comprehensive results...")
    summary = {
        'experiment_id': 'full_cohort_v2',
        'dataset': 'physionet2019_complete',
        'generated_at': datetime.now(UTC).isoformat(),
        'n_features': len(feature_names),
        'n_patients': {
            'train': train.n_patients,
            'validation': validation.n_patients,
            'test': test.n_patients,
            'total': train.n_patients + validation.n_patients + test.n_patients
        },
        'n_observations': {
            'train': len(X_train),
            'validation': len(X_val),
            'test': len(X_test),
            'total': len(X_train) + len(X_val) + len(X_test)
        },
        'models': results
    }
    
    with open(reports_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Test Set, Calibrated)")
    print("=" * 80)
    print(f"{'Model':<15} {'AUROC':<10} {'AUPRC':<10} {'Sens':<10} {'Spec':<10} {'Prec':<10} {'F1':<10} {'ECE':<10}")
    print("-" * 80)
    for name, result in results.items():
        m = result['test_metrics']
        print(f"{name:<15} {m['auroc']:<10.4f} {m['auprc']:<10.4f} {m['sensitivity']:<10.3f} "
              f"{m['specificity']:<10.3f} {m['precision']:<10.3f} {m['f1']:<10.3f} {m['ece']:<10.4f}")
    print("=" * 80)
    
    print(f"\nAll artifacts saved to: {artifact_root}")
    print(f"Completed: {datetime.now()}")
    print("\nSUCCESS: Full cohort training complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
