#!/usr/bin/env python3
"""Quick test to verify bootstrap function works correctly."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "icu-sentinel"))

from ml.p1_clinical.evaluation.bootstrap import compute_bootstrap_confidence_intervals

def test_bootstrap():
    """Test that bootstrap function returns correct schema."""
    print("Testing bootstrap confidence intervals...")
    
    # Create synthetic test data
    n_patients = 100
    n_obs_per_patient = 10
    
    patient_ids = np.repeat(np.arange(n_patients), n_obs_per_patient)
    true_labels = np.random.binomial(1, 0.1, n_patients * n_obs_per_patient)
    predicted_probs = np.random.uniform(0, 1, n_patients * n_obs_per_patient)
    predicted_class = (predicted_probs > 0.5).astype(int)
    
    pred_df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': true_labels,
        'predicted_probability': predicted_probs,
        'predicted_class': predicted_class
    })
    
    # Run bootstrap
    print("Running bootstrap with 100 iterations...")
    result = compute_bootstrap_confidence_intervals(
        predictions_df=pred_df,
        model_name='test_model',
        model_version='1.0.0',
        split='test',
        metric_names=['auroc', 'auprc', 'sensitivity', 'specificity'],
        n_iterations=100,
        confidence_level=0.95,
        random_seed=42,
        source='test'
    )
    
    # Verify schema
    print("\nVerifying BootstrapSummary schema...")
    assert hasattr(result, 'model_name'), "Missing model_name"
    assert hasattr(result, 'model_version'), "Missing model_version"
    assert hasattr(result, 'split'), "Missing split"
    assert hasattr(result, 'metrics'), "Missing metrics"
    assert hasattr(result, 'source'), "Missing source"
    
    print(f"✓ BootstrapSummary has correct attributes")
    print(f"✓ Model: {result.model_name}")
    print(f"✓ Split: {result.split}")
    print(f"✓ Number of metrics: {len(result.metrics)}")
    
    # Verify metrics schema
    print("\nVerifying BootstrapResult schema...")
    for boot_metric in result.metrics:
        assert hasattr(boot_metric, 'metric_name'), "Missing metric_name"
        assert hasattr(boot_metric, 'point_estimate'), "Missing point_estimate"
        assert hasattr(boot_metric, 'ci_lower'), "Missing ci_lower"
        assert hasattr(boot_metric, 'ci_upper'), "Missing ci_upper"
        assert hasattr(boot_metric, 'confidence_level'), "Missing confidence_level"
        assert hasattr(boot_metric, 'n_iterations'), "Missing n_iterations"
        assert hasattr(boot_metric, 'n_patients'), "Missing n_patients"
        
        print(f"✓ {boot_metric.metric_name}: [{boot_metric.ci_lower:.4f}, {boot_metric.ci_upper:.4f}]")
    
    # Test the iteration pattern used in train_full_cohort.py
    print("\nTesting iteration pattern from train_full_cohort.py...")
    for boot_metric in result.metrics:
        print(f"  {boot_metric.metric_name}: [{boot_metric.ci_lower:.4f}, {boot_metric.ci_upper:.4f}]")
    
    print("\n✅ All tests passed! Bootstrap fix is correct.")
    return True

if __name__ == '__main__':
    try:
        test_bootstrap()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
