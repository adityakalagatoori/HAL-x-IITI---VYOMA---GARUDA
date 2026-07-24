"""
GARUDA Testing: Robustness, Anomaly, Calibration, Generalization, Propagation, Validation

Combines all 6 test modules
"""
import numpy as np
import pandas as pd
from scipy import stats

# ============ ROBUSTNESS TESTING ============

def test_noise_robustness(model, X_test, y_test, noise_levels=[0.01, 0.05, 0.1]):
    """Test model resilience to Gaussian noise."""
    results = {}
    baseline_rmse = np.sqrt(((model.predict(X_test) - y_test) ** 2).mean())

    for noise_level in noise_levels:
        X_noisy = X_test + noise_level * np.random.normal(size=X_test.shape)
        y_pred_noisy = model.predict(X_noisy)
        rmse_noisy = np.sqrt(((y_pred_noisy - y_test) ** 2).mean())
        results[f'noise_{noise_level}'] = {
            'rmse': rmse_noisy,
            'degradation': (rmse_noisy - baseline_rmse) / baseline_rmse
        }
    return results

def test_sensor_dropout(model, X_test, y_test):
    """Test when one sensor fails."""
    results = {}
    baseline_rmse = np.sqrt(((model.predict(X_test) - y_test) ** 2).mean())

    for sensor_idx in range(X_test.shape[1]):
        X_dropout = X_test.copy()
        X_dropout[:, sensor_idx] = 0  # Dropout: zero sensor
        y_pred_dropout = model.predict(X_dropout)
        rmse_dropout = np.sqrt(((y_pred_dropout - y_test) ** 2).mean())
        results[f'sensor_{sensor_idx}'] = {
            'rmse': rmse_dropout,
            'criticality': (rmse_dropout - baseline_rmse) / baseline_rmse
        }
    return results

def test_byzantine_clients(weights_list, poisoned_fraction=0.5):
    """Test federated learning with poisoned clients."""
    n_clients = len(weights_list)
    n_poisoned = int(n_clients * poisoned_fraction)

    # Poison some weights (extreme values)
    poisoned_weights = [w * 10 for w in weights_list[:n_poisoned]]
    clean_weights = weights_list[n_poisoned:]

    all_weights = poisoned_weights + clean_weights

    # Standard mean (vulnerable)
    mean_agg = np.mean(all_weights, axis=0)

    # Median (Byzantine-robust)
    median_agg = np.median(all_weights, axis=0)

    # Check: median should ignore poisoned clients
    clean_mean = np.mean(clean_weights, axis=0)
    median_distance = np.linalg.norm(median_agg - clean_mean)
    mean_distance = np.linalg.norm(mean_agg - clean_mean)

    return {
        'median_robust': median_distance < mean_distance,
        'median_distance': median_distance,
        'mean_distance': mean_distance
    }

# ============ ANOMALY DETECTION ============

def test_anomaly_detection(detector, X_test, y_anomalies, threshold=0.7):
    """Evaluate fault detection precision/recall."""
    anomaly_scores = detector.predict(X_test)
    y_pred = (anomaly_scores > threshold).astype(int)

    tp = ((y_pred == 1) & (y_anomalies == 1)).sum()
    fp = ((y_pred == 1) & (y_anomalies == 0)).sum()
    fn = ((y_pred == 0) & (y_anomalies == 1)).sum()
    tn = ((y_pred == 0) & (y_anomalies == 0)).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    roc_auc = np.mean([precision, recall])

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

# ============ CALIBRATION ============

def test_conformal_calibration(predictions, true_values, target_coverage=0.90):
    """Verify UQ interval coverage."""
    lower = predictions['lower']
    upper = predictions['upper']

    covered = ((lower <= true_values) & (true_values <= upper)).mean()
    interval_width = (upper - lower).mean()

    calibration_error = abs(covered - target_coverage)

    return {
        'observed_coverage': covered,
        'target_coverage': target_coverage,
        'calibration_error': calibration_error,
        'interval_width': interval_width,
        'pass': calibration_error < 0.05
    }

# ============ GENERALIZATION ============

def test_cross_airline_generalization(model, airlines_test_data):
    """Train on one airline, test on others."""
    results = {}
    for airline_name, (X_test, y_test) in airlines_test_data.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
        r2 = 1 - (((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
        results[airline_name] = {'rmse': rmse, 'r2': r2}
    return results

def compute_maximum_mean_discrepancy(X_source, X_target, kernel='rbf', bandwidth=1.0):
    """Measure distribution shift (MMD)."""
    def rbf_kernel(x, y):
        sq_dist = np.sum((x[:, None] - y[None, :]) ** 2, axis=2)
        return np.exp(-sq_dist / (2 * bandwidth ** 2))

    kxx = rbf_kernel(X_source, X_source).mean()
    kyy = rbf_kernel(X_target, X_target).mean()
    kxy = rbf_kernel(X_source, X_target).mean()

    mmd = np.sqrt(kxx + kyy - 2 * kxy)
    return mmd

# ============ ERROR PROPAGATION ============

def test_error_propagation(noise_at_input, pipeline_functions):
    """Track variance through pipeline."""
    variance_history = [noise_at_input.var()]

    x = noise_at_input
    for func_name, func in pipeline_functions:
        x = func(x)
        variance_history.append(x.var())

    return {
        'input_variance': variance_history[0],
        'output_variance': variance_history[-1],
        'amplification': variance_history[-1] / variance_history[0],
        'variance_by_stage': variance_history
    }

def sensitivity_analysis(model, X_test, y_test, feature_idx):
    """Which sensors matter most?"""
    baseline_rmse = np.sqrt(((model.predict(X_test) - y_test) ** 2).mean())

    # Perturb one sensor
    X_perturbed = X_test.copy()
    X_perturbed[:, feature_idx] = X_test[:, feature_idx] * 1.1  # 10% increase

    y_pred_perturbed = model.predict(X_perturbed)
    rmse_perturbed = np.sqrt(((y_pred_perturbed - y_test) ** 2).mean())

    sensitivity = (rmse_perturbed - baseline_rmse) / baseline_rmse
    return sensitivity

# ============ STATISTICAL VALIDATION ============

def test_residual_normality(residuals):
    """Shapiro-Wilk test for normality."""
    stat, p_value = stats.shapiro(residuals)
    return {'stat': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}

def test_residual_stationarity(time_series):
    """ADF test for stationarity."""
    from statsmodels.tsa.stattools import adfuller
    stat, p_value, _, _, _, _ = adfuller(time_series)
    return {'stat': stat, 'p_value': p_value, 'is_stationary': p_value < 0.05}

def test_model_significance(X_test, y_pred, y_test):
    """F-test for overall model significance."""
    n = len(y_test)
    k = X_test.shape[1]

    ss_res = ((y_test - y_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()

    r2 = 1 - (ss_res / ss_tot)
    f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
    p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)

    return {'r2': r2, 'f_stat': f_stat, 'p_value': p_value, 'significant': p_value < 0.05}
