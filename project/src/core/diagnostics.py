"""
Sensor Perturbation & Robustness Testing

Tests model robustness to realistic sensor drift (±2%, ±5%, ±10%).
Airlines report 2-5% annual calibration drift on pressure/temperature sensors.

This validates that health estimates degrade gracefully under sensor noise,
not catastrophically.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
import json
from sklearn.metrics import mean_squared_error


class SensorPerturbationTester:
    """
    Add Gaussian noise to sensors and measure prediction degradation.
    """

    def __init__(self, df: pd.DataFrame, model_predictions_col: str = "OverallHealth_pred",
                 true_col: str = "OverallHealth_true"):
        """
        Args:
            df: DataFrame with sensor columns and predictions
            model_predictions_col: Name of prediction column to evaluate
            true_col: Name of true value column
        """
        self.df_clean = df.copy()
        self.predictions_col = model_predictions_col
        self.true_col = true_col
        self.sensors_to_test = [
            'P2_Pa', 'T2_K', 'P3_Pa', 'T3_K', 'P4_Pa', 'T4_K',
            'RPM_rev_min', 'FuelFlow_kg_s'
        ]

    def perturb_sensor(self, sensor: str, perturbation: float) -> pd.DataFrame:
        """
        Add Gaussian noise to a sensor.

        Args:
            sensor: Sensor column name
            perturbation: Perturbation level (e.g., 0.05 for ±5%)

        Returns:
            DataFrame with perturbed sensor values
        """
        df_perturbed = self.df_clean.copy()
        if sensor not in df_perturbed.columns:
            raise ValueError(f"Sensor {sensor} not found")

        # Gaussian noise: N(0, perturbation*value)
        noise = np.random.normal(0, perturbation * df_perturbed[sensor].values)
        df_perturbed[sensor] = df_perturbed[sensor] + noise
        return df_perturbed

    def evaluate_robustness(self, model_predict_fn, perturbation_levels: List[float] = None) -> Dict:
        """
        Evaluate model robustness across sensor perturbations.

        Args:
            model_predict_fn: Function(df) -> predictions array
            perturbation_levels: List of perturbations to test

        Returns:
            Dictionary of results by sensor and perturbation level
        """
        if perturbation_levels is None:
            perturbation_levels = [0.02, 0.05, 0.10]

        # Baseline: clean data
        y_true = self.df_clean[self.true_col].values
        pred_clean = model_predict_fn(self.df_clean)
        rmse_clean = np.sqrt(mean_squared_error(y_true, pred_clean))

        results = {
            'baseline_rmse': float(rmse_clean),
            'sensors_tested': {},
            'summary': {}
        }

        # Test each sensor
        for sensor in self.sensors_to_test:
            results['sensors_tested'][sensor] = {}

            for pert in perturbation_levels:
                df_perturbed = self.perturb_sensor(sensor, pert)
                try:
                    pred_perturbed = model_predict_fn(df_perturbed)
                    rmse_pert = np.sqrt(mean_squared_error(y_true, pred_perturbed))
                    rmse_increase = (rmse_pert - rmse_clean) / rmse_clean
                    status = self._classify_robustness(rmse_increase)
                except Exception as e:
                    rmse_pert = np.nan
                    rmse_increase = np.nan
                    status = "ERROR"

                results['sensors_tested'][sensor][f"{pert:.0%}_drift"] = {
                    'rmse': float(rmse_pert) if not np.isnan(rmse_pert) else None,
                    'rmse_increase_pct': float(rmse_increase * 100) if not np.isnan(rmse_increase) else None,
                    'status': status
                }

        # Aggregate statistics
        all_increases = []
        for sensor_results in results['sensors_tested'].values():
            for pert_results in sensor_results.values():
                if pert_results['rmse_increase_pct'] is not None:
                    all_increases.append(pert_results['rmse_increase_pct'])

        if all_increases:
            results['summary'] = {
                'mean_rmse_increase_pct': float(np.mean(all_increases)),
                'max_rmse_increase_pct': float(np.max(all_increases)),
                'min_rmse_increase_pct': float(np.min(all_increases)),
                'std_rmse_increase_pct': float(np.std(all_increases))
            }

        return results

    def _classify_robustness(self, rmse_increase: float) -> str:
        """Classify robustness based on RMSE increase."""
        if np.isnan(rmse_increase):
            return "UNKNOWN"
        increase_pct = rmse_increase * 100
        if increase_pct < 2:
            return "EXCELLENT"
        elif increase_pct < 5:
            return "GOOD"
        elif increase_pct < 10:
            return "ACCEPTABLE"
        else:
            return "CONCERNING"

    def report(self, results: Dict) -> str:
        """Generate human-readable robustness report."""
        report = []
        report.append("=" * 70)
        report.append("SENSOR PERTURBATION ROBUSTNESS TEST REPORT")
        report.append("=" * 70)

        baseline = results['baseline_rmse']
        report.append(f"\nBaseline RMSE (clean sensors): {baseline:.5f}")

        report.append(f"\nRobustness by Sensor and Perturbation Level:")
        report.append("")

        # Organize by sensor
        for sensor in sorted(results['sensors_tested'].keys()):
            report.append(f"  {sensor}:")
            sensor_results = results['sensors_tested'][sensor]
            for pert_key in sorted(sensor_results.keys()):
                pert_result = sensor_results[pert_key]
                rmse_increase = pert_result['rmse_increase_pct']
                status = pert_result['status']
                if rmse_increase is not None:
                    report.append(f"    {pert_key:10s} → RMSE increase {rmse_increase:+6.2f}%  [{status}]")
                else:
                    report.append(f"    {pert_key:10s} → ERROR")

        # Summary
        if 'summary' in results and results['summary']:
            report.append(f"\nAggregate Statistics:")
            report.append(f"  Mean RMSE increase: {results['summary']['mean_rmse_increase_pct']:+6.2f}%")
            report.append(f"  Max RMSE increase: {results['summary']['max_rmse_increase_pct']:+6.2f}%")
            report.append(f"  Std dev: {results['summary']['std_rmse_increase_pct']:6.2f}%")

        report.append("\n" + "=" * 70)
        report.append("CONCLUSION:")
        if results['summary']['max_rmse_increase_pct'] < 5:
            report.append("✓ Model is ROBUST to typical sensor drift (±2-5%)")
        elif results['summary']['max_rmse_increase_pct'] < 10:
            report.append("~ Model has ACCEPTABLE robustness to sensor drift")
        else:
            report.append("✗ Model is SENSITIVE to sensor drift; recalibration recommended")
        report.append("=" * 70)

        return "\n".join(report)


def test_health_model_robustness(predictions_file: str, model_predict_fn) -> Dict:
    """
    Convenience function: test robustness using saved predictions.

    Args:
        predictions_file: CSV file with predictions
        model_predict_fn: Function to generate predictions on perturbed data

    Returns:
        Robustness test results
    """
    df = pd.read_csv(predictions_file)
    tester = SensorPerturbationTester(df, "OverallHealth_pred", "OverallHealth_true")
    results = tester.evaluate_robustness(model_predict_fn)
    print(tester.report(results))
    return results


if __name__ == "__main__":
    # Example: test with dummy data
    np.random.seed(42)
    n_samples = 100

    # Create synthetic data
    df = pd.DataFrame({
        'P2_Pa': np.random.normal(200000, 50000, n_samples),
        'T2_K': np.random.normal(400, 50, n_samples),
        'P3_Pa': np.random.normal(300000, 70000, n_samples),
        'T3_K': np.random.normal(3000, 400, n_samples),
        'P4_Pa': np.random.normal(150000, 40000, n_samples),
        'T4_K': np.random.normal(2500, 300, n_samples),
        'RPM_rev_min': np.random.normal(50000, 10000, n_samples),
        'FuelFlow_kg_s': np.random.normal(1.0, 0.2, n_samples),
        'OverallHealth_true': np.random.uniform(0.5, 1.0, n_samples),
        'OverallHealth_pred': np.random.uniform(0.5, 1.0, n_samples)
    })

    # Simple dummy prediction function (returns same as input)
    def dummy_predict(df_input):
        return df_input['OverallHealth_pred'].values

    # Test robustness
    tester = SensorPerturbationTester(df, "OverallHealth_pred", "OverallHealth_true")
    results = tester.evaluate_robustness(dummy_predict, perturbation_levels=[0.02, 0.05, 0.10])
    print(tester.report(results))

    # Save results
    with open("../data/sensor_robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved sensor_robustness_results.json")
