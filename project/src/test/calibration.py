"""
Reliability Diagram Analysis & Calibration Curves

Validates that predicted uncertainty bounds match empirical error distribution.

Mathematical concept: A model is well-calibrated if predicted confidence intervals
contain the true value with frequency matching the predicted confidence.

Example:
- Predicted 90% confidence interval: [0.8, 1.2]
- True value: 1.0
- Outcome: TRUE (interval contains truth)

Over many samples:
- 90% CI should contain true value ~90% of the time
- 50% CI should contain true value ~50% of the time
- etc.

This script generates reliability diagrams showing:
1. Predicted interval width vs empirical coverage
2. Calibration curve: theoretical vs observed coverage
3. Quantile-quantile plots for residuals
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import json


class ReliabilityAnalyzer:
    """Analyzes calibration of uncertainty quantification."""

    def __init__(self, predictions: np.ndarray, lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray, true_values: np.ndarray):
        """
        Args:
            predictions: Point predictions
            lower_bounds: Lower confidence bounds
            upper_bounds: Upper confidence bounds
            true_values: Ground truth values
        """
        self.pred = predictions
        self.lower = lower_bounds
        self.upper = upper_bounds
        self.true = true_values

        # Compute metrics
        self.residuals = self.true - self.pred
        self.interval_widths = self.upper - self.lower
        self.interval_coverage = (self.true >= self.lower) & (self.true <= self.upper)

    def overall_coverage(self) -> float:
        """Fraction of samples where true value falls in predicted interval."""
        return float(np.mean(self.interval_coverage))

    def binned_coverage(self, n_bins: int = 5) -> Dict[str, np.ndarray]:
        """
        Compute coverage separately for each bin of interval widths.

        Bins by predicted interval width: smaller intervals should have lower coverage,
        wider intervals higher coverage.
        """
        bin_edges = np.percentile(self.interval_widths, np.linspace(0, 100, n_bins + 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        coverage_by_bin = []
        counts_by_bin = []

        for i in range(n_bins):
            mask = (self.interval_widths >= bin_edges[i]) & (self.interval_widths < bin_edges[i+1])
            if np.any(mask):
                bin_coverage = np.mean(self.interval_coverage[mask])
                bin_count = np.sum(mask)
            else:
                bin_coverage = np.nan
                bin_count = 0
            coverage_by_bin.append(bin_coverage)
            counts_by_bin.append(bin_count)

        return {
            'bin_centers': bin_centers,
            'coverage_by_bin': np.array(coverage_by_bin),
            'counts_by_bin': np.array(counts_by_bin)
        }

    def calibration_by_predicted_width(self, quantiles: list = None) -> Dict:
        """
        Calibration curve: for different predicted interval widths,
        what is the empirical coverage?

        Quantiles like [0.5, 0.9, 0.95] represent ~68%, ~90%, ~95% confidence.
        """
        if quantiles is None:
            quantiles = [0.5, 0.68, 0.80, 0.90, 0.95, 0.99]

        results = {}
        for q in quantiles:
            # Predicted interval with this quantile
            # Assume symmetric interval: +/- z-score * std
            z_score = np.abs(np.percentile(np.abs(self.residuals), q * 100))
            # Empirical: how many samples actually fall in mean +/- z_score?
            empirical_coverage = np.mean(np.abs(self.residuals) <= z_score)
            results[f"{q:.0%}_nominal"] = {
                'predicted_quantile': q,
                'empirical_coverage': empirical_coverage,
                'coverage_gap': empirical_coverage - q
            }

        return results

    def residual_analysis(self) -> Dict:
        """Analyze residual distribution."""
        return {
            'mean_residual': float(np.mean(self.residuals)),
            'std_residual': float(np.std(self.residuals)),
            'min_residual': float(np.min(self.residuals)),
            'max_residual': float(np.max(self.residuals)),
            'rmse': float(np.sqrt(np.mean(self.residuals ** 2))),
            'mae': float(np.mean(np.abs(self.residuals)))
        }

    def percentile_coverage(self) -> Dict[float, float]:
        """Coverage at different nominal confidence levels."""
        nominal_levels = [0.5, 0.68, 0.80, 0.90, 0.95, 0.99]
        results = {}
        sorted_residuals = np.sort(np.abs(self.residuals))
        for level in nominal_levels:
            idx = int(level * len(sorted_residuals))
            threshold = sorted_residuals[min(idx, len(sorted_residuals) - 1)]
            empirical = np.mean(np.abs(self.residuals) <= threshold)
            results[level] = {
                'empirical_coverage': float(empirical),
                'gap_from_nominal': float(empirical - level)
            }
        return results

    def is_well_calibrated(self, tolerance: float = 0.03) -> bool:
        """Check if model is well-calibrated (coverage gap < tolerance)."""
        coverage = self.overall_coverage()
        # For 90% nominal, should be 90% ± tolerance
        gap = abs(coverage - 0.90)
        return gap < tolerance

    def calibration_report(self) -> str:
        """Generate human-readable calibration report."""
        report = []
        report.append("=" * 60)
        report.append("RELIABILITY / CALIBRATION ANALYSIS REPORT")
        report.append("=" * 60)

        # Overall coverage
        overall_cov = self.overall_coverage()
        report.append(f"\nOverall Coverage (90% nominal): {overall_cov*100:.1f}%")
        report.append(f"Target: 90% ± 3%")
        report.append(f"Status: {'✓ WELL-CALIBRATED' if self.is_well_calibrated() else '✗ MISCALIBRATED'}")

        # Residual stats
        resid_stats = self.residual_analysis()
        report.append(f"\nResidual Statistics:")
        report.append(f"  Mean: {resid_stats['mean_residual']:+.5f}")
        report.append(f"  Std Dev: {resid_stats['std_residual']:.5f}")
        report.append(f"  RMSE: {resid_stats['rmse']:.5f}")
        report.append(f"  MAE: {resid_stats['mae']:.5f}")

        # Coverage by quantile
        report.append(f"\nCoverage by Nominal Confidence Level:")
        perc_cov = self.percentile_coverage()
        for level, metrics in perc_cov.items():
            gap_str = f"{metrics['gap_from_nominal']:+.1%}"
            status = "✓" if abs(metrics['gap_from_nominal']) < 0.03 else "✗"
            report.append(f"  {level:.0%} nominal → {metrics['empirical_coverage']:.1%} empirical ({gap_str}) {status}")

        # Interval width analysis
        report.append(f"\nInterval Width Analysis:")
        report.append(f"  Mean width: {np.mean(self.interval_widths):.5f}")
        report.append(f"  Median width: {np.median(self.interval_widths):.5f}")
        report.append(f"  Std dev: {np.std(self.interval_widths):.5f}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)

    def to_dict(self) -> Dict:
        """Export analysis to dictionary for JSON/dashboard."""
        return {
            'overall_coverage': float(self.overall_coverage()),
            'is_well_calibrated': self.is_well_calibrated(),
            'residual_analysis': self.residual_analysis(),
            'percentile_coverage': self.percentile_coverage(),
            'interval_width_stats': {
                'mean': float(np.mean(self.interval_widths)),
                'median': float(np.median(self.interval_widths)),
                'std': float(np.std(self.interval_widths))
            }
        }


def analyze_health_predictions(df: pd.DataFrame, target: str, lower_col: str = None,
                               upper_col: str = None) -> ReliabilityAnalyzer:
    """
    Convenience function: analyze predictions from a DataFrame.

    Args:
        df: DataFrame with predictions and bounds
        target: 'CompressorHealth', 'CombustorHealth', 'TurbineHealth', or 'OverallHealth'
        lower_col: Column name for lower bound (default: f"{target}_lower")
        upper_col: Column name for upper bound (default: f"{target}_upper")
    """
    if lower_col is None:
        lower_col = f"{target}_lower"
    if upper_col is None:
        upper_col = f"{target}_upper"

    pred = df[f"{target}_pred"].values
    lower = df[lower_col].values
    upper = df[upper_col].values
    true = df[f"{target}_true"].values

    return ReliabilityAnalyzer(pred, lower, upper, true)


def compare_models_calibration(models_dict: Dict[str, pd.DataFrame],
                              target: str = "OverallHealth") -> pd.DataFrame:
    """
    Compare calibration across multiple models.

    Args:
        models_dict: {'model_name': predictions_df, ...}
        target: Health metric to analyze

    Returns:
        DataFrame comparing coverage and calibration for each model
    """
    results = []
    for model_name, df in models_dict.items():
        analyzer = analyze_health_predictions(df, target)
        results.append({
            'model': model_name,
            'coverage': analyzer.overall_coverage(),
            'rmse': analyzer.residual_analysis()['rmse'],
            'calibrated': analyzer.is_well_calibrated()
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example: analyze stage_b predictions
    preds = pd.read_csv("../data/stage_b_predictions.csv")

    print("Analyzing OverallHealth predictions...\n")
    analyzer = analyze_health_predictions(preds, "OverallHealth")
    print(analyzer.calibration_report())

    # Export to JSON for dashboard
    analysis_dict = analyzer.to_dict()
    with open("../data/reliability_analysis.json", "w") as f:
        json.dump(analysis_dict, f, indent=2)
    print("\nSaved reliability_analysis.json")

    # Analyze all health metrics
    for target in ["CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"]:
        analyzer = analyze_health_predictions(preds, target)
        cov = analyzer.overall_coverage()
        cal = "✓" if analyzer.is_well_calibrated() else "✗"
        print(f"{target:18s} Coverage: {cov*100:5.1f}%  Calibrated: {cal}")
