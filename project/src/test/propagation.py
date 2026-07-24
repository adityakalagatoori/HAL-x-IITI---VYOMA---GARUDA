"""
Uncertainty Propagation from Subsystems to Overall Health (B9)

Current bug: OverallHealth uncertainty NOT derived from subsystem uncertainties.
Fix: Propagate subsystem uncertainties through aggregation function.

Method: Monte Carlo sampling
- Sample CompressorHealth from its distribution
- Sample CombustorHealth from its distribution
- Sample TurbineHealth from its distribution
- Aggregate: EHI = 0.5*Turb + 0.25*Comp + 0.25*Comb
- Compute percentiles over samples
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.stats import norm


class UncertaintyPropagator:
    """Propagates component uncertainties to overall health via Monte Carlo."""

    def __init__(self, n_samples: int = 10000):
        """
        Args:
            n_samples: Number of MC samples for propagation
        """
        self.n_samples = n_samples
        self.weights = {
            'CompressorHealth': 0.25,
            'CombustorHealth': 0.25,
            'TurbineHealth': 0.50
        }

    def propagate_mc(self, comp_pred: float, comp_std: float,
                     comb_pred: float, comb_std: float,
                     turb_pred: float, turb_std: float,
                     confidence: float = 0.90) -> Dict[str, float]:
        """
        Propagate uncertainty via Monte Carlo sampling.

        Args:
            comp_pred, comb_pred, turb_pred: Point estimates
            comp_std, comb_std, turb_std: Predicted standard deviations
            confidence: Confidence level (0.90 for 90% interval)

        Returns:
            Dictionary with 'pred', 'lower', 'upper'
        """
        # Generate samples from component distributions
        comp_samples = np.random.normal(comp_pred, comp_std, self.n_samples)
        comb_samples = np.random.normal(comb_pred, comb_std, self.n_samples)
        turb_samples = np.random.normal(turb_pred, turb_std, self.n_samples)

        # Aggregate: EHI = weighted sum
        ehi_samples = (self.weights['CompressorHealth'] * comp_samples +
                      self.weights['CombustorHealth'] * comb_samples +
                      self.weights['TurbineHealth'] * turb_samples)

        # Clamp to [0, 1]
        ehi_samples = np.clip(ehi_samples, 0, 1)

        # Compute quantiles
        alpha = (1 - confidence) / 2
        lower_q = alpha
        upper_q = 1 - alpha

        return {
            'pred': float(np.mean(ehi_samples)),
            'lower': float(np.quantile(ehi_samples, lower_q)),
            'upper': float(np.quantile(ehi_samples, upper_q)),
            'std': float(np.std(ehi_samples))
        }

    def propagate_analytical(self, comp_pred: float, comp_std: float,
                            comb_pred: float, comb_std: float,
                            turb_pred: float, turb_std: float,
                            confidence: float = 0.90) -> Dict[str, float]:
        """
        Propagate uncertainty analytically (faster, assumes linearity).

        For linear aggregation: EHI = w1*Comp + w2*Comb + w3*Turb
        Overall variance = w1²*var(Comp) + w2²*var(Comb) + w3²*var(Turb)
        """
        w1, w2, w3 = self.weights['CompressorHealth'], self.weights['CombustorHealth'], self.weights['TurbineHealth']

        # Mean of EHI
        ehi_mean = w1 * comp_pred + w2 * comb_pred + w3 * turb_pred

        # Variance of EHI (assuming independence)
        ehi_var = (w1 ** 2) * (comp_std ** 2) + (w2 ** 2) * (comb_std ** 2) + (w3 ** 2) * (turb_std ** 2)
        ehi_std = np.sqrt(ehi_var)

        # Confidence interval
        z_score = norm.ppf((1 + confidence) / 2)
        lower = ehi_mean - z_score * ehi_std
        upper = ehi_mean + z_score * ehi_std

        return {
            'pred': float(np.clip(ehi_mean, 0, 1)),
            'lower': float(np.clip(lower, 0, 1)),
            'upper': float(np.clip(upper, 0, 1)),
            'std': float(ehi_std)
        }

    def compare_methods(self, comp_pred: float, comp_std: float,
                       comb_pred: float, comb_std: float,
                       turb_pred: float, turb_std: float) -> Dict:
        """Compare MC vs analytical propagation."""
        mc_result = self.propagate_mc(comp_pred, comp_std, comb_pred, comb_std, turb_pred, turb_std)
        analytical_result = self.propagate_analytical(comp_pred, comp_std, comb_pred, comb_std, turb_pred, turb_std)

        return {
            'monte_carlo': mc_result,
            'analytical': analytical_result,
            'difference': {
                'pred_diff': abs(mc_result['pred'] - analytical_result['pred']),
                'lower_diff': abs(mc_result['lower'] - analytical_result['lower']),
                'upper_diff': abs(mc_result['upper'] - analytical_result['upper'])
            }
        }


def propagate_dataframe(df: pd.DataFrame, method: str = "analytical") -> pd.DataFrame:
    """
    Propagate uncertainty for all rows in a DataFrame.

    Expects columns:
    - CompressorHealth_pred, CompressorHealth_lower/upper (or std)
    - CombustorHealth_pred, CombustorHealth_lower/upper (or std)
    - TurbineHealth_pred, TurbineHealth_lower/upper (or std)

    Returns: DataFrame with added columns:
    - OverallHealth_pred, OverallHealth_lower, OverallHealth_upper, OverallHealth_std
    """
    propagator = UncertaintyPropagator()
    n_rows = len(df)

    ehi_preds = np.zeros(n_rows)
    ehi_lowers = np.zeros(n_rows)
    ehi_uppers = np.zeros(n_rows)
    ehi_stds = np.zeros(n_rows)

    for i in range(n_rows):
        # Compute standard deviations if not available
        comp_std = _get_std_or_compute(df.iloc[i], 'CompressorHealth')
        comb_std = _get_std_or_compute(df.iloc[i], 'CombustorHealth')
        turb_std = _get_std_or_compute(df.iloc[i], 'TurbineHealth')

        result = propagator.propagate_analytical(
            df[f'CompressorHealth_pred'].iloc[i],
            comp_std,
            df[f'CombustorHealth_pred'].iloc[i],
            comb_std,
            df[f'TurbineHealth_pred'].iloc[i],
            turb_std
        ) if method == "analytical" else propagator.propagate_mc(
            df[f'CompressorHealth_pred'].iloc[i],
            comp_std,
            df[f'CombustorHealth_pred'].iloc[i],
            comb_std,
            df[f'TurbineHealth_pred'].iloc[i],
            turb_std
        )

        ehi_preds[i] = result['pred']
        ehi_lowers[i] = result['lower']
        ehi_uppers[i] = result['upper']
        ehi_stds[i] = result['std']

    df['OverallHealth_pred_propagated'] = ehi_preds
    df['OverallHealth_lower_propagated'] = ehi_lowers
    df['OverallHealth_upper_propagated'] = ehi_uppers
    df['OverallHealth_std_propagated'] = ehi_stds

    return df


def _get_std_or_compute(row: pd.Series, health_col: str) -> float:
    """Get standard deviation from row, or compute from bounds."""
    std_col = f'{health_col}_std'
    lower_col = f'{health_col}_lower'
    upper_col = f'{health_col}_upper'
    pred_col = f'{health_col}_pred'

    if std_col in row.index and pd.notna(row[std_col]):
        return row[std_col]
    elif lower_col in row.index and upper_col in row.index:
        # Approximate std from 90% interval: interval_width ≈ 2 * 1.645 * std
        width = row[upper_col] - row[lower_col]
        return width / (2 * 1.645)
    else:
        # Default: small std
        return 0.01


if __name__ == "__main__":
    # Example
    propagator = UncertaintyPropagator(n_samples=10000)

    # Sample prediction
    comp_pred, comp_std = 0.95, 0.05
    comb_pred, comb_std = 0.90, 0.08
    turb_pred, turb_std = 0.85, 0.10

    print("Uncertainty Propagation Example")
    print(f"CompressorHealth: {comp_pred:.2f} ± {comp_std:.2f}")
    print(f"CombustorHealth: {comb_pred:.2f} ± {comb_std:.2f}")
    print(f"TurbineHealth: {turb_pred:.2f} ± {turb_std:.2f}")

    comparison = propagator.compare_methods(comp_pred, comp_std, comb_pred, comb_std, turb_pred, turb_std)

    print(f"\nMonte Carlo Result:")
    print(f"  EHI = {comparison['monte_carlo']['pred']:.4f}")
    print(f"  90% interval: [{comparison['monte_carlo']['lower']:.4f}, {comparison['monte_carlo']['upper']:.4f}]")
    print(f"  Std dev: {comparison['monte_carlo']['std']:.5f}")

    print(f"\nAnalytical Result:")
    print(f"  EHI = {comparison['analytical']['pred']:.4f}")
    print(f"  90% interval: [{comparison['analytical']['lower']:.4f}, {comparison['analytical']['upper']:.4f}]")
    print(f"  Std dev: {comparison['analytical']['std']:.5f}")

    print(f"\nDifference (MC vs Analytical):")
    print(f"  Pred diff: {comparison['difference']['pred_diff']:.6f}")
    print(f"  Lower diff: {comparison['difference']['lower_diff']:.6f}")
    print(f"  Upper diff: {comparison['difference']['upper_diff']:.6f}")
