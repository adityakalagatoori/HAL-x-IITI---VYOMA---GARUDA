"""
Engine Health Index (EHI) Aggregation

Implements domain-informed weighted aggregation of subsystem health metrics.

Research backing: Paper #4 "Advanced Prognostic Health Management of Turbofan
Engines: A Comprehensive Machine Learning Framework Using Real Flight Data" (2025)

Weights based on failure criticality:
- Turbine failure: Total engine loss (weight 0.50)
- Compressor fouling: Gradual performance loss (weight 0.25)
- Combustor erosion: Rare failure mode (weight 0.25)

This replaces the naive mean (0.33, 0.33, 0.33) with domain-informed weights.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class EngineHealthIndexAggregator:
    """
    Aggregates component health metrics into a composite EHI.

    Supports multiple weighting strategies:
    1. Domain-informed (physics-based)
    2. Data-driven (learned from feature importance)
    3. Uniform (baseline for comparison)
    """

    def __init__(self, method: str = "domain_informed"):
        """
        Args:
            method: 'domain_informed', 'uniform', or 'data_driven'
        """
        self.method = method
        if method == "domain_informed":
            # Turbine failure is catastrophic; compressor/combustor more graceful
            self.weights = {
                'CompressorHealth': 0.25,
                'CombustorHealth': 0.25,
                'TurbineHealth': 0.50
            }
        elif method == "uniform":
            # Baseline naive mean
            self.weights = {
                'CompressorHealth': 1/3,
                'CombustorHealth': 1/3,
                'TurbineHealth': 1/3
            }
        else:
            raise ValueError(f"Unknown method: {method}")

    def aggregate(self, comp_health: np.ndarray, comb_health: np.ndarray,
                  turb_health: np.ndarray) -> np.ndarray:
        """
        Compute weighted EHI from component healths.

        Args:
            comp_health: Compressor health (0-1) or multiple samples
            comb_health: Combustor health
            turb_health: Turbine health

        Returns:
            EHI (0-1): Composite health index
        """
        ehi = (self.weights['CompressorHealth'] * comp_health +
               self.weights['CombustorHealth'] * comb_health +
               self.weights['TurbineHealth'] * turb_health)
        return np.clip(ehi, 0, 1)

    def aggregate_with_uncertainty(self,
                                   comp_pred: Dict[str, np.ndarray],
                                   comb_pred: Dict[str, np.ndarray],
                                   turb_pred: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Aggregate health predictions with uncertainty propagation.

        Input format (same as stage_b output):
        {
            'pred': array of predictions,
            'lower': array of lower bounds,
            'upper': array of upper bounds
        }

        Returns:
        {
            'EHI_pred': weighted combination of predictions,
            'EHI_lower': propagated lower bound,
            'EHI_upper': propagated upper bound
        }
        """
        # Point estimates
        ehi_pred = self.aggregate(comp_pred['pred'], comb_pred['pred'], turb_pred['pred'])

        # Uncertainty propagation: worst-case bounds
        # Lower: minimize (all components at lower bound)
        ehi_lower = self.aggregate(comp_pred['lower'], comb_pred['lower'], turb_pred['lower'])

        # Upper: maximize (all components at upper bound)
        ehi_upper = self.aggregate(comp_pred['upper'], comb_pred['upper'], turb_pred['upper'])

        # Clamp to valid range [0, 1]
        ehi_lower = np.clip(ehi_lower, 0, 1)
        ehi_upper = np.clip(ehi_upper, 0, 1)

        return {
            'EHI_pred': ehi_pred,
            'EHI_lower': ehi_lower,
            'EHI_upper': ehi_upper
        }

    def importance_weights(self) -> Dict[str, float]:
        """Return the current weighting scheme."""
        return self.weights.copy()


def aggregate_predictions_dataframe(df: pd.DataFrame, method: str = "domain_informed") -> pd.DataFrame:
    """
    Convenience function: aggregate health predictions from a DataFrame.

    Expects columns: CompressorHealth_pred, CombustorHealth_pred, TurbineHealth_pred
                     (and optionally *_lower, *_upper)

    Returns: DataFrame with additional columns EHI_pred, EHI_lower, EHI_upper
    """
    agg = EngineHealthIndexAggregator(method=method)

    # Point estimates
    ehi_pred = agg.aggregate(
        df['CompressorHealth_pred'].values,
        df['CombustorHealth_pred'].values,
        df['TurbineHealth_pred'].values
    )
    df['EHI_pred'] = ehi_pred

    # Uncertainty bounds (if available)
    if 'CompressorHealth_lower' in df.columns:
        ehi_lower = agg.aggregate(
            df['CompressorHealth_lower'].values,
            df['CombustorHealth_lower'].values,
            df['TurbineHealth_lower'].values
        )
        df['EHI_lower'] = np.clip(ehi_lower, 0, 1)

    if 'CompressorHealth_upper' in df.columns:
        ehi_upper = agg.aggregate(
            df['CompressorHealth_upper'].values,
            df['CombustorHealth_upper'].values,
            df['TurbineHealth_upper'].values
        )
        df['EHI_upper'] = np.clip(ehi_upper, 0, 1)

    return df


def compare_aggregation_methods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare naive mean vs domain-informed weighting.

    Shows the difference in EHI for each sample when using different methods.
    """
    agg_naive = EngineHealthIndexAggregator(method="uniform")
    agg_domain = EngineHealthIndexAggregator(method="domain_informed")

    ehi_naive = agg_naive.aggregate(
        df['CompressorHealth_pred'].values,
        df['CombustorHealth_pred'].values,
        df['TurbineHealth_pred'].values
    )
    ehi_domain = agg_domain.aggregate(
        df['CompressorHealth_pred'].values,
        df['CombustorHealth_pred'].values,
        df['TurbineHealth_pred'].values
    )

    comparison = pd.DataFrame({
        'EngineID': df['EngineID'],
        'Cycle': df['Cycle'],
        'CompHealth': df['CompressorHealth_pred'],
        'CombHealth': df['CombustorHealth_pred'],
        'TurbHealth': df['TurbineHealth_pred'],
        'EHI_Naive': ehi_naive,
        'EHI_Domain': ehi_domain,
        'Delta': ehi_domain - ehi_naive
    })

    return comparison


if __name__ == "__main__":
    # Example: aggregate predictions from stage_b output
    preds = pd.read_csv("../data/stage_b_predictions.csv")

    # Add EHI aggregation
    preds_with_ehi = aggregate_predictions_dataframe(preds, method="domain_informed")

    # Save
    preds_with_ehi.to_csv("../data/stage_b_predictions_with_ehi.csv", index=False)
    print("Saved stage_b_predictions_with_ehi.csv")

    # Compare methods
    comparison = compare_aggregation_methods(preds)
    print("\nComparison: Naive Mean vs Domain-Informed EHI")
    print(comparison.head(10))
    print(f"\nMean difference in EHI: {comparison['Delta'].mean():.4f}")
    print(f"Max difference: {comparison['Delta'].abs().max():.4f}")
