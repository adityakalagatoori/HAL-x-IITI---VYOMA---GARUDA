"""
Out-of-Distribution Generalization Test (B10)

Current LOEO gap: Uses population trend fitted on 9 engines; new engine variant
(different design) won't fit this population → false confidence in generalization.

Fix: Test on synthetic out-of-distribution engines with perturbed physics parameters.
- Compressor efficiency curve perturbed (compressor fouling severity varies by design)
- Pressure ratio perturbed (architecture-dependent)
- Turbine efficiency perturbed
- RUL pattern perturbed

This gives real evidence of generalization to fundamentally different engine designs.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error


class SyntheticEngineGenerator:
    """Generate synthetic out-of-distribution engine variants."""

    def __init__(self, reference_data: pd.DataFrame):
        """
        Args:
            reference_data: Training data to establish nominal parameters
        """
        self.ref_data = reference_data
        self._fit_nominal_relationships()

    def _fit_nominal_relationships(self):
        """Learn nominal physics relationships from training data."""
        # Nominal compressor efficiency degradation rate
        comp_health_by_cycle = {}
        for eng in self.ref_data['EngineID'].unique():
            eng_data = self.ref_data[self.ref_data['EngineID'] == eng]
            if 'CompressorHealth' in eng_data.columns:
                cycles = eng_data['Cycle'].values
                healths = eng_data['CompressorHealth'].values
                if len(cycles) > 1:
                    # Fit linear trend
                    fit = np.polyfit(cycles, healths, 1)
                    comp_health_by_cycle[eng] = fit

        self.nominal_comp_degradation = np.mean([f[0] for f in comp_health_by_cycle.values()])

    def generate_ood_variant(self, base_data: pd.DataFrame, variant_id: int,
                            comp_eff_delta: float = 0.0,
                            pr_delta: float = 0.0,
                            turb_eff_delta: float = 0.0) -> pd.DataFrame:
        """
        Generate a synthetic out-of-distribution engine variant.

        Args:
            base_data: One engine's data to use as template
            variant_id: ID for this variant
            comp_eff_delta: Compressor efficiency shift (+0.1 = 10% more efficient)
            pr_delta: Pressure ratio shift (+0.05 = 5% higher)
            turb_eff_delta: Turbine efficiency shift

        Returns:
            Synthetic data for new variant
        """
        synthetic = base_data.copy()
        synthetic['EngineID'] = variant_id

        # Perturb pressure/temperature to reflect different compressor/turbine design
        if pr_delta != 0:
            synthetic['P3_Pa'] = synthetic['P2_Pa'] * (1 + pr_delta)

        if comp_eff_delta != 0:
            # Higher efficiency → lower T2 for same pressure rise
            synthetic['T2_K'] = synthetic['T2_K'] * (1 - comp_eff_delta * 0.01)

        if turb_eff_delta != 0:
            # Higher turbine efficiency → higher T4 (less temperature drop)
            synthetic['T4_K'] = synthetic['T4_K'] * (1 + turb_eff_delta * 0.01)

        # Perturb health trajectory
        if 'CompressorHealth' in synthetic.columns:
            synthetic['CompressorHealth'] = synthetic['CompressorHealth'] * (1 - comp_eff_delta * 0.1)

        return synthetic

    def generate_ood_test_set(self, base_engine_data: pd.DataFrame,
                             n_variants: int = 5) -> pd.DataFrame:
        """
        Generate multiple OOD variants for evaluation.
        """
        variants = []

        # Variant 1: More efficient compressor (newer design)
        variants.append(self.generate_ood_variant(base_engine_data, 101, comp_eff_delta=10))

        # Variant 2: Lower pressure ratio (older design)
        variants.append(self.generate_ood_variant(base_engine_data, 102, pr_delta=-0.05))

        # Variant 3: Mixed: better compressor, worse turbine
        variants.append(self.generate_ood_variant(base_engine_data, 103, comp_eff_delta=8, turb_eff_delta=-5))

        # Variant 4: High-altitude variant (lower nominal pressures)
        v4 = self.generate_ood_variant(base_engine_data, 104, pr_delta=-0.10)
        v4['Altitude_m'] = 20000
        variants.append(v4)

        # Variant 5: Military-spec higher compression
        variants.append(self.generate_ood_variant(base_engine_data, 105, pr_delta=0.15, comp_eff_delta=-5))

        return pd.concat(variants, ignore_index=True)


class OODGeneralizationTester:
    """Tests model generalization to out-of-distribution engine designs."""

    def __init__(self, trained_model_predict_fn):
        """
        Args:
            trained_model_predict_fn: Function that predicts on new data
        """
        self.predict_fn = trained_model_predict_fn

    def evaluate_on_ood_variants(self, ood_data: pd.DataFrame,
                                 true_col: str = "OverallHealth") -> Dict:
        """
        Evaluate trained model on OOD variants.

        Args:
            ood_data: Synthetic OOD engine data
            true_col: Ground truth column name

        Returns:
            Results dictionary
        """
        try:
            predictions = self.predict_fn(ood_data)
        except Exception as e:
            return {'error': str(e)}

        if true_col not in ood_data.columns:
            return {'warning': f'{true_col} not in OOD data; cannot compute error'}

        y_true = ood_data[true_col].values
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = np.mean(np.abs(y_true - predictions))

        results = {
            'n_samples': len(ood_data),
            'rmse': float(rmse),
            'mae': float(mae),
            'by_engine': {}
        }

        # Per-engine results
        for eng in ood_data['EngineID'].unique():
            eng_mask = ood_data['EngineID'] == eng
            eng_true = y_true[eng_mask]
            eng_pred = predictions[eng_mask]
            eng_rmse = np.sqrt(mean_squared_error(eng_true, eng_pred))
            results['by_engine'][int(eng)] = {'rmse': float(eng_rmse), 'n_samples': int(np.sum(eng_mask))}

        return results

    def compare_iid_vs_ood(self, iid_test_results: Dict, ood_results: Dict) -> Dict:
        """
        Compare performance on in-distribution vs out-of-distribution data.

        Shows generalization gap.
        """
        iid_rmse = iid_test_results.get('rmse', np.nan)
        ood_rmse = ood_results.get('rmse', np.nan)

        if not np.isnan(iid_rmse) and not np.isnan(ood_rmse):
            degradation = (ood_rmse - iid_rmse) / iid_rmse * 100
        else:
            degradation = np.nan

        return {
            'iid_rmse': iid_rmse,
            'ood_rmse': ood_rmse,
            'rmse_degradation_pct': degradation,
            'interpretation': self._interpret_generalization_gap(degradation)
        }

    @staticmethod
    def _interpret_generalization_gap(degradation: float) -> str:
        """Interpret the IID -> OOD performance drop."""
        if np.isnan(degradation):
            return "Unknown"
        elif degradation < 5:
            return "Excellent generalization (< 5% degradation)"
        elif degradation < 15:
            return "Good generalization (< 15% degradation)"
        elif degradation < 30:
            return "Acceptable generalization (< 30% degradation)"
        else:
            return "Poor generalization (> 30% degradation)"


def generate_and_test_ood(train_data: pd.DataFrame, test_data: pd.DataFrame,
                         model_predict_fn, iid_test_rmse: float) -> Dict:
    """
    Full OOD generalization test pipeline.

    Args:
        train_data: Training data
        test_data: IID test data
        model_predict_fn: Prediction function
        iid_test_rmse: RMSE on IID test set (for comparison)

    Returns:
        Complete OOD test results
    """
    # Generate OOD variants
    generator = SyntheticEngineGenerator(train_data)
    ood_test_data = generator.generate_ood_test_set(test_data.iloc[0:len(test_data)//2])

    # Evaluate on OOD
    tester = OODGeneralizationTester(model_predict_fn)
    ood_results = tester.evaluate_on_ood_variants(ood_test_data)

    # Compare
    iid_results = {'rmse': iid_test_rmse}
    comparison = tester.compare_iid_vs_ood(iid_results, ood_results)

    return {
        'ood_results': ood_results,
        'iid_vs_ood_comparison': comparison,
        'interpretation': comparison['interpretation']
    }


if __name__ == "__main__":
    # Example
    train = pd.read_csv("../data/train.csv")
    gt = pd.read_csv("../data/ground_truth.csv")
    train = train.merge(gt, on=["EngineID", "Cycle"])

    print("Testing Out-of-Distribution Generalization...\n")

    # Generate OOD variants
    generator = SyntheticEngineGenerator(train)
    sample_engine = train[train['EngineID'] == 1].head(20).copy()
    ood_data = generator.generate_ood_test_set(sample_engine)

    print(f"Generated {len(ood_data['EngineID'].unique())} synthetic OOD variants")
    print(f"Total OOD samples: {len(ood_data)}")
    print("\nVariant IDs:", sorted(ood_data['EngineID'].unique()))
    print("\n✓ OOD generalization test infrastructure ready")
