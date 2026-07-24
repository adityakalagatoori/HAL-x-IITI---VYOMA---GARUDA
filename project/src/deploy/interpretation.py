"""
Feature Importance Analysis (M4)

Understand which sensors matter most for health estimation.
Uses SHAP + permutation importance to identify critical sensors.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from typing import Dict


def compute_permutation_importance(X_test: np.ndarray, y_test: np.ndarray,
                                  model: GradientBoostingRegressor,
                                  feature_names: list) -> Dict[str, float]:
    """
    Compute permutation feature importance.
    """
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, result.importances_mean)}
    # Sort by importance
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def feature_importance_report(importance_dict: Dict[str, float]) -> str:
    """Generate readability report."""
    report = []
    report.append("=" * 60)
    report.append("FEATURE IMPORTANCE ANALYSIS")
    report.append("=" * 60)
    total_importance = sum(importance_dict.values())
    cumulative = 0
    for i, (feat, imp) in enumerate(importance_dict.items(), 1):
        cumulative += imp
        pct = (imp / total_importance * 100) if total_importance > 0 else 0
        cum_pct = (cumulative / total_importance * 100) if total_importance > 0 else 0
        report.append(f"{i:2d}. {feat:20s} {pct:6.2f}%  (cumulative: {cum_pct:6.2f}%)")
    report.append("=" * 60)
    return "\n".join(report)


if __name__ == "__main__":
    print("Feature Importance module ready")
