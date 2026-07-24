"""
Stage B with Proper Train/Validation/Test Split

This version implements correct hyperparameter tuning:
- 60% training set (fit models)
- 20% validation set (tune hyperparameters)
- 20% test set (final evaluation, reported metrics only)

The official train.csv / test.csv split is respected:
- everything in official train.csv goes to our 60/20 split
- official test.csv is held as final test (no tuning on it)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from conformal_rul_wrapper import ConformalRULPredictor
from physics_layer import run_physics_layer

HEALTH_COLS = ["CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"]
PHYSICS_FEATS = ["eta_c_raw", "eta_t_raw", "combustor_pressure_retention", "combustor_specific_rise"]
RAW_SENSOR_FEATS = ["Altitude_m", "Mach", "Tamb_K", "Pamb_Pa", "RPM_rev_min", "FuelFlow_kg_s",
                     "P2_Pa", "T2_K", "P3_Pa", "T3_K", "P4_Pa", "T4_K"]
ALL_FEATS = ["Cycle"] + PHYSICS_FEATS + RAW_SENSOR_FEATS
MONOTONIC_CST = [-1] + [0] * (len(ALL_FEATS) - 1)


def load_split_with_proper_division():
    """
    Load data with proper train/val/test split.

    Official split:
    - train.csv -> split 60/20 (our train and our val)
    - test.csv -> our test (never tuned on)
    """
    train_official = pd.read_csv("../data/train.csv")
    test_official = pd.read_csv("../data/test.csv")
    gt = pd.read_csv("../data/ground_truth.csv")

    train_official = train_official.merge(gt, on=["EngineID", "Cycle"])
    test_official = test_official.merge(gt, on=["EngineID", "Cycle"])

    train_official = run_physics_layer(train_official)
    test_official = run_physics_layer(test_official)

    # Further split official train: 75% train, 25% val (of 80% of official data)
    # This gives us ~60% / 20% / 20% split overall
    train_sub, val_sub = train_test_split(
        train_official, test_size=0.25, random_state=42
    )

    return train_sub, val_sub, test_official


class HierarchicalBaseline:
    """Population trend + per-engine deviation, empirical-Bayes shrunk."""

    def fit(self, train: pd.DataFrame, target: str, shrinkage_k: float = 8.0):
        pop_coef = np.polyfit(train["Cycle"], train[target], 1)
        self.pop_slope, self.pop_intercept = pop_coef

        self.engine_models = {}
        for eng, sub in train.groupby("EngineID"):
            n = len(sub)
            if n >= 2:
                slope, intercept = np.polyfit(sub["Cycle"], sub[target], 1)
            else:
                slope, intercept = self.pop_slope, self.pop_intercept
            w = n / (n + shrinkage_k)
            shrunk_slope = w * slope + (1 - w) * self.pop_slope
            shrunk_intercept = w * intercept + (1 - w) * self.pop_intercept
            pred = shrunk_slope * sub["Cycle"] + shrunk_intercept
            resid_std = float(np.std(sub[target].values - pred.values))
            self.engine_models[eng] = (shrunk_slope, shrunk_intercept, resid_std)
        self.pop_resid_std = float(np.std(train[target] - (self.pop_slope * train["Cycle"] + self.pop_intercept)))
        return self

    def predict(self, engine_ids, cycles):
        preds, stds = [], []
        for eng, cyc in zip(engine_ids, cycles):
            if eng in self.engine_models:
                slope, intercept, resid_std = self.engine_models[eng]
            else:
                slope, intercept, resid_std = self.pop_slope, self.pop_intercept, self.pop_resid_std
            preds.append(slope * cyc + intercept)
            stds.append(resid_std)
        return np.array(preds), np.array(stds)


def tune_hyperparams_on_validation(train, val, target, param_grid=None):
    """
    Hyperparameter tuning on validation set only.
    Returns best hyperparams to then evaluate on test set.
    """
    if param_grid is None:
        param_grid = {
            'max_iter': [100, 150, 200],
            'max_depth': [2, 3, 4],
            'learning_rate': [0.01, 0.05, 0.1]
        }

    baseline = HierarchicalBaseline().fit(train, target)
    base_pred_train, _ = baseline.predict(train["EngineID"], train["Cycle"])
    base_pred_val, _ = baseline.predict(val["EngineID"], val["Cycle"])

    resid_train = train[target].values - base_pred_train
    X_train = train[ALL_FEATS].values
    X_val = val[ALL_FEATS].values
    y_val = val[target].values

    best_rmse = float('inf')
    best_params = None

    # Grid search on validation set
    for max_iter in param_grid['max_iter']:
        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                model = HistGradientBoostingRegressor(
                    max_iter=max_iter, max_depth=max_depth, learning_rate=learning_rate,
                    monotonic_cst=MONOTONIC_CST, random_state=42
                )
                model.fit(X_train, resid_train)
                resid_pred_val = model.predict(X_val)
                final_pred_val = base_pred_val + resid_pred_val
                rmse_val = np.sqrt(mean_squared_error(y_val, final_pred_val))

                if rmse_val < best_rmse:
                    best_rmse = rmse_val
                    best_params = {'max_iter': max_iter, 'max_depth': max_depth, 'learning_rate': learning_rate}

    return best_params


def run_health_pipeline_proper_split(train, val, test, target, alpha=0.10):
    """
    Train on train set, tune on val set, evaluate on test set.
    Uses Barber et al. (2024) conformal prediction with proper calibration.
    """
    # Fit baseline on train set only
    baseline = HierarchicalBaseline().fit(train, target)
    base_pred_train, _ = baseline.predict(train["EngineID"], train["Cycle"])
    base_pred_val, _ = baseline.predict(val["EngineID"], val["Cycle"])
    base_pred_test, _ = baseline.predict(test["EngineID"], test["Cycle"])

    resid_train = train[target].values - base_pred_train
    resid_val = val[target].values - base_pred_val
    X_train = train[ALL_FEATS].values
    X_val = val[ALL_FEATS].values
    X_test = test[ALL_FEATS].values

    # Use validation set to find best hyperparameters
    best_params = tune_hyperparams_on_validation(train, val, target)

    # Train final model on full train set (not val) with best hyperparams
    model = HistGradientBoostingRegressor(
        max_iter=best_params['max_iter'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        monotonic_cst=MONOTONIC_CST,
        random_state=42
    )
    model.fit(X_train, resid_train)

    # Use val set to calibrate conformal prediction (proper conformal calibration)
    conf_pred = ConformalRULPredictor(model, X_val, resid_val, alpha=alpha)

    # Predict on test set
    resid_pred_test, resid_lower, resid_upper = conf_pred.predict(X_test)
    final_pred = base_pred_test + resid_pred_test
    lower = base_pred_test + resid_lower
    upper = base_pred_test + resid_upper

    y_true = test[target].values
    rmse = np.sqrt(mean_squared_error(y_true, final_pred))
    r2 = r2_score(y_true, final_pred)
    coverage = np.mean((y_true >= lower) & (y_true <= upper))

    rmse_baseline_only = np.sqrt(mean_squared_error(y_true, base_pred_test))

    return {
        "target": target,
        "best_params": best_params,
        "rmse": rmse,
        "r2": r2,
        "rmse_baseline_only": rmse_baseline_only,
        "uq_90pct_coverage": coverage,
        "q_hat": conf_pred.q_hat,
        "pred": final_pred,
        "lower": lower,
        "upper": upper,
        "true": y_true,
    }


def run_performance_pipeline_proper_split(train, val, test, target):
    """Tune performance model on validation set, evaluate on test."""
    X_train = train[RAW_SENSOR_FEATS].values
    y_train = train[target].values
    X_val = val[RAW_SENSOR_FEATS].values
    y_val = val[target].values
    X_test = test[RAW_SENSOR_FEATS].values
    y_test = test[target].values

    # Tune on validation set
    best_rmse = float('inf')
    best_model = None
    for n_est in [100, 200, 300]:
        for depth in [2, 3, 4]:
            model = HistGradientBoostingRegressor(
                max_iter=n_est, max_depth=depth, learning_rate=0.05, random_state=42
            )
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_model = model

    # Evaluate on test
    pred_test = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2 = r2_score(y_test, pred_test)
    return {"target": target, "rmse": rmse, "r2": r2, "pred": pred_test, "true": y_test}


if __name__ == "__main__":
    train, val, test = load_split_with_proper_division()
    print(f"Train rows: {len(train)}  Val rows: {len(val)}  Test rows: {len(test)}\n")

    print("=== Health Estimation (Proper 60/20/20 Split) ===")
    health_results = {}
    for target in HEALTH_COLS:
        res = run_health_pipeline_proper_split(train, val, test, target)
        health_results[target] = res
        print(f"{target:18s} Best params: {res['best_params']}")
        print(f"  RMSE={res['rmse']:.5f}  R2={res['r2']:.4f}  "
              f"Coverage={res['uq_90pct_coverage']*100:.1f}%  q_hat={res['q_hat']:.5f}")

    print("\n=== Performance Prediction (Proper Split) ===")
    perf_results = {}
    for target in ["Thrust_N", "TSFC_g_N_s"]:
        res = run_performance_pipeline_proper_split(train, val, test, target)
        perf_results[target] = res
        print(f"{target:12s} RMSE={res['rmse']:.4f}  R2={res['r2']:.4f}")

    # Save predictions
    out = test[["EngineID", "Cycle"]].copy()
    for t in HEALTH_COLS:
        out[f"{t}_true"] = health_results[t]["true"]
        out[f"{t}_pred"] = health_results[t]["pred"]
        out[f"{t}_lower"] = health_results[t]["lower"]
        out[f"{t}_upper"] = health_results[t]["upper"]
    for t in ["Thrust_N", "TSFC_g_N_s"]:
        out[f"{t}_true"] = perf_results[t]["true"]
        out[f"{t}_pred"] = perf_results[t]["pred"]
    out.to_csv("../data/stage_b_proper_split_predictions.csv", index=False)
    print("\nSaved stage_b_proper_split_predictions.csv")
