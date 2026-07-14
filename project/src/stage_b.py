"""
Stage B: data-driven layer.

(1) Health estimation: per-engine adaptive baseline (health trend vs cycle,
    fit only from that engine's own TRAIN cycles) + a physics-residual
    correction model (captures any sensor-driven deviation from the
    expected schedule -- the anomaly/early-warning signal) + uncertainty
    quantification via bootstrap ensembling.
(2) Performance prediction: Thrust_N / TSFC regressed directly from
    operating condition + gas-path sensors (these do NOT follow the cycle
    trend -- they are driven by instantaneous flight condition, confirmed
    by EDA correlation check).

Uses the OFFICIAL train.csv / test.csv split (joined to ground_truth.csv),
exactly mirroring how the PS's own evaluation would be run.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from physics_layer import run_physics_layer

HEALTH_COLS = ["CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"]
PHYSICS_FEATS = ["eta_c_raw", "eta_t_raw", "combustor_pressure_retention", "combustor_specific_rise"]
RAW_SENSOR_FEATS = ["Altitude_m", "Mach", "Tamb_K", "Pamb_Pa", "RPM_rev_min", "FuelFlow_kg_s",
                     "P2_Pa", "T2_K", "P3_Pa", "T3_K", "P4_Pa", "T4_K"]


def load_split():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    gt = pd.read_csv("../data/ground_truth.csv")
    train = train.merge(gt, on=["EngineID", "Cycle"])
    test_labeled = test.merge(gt, on=["EngineID", "Cycle"])
    train = run_physics_layer(train)
    test_labeled = run_physics_layer(test_labeled)
    return train, test_labeled


def fit_per_engine_baseline(train: pd.DataFrame, target: str):
    """Linear health-vs-cycle trend fit independently per engine, using only
    that engine's own TRAIN cycles. Returns dict[engine_id] -> (slope, intercept, resid_std)."""
    models = {}
    for eng, sub in train.groupby("EngineID"):
        X = sub[["Cycle"]].values
        y = sub[target].values
        m = LinearRegression().fit(X, y)
        resid = y - m.predict(X)
        models[eng] = (m, resid.std())
    return models


def baseline_predict(models, engine_ids, cycles):
    preds, stds = [], []
    for eng, cyc in zip(engine_ids, cycles):
        m, resid_std = models[eng]
        preds.append(m.predict([[cyc]])[0])
        stds.append(resid_std)
    return np.array(preds), np.array(stds)


def run_health_pipeline(train, test, target, n_bootstrap=50):
    baseline_models = fit_per_engine_baseline(train, target)
    base_pred_train, _ = baseline_predict(baseline_models, train["EngineID"], train["Cycle"])
    base_pred_test, base_std_test = baseline_predict(baseline_models, test["EngineID"], test["Cycle"])

    resid_train = train[target].values - base_pred_train
    X_train = train[PHYSICS_FEATS + RAW_SENSOR_FEATS].values
    X_test = test[PHYSICS_FEATS + RAW_SENSOR_FEATS].values

    # bootstrap ensemble of residual-correction models -> point estimate + uncertainty
    rng = np.random.default_rng(42)
    boot_preds = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(X_train), len(X_train))
        model = GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, random_state=int(rng.integers(0, 1e6)))
        model.fit(X_train[idx], resid_train[idx])
        boot_preds.append(model.predict(X_test))
    boot_preds = np.array(boot_preds)  # (n_bootstrap, n_test)

    resid_correction_mean = boot_preds.mean(axis=0)
    resid_correction_std = boot_preds.std(axis=0)

    final_pred = base_pred_test + resid_correction_mean
    # combine baseline residual spread and bootstrap spread in quadrature for a 90% interval
    total_std = np.sqrt(base_std_test ** 2 + resid_correction_std ** 2)
    lower = final_pred - 1.645 * total_std
    upper = final_pred + 1.645 * total_std

    y_true = test[target].values
    rmse = np.sqrt(mean_squared_error(y_true, final_pred))
    r2 = r2_score(y_true, final_pred)
    coverage = np.mean((y_true >= lower) & (y_true <= upper))

    # also report baseline-only (no physics residual) for comparison
    rmse_baseline_only = np.sqrt(mean_squared_error(y_true, base_pred_test))

    return {
        "target": target,
        "rmse": rmse,
        "r2": r2,
        "rmse_baseline_only": rmse_baseline_only,
        "uq_90pct_coverage": coverage,
        "pred": final_pred,
        "lower": lower,
        "upper": upper,
        "true": y_true,
    }


def run_performance_pipeline(train, test, target):
    X_train = train[RAW_SENSOR_FEATS].values
    y_train = train[target].values
    X_test = test[RAW_SENSOR_FEATS].values
    y_test = test[target].values

    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    return {"target": target, "rmse": rmse, "r2": r2, "pred": pred, "true": y_test}


if __name__ == "__main__":
    train, test = load_split()
    print(f"Train rows: {len(train)}  Test rows: {len(test)}\n")

    print("=== Health estimation (baseline + physics-residual + bootstrap UQ) ===")
    health_results = {}
    for target in HEALTH_COLS:
        res = run_health_pipeline(train, test, target)
        health_results[target] = res
        print(f"{target:18s} RMSE={res['rmse']:.5f}  R2={res['r2']:.4f}  "
              f"(baseline-only RMSE={res['rmse_baseline_only']:.5f})  "
              f"90% CI coverage={res['uq_90pct_coverage']*100:.1f}%")

    print("\n=== Performance prediction (direct sensor regression) ===")
    perf_results = {}
    for target in ["Thrust_N", "TSFC_g_N_s"]:
        res = run_performance_pipeline(train, test, target)
        perf_results[target] = res
        print(f"{target:12s} RMSE={res['rmse']:.4f}  R2={res['r2']:.4f}")

    # persist test-set predictions for the dashboard
    out = test[["EngineID", "Cycle"]].copy()
    for t in HEALTH_COLS:
        out[f"{t}_true"] = health_results[t]["true"]
        out[f"{t}_pred"] = health_results[t]["pred"]
        out[f"{t}_lower"] = health_results[t]["lower"]
        out[f"{t}_upper"] = health_results[t]["upper"]
    for t in ["Thrust_N", "TSFC_g_N_s"]:
        out[f"{t}_true"] = perf_results[t]["true"]
        out[f"{t}_pred"] = perf_results[t]["pred"]
    out.to_csv("../data/stage_b_predictions.csv", index=False)
    print("\nSaved stage_b_predictions.csv")
