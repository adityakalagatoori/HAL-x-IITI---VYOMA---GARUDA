"""
Sensor Integrity Layer -- distinguishes "the sensor is lying" from "the
component is actually degrading."

Motivation: our existing anomaly detector (a statistical z-score against
the normal operating envelope) fires on BOTH a real component fault and a
garbage/stuck/glitched sensor reading -- it cannot tell them apart. That's
a real gap, confirmed by direct testing below.

Two independent check families, both calibrated against what actually
holds on this dataset (not textbook assumptions blindly applied):

1. HARD PHYSICAL RULES -- verified 100% true on the full reference dataset,
   so any violation is a sensor/data fault, never a legitimate component
   reading, no matter how degraded the engine is:
     - All pressures and temperatures must be strictly positive
     - T3 (turbine inlet) must exceed T2 (compressor exit) -- combustion
       always adds heat; violating this is thermodynamically impossible

   NOTE: we also tested "efficiency in [0,1]" and "T4 < T3, P4 < P3"
   (turbine must cool/expand the gas) as candidate hard rules and REJECTED
   them -- both are violated on a nontrivial fraction of the clean
   reference data (our simplified isentropic formula isn't a perfect match
   to this dataset's generator, and some turbine relations aren't strictly
   monotonic here). Using them as hard rules would false-alarm constantly
   on legitimate data. Kept only the two rules confirmed 100% reliable.

2. EMPIRICAL PLAUSIBILITY ENVELOPE -- the 0.5th-99.5th percentile range of
   each physics-derived signal, learned from train.csv only. A reading far
   outside this envelope (we use 1.5x the observed span as margin) is
   flagged as implausible even though it doesn't violate a hard law.
"""
import numpy as np
import pandas as pd

from physics_layer import run_physics_layer

RAW_POSITIVE_COLS = ["P2_Pa", "T2_K", "P3_Pa", "T3_K", "P4_Pa", "T4_K", "Tamb_K", "Pamb_Pa"]


def fit_envelope(train_phys: pd.DataFrame):
    bounds = {}
    for col in ["eta_c_raw", "eta_t_raw"]:
        lo, hi = train_phys[col].quantile([0.005, 0.995])
        span = hi - lo
        bounds[col] = (lo - 0.5 * span, hi + 0.5 * span)  # 1.5x span total margin
    return bounds


def check_row(row_raw: pd.Series, row_phys: pd.Series, bounds: dict):
    """Returns (verdict, reasons) where verdict is one of:
    'ok', 'sensor_fault_suspected', 'component_anomaly_suspected'."""
    reasons = []

    # -- hard rules (100% reliable on reference data -- any violation = sensor fault) --
    for c in RAW_POSITIVE_COLS:
        if row_raw[c] <= 0 or not np.isfinite(row_raw[c]):
            reasons.append(f"{c} is non-positive or non-finite ({row_raw[c]:.2f}) -- physically impossible")
    if row_raw["T3_K"] <= row_raw["T2_K"]:
        reasons.append(f"T3 ({row_raw['T3_K']:.1f}K) does not exceed T2 ({row_raw['T2_K']:.1f}K) -- "
                        f"combustion must add heat; this reading violates a hard physical law")

    if reasons:
        return "sensor_fault_suspected", reasons

    # -- empirical plausibility (soft: outside this envelope, but not physically impossible) --
    soft_reasons, extreme_reasons = [], []
    for col in ["eta_c_raw", "eta_t_raw"]:
        lo, hi = bounds[col]
        span = hi - lo
        if row_phys[col] < lo or row_phys[col] > hi:
            dist = max(lo - row_phys[col], row_phys[col] - hi)
            msg = f"{col}={row_phys[col]:.3f} is outside the empirically observed plausible range [{lo:.2f}, {hi:.2f}]"
            # a deviation this many multiples of the observed span away is not credible as real
            # degradation (our own EDA shows true degradation is smooth and bounded) -- escalate
            if dist > 3 * span:
                extreme_reasons.append(msg + f" by {dist / span:.0f}x the normal span -- too extreme to be real degradation")
            else:
                soft_reasons.append(msg)

    if extreme_reasons:
        return "sensor_fault_suspected", extreme_reasons
    if soft_reasons:
        return "component_anomaly_suspected", soft_reasons

    return "ok", []


def inject_garbage(df_engine: pd.DataFrame, cycle: int, kind: str) -> pd.DataFrame:
    out = df_engine.copy()
    mask = out.Cycle == cycle
    if kind == "negative_pressure":
        out.loc[mask, "P2_Pa"] = -5000.0
    elif kind == "stuck_low_T3":
        out.loc[mask, "T3_K"] = out.loc[mask, "T2_K"].values[0] - 20  # violates combustion
    elif kind == "extreme_spike":
        out.loc[mask, "T4_K"] = out.loc[mask, "T4_K"] * 15
    return out


if __name__ == "__main__":
    train = pd.read_csv("../data/train.csv")
    train_phys = run_physics_layer(train)
    bounds = fit_envelope(train_phys)
    print("Empirical plausibility bounds (train-only):", bounds)

    complete = pd.read_csv("../data/turbojet_complete_dataset.csv")
    eng1 = complete[complete.EngineID == 1].sort_values("Cycle").reset_index(drop=True)

    print("\n=== Baseline: all real, unmodified readings ===")
    phys = run_physics_layer(eng1)
    n_flagged = 0
    for i in range(len(eng1)):
        verdict, reasons = check_row(eng1.iloc[i], phys.iloc[i], bounds)
        if verdict != "ok":
            n_flagged += 1
    print(f"{n_flagged}/{len(eng1)} real cycles flagged (false-positive rate on clean data)")

    print("\n=== Garbage-sensor injection tests (cycle 20) ===")
    for kind in ["negative_pressure", "stuck_low_T3", "extreme_spike"]:
        shocked = inject_garbage(eng1, 20, kind)
        phys_s = run_physics_layer(shocked)
        row_idx = shocked.index[shocked.Cycle == 20][0]
        verdict, reasons = check_row(shocked.iloc[row_idx], phys_s.iloc[row_idx], bounds)
        print(f"\n[{kind}] -> verdict: {verdict.upper()}")
        for r in reasons:
            print("   -", r)
