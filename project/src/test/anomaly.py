"""
Validates the core claim we've been making narratively but hadn't tested:
"the physics-informed residual layer catches abnormal degradation that a
pure cycle-based model cannot, because it only sees Cycle number."

Experiment:
  1. Take Engine 1's real trajectory (30 cycles, real sensor values).
  2. From cycle 15 onward, inject a physically-realistic ADDITIONAL shock
     (simulating sudden compressor fouling/FOD): raise T2 by a fixed
     increment for the same pressure ratio P2/P1, i.e. genuinely lower
     compressor efficiency than the smooth schedule implies. This is NOT
     a label change -- it only touches the sensor stream, which is all a
     real deployed system would ever see.
  3. Tier 1 (exact cycle-based reconstruction) predicts purely from Cycle
     number -- by construction, it CANNOT see the shock at all.
  4. Tier 2's physics layer (Stage A) recomputes eta_c_raw from the
     (shocked) sensor stream every cycle. An "expected normal envelope"
     for eta_c_raw is fit on pooled healthy training data (all engines,
     all cycles) as a function of operating condition only. Post-shock
     cycles should show eta_c_raw falling well outside that envelope --
     a concrete, testable anomaly flag.

This is what a black-box cycle-position or LSTM-on-raw-sensors model
without a physics layer structurally cannot distinguish from normal
scatter -- because it was never taught what a physically consistent
compressor efficiency looks like in the first place.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from physics_layer import stagnation_conditions, compressor_efficiency, GAMMA_AIR

SHOCK_CYCLE = 15
SHOCK_DELTA_T2_K = 35.0  # extra temperature rise at compressor exit -> efficiency loss


def inject_shock(df_engine: pd.DataFrame) -> pd.DataFrame:
    shocked = df_engine.copy()
    mask = shocked["Cycle"] >= SHOCK_CYCLE
    shocked.loc[mask, "T2_K"] = shocked.loc[mask, "T2_K"] + SHOCK_DELTA_T2_K
    return shocked


def compute_eta_c_raw(df):
    T01, P01 = stagnation_conditions(df["Tamb_K"], df["Pamb_Pa"], df["Mach"])
    eta, _ = compressor_efficiency(T01, P01, df["P2_Pa"], df["T2_K"])
    return eta


if __name__ == "__main__":
    train = pd.read_csv("../data/train.csv")
    complete = pd.read_csv("../data/turbojet_complete_dataset.csv")

    # "normal operating envelope" for eta_c_raw, fit on pooled TRAIN data only
    # (i.e. only ever-normal, never-shocked engines -- exactly what we'd have
    # in a real fleet before any fault has occurred)
    train = train.copy()
    train["eta_c_raw"] = compute_eta_c_raw(train)
    env_feats = ["Altitude_m", "Mach", "Tamb_K", "Pamb_Pa"]
    envelope_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    envelope_model.fit(train[env_feats], train["eta_c_raw"])
    resid = train["eta_c_raw"] - envelope_model.predict(train[env_feats])
    resid_std = resid.std()

    eng1 = complete[complete.EngineID == 1].sort_values("Cycle").reset_index(drop=True)
    shocked = inject_shock(eng1)

    for label, data in [("UNSHOCKED (original)", eng1), ("SHOCKED from cycle 15", shocked)]:
        data = data.copy()
        data["eta_c_raw"] = compute_eta_c_raw(data)
        expected = envelope_model.predict(data[env_feats])
        z = (data["eta_c_raw"] - expected) / resid_std
        flagged = z.abs() > 3.0
        print(f"\n=== {label} ===")
        print(f"Cycles flagged as anomalous (|z|>3): {list(data.loc[flagged, 'Cycle'])}")
        print(f"Max |z-score| pre-cycle-15: {z[data.Cycle < SHOCK_CYCLE].abs().max():.2f}")
        print(f"Max |z-score| post-cycle-15: {z[data.Cycle >= SHOCK_CYCLE].abs().max():.2f}")

    print("\n=== Tier 1 (cycle-only) blindness check ===")
    print("Tier 1 predicts health purely as f(Cycle); it uses NO sensor columns at all,")
    print("so its prediction for the shocked engine is IDENTICAL to the unshocked engine")
    print("at every cycle -- it cannot detect the injected fault by construction.")
