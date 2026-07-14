"""
Stage A: 0-D Brayton-cycle gas-path physics layer.

Per-cycle, closed-form inversion of measured station data (P,T at inlet/
compressor-exit/combustor-exit/turbine-exit) into interpretable component
health parameters -- no training required. This is the physics half of the
hybrid digital twin (see PS-2, tasks 4.1/4.2).

Station map (matches PS-2 Section 3.2 sensor schema exactly):
  1 = ambient/inlet   (Tamb, Pamb)
  2 = compressor exit (P2, T2)
  3 = combustor exit / turbine inlet (P3, T3)
  4 = turbine exit    (P4, T4)
"""
import numpy as np
import pandas as pd

GAMMA_AIR = 1.4
GAMMA_GAS = 1.33


def stagnation_conditions(Tamb, Pamb, Mach, gamma=GAMMA_AIR):
    """Ram-corrected stagnation T0/P0 at the inlet (station 1)."""
    factor = 1 + (gamma - 1) / 2 * Mach ** 2
    T01 = Tamb * factor
    P01 = Pamb * factor ** (gamma / (gamma - 1))
    return T01, P01


def compressor_efficiency(T01, P01, P2, T2, gamma=GAMMA_AIR):
    """Isentropic efficiency eta_c = (T2s - T01) / (T2 - T01), using
    ram-corrected inlet stagnation conditions (not static ambient) since
    Altitude/Mach vary every cycle and static ambient alone injects flight-
    condition noise into the efficiency estimate."""
    pr = P2 / P01
    T2s = T01 * pr ** ((gamma - 1) / gamma)
    denom = (T2 - T01)
    eta = np.where(np.abs(denom) > 1e-6, (T2s - T01) / denom, np.nan)
    return eta, T2s


def turbine_efficiency(P3, T3, P4, T4, gamma=GAMMA_GAS):
    """Isentropic efficiency eta_t = (T3 - T4) / (T3 - T4s)."""
    pr = P4 / P3
    T4s = T3 * pr ** ((gamma - 1) / gamma)
    denom = (T3 - T4s)
    eta = np.where(np.abs(denom) > 1e-6, (T3 - T4) / denom, np.nan)
    return eta, T4s


def combustor_performance_proxy(P2, T2, P3, T3, fuel_flow):
    """
    No air-mass-flow sensor is provided, so a full fuel-air energy balance
    (m_f * LHV * eta_b = m_a*cp_g*T3 - m_a*cp_a*T2) is not directly solvable.
    Proxy combines:
      - pressure-retention ratio P3/P2 (combustor pressure loss -> should be
        close to 1, degrades toward <1 with liner/duct issues)
      - specific temperature rise per unit fuel flow, (T3-T2)/fuel_flow
        (higher = more effective heat release per kg of fuel burned)
    Both are directly measurable and documented as an engineering assumption
    per PS-2 Section 7 (mass-flow sensor not in the given schema).
    """
    pressure_retention = P3 / P2
    specific_rise = (T3 - T2) / fuel_flow
    return pressure_retention, specific_rise


def run_physics_layer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    T01, P01 = stagnation_conditions(df["Tamb_K"], df["Pamb_Pa"], df["Mach"])
    eta_c, T2s = compressor_efficiency(T01, P01, df["P2_Pa"], df["T2_K"])
    eta_t, T4s = turbine_efficiency(df["P3_Pa"], df["T3_K"], df["P4_Pa"], df["T4_K"])
    pr, spec_rise = combustor_performance_proxy(df["P2_Pa"], df["T2_K"], df["P3_Pa"], df["T3_K"], df["FuelFlow_kg_s"])

    out["eta_c_raw"] = eta_c
    out["eta_t_raw"] = eta_t
    out["combustor_pressure_retention"] = pr
    out["combustor_specific_rise"] = spec_rise
    return out


if __name__ == "__main__":
    df = pd.read_csv("../data/turbojet_complete_dataset.csv")
    result = run_physics_layer(df)
    result.to_csv("../data/physics_layer_output.csv", index=False)

    print("=== Correlation: physics-recovered signal vs true ground-truth health ===")
    print(f"eta_c_raw                     vs CompressorHealth : "
          f"{result['eta_c_raw'].corr(result['CompressorHealth']):.4f}")
    print(f"eta_t_raw                     vs TurbineHealth    : "
          f"{result['eta_t_raw'].corr(result['TurbineHealth']):.4f}")
    print(f"combustor_pressure_retention  vs CombustorHealth  : "
          f"{result['combustor_pressure_retention'].corr(result['CombustorHealth']):.4f}")
    print(f"combustor_specific_rise       vs CombustorHealth  : "
          f"{result['combustor_specific_rise'].corr(result['CombustorHealth']):.4f}")

    print("\nSaved physics_layer_output.csv")
