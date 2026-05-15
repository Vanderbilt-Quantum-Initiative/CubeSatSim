"""
params/scenarios.py — Named scenario definitions as deltas from the baseline.

A scenario is a named dict of parameter overrides applied on top of the
baseline defaults defined in params/definitions.py.  Physics modules never
see scenarios — only the typed config objects produced by the registry.

Usage
-----
    from params.scenarios import get_scenario
    from params.registry import ParameterRegistry

    reg = ParameterRegistry()
    reg.update(get_scenario("vqi_400km"))
    channel = reg.build_channel()
    source  = reg.build_source()

Adding a scenario
-----------------
    Add an entry to SCENARIOS below.  Keys must match names in definitions.py.
    Only override what differs from baseline — the rest inherits from defaults.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {

    # ── Baseline: design midpoint, no overrides ───────────────────────────────
    "baseline": {},

    # ── VQI primary scenario: 400 km, 850 nm, D_rx=1 m, SNSPD ───────────────
    "vqi_400km": {
        "h_orbit":     400e3,
        "lambda_":     850e-9,
        "w0":          0.08,
        "D_rx":        1.0,
        "eta_tx":      0.50,
        "eta_rx":      0.45,
        "mu":          0.5,
        "nu":          0.1,
        "P_mu":        0.6,
        "P_nu":        0.3,
        "P_X":         0.9,
        "f_clock":     100e6,
        "alpha":       0.329 / 400e3,
        "Cn2_0":       1.7e-14,
        "sigma_pnt":   2e-6,
        "e_opt":       0.03,
        "dark_count_rate": 500.0,
        "eta_det":     0.60,
        "tau_dead":    50e-9,
        "r_PE":        0.10,
        "ec_algorithm":"ldpc",
        "epsilon_PA":  1e-10,
        "rf_bandwidth":10e6,
        "gs_lat":      36.1,
        "gs_lon":      -86.7,
        "gs_alt_m":    182.0,
        "theta_el_min":10.0,
        "t_acq":       15.0,
        "dt_sim":      1.0,
    },

    # ── Bourgoin 2013 validation: 600 km, 670 nm, D_rx=0.5 m ─────────────────
    "bourgoin_2013": {
        "h_orbit":     600e3,
        "lambda_":     670e-9,
        "w0":          0.06,
        "D_rx":        0.50,
        "eta_tx":      0.50,
        "eta_rx":      0.45,
        "mu":          0.5,
        "nu":          0.1,
        "P_mu":        0.6,
        "P_nu":        0.3,
        "P_X":         0.9,
        "f_clock":     300e6,
        "alpha":       0.329 / 600e3,
        "Cn2_0":       1.7e-14,
        "sigma_pnt":   2e-6,
        "e_opt":       0.03,
        "dark_count_rate": 20.0,
        "eta_det":     0.65,
        "tau_dead":    50e-9,
        "r_PE":        0.10,
        "ec_algorithm":"ldpc",
        "epsilon_PA":  1e-10,
        "rf_bandwidth":10e6,
        "gs_lat":      36.1,
        "gs_lon":      -86.7,
        "gs_alt_m":    182.0,
        "theta_el_min":10.0,
        "t_acq":       15.0,
        "dt_sim":      1.0,
    },

    # ── Optimistic: D_rx=1.5 m, SNSPD, low noise ─────────────────────────────
    "optimistic": {
        "h_orbit":     400e3,
        "lambda_":     850e-9,
        "w0":          0.10,    # near analytic optimum w0* ≈ 9.6 cm
        "D_rx":        1.5,
        "eta_tx":      0.60,
        "eta_rx":      0.55,
        "mu":          0.5,
        "nu":          0.1,
        "P_mu":        0.6,
        "P_nu":        0.3,
        "P_X":         0.9,
        "f_clock":     100e6,
        "alpha":       0.20 / 400e3,  # good atmospheric site
        "Cn2_0":       1e-15,
        "sigma_pnt":   1e-6,
        "e_opt":       0.02,
        "dark_count_rate": 100.0,
        "eta_det":     0.85,
        "tau_dead":    10e-9,
        "r_PE":        0.10,
        "ec_algorithm":"ldpc",
        "epsilon_PA":  1e-10,
        "rf_bandwidth":10e6,
        "theta_el_min":10.0,
        "t_acq":       10.0,
        "dt_sim":      1.0,
    },

    # ── Conservative: D_rx=0.5 m, Si-SPAD, typical atmosphere ────────────────
    "conservative": {
        "h_orbit":     400e3,
        "lambda_":     850e-9,
        "w0":          0.08,
        "D_rx":        0.50,
        "eta_tx":      0.40,
        "eta_rx":      0.35,
        "mu":          0.5,
        "nu":          0.1,
        "P_mu":        0.6,
        "P_nu":        0.3,
        "P_X":         0.9,
        "f_clock":     50e6,
        "alpha":       0.40 / 400e3,
        "Cn2_0":       5e-14,
        "sigma_pnt":   5e-6,
        "e_opt":       0.05,
        "dark_count_rate": 1000.0,
        "eta_det":     0.50,
        "tau_dead":    50e-9,
        "r_PE":        0.10,
        "ec_algorithm":"ldpc",
        "epsilon_PA":  1e-10,
        "rf_bandwidth":10e6,
        "theta_el_min":20.0,
        "t_acq":       20.0,
        "dt_sim":      1.0,
    },

    # ── ISS-like: 400 km, 51.6° inclination ──────────────────────────────────
    "iss_like": {
        "h_orbit":     400e3,
        "inclination": 51.6,
        "lambda_":     850e-9,
        "w0":          0.08,
        "D_rx":        1.0,
        "eta_tx":      0.50,
        "eta_rx":      0.45,
        "mu":          0.5,
        "nu":          0.1,
        "P_mu":        0.6,
        "P_nu":        0.3,
        "P_X":         0.9,
        "f_clock":     100e6,
        "alpha":       0.329 / 400e3,
        "Cn2_0":       1.7e-14,
        "sigma_pnt":   2e-6,
        "e_opt":       0.03,
        "dark_count_rate": 500.0,
        "eta_det":     0.60,
        "tau_dead":    50e-9,
        "r_PE":        0.10,
        "ec_algorithm":"ldpc",
        "epsilon_PA":  1e-10,
        "rf_bandwidth":10e6,
        "gs_lat":      36.1,
        "gs_lon":      -86.7,
        "gs_alt_m":    182.0,
        "theta_el_min":10.0,
        "t_acq":       15.0,
        "dt_sim":      1.0,
    },
}


def get_scenario(name: str) -> dict:
    """Return the override dict for the named scenario.

    Raises KeyError if the scenario is not registered.
    """
    if name not in SCENARIOS:
        available = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario {name!r}.  Available: {available}.")
    return dict(SCENARIOS[name])


def list_scenarios() -> list[str]:
    """Return names of all registered scenarios."""
    return sorted(SCENARIOS)
