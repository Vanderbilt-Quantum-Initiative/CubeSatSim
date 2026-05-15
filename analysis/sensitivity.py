"""
analysis/sensitivity.py — Local parameter sensitivity around a baseline scenario.

Computes the normalised elasticity of ℓ_finite (and R) with respect to each
parameter using central finite differences.  Outputs a tornado plot sorted by
absolute impact.

    elasticity(p) = (Δℓ / ℓ_baseline) / (Δp / p_baseline)

Interpretation: elasticity = 2.0 means a 1% increase in p raises ℓ_finite by 2%.
Negative elasticity means the parameter and ℓ_finite move in opposite directions.

Usage
-----
    from analysis.sensitivity import compute_sensitivity, plot_tornado

    sens = compute_sensitivity("optimistic")
    fig  = plot_tornado(sens, "optimistic")
    fig.savefig("tornado.png", dpi=150)
"""

from __future__ import annotations

import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from params.registry import ParameterRegistry
from params.scenarios import get_scenario
from orbit.pass_sim import simulate_pass

_T_START = datetime(2025, 1, 1, 7, 0, tzinfo=timezone.utc)
_T_END   = datetime(2025, 1, 1, 8, 30, tzinfo=timezone.utc)
_COLORS  = plt.cm.tab10.colors
_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "viz", "out", "analysis")

# Parameters to include in the sensitivity analysis.
# Each entry: (registry_key, display_label, is_str_param)
_PARAMS: list[tuple[str, str, bool]] = [
    ("D_rx",       "D_rx  (aperture)",          False),
    ("w0",         "w₀  (beam waist)",           False),
    ("sigma_pnt",  "σ_pnt  (pointing jitter)",   False),
    ("eta_tx",     "η_tx  (TX optics)",          False),
    ("eta_rx",     "η_rx  (RX optics)",          False),
    ("eta_det",    "η_det  (detector QE)",       False),
    ("alpha",      "α  (extinction coeff)",      False),
    ("Cn2_0",      "C_n²(0)  (turbulence)",      False),
    ("e_opt",      "e_opt  (optical QBER floor)",False),
    ("mu",         "μ  (signal intensity)",      False),
    ("nu",         "ν  (decoy intensity)",       False),
    ("f_clock",    "f_clock  (clock rate)",      False),
    ("r_PE",       "r_PE  (PE fraction)",        False),
    ("epsilon_PA", "ε_PA  (security param)",     False),
]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _run_once(overrides: dict[str, Any], scenario: str,
              decoy_mode: str) -> tuple[float, float]:
    """Return (ell_finite, R) for a given set of overrides."""
    reg = ParameterRegistry()
    reg.update(get_scenario(scenario))
    for k, v in overrides.items():
        try:
            reg.set(k, v)
        except (KeyError, ValueError):
            pass
    try:
        result = simulate_pass(reg, t_start=_T_START, t_end=_T_END,
                               decoy_mode=decoy_mode)
        return result.ell_finite, result.key_rate.R
    except RuntimeError:
        return 0.0, 0.0


def compute_sensitivity(
    scenario: str = "optimistic",
    rel_step: float = 0.05,
    decoy_mode: str = "finite",
    params: list[tuple[str, str, bool]] | None = None,
) -> dict[str, dict]:
    """Compute normalised elasticity of ℓ_finite and R for each parameter.

    Parameters
    ----------
    scenario
        Baseline scenario name.  Should be one with ℓ_finite > 0 so that
        elasticities are well-defined (use 'optimistic').
    rel_step
        Fractional step size for finite differences (default 5%).
    decoy_mode
        'finite' or 'asymptotic'.
    params
        List of (key, label, is_str) tuples.  Defaults to _PARAMS.

    Returns
    -------
    dict mapping param_key → {
        'label': str,
        'baseline': float,
        'ell_lo': float, 'ell_hi': float,
        'R_lo': float,   'R_hi': float,
        'ell_elasticity': float,
        'R_elasticity': float,
    }
    """
    if params is None:
        params = _PARAMS

    reg0 = ParameterRegistry()
    reg0.update(get_scenario(scenario))

    ell_base, R_base = _run_once({}, scenario, decoy_mode)
    print(f"  Baseline ({scenario}): ℓ_finite={ell_base:.0f}  R={R_base:.4e}")

    results: dict[str, dict] = {}

    for key, label, is_str in params:
        if is_str:
            continue
        try:
            p0 = reg0.get(key)
        except KeyError:
            continue

        if p0 == 0.0:
            continue

        dp   = abs(p0) * rel_step
        p_lo = p0 - dp
        p_hi = p0 + dp

        ell_lo, R_lo = _run_once({key: p_lo}, scenario, decoy_mode)
        ell_hi, R_hi = _run_once({key: p_hi}, scenario, decoy_mode)

        # Central-difference elasticity (normalised)
        if ell_base > 0:
            ell_elast = ((ell_hi - ell_lo) / ell_base) / (2 * rel_step)
        else:
            ell_elast = (ell_hi - ell_lo) / max(abs(ell_hi - ell_lo), 1)

        if R_base != 0:
            R_elast = ((R_hi - R_lo) / R_base) / (2 * rel_step)
        else:
            R_elast = 0.0

        results[key] = {
            "label":          label,
            "baseline":       p0,
            "ell_lo":         ell_lo,
            "ell_hi":         ell_hi,
            "R_lo":           R_lo,
            "R_hi":           R_hi,
            "ell_elasticity": ell_elast,
            "R_elasticity":   R_elast,
        }
        print(f"  {key:15s}  ε(ℓ)={ell_elast:+.2f}  ε(R)={R_elast:+.2f}")

    return results


# ---------------------------------------------------------------------------
# Tornado plot
# ---------------------------------------------------------------------------

def plot_tornado(
    sensitivities: dict[str, dict],
    scenario: str = "optimistic",
    metric: str = "ell_elasticity",
    top_n: int = 12,
) -> plt.Figure:
    """Tornado plot of normalised elasticities sorted by absolute magnitude.

    Parameters
    ----------
    sensitivities
        Output of compute_sensitivity().
    metric
        'ell_elasticity' (ℓ_finite) or 'R_elasticity' (key fraction R).
    top_n
        Show only the top-N parameters by absolute elasticity.
    """
    # Sort by absolute elasticity
    items = sorted(sensitivities.items(),
                   key=lambda kv: abs(kv[1][metric]), reverse=True)[:top_n]

    labels      = [v["label"]  for _, v in items]
    elasticities= [v[metric]   for _, v in items]

    fig, ax = plt.subplots(figsize=(9, max(4, len(items) * 0.52 + 1.5)))

    colors = [_COLORS[0] if e >= 0 else _COLORS[3] for e in elasticities]
    bars = ax.barh(range(len(items)), elasticities, color=colors, alpha=0.85, height=0.65)

    for i, (bar, val) in enumerate(zip(bars, elasticities)):
        ha = "left" if val >= 0 else "right"
        offset = 0.05 if val >= 0 else -0.05
        ax.text(val + offset, i, f"{val:+.2f}", va="center", ha=ha, fontsize=9)

    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlabel("Normalised elasticity  (Δℓ/ℓ) / (Δp/p)", fontsize=11)

    metric_label = "ℓ_finite" if metric == "ell_elasticity" else "R"
    ax.set_title(
        f"Sensitivity Tornado — {metric_label} vs hardware/channel parameters\n"
        f"scenario: {scenario}   (central finite difference, ±5%)",
        fontsize=10,
    )

    from matplotlib.patches import Patch
    legend = [Patch(color=_COLORS[0], label="Positive: ↑p → ↑ℓ"),
              Patch(color=_COLORS[3], label="Negative: ↑p → ↓ℓ")]
    ax.legend(handles=legend, fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(_OUT_DIR, exist_ok=True)
    print("Computing sensitivity (optimistic scenario)...")
    sens = compute_sensitivity("optimistic", rel_step=0.05, decoy_mode="finite")

    fig = plot_tornado(sens, "optimistic", metric="ell_elasticity")
    path = os.path.join(_OUT_DIR, "tornado_ell.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")

    fig2 = plot_tornado(sens, "optimistic", metric="R_elasticity")
    path2 = os.path.join(_OUT_DIR, "tornado_R.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved → {path2}")
    plt.show()
