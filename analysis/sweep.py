"""
analysis/sweep.py — 1D and 2D parameter sweeps over ℓ_finite (and R).

Sweeps vary one or two parameters across a grid while holding everything
else at the baseline scenario values.  Uses asymptotic decoy bounds by
default so the output is smooth (finite bounds can collapse to zero over
isolated grid points, obscuring the underlying trend).

    sweep_1d   — line plot: ℓ_finite vs one parameter
    sweep_2d   — filled contour heatmap: ℓ_finite vs two parameters

Usage
-----
    from analysis.sweep import sweep_1d, sweep_2d, plot_sweep_1d, plot_sweep_2d

    vals, ells = sweep_1d("optimistic", "D_rx", np.linspace(0.3, 2.0, 20))
    fig = plot_sweep_1d("D_rx", vals, ells)
"""

from __future__ import annotations

import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

from params.registry import ParameterRegistry
from params.scenarios import get_scenario
from orbit.pass_sim import simulate_pass
from params.definitions import PARAM_DEFS

_T_START = datetime(2025, 1, 1, 7, 0, tzinfo=timezone.utc)
_T_END   = datetime(2025, 1, 1, 8, 30, tzinfo=timezone.utc)
_COLORS  = plt.cm.tab10.colors
_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "viz", "out", "analysis")


def _label(key: str) -> str:
    """Human-readable label for a parameter key."""
    pdef = PARAM_DEFS.get(key)
    if pdef:
        unit = f"  ({pdef.unit})" if pdef.unit not in ("dimensionless", "string") else ""
        return f"{pdef.symbol}{unit}"
    return key


def _run(overrides: dict[str, Any], scenario: str,
         decoy_mode: str, dt: float) -> tuple[float, float]:
    """Return (ell_finite, R) for a registry with given overrides."""
    reg = ParameterRegistry()
    reg.update(get_scenario(scenario))
    reg.set("dt_sim", dt)
    for k, v in overrides.items():
        try:
            reg.set(k, v)
        except (KeyError, ValueError):
            pass
    try:
        r = simulate_pass(reg, t_start=_T_START, t_end=_T_END, decoy_mode=decoy_mode)
        return r.ell_finite, r.key_rate.R
    except RuntimeError:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# 1D sweep
# ---------------------------------------------------------------------------

def sweep_1d(
    scenario: str,
    param: str,
    values: np.ndarray,
    decoy_mode: str = "asymptotic",
    dt: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep one parameter and return (values, ell_finite_array, R_array).

    Uses dt=5s and asymptotic bounds by default for speed and smoothness.
    """
    ells, Rs = [], []
    for i, v in enumerate(values):
        ell, R = _run({param: v}, scenario, decoy_mode, dt)
        ells.append(ell)
        Rs.append(R)
        print(f"    {param}={v:.4g}  ℓ={ell:.0f}  R={R:.4e}")
    return np.array(values), np.array(ells), np.array(Rs)


def plot_sweep_1d(
    param: str,
    values: np.ndarray,
    ells: np.ndarray,
    Rs: np.ndarray,
    scenario: str = "",
    extra_lines: list[tuple[np.ndarray, np.ndarray, str]] | None = None,
) -> plt.Figure:
    """Two-panel plot: ℓ_finite (top) and R (bottom) vs swept parameter."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    xlabel = _label(param)

    # ── ℓ_finite panel ────────────────────────────────────────────────────
    ax1.plot(values, ells / 1e3, color=_COLORS[0], lw=2, label=scenario or "baseline")
    if extra_lines:
        for xs, ys, lbl in extra_lines:
            ax1.plot(xs, ys / 1e3, lw=1.8, ls="--", label=lbl)
    ax1.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax1.fill_between(values, 0, np.maximum(ells / 1e3, 0), alpha=0.12, color=_COLORS[0])
    ax1.set_ylabel("ℓ_finite  (kbits)", fontsize=11)
    ax1.set_title(f"Parameter Sweep: {xlabel}\n(scenario: {scenario})", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Mark go/no-go boundary
    pos = ells > 0
    if pos.any() and (~pos).any():
        # Find first positive value
        idx = np.where(pos)[0][0]
        ax1.axvline(values[idx], color="red", ls=":", lw=1.2, alpha=0.8)
        ax1.text(values[idx], ax1.get_ylim()[1] * 0.9,
                 f" go: {values[idx]:.3g}", fontsize=8, color="red")

    # ── R panel ──────────────────────────────────────────────────────────
    ax2.plot(values, Rs * 1e3, color=_COLORS[1], lw=2)
    ax2.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax2.fill_between(values, 0, np.maximum(Rs * 1e3, 0), alpha=0.12, color=_COLORS[1])
    ax2.set_ylabel("Key fraction R  (×10⁻³ bits/pulse)", fontsize=11)
    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2D sweep
# ---------------------------------------------------------------------------

def sweep_2d(
    scenario: str,
    param1: str,
    values1: np.ndarray,
    param2: str,
    values2: np.ndarray,
    decoy_mode: str = "asymptotic",
    dt: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep two parameters over a grid.  Returns (values1, values2, Z_ell_finite)."""
    Z = np.zeros((len(values2), len(values1)))
    total = len(values1) * len(values2)
    done  = 0
    for j, v2 in enumerate(values2):
        for i, v1 in enumerate(values1):
            ell, _ = _run({param1: v1, param2: v2}, scenario, decoy_mode, dt)
            Z[j, i] = ell
            done += 1
            if done % 10 == 0 or done == total:
                print(f"    {done}/{total}  {param1}={v1:.3g}  {param2}={v2:.3g}  ℓ={ell:.0f}")
    return values1, values2, Z


def plot_sweep_2d(
    param1: str,
    values1: np.ndarray,
    param2: str,
    values2: np.ndarray,
    Z: np.ndarray,
    scenario: str = "",
) -> plt.Figure:
    """Filled contour heatmap of ℓ_finite over a 2D parameter grid."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Clip negatives to zero for display
    Z_plot = np.maximum(Z, 0)

    # Use log scale if range spans > 2 orders of magnitude
    pos_vals = Z_plot[Z_plot > 0]
    use_log  = pos_vals.size > 0 and (pos_vals.max() / max(pos_vals.min(), 1)) > 100

    if use_log and pos_vals.size > 0:
        Z_display = np.where(Z_plot > 0, np.log10(np.maximum(Z_plot, 1)), 0)
        norm  = None
        label = "ℓ_finite  (log₁₀ bits)"
    else:
        Z_display = Z_plot / 1e3
        norm  = None
        label = "ℓ_finite  (kbits)"

    levels = 20
    cf = ax.contourf(values1, values2, Z_display, levels=levels, cmap="viridis")
    cb = fig.colorbar(cf, ax=ax, label=label)

    # Go/no-go boundary (ℓ_finite = 0 contour)
    try:
        ax.contour(values1, values2, Z, levels=[0], colors="red", linewidths=1.8)
        # Label
        ax.plot([], [], color="red", lw=1.8, label="go/no-go boundary (ℓ = 0)")
        ax.legend(fontsize=9, loc="upper right")
    except Exception:
        pass

    # Mark baseline
    reg0 = ParameterRegistry()
    reg0.update(get_scenario(scenario))
    try:
        p1_base = reg0.get(param1)
        p2_base = reg0.get(param2)
        ax.plot(p1_base, p2_base, "w*", ms=14, label=f"Baseline ({p1_base:.3g}, {p2_base:.3g})")
        ax.legend(fontsize=9, loc="upper right")
    except KeyError:
        pass

    ax.set_xlabel(_label(param1), fontsize=11)
    ax.set_ylabel(_label(param2), fontsize=11)
    ax.set_title(
        f"ℓ_finite vs {param1} and {param2}\n"
        f"(scenario: {scenario}, asymptotic decoy bounds)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Canned sweeps for the report
# ---------------------------------------------------------------------------

def run_all_sweeps(save_dir: str | None = None) -> list[plt.Figure]:
    out = save_dir or _OUT_DIR
    os.makedirs(out, exist_ok=True)
    figs = []

    scenario = "optimistic"

    # ── 1D: ℓ_finite vs D_rx ─────────────────────────────────────────────
    print("1D sweep: D_rx...")
    D_rx_vals = np.linspace(0.3, 2.0, 22)
    v, ells, Rs = sweep_1d(scenario, "D_rx", D_rx_vals)
    fig = plot_sweep_1d("D_rx", v, ells, Rs, scenario)
    path = os.path.join(out, "sweep1d_D_rx.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    figs.append(fig)

    # ── 1D: ℓ_finite vs orbital altitude ─────────────────────────────────
    print("1D sweep: h_orbit...")
    h_vals = np.linspace(200e3, 800e3, 18)
    v, ells, Rs = sweep_1d(scenario, "h_orbit", h_vals)
    fig = plot_sweep_1d("h_orbit", v / 1e3, ells, Rs, scenario)
    # Fix x-axis label (values in km now)
    fig.axes[1].set_xlabel("Orbital altitude  h  (km)", fontsize=11)
    fig.axes[0].set_title(f"Parameter Sweep: Orbital altitude (km)\n(scenario: {scenario})", fontsize=10)
    path = os.path.join(out, "sweep1d_h_orbit.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    figs.append(fig)

    # ── 1D: ℓ_finite vs beam waist w0 (two scenarios) ────────────────────
    print("1D sweep: w0...")
    w0_vals = np.linspace(0.02, 0.25, 20)
    v1, e1, r1 = sweep_1d("optimistic",  "w0", w0_vals)
    v2, e2, r2 = sweep_1d("vqi_400km",   "w0", w0_vals)
    fig = plot_sweep_1d("w0", v1, e1, r1, "optimistic",
                        extra_lines=[(v2, e2, "vqi_400km")])
    # Annotate analytic optimum  w0* = λ/(π·σ_pnt·√2)
    reg_opt = ParameterRegistry(); reg_opt.update(get_scenario("optimistic"))
    lam     = reg_opt.get("lambda_")
    sig_pnt = reg_opt.get("sigma_pnt")
    w0_star = lam / (math.pi * sig_pnt * math.sqrt(2))
    for ax in fig.axes:
        ax.axvline(w0_star, color="gray", ls="--", lw=1.0, alpha=0.8)
        ax.text(w0_star + 0.002, ax.get_ylim()[1] * 0.8,
                f"w₀* = {w0_star*100:.1f} cm", fontsize=8, color="gray")
    path = os.path.join(out, "sweep1d_w0.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    figs.append(fig)

    # ── 1D: ℓ_finite vs r_PE ─────────────────────────────────────────────
    print("1D sweep: r_PE...")
    rpe_vals = np.linspace(0.02, 0.45, 20)
    v, ells, Rs = sweep_1d(scenario, "r_PE", rpe_vals, decoy_mode="finite")
    fig = plot_sweep_1d("r_PE", v, ells, Rs, scenario)
    path = os.path.join(out, "sweep1d_r_PE.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    figs.append(fig)

    # ── 2D: D_rx vs h_orbit ──────────────────────────────────────────────
    print("2D sweep: D_rx × h_orbit...")
    D_rx_2d  = np.linspace(0.4, 2.0, 13)
    h_orb_2d = np.linspace(250e3, 700e3, 12)
    v1, v2, Z = sweep_2d(scenario, "D_rx", D_rx_2d, "h_orbit", h_orb_2d)
    fig = plot_sweep_2d("D_rx", v1, "h_orbit", v2 / 1e3, Z, scenario)
    fig.axes[0].set_ylabel("Orbital altitude  h  (km)", fontsize=11)
    path = os.path.join(out, "sweep2d_Drx_horbit.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    figs.append(fig)

    # ── 2D: w0 vs sigma_pnt ──────────────────────────────────────────────
    print("2D sweep: w0 × sigma_pnt...")
    w0_2d   = np.linspace(0.03, 0.20, 12)
    sig_2d  = np.linspace(0.5e-6, 8e-6, 12)
    v1, v2, Z = sweep_2d(scenario, "w0", w0_2d, "sigma_pnt", sig_2d)
    fig = plot_sweep_2d("w0", v1 * 100, "sigma_pnt", v2 * 1e6, Z, scenario)
    fig.axes[0].set_xlabel("Beam waist w₀  (cm)", fontsize=11)
    fig.axes[0].set_ylabel("Pointing jitter σ_pnt  (µrad)", fontsize=11)
    path = os.path.join(out, "sweep2d_w0_sigpnt.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    figs.append(fig)

    return figs


if __name__ == "__main__":
    run_all_sweeps()
    plt.show()
