"""
viz/pass_sim_plots.py — Full-pass simulation visualisations.

Plots
-----
    plot_pass_profile       4-panel time-series: elevation, η₀, QBER, cumulative bits.
    plot_loss_waterfall     Per-term loss breakdown at peak elevation (dB waterfall).
    plot_key_budget         Bit-budget funnel: pulses → sifted → n_key → ℓ_finite.
    plot_scenario_compare   Multi-scenario bar comparison across key metrics.
    plot_all                Render all four → viz/out/pass_sim/.
"""

from __future__ import annotations

import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from params.registry import ParameterRegistry
from params.scenarios import get_scenario
from orbit.pass_sim import simulate_pass
from core.types import PassResult

_COLORS = plt.cm.tab10.colors
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "pass_sim")

# Best pass window (07:00 UTC, 54° max elevation)
_T_START = datetime(2025, 1, 1, 7, 0, tzinfo=timezone.utc)
_T_END   = datetime(2025, 1, 1, 8, 30, tzinfo=timezone.utc)


def _run_scenario(name: str, decoy_mode: str = "finite") -> PassResult:
    reg = ParameterRegistry()
    reg.update(get_scenario(name))
    return simulate_pass(reg, t_start=_T_START, t_end=_T_END, decoy_mode=decoy_mode)


# ---------------------------------------------------------------------------
# Figure 1 — Pass profile (4-panel time series)
# ---------------------------------------------------------------------------

def plot_pass_profile(result: PassResult | None = None,
                      scenario: str = "vqi_400km") -> plt.Figure:
    """4-panel time-series showing how every key quantity evolves through a pass."""
    if result is None:
        result = _run_scenario(scenario)

    t = result.time / 60.0          # seconds → minutes
    el_deg = np.degrees(result.elevation)
    eta_dB = 10.0 * np.log10(np.clip(result.eta_0, 1e-10, None))
    qber   = result.qber_instant * 100.0   # fraction → %

    # AEP breakeven: ℓ_finite > 0 when cumulative_n * R > AEP correction
    # Mark the running AEP threshold on the cumulative photon panel
    R    = result.key_rate.R
    eps  = 1e-10
    aep_bits = lambda n: 4.0 * math.sqrt(max(n, 1) * math.log2(6.0 / eps))

    fig = plt.figure(figsize=(10, 9))
    gs  = gridspec.GridSpec(4, 1, hspace=0.08, figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # ── Panel 1: Elevation ────────────────────────────────────────────────
    axes[0].plot(t, el_deg, color=_COLORS[0], lw=1.8)
    axes[0].axhline(10, color="gray", ls=":", lw=0.9, alpha=0.7)
    axes[0].fill_between(t, 0, el_deg, alpha=0.12, color=_COLORS[0])
    axes[0].set_ylabel("Elevation (°)", fontsize=10)
    axes[0].set_ylim(0, max(el_deg) * 1.15)
    axes[0].text(t[-1] * 0.02, 11.5, "θ_el,min = 10°", fontsize=8, color="gray")

    # ── Panel 2: Static transmissivity ───────────────────────────────────
    axes[1].plot(t, eta_dB, color=_COLORS[1], lw=1.8)
    axes[1].fill_between(t, eta_dB, eta_dB.min() - 1, alpha=0.12, color=_COLORS[1])
    axes[1].set_ylabel("η₀  (dB)", fontsize=10)
    # Annotate peak
    pk = np.argmax(result.eta_0)
    axes[1].annotate(f"peak {eta_dB[pk]:.1f} dB",
                     xy=(t[pk], eta_dB[pk]),
                     xytext=(t[pk] + t[-1] * 0.05, eta_dB[pk] - 0.5),
                     fontsize=8, color=_COLORS[1],
                     arrowprops=dict(arrowstyle="->", color=_COLORS[1], lw=0.8))

    # ── Panel 3: QBER ─────────────────────────────────────────────────────
    axes[2].plot(t, qber, color=_COLORS[2], lw=1.8, label="E_μ (fading-averaged)")
    axes[2].axhline(11, color="red", ls="--", lw=1.0, alpha=0.8, label="BB84 threshold (11%)")
    axes[2].axhline(result.E_mu_weighted * 100, color=_COLORS[2],
                    ls=":", lw=1.0, alpha=0.8,
                    label=f"Pass mean ({result.E_mu_weighted*100:.2f}%)")
    axes[2].set_ylabel("QBER (%)", fontsize=10)
    axes[2].set_ylim(0, max(qber.max() * 1.4, 12))
    axes[2].legend(fontsize=8, loc="upper right")

    # ── Panel 4: Cumulative sifted photons ────────────────────────────────
    cum = result.cumulative_n
    axes[3].plot(t, cum / 1e6, color=_COLORS[3], lw=1.8, label="Cumulative n_sifted")

    # AEP breakeven line (requires ~this many bits to extract any key)
    if R > 0:
        # gross = n * R must exceed AEP → n > AEP/R
        n_aep_break = (4.0 * math.sqrt(math.log2(6.0 / eps)) / R) ** 2
        axes[3].axhline(n_aep_break / 1e6, color="red", ls="--", lw=1.0, alpha=0.8,
                        label=f"AEP breakeven ({n_aep_break/1e6:.1f}M)")

    axes[3].set_ylabel("Sifted photons (×10⁶)", fontsize=10)
    axes[3].set_xlabel("Time since window open  (min)", fontsize=10)
    axes[3].legend(fontsize=8, loc="upper left")

    # Shared formatting
    for i, ax in enumerate(axes):
        ax.grid(True, alpha=0.25)
        ax.set_xlim(t[0], t[-1])
        if i < 3:
            ax.set_xticklabels([])

    fig.suptitle(
        f"Full Pass Simulation — scenario: {scenario}\n"
        f"T_pass = {result.T_pass:.0f} s  "
        f"n_sifted = {result.n_sifted:.2e}  "
        f"ℓ_finite = {result.ell_finite:.0f} bits  "
        f"go = {result.go}",
        fontsize=11, fontweight="bold",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Loss waterfall at peak elevation
# ---------------------------------------------------------------------------

def plot_loss_waterfall(result: PassResult | None = None,
                        scenario: str = "vqi_400km") -> plt.Figure:
    """Horizontal waterfall: per-term loss contribution at peak-elevation geometry."""
    if result is None:
        result = _run_scenario(scenario)

    # Reconstruct loss budget at peak elevation from the stored η₀ array.
    # We need to re-run evaluate_point at peak elevation to get LossBudget.
    reg = ParameterRegistry()
    reg.update(get_scenario(scenario))
    channel  = reg.build_channel()
    source   = reg.build_source()
    detector = reg.build_detector()

    pk_idx  = int(np.argmax(result.eta_0))
    from core.types import Geometry
    from core.evaluator import evaluate_point
    from physics.atmosphere import hufnagel_valley

    geom = Geometry(
        theta_el=float(result.elevation[pk_idx]),
        L=float(result.eta_0[pk_idx]),   # placeholder — recompute below
        zeta=math.pi / 2.0 - float(result.elevation[pk_idx]),
        h_orbit=channel.h_orbit if hasattr(channel, "h_orbit") else 400e3,
    )
    # Recompute slant range from elevation + altitude
    h_orb = reg.get("h_orbit")
    R_earth = 6.371e6
    el = float(result.elevation[pk_idx])
    L = math.sqrt((R_earth + h_orb)**2 - (R_earth * math.cos(el))**2) - R_earth * math.sin(el)
    geom = Geometry(theta_el=el, L=L, zeta=math.pi/2 - el, h_orbit=h_orb)

    cn2 = hufnagel_valley(channel.Cn2_0, channel.v_wind)
    state = evaluate_point(geom, channel, source, detector, cn2_profile=cn2)
    lb = state.loss_budget

    labels = ["η_tx (optics)", "η_atm (Beer-Lambert)", "η_diff (diffraction)",
              "η_pnt (pointing)", "η_rx (receiver)"]
    values_lin = [lb.eta_tx, lb.eta_atm, lb.eta_diff, lb.eta_pnt, lb.eta_rx]
    values_dB  = [10*math.log10(v) if v > 0 else -100 for v in values_lin]
    colors_bar = [_COLORS[i] for i in range(len(labels))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: dB bar chart
    ax = axes[0]
    bars = ax.barh(range(len(labels)), values_dB, color=colors_bar, alpha=0.85, height=0.6)
    for i, (bar, val) in enumerate(zip(bars, values_dB)):
        ax.text(val - 0.2, i, f"{val:.2f} dB", va="center", ha="right",
                fontsize=9, color="white", fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Loss  (dB)", fontsize=11)
    ax.set_title("Per-Term Loss at Peak Elevation", fontsize=10)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.grid(True, which="both", axis="x", alpha=0.25)
    ax.invert_yaxis()

    # Add total
    total_dB = 10*math.log10(lb.eta_0)
    ax.set_xlabel(f"Loss  (dB)     [total η₀ = {total_dB:.2f} dB]", fontsize=11)

    # Right: cumulative waterfall (cascade from 0 dB down)
    ax2 = axes[1]
    cumulative = 0.0
    for i, (label, dB, color) in enumerate(zip(labels, values_dB, colors_bar)):
        ax2.barh(i, dB, left=cumulative, color=color, alpha=0.85, height=0.6)
        ax2.text(cumulative + dB/2, i, f"{dB:.1f}", va="center", ha="center",
                 fontsize=8, color="white", fontweight="bold")
        cumulative += dB

    # Total arrow
    ax2.annotate(f"η₀ = {total_dB:.2f} dB",
                 xy=(cumulative, len(labels) - 0.5),
                 xytext=(cumulative - 3, len(labels) + 0.2),
                 fontsize=9, color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("Cumulative loss from transmitter  (dB)", fontsize=11)
    ax2.set_title("Cascaded Loss Budget  (Waterfall)", fontsize=10)
    ax2.grid(True, axis="x", alpha=0.25)
    ax2.invert_yaxis()

    el_deg = math.degrees(float(result.elevation[pk_idx]))
    fig.suptitle(
        f"Link Loss Breakdown — scenario: {scenario}\n"
        f"Peak elevation {el_deg:.1f}°   slant range {L/1e3:.0f} km   "
        f"η₀ = {lb.eta_0:.4f}  ({total_dB:.2f} dB)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Key bit budget funnel
# ---------------------------------------------------------------------------

def plot_key_budget(result: PassResult | None = None,
                    scenario: str = "vqi_400km") -> plt.Figure:
    """Horizontal funnel showing how the total pulse budget shrinks to ℓ_finite."""
    if result is None:
        result = _run_scenario(scenario)

    reg = ParameterRegistry()
    reg.update(get_scenario(scenario))
    source = reg.build_source()
    pp     = result.post_processing
    kr     = result.key_rate

    N_total   = source.f_clock * result.T_pass
    N_signal  = N_total * source.P_mu
    N_det     = result.n_sifted / source.sifting_factor()   # un-sift
    N_sifted  = result.n_sifted
    N_key     = pp.n_key
    N_gross   = N_key * max(kr.R, 0)
    N_finite  = result.ell_finite

    stages = [
        ("Total pulses",              N_total,   _COLORS[0]),
        ("Signal pulses  (×P_μ)",     N_signal,  _COLORS[1]),
        ("Detections  (×Q_μ/q)",      N_det,     _COLORS[2]),
        ("Sifted bits  (×q)",         N_sifted,  _COLORS[3]),
        ("Key block  (1−r_PE)",       N_key,     _COLORS[4]),
        ("Gross key  (×R)",           N_gross,   _COLORS[5]),
        ("ℓ_finite  (−AEP)",          N_finite,  _COLORS[6] if N_finite > 0 else "red"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, val, color) in enumerate(stages):
        if val <= 0:
            ax.barh(i, 0.5, color="red", alpha=0.5, height=0.6)
            ax.text(1.2, i, f"0  (AEP penalty exceeds gross key)", va="center", fontsize=9, color="red")
        else:
            ax.barh(i, val, color=color, alpha=0.85, height=0.6)
            ax.text(val * 1.05, i, f"{val:.2e}", va="center", fontsize=9)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([s[0] for s in stages], fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Count  (log scale)", fontsize=11)
    ax.set_title(
        f"Key Bit Budget — scenario: {scenario}\n"
        f"f_clock={source.f_clock/1e6:.0f} MHz   T_pass={result.T_pass:.0f}s   "
        f"f_EC={pp.f_EC:.3f}   R={kr.R:.4e}",
        fontsize=10,
    )
    ax.invert_yaxis()
    ax.set_xlim(0.1, N_total * 5)
    ax.grid(True, which="both", axis="x", alpha=0.2)

    # Annotate each reduction factor
    vals = [s[1] for s in stages]
    for i in range(len(vals) - 1):
        if vals[i] > 0 and vals[i+1] > 0:
            ratio = vals[i+1] / vals[i]
            ax.annotate(f"×{ratio:.3f}", xy=(vals[i], i + 0.5),
                        xytext=(vals[i] * 0.35, i + 0.55),
                        fontsize=7.5, color="dimgray",
                        ha="center",
                        arrowprops=dict(arrowstyle="-", color="lightgray", lw=0.6))

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Multi-scenario comparison
# ---------------------------------------------------------------------------

def plot_scenario_compare(scenarios: list[str] | None = None) -> plt.Figure:
    """Grouped bar comparison of key metrics across scenarios."""
    if scenarios is None:
        scenarios = ["conservative", "vqi_400km", "optimistic"]

    results = {}
    for sc in scenarios:
        try:
            results[sc] = _run_scenario(sc, decoy_mode="finite")
        except RuntimeError as e:
            print(f"  {sc}: {e}")
            results[sc] = None

    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        raise RuntimeError("No scenarios produced valid results.")

    labels = list(valid.keys())
    n = len(labels)
    x = np.arange(n)

    metrics = {
        "n_sifted  (bits)":     [v.n_sifted                  for v in valid.values()],
        "E_μ  (%)":             [v.E_mu_weighted * 100        for v in valid.values()],
        "R  (bits/pulse)":      [max(v.key_rate.R, 0)         for v in valid.values()],
        "ℓ_finite  (bits)":     [v.ell_finite                 for v in valid.values()],
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    display_labels = [s.replace("_", "\n") for s in labels]

    for ax, (metric, vals) in zip(axes, metrics.items()):
        colors = [_COLORS[i % 10] for i in range(n)]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.6)

        # Colour ℓ_finite = 0 bars red
        if "ℓ_finite" in metric:
            for bar, val in zip(bars, vals):
                if val <= 0:
                    bar.set_color("red")
                    bar.set_alpha(0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, fontsize=9)
        ax.set_title(metric, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                        f"{val:.2e}" if val > 1000 else f"{val:.4f}",
                        ha="center", va="bottom", fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, ax.get_ylim()[1] * 0.05,
                        "0", ha="center", va="bottom", fontsize=9, color="red")

        # Log scale for large-range metrics
        if "n_sifted" in metric or "ℓ_finite" in metric:
            pos_vals = [v for v in vals if v > 0]
            if pos_vals:
                ax.set_yscale("log")
                ax.set_ylim(min(pos_vals) * 0.1, max(pos_vals) * 10)

    # Legend
    patches = [Patch(color=_COLORS[i], label=labels[i]) for i in range(n)]
    fig.legend(handles=patches, fontsize=9, loc="lower center",
               ncol=n, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Scenario Comparison — Key Pass Metrics\n"
                 "(best pass: 07:00 UTC, max elevation ≈ 54°)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# plot_all
# ---------------------------------------------------------------------------

def plot_all(save_dir: str | None = None) -> list[plt.Figure]:
    out = save_dir or _OUT_DIR
    os.makedirs(out, exist_ok=True)

    print("Running vqi_400km pass...")
    result_vqi = _run_scenario("vqi_400km")

    print("Running optimistic pass...")
    result_opt = _run_scenario("optimistic")

    plots = [
        ("pass_profile_vqi",      lambda: plot_pass_profile(result_vqi,  "vqi_400km")),
        ("pass_profile_optimistic",lambda: plot_pass_profile(result_opt, "optimistic")),
        ("loss_waterfall_vqi",    lambda: plot_loss_waterfall(result_vqi, "vqi_400km")),
        ("key_budget_vqi",        lambda: plot_key_budget(result_vqi,     "vqi_400km")),
        ("key_budget_optimistic", lambda: plot_key_budget(result_opt,     "optimistic")),
        ("scenario_compare",      lambda: plot_scenario_compare()),
    ]

    figs = []
    for name, fn in plots:
        print(f"  plotting {name}...")
        fig = fn()
        path = os.path.join(out, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved → {path}")
        figs.append(fig)

    return figs


if __name__ == "__main__":
    plot_all()
    plt.show()
