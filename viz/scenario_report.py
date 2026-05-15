"""
viz/scenario_report.py — Per-scenario full report generator.

Generates all visualisations for one or more scenarios and saves them under:
    viz/out/<scenario_name>/<category>/

Category subfolders
-------------------
    pass/       Time-series pass profile, loss waterfall, key-budget funnel
    keyrate/    EUR breakdown, EUR-vs-AEP comparison, s_Z1 bounds
    comparison/ Cross-scenario comparison charts (written once at the end)

Usage
-----
    # From repo root — all proposal scenarios
    python viz/scenario_report.py

    # Specific scenarios
    python viz/scenario_report.py proposal_1 proposal1_revised proposal1_stretch optimistic

    # Single scenario
    python viz/scenario_report.py optimistic
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for scripted runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from params.registry import ParameterRegistry
from params.scenarios import get_scenario, load_yaml
from orbit.pass_sim import simulate_pass
from core.types import PassResult, EURDecoyBounds
from physics.keyrate import binary_entropy

# ── Run config ────────────────────────────────────────────────────────────────
_T_START = datetime(2026, 1, 15, 4, 0, tzinfo=timezone.utc)
_T_END   = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

_GS = dict(gs_lat=36.1, gs_lon=-86.7, gs_alt_m=182.0, inclination=53.0)

_COLORS = plt.cm.tab10.colors

_OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

_DEFAULT_SCENARIOS = [
    "proposal_1",
    "proposal1_revised",
    "proposal1_stretch",
    "optimistic",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _out_dir(scenario_name: str, category: str) -> str:
    """Return and create `viz/out/<scenario_name>/<category>/`."""
    d = os.path.join(_OUT_ROOT, scenario_name, category)
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved → {os.path.relpath(path)}")


def _load_registry(scenario_name: str) -> ParameterRegistry:
    """Load a scenario from YAML file (preferred) or built-in dict."""
    reg = ParameterRegistry(scenario_name)
    yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scenarios", f"{scenario_name}.yaml"
    )
    if os.path.isfile(yaml_path):
        reg.update(load_yaml(yaml_path))
    else:
        reg.update(get_scenario(scenario_name))
    for k, v in _GS.items():
        try:
            reg.set(k, v)
        except Exception:
            pass
    return reg


def _run(scenario_name: str, proof_method: str = "eur") -> PassResult:
    reg = _load_registry(scenario_name)
    return simulate_pass(reg, t_start=_T_START, t_end=_T_END,
                         proof_method=proof_method)


# ── Pass plots ────────────────────────────────────────────────────────────────

def plot_pass_profile(result: PassResult, scenario: str) -> plt.Figure:
    """4-panel time-series: elevation, η₀ (dB), QBER, cumulative sifted bits."""
    t      = result.time / 60.0
    el_deg = np.degrees(result.elevation)
    eta_dB = 10.0 * np.log10(np.clip(result.eta_0, 1e-15, None))
    qber   = result.qber_instant * 100.0

    fig = plt.figure(figsize=(10, 9))
    gs  = gridspec.GridSpec(4, 1, hspace=0.08, figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # Panel 1: elevation
    axes[0].plot(t, el_deg, color=_COLORS[0], lw=1.8)
    axes[0].fill_between(t, 0, el_deg, alpha=0.12, color=_COLORS[0])
    axes[0].set_ylabel("Elevation (°)", fontsize=10)
    axes[0].set_ylim(0, max(el_deg) * 1.15)

    # Panel 2: static transmissivity
    axes[1].plot(t, eta_dB, color=_COLORS[1], lw=1.8)
    axes[1].fill_between(t, eta_dB, eta_dB.min() - 1, alpha=0.12, color=_COLORS[1])
    axes[1].set_ylabel("η₀  (dB)", fontsize=10)
    pk = int(np.argmax(result.eta_0))
    axes[1].annotate(f"peak {eta_dB[pk]:.1f} dB",
                     xy=(t[pk], eta_dB[pk]),
                     xytext=(t[pk] + t[-1] * 0.06, eta_dB[pk] - 0.8),
                     fontsize=8, color=_COLORS[1],
                     arrowprops=dict(arrowstyle="->", color=_COLORS[1], lw=0.8))

    # Panel 3: QBER
    axes[2].plot(t, qber, color=_COLORS[2], lw=1.8, label="E_μ (fading-averaged)")
    axes[2].axhline(11, color="red", ls="--", lw=1.0, alpha=0.8, label="BB84 limit (11%)")
    axes[2].axhline(result.E_mu_weighted * 100, color=_COLORS[2], ls=":", lw=1.0,
                    label=f"Pass mean ({result.E_mu_weighted*100:.2f}%)")
    axes[2].set_ylabel("QBER (%)", fontsize=10)
    axes[2].set_ylim(0, max(qber.max() * 1.4, 12))
    axes[2].legend(fontsize=8, loc="upper right")

    # Panel 4: cumulative sifted photons
    cum = result.cumulative_n
    axes[3].plot(t, cum / 1e6, color=_COLORS[3], lw=1.8, label="Cumulative n_sifted")
    axes[3].set_ylabel("Sifted photons (×10⁶)", fontsize=10)
    axes[3].set_xlabel("Time since window open  (min)", fontsize=10)
    axes[3].legend(fontsize=8, loc="upper left")

    for i, ax in enumerate(axes):
        ax.grid(True, alpha=0.25)
        ax.set_xlim(t[0], t[-1])
        if i < 3:
            ax.set_xticklabels([])

    proof = "EUR" if result.eur_decoy_bounds is not None else "AEP"
    fig.suptitle(
        f"Pass Profile — {scenario}  [{proof}]\n"
        f"T_pass={result.T_pass:.0f} s   "
        f"n_sifted={result.n_sifted:.2e}   "
        f"ℓ_finite={result.ell_finite:,.0f} bits   "
        f"go={result.go}",
        fontsize=11, fontweight="bold",
    )
    return fig


def plot_loss_waterfall(result: PassResult, scenario: str) -> plt.Figure:
    """Horizontal dB waterfall at peak elevation."""
    from core.types import Geometry
    from core.evaluator import evaluate_point
    from physics.atmosphere import hufnagel_valley

    reg     = _load_registry(scenario)
    channel = reg.build_channel()
    source  = reg.build_source()
    detector = reg.build_detector()

    pk_idx = int(np.argmax(result.eta_0))
    el     = float(result.elevation[pk_idx])
    h_orb  = reg.get("h_orbit")
    R_earth = 6.371e6
    L = math.sqrt((R_earth + h_orb)**2 - (R_earth * math.cos(el))**2) - R_earth * math.sin(el)
    geom = Geometry(theta_el=el, L=L, zeta=math.pi/2 - el, h_orbit=h_orb)

    cn2 = hufnagel_valley(channel.Cn2_0, channel.v_wind)
    state = evaluate_point(geom, channel, source, detector, cn2_profile=cn2)
    lb = state.loss_budget

    labels     = ["η_tx (optics)", "η_atm (Beer-Lambert)", "η_diff (diffraction)",
                  "η_pnt (pointing)", "η_rx (receiver)"]
    values_lin = [lb.eta_tx, lb.eta_atm, lb.eta_diff, lb.eta_pnt, lb.eta_rx]
    values_dB  = [10*math.log10(v) if v > 0 else -100 for v in values_lin]
    colors_bar = list(_COLORS[:len(labels)])
    total_dB   = 10*math.log10(lb.eta_0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: per-term bars
    ax = axes[0]
    bars = ax.barh(range(len(labels)), values_dB, color=colors_bar, alpha=0.85, height=0.6)
    for bar, val in zip(bars, values_dB):
        ax.text(val - 0.15, bar.get_y() + bar.get_height()/2,
                f"{val:.2f} dB", va="center", ha="right",
                fontsize=9, color="white", fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(f"Loss  (dB)   [η₀ = {total_dB:.2f} dB]", fontsize=10)
    ax.set_title("Per-Term Loss", fontsize=10)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.grid(True, axis="x", alpha=0.25)
    ax.invert_yaxis()

    # Right: cascade waterfall
    ax2 = axes[1]
    cumulative = 0.0
    for i, (label, dB, color) in enumerate(zip(labels, values_dB, colors_bar)):
        ax2.barh(i, dB, left=cumulative, color=color, alpha=0.85, height=0.6)
        ax2.text(cumulative + dB/2, i, f"{dB:.1f}", va="center",
                 ha="center", fontsize=8, color="white", fontweight="bold")
        cumulative += dB
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("Cumulative loss from transmitter  (dB)", fontsize=10)
    ax2.set_title("Cascaded Waterfall", fontsize=10)
    ax2.grid(True, axis="x", alpha=0.25)
    ax2.invert_yaxis()

    fig.suptitle(
        f"Link Loss Breakdown — {scenario}\n"
        f"Peak elevation {math.degrees(el):.1f}°   slant range {L/1e3:.0f} km   "
        f"η₀ = {lb.eta_0:.4f}  ({total_dB:.2f} dB)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_key_budget(result: PassResult, scenario: str) -> plt.Figure:
    """Funnel from total pulses → ℓ_finite, labelled for both EUR and AEP paths."""
    reg    = _load_registry(scenario)
    source = reg.build_source()
    pp     = result.post_processing
    kr     = result.key_rate

    N_total  = source.f_clock * result.T_pass
    N_signal = N_total * source.P_mu
    N_det    = result.n_sifted / source.sifting_factor()
    N_sifted = result.n_sifted
    N_finite = result.ell_finite

    eur_b = result.eur_decoy_bounds
    proof = "EUR" if eur_b is not None else "AEP"

    if eur_b is not None:
        N_key_Z = N_sifted * source.P_X ** 2 / source.sifting_factor()
        N_s_Z1  = eur_b.s_Z1_lower
        N_s_Z0  = eur_b.s_Z0_lower
        N_pa    = N_s_Z1 * (1.0 - binary_entropy(eur_b.phi_Z_upper)) + N_s_Z0
        stages = [
            ("Total pulses",             N_total,   _COLORS[0]),
            ("Signal pulses  (×P_μ)",    N_signal,  _COLORS[1]),
            ("Detections  (×Q_μ/q)",     N_det,     _COLORS[2]),
            ("Sifted bits  (key+test)",  N_sifted,  _COLORS[3]),
            ("Key-basis signal  (×P_X²)",N_key_Z,   _COLORS[4]),
            ("s_Z1^L  (single-photon)",  N_s_Z1,    _COLORS[5]),
            ("PA bits  (×[1−H₂(φ)])",   N_pa,      _COLORS[6]),
            ("ℓ_finite  (−leak_EC−corr)", N_finite, _COLORS[7] if N_finite > 0 else "red"),
        ]
    else:
        N_key   = pp.n_key
        N_gross = N_key * max(kr.R, 0)
        stages = [
            ("Total pulses",             N_total,   _COLORS[0]),
            ("Signal pulses  (×P_μ)",    N_signal,  _COLORS[1]),
            ("Detections  (×Q_μ/q)",     N_det,     _COLORS[2]),
            ("Sifted bits",              N_sifted,  _COLORS[3]),
            ("Key block  (1−r_PE)",      N_key,     _COLORS[4]),
            ("Gross key  (×R)",          N_gross,   _COLORS[5]),
            ("ℓ_finite  (−AEP corr.)",  N_finite,  _COLORS[6] if N_finite > 0 else "red"),
        ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (label, val, color) in enumerate(stages):
        if val <= 0:
            ax.barh(i, 0.5, color="red", alpha=0.4, height=0.6)
            ax.text(1.5, i, "0  (infeasible)", va="center", fontsize=9, color="red")
        else:
            ax.barh(i, val, color=color, alpha=0.85, height=0.6)
            ax.text(val * 1.05, i, f"{val:,.0f}" if val < 1e6 else f"{val:.3e}",
                    va="center", fontsize=8.5)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([s[0] for s in stages], fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlabel("Count  (log scale)", fontsize=11)
    ax.set_title(
        f"Key Bit Budget [{proof}] — {scenario}\n"
        f"f_clock={source.f_clock/1e6:.0f} MHz   T_pass={result.T_pass:.0f} s   "
        f"E_μ={result.E_mu_weighted*100:.2f}%   "
        f"ℓ_finite={N_finite:,.0f} bits",
        fontsize=10,
    )
    ax.invert_yaxis()
    ax.set_xlim(0.1, N_total * 5)
    ax.grid(True, which="both", axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


# ── EUR-specific keyrate plots ────────────────────────────────────────────────

def plot_eur_breakdown(result_eur: PassResult, result_aep: PassResult,
                       scenario: str) -> plt.Figure:
    """Side-by-side bar: EUR term decomposition vs AEP term decomposition."""
    eur_b = result_eur.eur_decoy_bounds
    pp    = result_eur.post_processing
    kr    = result_eur.key_rate
    eps   = 1.0e-10
    eps_sub = eps / 6.0

    # EUR terms
    s_Z0  = eur_b.s_Z0_lower if eur_b else 0.0
    s_Z1  = eur_b.s_Z1_lower if eur_b else 0.0
    phi   = eur_b.phi_Z_upper if eur_b else 0.0
    pa_bits   = s_Z1 * (1.0 - binary_entropy(phi)) + s_Z0
    leak_eur  = result_eur.post_processing.leak_EC
    corr_eur  = 7.0 * math.log2(2.0 / eps_sub)
    ell_eur   = result_eur.ell_finite

    # AEP terms
    n_key_aep = result_aep.post_processing.n_key
    R_aep     = result_aep.key_rate.R
    gross_aep = max(0.0, n_key_aep * R_aep)
    corr_aep  = 4.0 * math.sqrt(n_key_aep * math.log2(6.0 / eps)) + math.log2(2.0 / eps)
    ell_aep   = result_aep.ell_finite

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    # Left — EUR breakdown
    ax = axes[0]
    eur_labels = ["s_Z0^L (vacuum)", "s_Z1^L·(1−H₂(φ)) (PA)", "−leak_EC", "−6·log₂(2/ε̃) corr.", "= ℓ_finite"]
    eur_vals   = [s_Z0, s_Z1*(1-binary_entropy(phi)), -leak_eur, -corr_eur, ell_eur]
    eur_colors = [_COLORS[0], _COLORS[1], _COLORS[2], _COLORS[3],
                  _COLORS[4] if ell_eur > 0 else "red"]

    bars = ax.barh(range(len(eur_labels)), eur_vals, color=eur_colors, alpha=0.85, height=0.6)
    for bar, val in zip(bars, eur_vals):
        x_pos = val + (abs(val) * 0.03 if val >= 0 else -abs(val) * 0.03)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{val:+,.0f}" if abs(val) < 1e6 else f"{val:+.2e}",
                va="center", ha="left" if val >= 0 else "right", fontsize=8.5)
    ax.set_yticks(range(len(eur_labels)))
    ax.set_yticklabels(eur_labels, fontsize=9.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Bits", fontsize=10)
    ax.set_title(f"EUR Formula Breakdown\nℓ = {ell_eur:,.0f} bits", fontsize=10)
    ax.grid(True, axis="x", alpha=0.25)
    ax.invert_yaxis()

    # Right — AEP breakdown
    ax2 = axes[1]
    aep_labels = ["Gross key  (n_key·R)", "−Δ_AEP  (4·√(n·log 6/ε))", "−log₂(2/ε_PA)", "= ℓ_finite"]
    delta_aep  = 4.0 * math.sqrt(n_key_aep * math.log2(6.0/eps))
    hash_cost  = math.log2(2.0 / eps)
    aep_vals   = [gross_aep, -delta_aep, -hash_cost, ell_aep]
    aep_colors = [_COLORS[0], _COLORS[2], _COLORS[3],
                  _COLORS[4] if ell_aep > 0 else "red"]

    bars2 = ax2.barh(range(len(aep_labels)), aep_vals, color=aep_colors, alpha=0.85, height=0.6)
    for bar, val in zip(bars2, aep_vals):
        x_pos = val + (abs(val) * 0.03 if val >= 0 else -abs(val) * 0.03)
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                 f"{val:+,.0f}" if abs(val) < 1e7 else f"{val:+.2e}",
                 va="center", ha="left" if val >= 0 else "right", fontsize=8.5)
    ax2.set_yticks(range(len(aep_labels)))
    ax2.set_yticklabels(aep_labels, fontsize=9.5)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Bits", fontsize=10)
    ax2.set_title(f"AEP Formula Breakdown\nℓ = {ell_aep:,.0f} bits", fontsize=10)
    ax2.grid(True, axis="x", alpha=0.25)
    ax2.invert_yaxis()

    fig.suptitle(
        f"EUR vs AEP Finite-Key Decomposition — {scenario}\n"
        f"EUR gain: {ell_eur - ell_aep:+,.0f} bits  "
        f"({'+∞' if ell_aep == 0 and ell_eur > 0 else f'{ell_eur/max(ell_aep,1):.1f}×'}× improvement)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_eur_bounds(result: PassResult, scenario: str) -> plt.Figure:
    """Bar chart: s_Z0^L, s_Z1^L, phi_Z^U, e1^U diagnostic."""
    eur_b = result.eur_decoy_bounds
    if eur_b is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No EUR bounds (AEP run)", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: count bounds
    ax = axes[0]
    labels  = ["s_Z0^L\n(vacuum key basis)", "s_Z1^L\n(single-photon key basis)"]
    vals    = [eur_b.s_Z0_lower, eur_b.s_Z1_lower]
    colors  = [_COLORS[0], _COLORS[1]]
    bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                f"{val:.2e}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Detected photons (lower bound)", fontsize=10)
    ax.set_title("Key-Basis Photon Count Bounds", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)

    # Right: error rate bounds
    ax2 = axes[1]
    labels2 = ["e1^U\n(test-basis\nsingle-photon QBER)", "φ_Z^U\n(phase error\nkey basis)"]
    vals2   = [eur_b.e1_upper, eur_b.phi_Z_upper]
    colors2 = [_COLORS[2], _COLORS[3]]
    ax2.bar(range(len(labels2)), vals2, color=colors2, alpha=0.85, width=0.5)
    ax2.axhline(0.11, color="red", ls="--", lw=1.0, label="BB84 QBER limit (11%)")
    ax2.axhline(0.5,  color="darkred", ls=":", lw=1.0, label="φ = 0.5 (no key)")
    for i, (label, val) in enumerate(zip(labels2, vals2)):
        ax2.text(i, val + 0.01, f"{val:.4f}", ha="center", fontsize=9)
    ax2.set_xticks(range(len(labels2)))
    ax2.set_xticklabels(labels2, fontsize=10)
    ax2.set_ylabel("Rate (dimensionless)", fontsize=10)
    ax2.set_ylim(0, 0.55)
    ax2.set_title("Error Rate Bounds (EUR Connection)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"EUR Decoy Bounds — {scenario}\n"
        f"Y1^L = {eur_b.Y1_lower:.4e}   Y0 = {eur_b.Y0_bound:.4e}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ── Cross-scenario comparison ─────────────────────────────────────────────────

def plot_comparison(all_results: dict[str, PassResult]) -> plt.Figure:
    """4-panel bar comparison: n_sifted, E_mu, ell_finite (EUR), ell_finite (AEP)."""
    labels  = list(all_results.keys())
    results = list(all_results.values())
    n       = len(labels)
    x       = np.arange(n)

    # Fetch AEP results stored as sub-key or re-run if needed
    ell_eur = [r[0].ell_finite for r in results]
    ell_aep = [r[1].ell_finite for r in results]
    n_sifted = [r[0].n_sifted  for r in results]
    e_mu     = [r[0].E_mu_weighted * 100 for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    def _bar(ax, vals, title, ylabel, log=False, threshold=None):
        bar_handles = []
        for i, v in enumerate(vals):
            color = _COLORS[i % 10]
            b = ax.bar(x[i], max(v, 1), color=color, alpha=0.85, width=0.6)
            bar_handles.append(b[0])
            ax.text(x[i], max(v, 1) * 1.05, f"{v:,.0f}" if v < 1e6 else f"{v:.2e}",
                    ha="center", fontsize=8.5)
        if threshold is not None:
            ax.axhline(threshold, color="red", ls="--", lw=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([lb.replace("_", "\n") for lb in labels], fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        if log:
            pos = [v for v in vals if v > 0]
            if pos:
                ax.set_yscale("log")
                ax.set_ylim(min(pos) * 0.1, max(pos) * 10)
        return bar_handles

    _bar(axes[0], n_sifted, "Total Sifted Bits", "bits", log=True)
    _bar(axes[1], e_mu,     "Signal QBER (E_μ)", "%", threshold=11.0)
    eur_bars = _bar(axes[2], ell_eur, "ℓ_finite — EUR", "bits", log=True)
    _bar(axes[3], ell_aep,  "ℓ_finite — AEP",  "bits", log=True)

    # Highlight EUR-only positive results
    for i, (ve, va) in enumerate(zip(ell_eur, ell_aep)):
        if ve > 0 and va == 0:
            eur_bars[i].set_edgecolor("gold")
            eur_bars[i].set_linewidth(2.5)

    patches = [Patch(color=_COLORS[i % 10], label=labels[i]) for i in range(n)]
    fig.legend(handles=patches, fontsize=9, loc="lower center", ncol=n,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Cross-Scenario Comparison  (EUR vs AEP)\n"
                 "Gold border = EUR certifies key; AEP gives zero",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ── Main report runner ────────────────────────────────────────────────────────

def generate_scenario_report(scenario_name: str) -> tuple[PassResult, PassResult]:
    """Generate all plots for one scenario. Returns (eur_result, aep_result)."""
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*60}")

    print("  Running EUR pass...")
    res_eur = _run(scenario_name, proof_method="eur")
    print(f"  EUR  ℓ_finite = {res_eur.ell_finite:,.0f} bits  "
          f"({'✓ go' if res_eur.go else '✗ zero'})")

    print("  Running AEP pass...")
    res_aep = _run(scenario_name, proof_method="aep")
    print(f"  AEP  ℓ_finite = {res_aep.ell_finite:,.0f} bits")

    # pass/ plots
    print("  Generating pass/ plots...")
    d_pass = _out_dir(scenario_name, "pass")
    _save(plot_pass_profile(res_eur, scenario_name),
          os.path.join(d_pass, "pass_profile.png"))
    _save(plot_loss_waterfall(res_eur, scenario_name),
          os.path.join(d_pass, "loss_waterfall.png"))
    _save(plot_key_budget(res_eur, scenario_name),
          os.path.join(d_pass, "key_budget.png"))

    # keyrate/ plots
    print("  Generating keyrate/ plots...")
    d_kr = _out_dir(scenario_name, "keyrate")
    _save(plot_eur_breakdown(res_eur, res_aep, scenario_name),
          os.path.join(d_kr, "eur_vs_aep.png"))
    _save(plot_eur_bounds(res_eur, scenario_name),
          os.path.join(d_kr, "eur_bounds.png"))

    return res_eur, res_aep


def run_all(scenario_names: list[str]) -> None:
    all_results: dict[str, tuple[PassResult, PassResult]] = {}

    for name in scenario_names:
        try:
            eur, aep = generate_scenario_report(name)
            all_results[name] = (eur, aep)
        except Exception as exc:
            print(f"  ERROR in {name}: {exc}")
            import traceback; traceback.print_exc()

    if len(all_results) >= 2:
        print("\n  Generating comparison/ plot...")
        d_cmp = os.path.join(_OUT_ROOT, "comparison")
        os.makedirs(d_cmp, exist_ok=True)
        fig = plot_comparison(all_results)
        path = os.path.join(d_cmp, "scenario_compare.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved → {os.path.relpath(path)}")

    print(f"\n  Done. Outputs under {os.path.relpath(_OUT_ROOT)}/")


if __name__ == "__main__":
    scenarios = sys.argv[1:] if len(sys.argv) > 1 else _DEFAULT_SCENARIOS
    run_all(scenarios)
