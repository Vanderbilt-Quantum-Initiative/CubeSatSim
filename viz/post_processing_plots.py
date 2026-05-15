"""
viz/post_processing_plots.py — Visualisations for physics/post_processing.py.

Plots:
    plot_ec_efficiency        f_EC vs block size for Cascade and LDPC.
    plot_sifting_funnel       Bit budget from pulses to key bits.
    plot_classical_bandwidth  Required RF bandwidth vs sifted bits.
    plot_pe_tradeoff          ℓ_finite vs r_PE for several n_sifted.
    plot_all                  Render all four → viz/out/post_processing/.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from physics.post_processing import (
    ec_efficiency, classical_bandwidth_check, pe_split,
)
from physics.decoy import asymptotic_bounds, finite_bounds
from physics.keyrate import gllp_asymptotic, finite_key_length, binary_entropy
from core.types import DetectionResult, SourceConfig, PostProcessingConfig

_COLORS = plt.cm.tab10.colors
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "post_processing")

# Reference scenario
REF_SIGNAL = DetectionResult(Q=0.024, E=0.031, intensity=0.5, n_counts=0.0)
REF_DECOY  = DetectionResult(Q=0.010, E=0.035, intensity=0.1, n_counts=0.0)
REF_VACUUM = DetectionResult(Q=1e-6,  E=0.5,   intensity=0.0, n_counts=0.0)
REF_MU, REF_NU, REF_Y0 = 0.5, 0.1, 1e-6


# ---------------------------------------------------------------------------
# Figure 1 — EC efficiency vs block size
# ---------------------------------------------------------------------------

def plot_ec_efficiency() -> plt.Figure:
    """f_EC vs n_key for Cascade and LDPC at three QBER values."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    n_keys = np.logspace(3, 8, 200)
    E_mus  = [0.02, 0.05, 0.10]

    for i, E_mu in enumerate(E_mus):
        f_cas  = [ec_efficiency(n, E_mu, "cascade") for n in n_keys]
        f_ldpc = [ec_efficiency(n, E_mu, "ldpc")    for n in n_keys]
        label_cas  = f"Cascade  E={E_mu:.2f}" if i == 0 else f"E={E_mu:.2f}"
        label_ldpc = f"LDPC  E={E_mu:.2f}"    if i == 0 else f"E={E_mu:.2f}"
        ax.semilogx(n_keys, f_cas,  color=_COLORS[i], lw=1.8, ls="-",  label=label_cas)
        ax.semilogx(n_keys, f_ldpc, color=_COLORS[i], lw=1.8, ls="--", label=label_ldpc)

    ax.axhline(1.0, color="black", ls=":", lw=1, alpha=0.7, label="f_EC = 1.0 (Shannon limit)")

    # Legend proxy for line styles
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], color="gray", ls="-",  lw=1.8, label="Cascade (interactive)"),
        Line2D([0], [0], color="gray", ls="--", lw=1.8, label="LDPC (one-way)"),
    ]
    ax.legend(handles=proxies + [
        Line2D([0], [0], color=_COLORS[i], lw=1.8,
               label=f"E_μ = {E_mus[i]:.2f}") for i in range(3)
    ] + [Line2D([0], [0], color="black", ls=":", lw=1, label="Shannon limit")],
              fontsize=8, ncol=2)

    ax.set_xlabel("Key block size  n_key  (sifted bits)", fontsize=11)
    ax.set_ylabel("EC efficiency  f_EC", fontsize=11)
    ax.set_title("Error-Correction Efficiency vs Block Size\n"
                 "Cascade (solid) vs LDPC (dashed)", fontsize=10)
    ax.set_ylim(0.95, 1.25)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Sifting funnel
# ---------------------------------------------------------------------------

def plot_sifting_funnel() -> plt.Figure:
    """Horizontal bar chart showing the bit budget from pulses to key bits."""
    fig, ax = plt.subplots(figsize=(8, 4))

    f_clock = 100e6
    T_pass  = 200.0
    P_mu    = 0.6
    Q_mu    = 0.024
    q       = 0.82
    r_PE    = 0.10

    N_total    = f_clock * T_pass
    N_signal   = N_total * P_mu
    N_detected = N_signal * Q_mu
    N_sifted   = N_detected * q
    N_key      = N_sifted * (1 - r_PE)

    stages = [
        ("Total pulses",       N_total,    _COLORS[0]),
        ("Signal pulses (P_μ)", N_signal,   _COLORS[1]),
        ("Detections (Q_μ)",   N_detected, _COLORS[2]),
        ("Sifted bits (q)",    N_sifted,   _COLORS[3]),
        ("Key bits (1−r_PE)",  N_key,      _COLORS[4]),
    ]

    for i, (label, val, color) in enumerate(stages):
        ax.barh(i, val, color=color, alpha=0.85)
        ax.text(val * 1.05, i, f"{val:.2e}", va="center", fontsize=9)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([s[0] for s in stages], fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Count  (log scale)", fontsize=11)
    ax.set_title("Sifting Funnel: Pulses to Key Bits\n"
                 f"(f_clock={f_clock/1e6:.0f} MHz, T_pass={T_pass:.0f}s, "
                 f"P_μ={P_mu}, Q_μ={Q_mu}, q={q}, r_PE={r_PE})", fontsize=10)
    ax.grid(True, which="both", axis="x", alpha=0.25)
    ax.set_xlim(1e4, N_total * 3)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Required classical bandwidth vs n_sifted
# ---------------------------------------------------------------------------

def plot_classical_bandwidth() -> plt.Figure:
    """Required RF bandwidth to support post-processing vs n_sifted."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    n_sifteds = np.logspace(4, 8, 200)
    E_mu  = 0.05
    f_EC  = 1.12
    T_pass = 200.0

    for i, alg in enumerate(["cascade", "ldpc"]):
        bws = []
        for n in n_sifteds:
            _, vol, _ = classical_bandwidth_check(n, E_mu, f_EC, alg, T_pass, rf_bandwidth=1e9)
            bws.append(vol / T_pass)
        ax.loglog(n_sifteds, bws, color=_COLORS[i], lw=1.8,
                  label=f"{alg.capitalize()}")

    # Reference RF bandwidth lines
    for bw, label in [(1e6, "1 Mbps"), (10e6, "10 Mbps"), (100e6, "100 Mbps")]:
        ax.axhline(bw, color="gray", ls="--", lw=0.9, alpha=0.7)
        ax.text(1.2e4, bw * 1.15, label, fontsize=8, color="gray")

    ax.set_xlabel("Sifted bits  n_sifted  (log scale)", fontsize=11)
    ax.set_ylabel("Required bandwidth  (bits/s)", fontsize=11)
    ax.set_title("Required Classical Bandwidth vs Sifted Bits\n"
                 f"(E_μ = {E_mu}, f_EC = {f_EC}, T_pass = {T_pass:.0f}s)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlim(1e4, 1e8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — ℓ_finite vs r_PE
# ---------------------------------------------------------------------------

def plot_pe_tradeoff() -> plt.Figure:
    """Optimal r_PE: more PE tightens decoy bounds but costs key bits."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    r_PE_vals = np.linspace(0.01, 0.5, 60)
    q    = 0.82
    eps  = 1e-10

    n_sifted_cases = [1e6, 5e6, 1e7]

    for i, n_sifted in enumerate(n_sifted_cases):
        ells = []
        for r_PE in r_PE_vals:
            n_PE, n_key = pe_split(n_sifted, r_PE)
            f_EC = ec_efficiency(n_key, REF_SIGNAL.E, "ldpc")

            # Finite decoy bounds using this PE budget
            b = finite_bounds(REF_SIGNAL, REF_DECOY, REF_VACUUM,
                              REF_MU, REF_NU, REF_Y0,
                              n_PE=n_PE, confidence=0.99)
            R = gllp_asymptotic(b, REF_SIGNAL, q, f_EC)
            ell = finite_key_length(n_key, R, REF_SIGNAL.E, f_EC, eps)
            ells.append(ell)

        ax.plot(r_PE_vals, ells, color=_COLORS[i], lw=1.8,
                label=f"n_sifted = {n_sifted:.0e}")

        # Mark optimum
        best_idx = int(np.argmax(ells))
        if ells[best_idx] > 0:
            ax.plot(r_PE_vals[best_idx], ells[best_idx], "*",
                    color=_COLORS[i], ms=10)

    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("PE fraction  r_PE", fontsize=11)
    ax.set_ylabel("Finite-key length  ℓ_finite  (bits)", fontsize=11)
    ax.set_title("Key Bits vs PE Fraction\n"
                 "(★ = optimum, confidence = 99%, ε_PA = 10⁻¹⁰)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_all
# ---------------------------------------------------------------------------

def plot_all(save_dir: str | None = None) -> list[plt.Figure]:
    out = save_dir or _OUT_DIR
    os.makedirs(out, exist_ok=True)

    plots = [
        ("ec_efficiency",        plot_ec_efficiency),
        ("sifting_funnel",       plot_sifting_funnel),
        ("classical_bandwidth",  plot_classical_bandwidth),
        ("pe_tradeoff",          plot_pe_tradeoff),
    ]

    figs = []
    for name, fn in plots:
        fig = fn()
        path = os.path.join(out, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved → {path}")
        figs.append(fig)
    return figs


if __name__ == "__main__":
    plot_all()
    plt.show()
