"""
viz/decoy_plots.py — Visualisations for physics/decoy.py.

Plots:
    plot_q1_bound               Q1_lower vs decoy intensity ratio ν/μ.
    plot_e1_bound               e1_upper vs Q1_lower for several decoy QBERs.
    plot_finite_vs_asymptotic   Bound tightening with PE sample size.
    plot_intensity_tradeoff     Q1_lower heatmap over (μ, ν/μ) space.
    plot_all                    Render all four → viz/out/decoy/.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from physics.decoy import asymptotic_bounds, finite_bounds
from core.types import DetectionResult

_COLORS = plt.cm.tab10.colors
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "decoy")

# Reference observables (realistic 400 km pass)
REF_SIGNAL = DetectionResult(Q=0.024, E=0.031, intensity=0.5, n_counts=0.0)
REF_DECOY  = DetectionResult(Q=0.010, E=0.035, intensity=0.1, n_counts=0.0)
REF_VACUUM = DetectionResult(Q=1e-6,  E=0.5,   intensity=0.0, n_counts=0.0)
REF_MU, REF_NU, REF_Y0 = 0.5, 0.1, 1e-6


# ---------------------------------------------------------------------------
# Figure 1 — Q1_lower vs ν/μ ratio
# ---------------------------------------------------------------------------

def plot_q1_bound() -> plt.Figure:
    """Q1_lower from asymptotic decoy bounds as ν varies from near 0 to μ."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    mu = 0.5
    nu_vals = np.linspace(0.02, 0.45, 200)
    Q_nu_cases = [
        (0.005, "Q_ν = 0.005 (poor link)"),
        (0.010, "Q_ν = 0.010 (reference)"),
        (0.020, "Q_ν = 0.020 (good link)"),
    ]

    for i, (Q_nu, label) in enumerate(Q_nu_cases):
        Q1_vals = []
        for nu in nu_vals:
            sig = DetectionResult(Q=0.024, E=0.031, intensity=mu,  n_counts=0.0)
            dec = DetectionResult(Q=Q_nu,  E=0.035, intensity=nu,   n_counts=0.0)
            vac = DetectionResult(Q=1e-6,  E=0.5,   intensity=0.0,  n_counts=0.0)
            try:
                b = asymptotic_bounds(sig, dec, vac, mu, nu, REF_Y0)
                Q1_vals.append(b.Q1_lower)
            except (ValueError, ZeroDivisionError):
                Q1_vals.append(float("nan"))
        ax.plot(nu_vals / mu, Q1_vals, color=_COLORS[i], lw=1.8, label=label)

    ax.axvline(REF_NU / mu, color="gray", ls=":", lw=1, label=f"Reference ν/μ = {REF_NU/mu:.1f}")
    ax.set_xlabel("Decoy intensity ratio  ν/μ", fontsize=11)
    ax.set_ylabel("Q1 lower bound", fontsize=11)
    ax.set_title("Single-Photon Gain Bound Q₁ vs Decoy Intensity Ratio\n"
                 "(μ = 0.5, Q_μ = 0.024, asymptotic bounds)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — e1_upper vs Q1_lower for several E_ν values
# ---------------------------------------------------------------------------

def plot_e1_bound() -> plt.Figure:
    """e1_upper as a function of decoy QBER E_ν at several Q_ν levels."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Vary nu to get different Q1_lower values, for several E_nu scenarios
    mu = 0.5
    nu_vals = np.linspace(0.03, 0.45, 150)
    E_nu_cases = [
        (0.03, "E_ν = 0.03"),
        (0.05, "E_ν = 0.05"),
        (0.08, "E_ν = 0.08"),
    ]

    for i, (E_nu, label) in enumerate(E_nu_cases):
        q1s, e1s = [], []
        for nu in nu_vals:
            sig = DetectionResult(Q=0.024, E=0.031, intensity=mu,  n_counts=0.0)
            dec = DetectionResult(Q=0.010, E=E_nu,  intensity=nu,   n_counts=0.0)
            vac = DetectionResult(Q=1e-6,  E=0.5,   intensity=0.0,  n_counts=0.0)
            try:
                b = asymptotic_bounds(sig, dec, vac, mu, nu, REF_Y0)
                if b.Q1_lower > 0:
                    q1s.append(b.Q1_lower)
                    e1s.append(b.e1_upper)
            except (ValueError, ZeroDivisionError):
                pass
        if q1s:
            # Sort by Q1_lower for a clean curve
            pairs = sorted(zip(q1s, e1s))
            q1s_s, e1s_s = zip(*pairs)
            ax.plot(q1s_s, e1s_s, color=_COLORS[i], lw=1.8, label=label)

    ax.axhline(0.5,  color="black", ls=":", lw=1, alpha=0.6, label="e1 = 0.5 (worst case)")
    ax.axhline(0.11, color="red",   ls="--", lw=1, alpha=0.8, label="BB84 threshold (11%)")
    ax.set_xlabel("Q1 lower bound", fontsize=11)
    ax.set_ylabel("e1 upper bound", fontsize=11)
    ax.set_title("Single-Photon Phase Error Bound e₁ vs Q₁\n"
                 "(varying ν, μ = 0.5, Q_μ = 0.024)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.55)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Finite vs asymptotic bounds vs n_PE
# ---------------------------------------------------------------------------

def plot_finite_vs_asymptotic() -> plt.Figure:
    """Q1_lower and e1_upper tighten toward asymptotic as n_PE grows."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    n_PE_vals = np.logspace(4, 9, 80)

    b_asym = asymptotic_bounds(REF_SIGNAL, REF_DECOY, REF_VACUUM,
                               REF_MU, REF_NU, REF_Y0)

    for conf_i, (conf, label) in enumerate([(0.90, "90%"), (0.99, "99%"), (0.999, "99.9%")]):
        q1_fin, e1_fin = [], []
        for n in n_PE_vals:
            b = finite_bounds(REF_SIGNAL, REF_DECOY, REF_VACUUM,
                              REF_MU, REF_NU, REF_Y0,
                              n_PE=n, confidence=conf)
            q1_fin.append(b.Q1_lower)
            e1_fin.append(b.e1_upper)

        axes[0].semilogx(n_PE_vals, q1_fin, color=_COLORS[conf_i], lw=1.8, label=label)
        axes[1].semilogx(n_PE_vals, e1_fin, color=_COLORS[conf_i], lw=1.8, label=label)

    # Asymptotic reference
    axes[0].axhline(b_asym.Q1_lower, color="black", ls="--", lw=1.2,
                    label=f"Asymptotic = {b_asym.Q1_lower:.4f}")
    axes[1].axhline(b_asym.e1_upper, color="black", ls="--", lw=1.2,
                    label=f"Asymptotic = {b_asym.e1_upper:.4f}")
    axes[1].axhline(0.11, color="red", ls=":", lw=1, alpha=0.7, label="BB84 threshold")

    axes[0].set_xlabel("PE sample size n_PE", fontsize=10)
    axes[0].set_ylabel("Q1 lower bound", fontsize=10)
    axes[0].set_title("Q₁ bound tightens with PE data", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].set_ylim(0, None)

    axes[1].set_xlabel("PE sample size n_PE", fontsize=10)
    axes[1].set_ylabel("e1 upper bound", fontsize=10)
    axes[1].set_title("e₁ bound tightens with PE data", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].set_ylim(0, 0.55)

    fig.suptitle("Finite-Key Decoy Bounds vs PE Sample Size\n"
                 "(signal Q=0.024 E=0.031 · decoy Q=0.010 E=0.035 · μ=0.5 ν=0.1)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Q1_lower heatmap over (μ, ν/μ)
# ---------------------------------------------------------------------------

def plot_intensity_tradeoff() -> plt.Figure:
    """2D heatmap: Q1_lower as a function of signal intensity μ and ratio ν/μ."""
    fig, ax = plt.subplots(figsize=(7, 5))

    mu_vals  = np.linspace(0.2, 1.0, 50)
    ratio_vals = np.linspace(0.05, 0.5, 50)   # nu/mu

    Q_mu_ref = 0.024   # roughly proportional to mu, fix for illustration
    E_nu_ref = 0.035
    Y_0 = 1e-6

    Z = np.zeros((len(ratio_vals), len(mu_vals)))
    for j, mu in enumerate(mu_vals):
        for i, ratio in enumerate(ratio_vals):
            nu = ratio * mu
            if nu <= 0 or nu >= mu:
                continue
            # Q_nu scales roughly linearly with nu (high-loss regime: Q_nu ≈ nu * eta_0)
            Q_nu = Q_mu_ref * (nu / 0.5)
            sig = DetectionResult(Q=Q_mu_ref, E=0.031, intensity=mu,  n_counts=0.0)
            dec = DetectionResult(Q=Q_nu,     E=E_nu_ref, intensity=nu, n_counts=0.0)
            vac = DetectionResult(Q=1e-6,     E=0.5,   intensity=0.0, n_counts=0.0)
            try:
                b = asymptotic_bounds(sig, dec, vac, mu, nu, Y_0)
                Z[i, j] = b.Q1_lower
            except (ValueError, ZeroDivisionError):
                Z[i, j] = 0.0

    im = ax.contourf(mu_vals, ratio_vals, Z, levels=20, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Q1 lower bound")
    ax.contour(mu_vals, ratio_vals, Z, levels=[0], colors="red", linewidths=1.5)

    # Mark reference point
    ax.plot(REF_MU, REF_NU / REF_MU, "w*", ms=12, label=f"Reference (μ={REF_MU}, ν/μ={REF_NU/REF_MU:.1f})")
    ax.set_xlabel("Signal intensity  μ  (photons/pulse)", fontsize=11)
    ax.set_ylabel("Decoy intensity ratio  ν/μ", fontsize=11)
    ax.set_title("Q₁ Lower Bound vs Signal/Decoy Intensity\n"
                 "(Q_μ = 0.024 fixed, Q_ν ∝ ν, E_ν = 0.035, asymptotic)",
                 fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_all
# ---------------------------------------------------------------------------

def plot_all(save_dir: str | None = None) -> list[plt.Figure]:
    out = save_dir or _OUT_DIR
    os.makedirs(out, exist_ok=True)

    plots = [
        ("q1_bound",              plot_q1_bound),
        ("e1_bound",              plot_e1_bound),
        ("finite_vs_asymptotic",  plot_finite_vs_asymptotic),
        ("intensity_tradeoff",    plot_intensity_tradeoff),
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
