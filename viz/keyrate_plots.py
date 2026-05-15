"""
viz/keyrate_plots.py — Visualisations for physics/keyrate.py.

Plots:
    plot_binary_entropy       H₂(x) and (1−H₂(x)) across [0, 1].
    plot_key_fraction_vs_qber R vs E_μ for several decoy bound qualities.
    plot_finite_key_length    ℓ_finite vs n_key for several ε_PA values.
    plot_skbr_vs_eta0         SKBR vs η₀ for several clock rates.
    plot_all                  Render all four → viz/out/keyrate/.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from physics.keyrate import binary_entropy, gllp_asymptotic, finite_key_length
from core.types import DetectionResult, DecoyBounds

_COLORS = plt.cm.tab10.colors
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "keyrate")


# ---------------------------------------------------------------------------
# Figure 1 — Binary entropy
# ---------------------------------------------------------------------------

def plot_binary_entropy() -> plt.Figure:
    """H₂(x) and the residual key fraction 1−H₂(x) over [0, 1]."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    xs = np.linspace(0.001, 0.999, 500)
    H2 = np.array([binary_entropy(x) for x in xs])

    ax.plot(xs, H2,       color=_COLORS[0], lw=2,   label="H₂(x)  — entropy")
    ax.plot(xs, 1.0 - H2, color=_COLORS[1], lw=2,   label="1 − H₂(x)  — key fraction (PA only)")

    ax.axvline(0.11, color="red",   ls="--", lw=1.2, label="BB84 threshold (11%)")
    ax.axvline(0.5,  color="gray",  ls=":",  lw=1,   alpha=0.7)
    ax.axhline(0.5,  color="gray",  ls=":",  lw=0.8, alpha=0.5)

    # Annotate thresholds
    ax.annotate("H₂(0.11) ≈ 0.50", xy=(0.11, binary_entropy(0.11)),
                xytext=(0.18, 0.55), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("H₂(0.5) = 1.0", xy=(0.5, 1.0),
                xytext=(0.55, 0.92), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("Error probability  x", fontsize=11)
    ax.set_ylabel("Bits", fontsize=11)
    ax.set_title("Binary Shannon Entropy H₂(x)\nand Residual Key Fraction 1 − H₂(x)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Key fraction R vs signal QBER
# ---------------------------------------------------------------------------

def plot_key_fraction_vs_qber() -> plt.Figure:
    """R from GLLP vs E_μ for three decoy bound qualities."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    E_mu_vals = np.linspace(0.001, 0.149, 300)
    Q_mu = 0.024
    q    = 0.82    # efficient BB84
    f_EC = 1.16

    # Three decoy bound scenarios: (Q1_lower, e1_upper, label)
    scenarios = [
        (0.040, 0.030, "Good bounds  (Q1=0.040, e1=0.030)"),
        (0.025, 0.040, "Medium bounds (Q1=0.025, e1=0.040)"),
        (0.010, 0.060, "Poor bounds  (Q1=0.010, e1=0.060)"),
    ]

    for i, (Q1, e1, label) in enumerate(scenarios):
        bounds = DecoyBounds(Q1_lower=Q1, e1_upper=e1, mode="asymptotic")
        Rs = []
        for E_mu in E_mu_vals:
            sig = DetectionResult(Q=Q_mu, E=E_mu, intensity=0.5, n_counts=0.0)
            Rs.append(gllp_asymptotic(bounds, sig, q, f_EC))
        Rs = np.array(Rs)
        ax.plot(E_mu_vals, Rs, color=_COLORS[i], lw=1.8, label=label)

    ax.axhline(0, color="black", lw=0.9, alpha=0.7)
    ax.axvline(0.11, color="red", ls="--", lw=1.2, alpha=0.8, label="BB84 threshold (11%)")
    ax.fill_between(E_mu_vals, -0.005, 0, color="red", alpha=0.08, label="R < 0 (no key)")

    ax.set_xlabel("Signal QBER  E_μ", fontsize=11)
    ax.set_ylabel("GLLP key fraction  R  (bits/sifted photon)", fontsize=11)
    ax.set_title("Key Fraction R vs Signal QBER\n"
                 "(Q_μ = 0.024, q = 0.82, f_EC = 1.16)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Finite-key length vs block size
# ---------------------------------------------------------------------------

def plot_finite_key_length() -> plt.Figure:
    """ℓ_finite vs n_key for several ε_PA; shows AEP correction dominance."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    R     = 0.018
    E_mu  = 0.031
    f_EC  = 1.16
    n_keys = np.logspace(4, 9, 300)

    eps_scenarios = [
        (1e-6,  "ε_PA = 10⁻⁶"),
        (1e-9,  "ε_PA = 10⁻⁹"),
        (1e-12, "ε_PA = 10⁻¹²"),
    ]

    for i, (eps, label) in enumerate(eps_scenarios):
        ells = np.array([finite_key_length(n, R, E_mu, f_EC, eps) for n in n_keys])
        ax.loglog(n_keys[ells > 0], ells[ells > 0], color=_COLORS[i], lw=1.8, label=label)
        # Show zero region
        zero_mask = ells == 0
        if zero_mask.any():
            ax.axvline(n_keys[~zero_mask][0] if (~zero_mask).any() else n_keys[-1],
                       color=_COLORS[i], ls=":", lw=0.8, alpha=0.5)

    # Asymptotic upper bound n*R
    ax.loglog(n_keys, n_keys * R, color="black", ls="--", lw=1.2,
              alpha=0.7, label="n · R  (asymptotic, no penalty)")

    ax.set_xlabel("Key block size  n_key  (sifted bits)", fontsize=11)
    ax.set_ylabel("Finite-key length  ℓ_finite  (bits)", fontsize=11)
    ax.set_title("Finite-Key Length vs Block Size\n"
                 f"(R = {R}, E_μ = {E_mu}, f_EC = {f_EC})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlim(1e4, 1e9)

    # Annotate AEP regime
    ax.text(2e5, 5, "← AEP correction\n   dominates", fontsize=8, color="dimgray")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — SKBR vs η₀
# ---------------------------------------------------------------------------

def plot_skbr_vs_eta0() -> plt.Figure:
    """SKBR vs mean channel transmissivity for several clock rates."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    eta0_vals = np.logspace(-4, -0.5, 200)
    mu   = 0.5
    e_opt = 0.03
    q    = 0.82
    f_EC = 1.16
    Y_0  = 1e-6
    e1   = 0.04    # conservative single-photon phase error

    f_clocks = [10e6, 50e6, 100e6]

    for i, f_clk in enumerate(f_clocks):
        skbrs = []
        for eta0 in eta0_vals:
            # High-loss approximation: Q_mu ≈ eta0*mu + Y_0, Q1 ≈ 0.7*Q_mu
            Q_mu = 1.0 - (1.0 - Y_0) * np.exp(-eta0 * mu)
            Q1   = 0.7 * Q_mu    # rough lower bound
            E_mu = (e_opt * eta0 * mu + 0.5 * Y_0) / (eta0 * mu + Y_0)
            bounds = DecoyBounds(Q1_lower=max(0, Q1), e1_upper=e1, mode="asymptotic")
            sig = DetectionResult(Q=Q_mu, E=E_mu, intensity=mu, n_counts=0.0)
            R = gllp_asymptotic(bounds, sig, q, f_EC)
            skbrs.append(f_clk * R)
        skbrs = np.array(skbrs)

        positive = skbrs > 0
        label = f"f_clock = {f_clk/1e6:.0f} MHz"
        if positive.any():
            ax.semilogx(eta0_vals[positive], skbrs[positive] / 1e3,
                        color=_COLORS[i], lw=1.8, label=label)
        # Break-even marker
        if positive.any() and (~positive).any():
            be_idx = np.where(positive)[0][0]
            ax.axvline(eta0_vals[be_idx], color=_COLORS[i], ls=":", lw=0.9, alpha=0.6)

    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("Mean channel transmissivity  η₀  (log scale)", fontsize=11)
    ax.set_ylabel("Secret key bit rate  SKBR  (kbps)", fontsize=11)
    ax.set_title("Secret Key Bit Rate vs Channel Transmissivity\n"
                 "(μ=0.5, e_opt=0.03, q=0.82, f_EC=1.16, asymptotic bounds)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlim(1e-4, 0.35)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_all
# ---------------------------------------------------------------------------

def plot_all(save_dir: str | None = None) -> list[plt.Figure]:
    out = save_dir or _OUT_DIR
    os.makedirs(out, exist_ok=True)

    plots = [
        ("binary_entropy",       plot_binary_entropy),
        ("key_fraction_vs_qber", plot_key_fraction_vs_qber),
        ("finite_key_length",    plot_finite_key_length),
        ("skbr_vs_eta0",         plot_skbr_vs_eta0),
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
