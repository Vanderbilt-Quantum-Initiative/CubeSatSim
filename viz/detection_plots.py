"""
viz/detection_plots.py — Visualisations for physics/detection.py.

Plots:
    plot_gain_vs_transmissivity   Q_μ(η) for several intensities; fading-averaged Q.
    plot_qber_vs_transmissivity   E(η) vs η; noise floor and signal-dominated regimes.
    plot_jensen_gaps              Gain and QBER gaps vs Rytov variance; validates integration.
    plot_fading_observables       ⟨Q⟩ and ⟨E⟩ vs mean η for LogNormal and GammaGamma.
    plot_all                      Render all four figures → viz/out/detection/.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from physics.detection import (
    instantaneous_qber, expected_gain, expected_qber,
    jensen_gaps, compute_detection, noise_yield,
)
from physics.detector import DetectorModel
from physics.turbulence import LogNormalFading, GammaGammaFading, select_fading_model

_COLORS = plt.cm.tab10.colors
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "detection")

# Reference detector (Si-SPAD, good cryo performance)
REF_DET = DetectorModel(eta_det=0.65, p_d=1e-6, tau_d=50e-9, delta_t=1e-9)

# Baseline noise yield (dark-only, night sky)
REF_Y0 = noise_yield(REF_DET, H_bg=0.0, Omega_FOV=1e-8,
                     A_rx=0.2, delta_lambda=1e-9, eta_rx=0.7)

# Reference geometry: 400 km zenith pass, mild turbulence
REF_ETA0 = 0.05      # mean static channel transmissivity
REF_SIGMA_R2 = 0.3   # Rytov variance → LogNormal regime


# ---------------------------------------------------------------------------
# Figure 1 — Gain Q_μ(η) vs transmissivity for several intensities
# ---------------------------------------------------------------------------

def plot_gain_vs_transmissivity() -> plt.Figure:
    """
    Q_μ(η) = 1 − (1−Y₀)·e^{−ημ} for μ ∈ {0.1, 0.3, 0.5, 1.0}.

    A horizontal band shows the fading-averaged ⟨Q_μ⟩ for the reference
    LogNormal channel (η₀ = 0.05, σ_R² = 0.3) at μ = 0.5. The gap between
    Q_μ(⟨η⟩) and ⟨Q_μ⟩ visualises the Jensen correction.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    etas = np.linspace(0.0, 0.3, 400)
    intensities = [0.1, 0.3, 0.5, 1.0]

    for i, mu in enumerate(intensities):
        Qs = [instantaneous_qber.__module__ and
              1.0 - (1.0 - REF_Y0) * np.exp(-eta * mu)
              for eta in etas]
        ax.plot(etas, Qs, color=_COLORS[i], lw=1.8, label=f"μ = {mu}")

    # Jensen illustration for μ = 0.5
    fading = LogNormalFading(eta_0=REF_ETA0, sigma_R2=REF_SIGMA_R2)
    mu_ref = 0.5
    Q_mean_eta = 1.0 - (1.0 - REF_Y0) * np.exp(-fading.mean_eta() * mu_ref)
    Q_avg = expected_gain(fading, mu_ref, REF_Y0)

    ax.axvline(fading.mean_eta(), color="gray", ls=":", lw=1, label=f"⟨η⟩ = {fading.mean_eta():.3f}")
    ax.axhline(Q_mean_eta, color=_COLORS[2], ls="--", lw=1, alpha=0.7,
               label=f"Q(⟨η⟩) = {Q_mean_eta:.4f}")
    ax.axhline(Q_avg, color=_COLORS[2], ls="-.", lw=1.5, alpha=0.9,
               label=f"⟨Q(η)⟩ = {Q_avg:.4f}")
    ax.fill_between([0, 0.3], Q_avg, Q_mean_eta, color=_COLORS[2],
                    alpha=0.12, label="Jensen gap")

    ax.set_xlabel("Transmissivity η", fontsize=11)
    ax.set_ylabel("Gain  Q(η)", fontsize=11)
    ax.set_title("Detection Gain vs Transmissivity\n"
                 "Jensen gap shown for μ = 0.5 (LogNormal, η₀ = 0.05, σ²_R = 0.3)",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — QBER E(η) vs transmissivity
# ---------------------------------------------------------------------------

def plot_qber_vs_transmissivity() -> plt.Figure:
    """
    E(η) = (e_opt·ημ + 0.5·Y₀) / (ημ + Y₀) across the full transmissivity range.

    Three regimes are visible:
        η → 0:   noise-dominated → E → 0.5 (random background detections)
        η ≈ Y₀/μ: crossover between noise and signal
        η → ∞:   signal-dominated → E → e_opt (optical alignment limit)
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    e_opts = [0.01, 0.02, 0.03, 0.05]
    etas = np.logspace(-7, 0, 600)

    # Left: QBER vs η for several e_opt values
    ax = axes[0]
    for i, e_opt in enumerate(e_opts):
        Es = [instantaneous_qber(eta, mu=0.5, e_opt=e_opt, Y_0=REF_Y0) for eta in etas]
        ax.semilogx(etas, Es, color=_COLORS[i], lw=1.8, label=f"e_opt = {e_opt:.2f}")

    ax.axhline(0.5, color="black", ls=":", lw=0.8, alpha=0.5, label="E = 0.5 (noise floor)")
    ax.axhline(0.11, color="red", ls="--", lw=1, alpha=0.7, label="BB84 threshold (11%)")
    ax.axvline(REF_Y0, color="gray", ls=":", lw=1, label=f"Y₀ = {REF_Y0:.0e}")

    # Mark mean eta for reference
    fading = LogNormalFading(eta_0=REF_ETA0, sigma_R2=REF_SIGMA_R2)
    ax.axvline(fading.mean_eta(), color="steelblue", ls="--", lw=1,
               label=f"⟨η⟩ = {fading.mean_eta():.3f}")

    ax.set_xlabel("Transmissivity η  (log scale)", fontsize=10)
    ax.set_ylabel("QBER  E(η)", fontsize=10)
    ax.set_title("QBER vs Transmissivity\n(μ = 0.5, Y₀ = 1×10⁻⁶)", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.set_xlim(1e-7, 1.0)
    ax.set_ylim(0, 0.55)
    ax.grid(True, which="both", alpha=0.2)

    # Right: QBER vs μ for several η values (signal-to-noise)
    ax = axes[1]
    mus = np.linspace(0.01, 1.5, 400)
    eta_vals = [0.001, 0.01, 0.05, 0.1, 0.3]
    for i, eta in enumerate(eta_vals):
        Es = [instantaneous_qber(eta, mu=mu, e_opt=0.03, Y_0=REF_Y0) for mu in mus]
        ax.plot(mus, Es, color=_COLORS[i], lw=1.8, label=f"η = {eta}")

    ax.axhline(0.11, color="red", ls="--", lw=1, alpha=0.7, label="BB84 threshold")
    ax.set_xlabel("Source intensity  μ  (photons/pulse)", fontsize=10)
    ax.set_ylabel("QBER  E(η, μ)", fontsize=10)
    ax.set_title("QBER vs Intensity\n(e_opt = 0.03, Y₀ = 1×10⁻⁶)", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 0.55)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Instantaneous QBER Structure", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Jensen gaps vs Rytov variance
# ---------------------------------------------------------------------------

def plot_jensen_gaps() -> plt.Figure:
    """
    Gain gap and QBER gap as a function of σ²_R (Rytov variance).

    The gap is zero at σ_R² = 0 (no fading; delta function at η₀) and grows
    monotonically with turbulence strength. Transition from LogNormal to
    GammaGamma at σ_R² = 0.75 is marked.

    A negative gap at any point would indicate an integration bug.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sigma_R2_vals = np.concatenate([
        np.linspace(0.01, 0.74, 60),   # LogNormal regime
        np.linspace(0.76, 3.0, 60),    # GammaGamma regime
    ])
    mu_vals = [0.1, 0.3, 0.5]
    threshold = 0.75

    for i, mu in enumerate(mu_vals):
        gain_gaps_ln, qber_gaps_ln = [], []
        gain_gaps_gg, qber_gaps_gg = [], []
        sr2_ln, sr2_gg = [], []

        for sr2 in sigma_R2_vals:
            fading = select_fading_model(sr2, REF_ETA0, threshold=threshold)
            gg, qg = jensen_gaps(fading, mu=mu, e_opt=0.03, Y_0=REF_Y0)
            if sr2 < threshold:
                gain_gaps_ln.append(gg)
                qber_gaps_ln.append(qg)
                sr2_ln.append(sr2)
            else:
                gain_gaps_gg.append(gg)
                qber_gaps_gg.append(qg)
                sr2_gg.append(sr2)

        axes[0].plot(sr2_ln, gain_gaps_ln, color=_COLORS[i], lw=1.8,
                     label=f"μ={mu}" if i == 0 else "_nolegend_")
        axes[0].plot(sr2_gg, gain_gaps_gg, color=_COLORS[i], lw=1.8, ls="--")
        axes[1].plot(sr2_ln, qber_gaps_ln, color=_COLORS[i], lw=1.8, label=f"μ = {mu}")
        axes[1].plot(sr2_gg, qber_gaps_gg, color=_COLORS[i], lw=1.8, ls="--")

    for ax, title in zip(axes, ["Gain Jensen Gap\nQ(⟨η⟩) − ⟨Q(η)⟩  (must be > 0)",
                                  "QBER Jensen Gap\n⟨E(η)⟩ − E(⟨η⟩)  (must be > 0)"]):
        ax.axvline(threshold, color="red", ls=":", lw=1.2, label="LN→GG threshold")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_xlabel("Rytov variance  σ²_R", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 3.0)

    axes[0].set_ylabel("Gain gap  (dimensionless)", fontsize=10)
    axes[1].set_ylabel("QBER gap  (dimensionless)", fontsize=10)

    # Add regime labels
    axes[0].text(0.35, axes[0].get_ylim()[1] * 0.9, "LogNormal", fontsize=8,
                 color="dimgray", ha="center")
    axes[0].text(1.5, axes[0].get_ylim()[1] * 0.9, "Gamma-Gamma\n(dashed)", fontsize=8,
                 color="dimgray", ha="center")

    fig.suptitle("Jensen Correction Gaps vs Turbulence Strength\n"
                 "(η₀ = 0.05, e_opt = 0.03, Y₀ = 1×10⁻⁶)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Fading-averaged ⟨Q⟩ and ⟨E⟩ vs mean channel transmissivity
# ---------------------------------------------------------------------------

def plot_fading_observables() -> plt.Figure:
    """
    ⟨Q_μ⟩ and ⟨E_μ⟩ vs η₀ (mean static transmissivity) for three turbulence levels.

    Compares:
      - No fading (σ_R² → 0): gain/QBER computed at η₀ directly.
      - Weak turbulence (σ_R² = 0.3, LogNormal).
      - Strong turbulence (σ_R² = 1.5, GammaGamma).

    Shows that fading always reduces gain and always raises QBER relative to
    the no-fading baseline — consistent with Jensen's inequality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    eta0_vals = np.logspace(-4, -0.5, 80)
    mu = 0.5
    e_opt = 0.03

    scenarios = [
        ("No fading (δ fn)",  0.001, "black",     "-"),
        ("LogNormal σ²=0.3",  0.3,   _COLORS[0],  "-"),
        ("LogNormal σ²=0.6",  0.6,   _COLORS[1],  "-"),
        ("GammaGamma σ²=1.0", 1.0,   _COLORS[2],  "--"),
        ("GammaGamma σ²=2.0", 2.0,   _COLORS[3],  "--"),
    ]

    for label, sr2, color, ls in scenarios:
        Qs, Es = [], []
        for eta0 in eta0_vals:
            fading = select_fading_model(sr2, eta0, threshold=0.75)
            Q = expected_gain(fading, mu, REF_Y0)
            E = expected_qber(fading, mu, e_opt, REF_Y0)
            Qs.append(Q)
            Es.append(E)

        axes[0].loglog(eta0_vals, Qs, color=color, ls=ls, lw=1.8, label=label)
        axes[1].semilogx(eta0_vals, Es, color=color, ls=ls, lw=1.8, label=label)

    axes[0].set_xlabel("Mean transmissivity  η₀  (log scale)", fontsize=10)
    axes[0].set_ylabel("Fading-averaged gain  ⟨Q_μ⟩", fontsize=10)
    axes[0].set_title("⟨Q_μ⟩ vs η₀\n(μ = 0.5, Y₀ = 1×10⁻⁶)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].set_xlim(1e-4, 0.35)

    axes[1].axhline(0.11, color="red", ls=":", lw=1, alpha=0.8, label="BB84 threshold")
    axes[1].set_xlabel("Mean transmissivity  η₀  (log scale)", fontsize=10)
    axes[1].set_ylabel("Fading-averaged QBER  ⟨E_μ⟩", fontsize=10)
    axes[1].set_title("⟨E_μ⟩ vs η₀\n(μ = 0.5, e_opt = 0.03)", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].set_xlim(1e-4, 0.35)
    axes[1].set_ylim(0, 0.5)

    fig.suptitle("Fading-Averaged Detection Observables vs Channel Transmissivity",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_all
# ---------------------------------------------------------------------------

def plot_all(save_dir: str | None = None) -> list[plt.Figure]:
    out = save_dir or _OUT_DIR
    os.makedirs(out, exist_ok=True)

    plots = [
        ("gain_vs_transmissivity",   plot_gain_vs_transmissivity),
        ("qber_vs_transmissivity",   plot_qber_vs_transmissivity),
        ("jensen_gaps",              plot_jensen_gaps),
        ("fading_observables",       plot_fading_observables),
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
