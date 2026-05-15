"""
viz/source_plots.py — Visualisations for physics/source.py.

Plots:
    plot_sifting_factor      q vs P_X: efficient BB84 sifting recovery.
    plot_prep_probability    Valid (P_mu, P_nu) preparation space.
    plot_qrng_constraint     QRNG rate required to sustain f_clock.
    plot_pulse_budget        Stacked pulse-type budget for a reference config.
    plot_all                 Render all four figures → viz/out/source/.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

from physics.source import SourceConfig, QRNGModel

_COLORS = plt.cm.tab10.colors
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "source")

# Reference source configuration (representative CubeSat QKD)
REF_SRC = SourceConfig(
    mu=0.5, nu=0.1, P_mu=0.6, P_nu=0.3, P_X=0.9, f_clock=50e6
)


# ---------------------------------------------------------------------------
# Figure 1 — Sifting factor vs P_X
# ---------------------------------------------------------------------------

def plot_sifting_factor() -> plt.Figure:
    """
    q = P_X² + (1−P_X)² vs basis-choice probability P_X.

    Shows how efficient / asymmetric BB84 (P_X → 1) recovers the factor-of-2
    sifting penalty of standard BB84 (P_X = 0.5, q = 0.5) with no hardware change.
    Key design decision owned by Software / Mission Operations.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    P_X = np.linspace(0.5, 1.0, 500)
    q = P_X**2 + (1 - P_X)**2

    ax.plot(P_X, q, color=_COLORS[0], lw=2.5, label=r"$q = P_X^2 + (1-P_X)^2$")

    # Annotate standard and efficient limits
    ax.axvline(0.5, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax.axhline(0.5, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax.text(0.505, 0.46, r"Standard BB84 ($P_X=0.5$, $q=0.5$)", fontsize=8.5, color="gray")

    # Reference config
    q_ref = REF_SRC.sifting_factor()
    ax.scatter([REF_SRC.P_X], [q_ref], color=_COLORS[1], zorder=5, s=80,
               label=f"Reference ($P_X={REF_SRC.P_X}$, $q={q_ref:.3f}$)")
    ax.annotate(f"  q = {q_ref:.3f}\n  +{(q_ref/0.5 - 1)*100:.0f}% vs standard",
                xy=(REF_SRC.P_X, q_ref),
                xytext=(REF_SRC.P_X - 0.12, q_ref - 0.08),
                fontsize=8.5, color=_COLORS[1],
                arrowprops=dict(arrowstyle="->", color=_COLORS[1], lw=1.2))

    ax.fill_between(P_X, 0.5, q, alpha=0.12, color=_COLORS[0],
                    label="Gain over standard BB84")

    ax.set_xlabel(r"Basis-choice probability $P_X$", fontsize=11)
    ax.set_ylabel(r"Sifting factor $q$", fontsize=11)
    ax.set_title("Sifting Factor vs Basis-Choice Probability\n"
                 r"(Efficient BB84: $P_X \to 1$ recovers the factor-of-2 penalty)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Preparation probability space
# ---------------------------------------------------------------------------

def plot_prep_probability() -> plt.Figure:
    """
    Valid (P_mu, P_nu) region: the triangle P_mu ≥ 0, P_nu ≥ 0, P_mu+P_nu ≤ 1.

    Contours show P_vac = 1 − P_mu − P_nu (vacuum fraction).
    The vacuum fraction is not wasted: vacuum pulses are used as a third
    decoy intensity and tighten the decoy bounds on Y_0.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    P_mu_vals = np.linspace(0, 1, 400)
    P_nu_vals = np.linspace(0, 1, 400)
    PM, PN = np.meshgrid(P_mu_vals, P_nu_vals)
    PV = 1.0 - PM - PN

    # Valid region mask
    valid = PV >= 0
    PV_plot = np.where(valid, PV, np.nan)

    cf = ax.contourf(PM, PN, PV_plot, levels=np.linspace(0, 1, 21), cmap="YlOrRd_r")
    cs = ax.contour(PM, PN, PV_plot, levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                    colors="white", linewidths=0.8, alpha=0.7)
    ax.clabel(cs, fmt=r"$P_{vac}=%.2f$", fontsize=7.5, inline=True)
    fig.colorbar(cf, ax=ax, label=r"Vacuum fraction $P_{vac} = 1 - P_\mu - P_\nu$")

    # Triangle boundary
    tri_x = [0, 1, 0, 0]
    tri_y = [0, 0, 1, 0]
    ax.plot(tri_x, tri_y, "k-", lw=1.5, alpha=0.6)

    # Reference point
    ax.scatter([REF_SRC.P_mu], [REF_SRC.P_nu], color="white", edgecolors="black",
               s=120, zorder=6, linewidths=1.5)
    ax.annotate(
        f" Ref: $P_\\mu={REF_SRC.P_mu}$, $P_\\nu={REF_SRC.P_nu}$\n"
        f" $P_{{vac}}={REF_SRC.P_vac:.2f}$",
        xy=(REF_SRC.P_mu, REF_SRC.P_nu),
        xytext=(REF_SRC.P_mu + 0.08, REF_SRC.P_nu + 0.08),
        fontsize=8.5, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2)
    )

    # Region labels
    ax.text(0.1, 0.05, "Invalid\n($P_{vac} < 0$)", fontsize=8, color="gray",
            transform=ax.transData)
    ax.text(0.6, 0.28, "Invalid\nregion", fontsize=7, color="gray", alpha=0.6,
            rotation=-45)

    ax.set_xlabel(r"Signal probability $P_\mu$", fontsize=11)
    ax.set_ylabel(r"Decoy probability $P_\nu$", fontsize=11)
    ax.set_title("Preparation Probability Space\n"
                 r"($P_{vac} = 1 - P_\mu - P_\nu \geq 0$ required)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — QRNG rate constraint
# ---------------------------------------------------------------------------

def plot_qrng_constraint() -> plt.Figure:
    """
    Two panels:
      Left:  Required QRNG rate vs f_clock for different bits_per_pulse.
      Right: Max sustainable f_clock vs QRNG rate — technology comparison.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    f_clock_MHz = np.linspace(1, 500, 500)

    # --- Left: required rate vs f_clock ---
    bpp_configs = [
        (2, "2 bits/pulse (basis only)", _COLORS[0], "-"),
        (3, "3 bits/pulse (basis + intensity)", _COLORS[1], "-"),
        (4, "4 bits/pulse (basis + intensity + spare)", _COLORS[2], "--"),
    ]
    for bpp, label, color, ls in bpp_configs:
        req_gbps = f_clock_MHz * bpp / 1e3   # Gbps
        ax_l.plot(f_clock_MHz, req_gbps, color=color, ls=ls, lw=2, label=label)

    # QRNG technology lines
    for rate_gbps, tech, color in [(0.1, "100 Mbps QRNG", "gray"),
                                    (1.0, "1 Gbps QRNG",   _COLORS[3]),
                                    (10.0, "10 Gbps QRNG", _COLORS[4])]:
        ax_l.axhline(rate_gbps, color=color, lw=1.2, ls=":", alpha=0.8)
        ax_l.text(505, rate_gbps, f" {tech}", fontsize=7.5, color=color, va="center")

    # Reference operating point
    qrng_ref = QRNGModel(rate=300e6, bits_per_pulse=3)
    ax_l.scatter([REF_SRC.f_clock / 1e6], [REF_SRC.f_clock * 3 / 1e9],
                 color=_COLORS[1], s=80, zorder=5)
    ax_l.annotate(f"Reference\n({REF_SRC.f_clock/1e6:.0f} MHz, {REF_SRC.f_clock*3/1e6:.0f} Mbps)",
                  xy=(REF_SRC.f_clock / 1e6, REF_SRC.f_clock * 3 / 1e9),
                  xytext=(REF_SRC.f_clock / 1e6 + 50, REF_SRC.f_clock * 3 / 1e9 + 0.15),
                  fontsize=8, arrowprops=dict(arrowstyle="->", color="gray", lw=1))

    ax_l.set_xlabel(r"Clock rate $f_{clock}$ (MHz)", fontsize=11)
    ax_l.set_ylabel("Required QRNG output rate (Gbps)", fontsize=11)
    ax_l.set_title("QRNG Rate Requirement vs Clock Rate", fontsize=11, fontweight="bold")
    ax_l.legend(fontsize=9, loc="upper left")
    ax_l.set_xlim(0, 500)
    ax_l.set_ylim(0, 2)
    ax_l.grid(True, alpha=0.3)

    # --- Right: max f_clock vs QRNG rate ---
    rate_Mbps = np.logspace(1, 4, 300)  # 10 Mbps to 10 Gbps
    for bpp, label, color, ls in bpp_configs:
        f_max_MHz = rate_Mbps / bpp
        ax_r.loglog(rate_Mbps, f_max_MHz, color=color, ls=ls, lw=2, label=label)

    # Detector dead-time limits (horizontal lines)
    for tau_ns, tech, color in [(100, "Si-SPAD τ_d=100ns → 10 MHz", _COLORS[5]),
                                  (20,  "Si-SPAD τ_d=20ns → 50 MHz",  _COLORS[6]),
                                  (10,  "SNSPD τ_d=10ns → 100 MHz",   _COLORS[7])]:
        f_det_MHz = 1e3 / tau_ns
        ax_r.axhline(f_det_MHz, color=color, lw=1.3, ls="-.", alpha=0.8)
        ax_r.text(12, f_det_MHz * 1.08, tech, fontsize=7.5, color=color)

    ax_r.set_xlabel("QRNG output rate (Mbps)", fontsize=11)
    ax_r.set_ylabel(r"Max $f_{clock}$ (MHz)", fontsize=11)
    ax_r.set_title("Max Clock Rate vs QRNG Rate\n(detector dead-time limits shown)",
                   fontsize=11, fontweight="bold")
    ax_r.legend(fontsize=9, loc="upper left")
    ax_r.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Pulse budget breakdown
# ---------------------------------------------------------------------------

def plot_pulse_budget() -> plt.Figure:
    """
    Stacked horizontal bar showing how each pulse type contributes to the
    sifted key, from raw f_clock pulses down to signal detections.

    Illustrates the sifting funnel: f_clock → signal pulses → basis-matched
    → detected. Decoy and vacuum pulses contribute to decoy bounds, not key.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    src = REF_SRC
    q = src.sifting_factor()

    # Pulse fractions (per clock cycle)
    fracs = {
        r"Signal ($P_\mu$)":  src.P_mu,
        r"Decoy ($P_\nu$)":   src.P_nu,
        r"Vacuum ($P_{vac}$)": src.P_vac,
    }
    colors_pie = [_COLORS[0], _COLORS[1], _COLORS[2]]

    # Top row: pulse type breakdown
    left = 0.0
    y_top = 1.0
    for (label, frac), color in zip(fracs.items(), colors_pie):
        ax.barh(y_top, frac, left=left, height=0.35, color=color,
                edgecolor="white", linewidth=1.2)
        if frac > 0.04:
            ax.text(left + frac / 2, y_top, f"{frac:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        left += frac

    ax.text(-0.01, y_top, "Pulse types", ha="right", va="center", fontsize=9)

    # Second row: signal pulses that are key-generating (basis-matched)
    y_mid = 0.0
    key_frac = src.P_mu * q
    decoy_frac = src.P_nu
    vac_frac = src.P_vac
    non_key_sig = src.P_mu * (1 - q)   # signal pulses lost to basis mismatch

    ax.barh(y_mid, key_frac, left=0, height=0.35, color=_COLORS[0],
            edgecolor="white", linewidth=1.2, label=f"Signal, basis-matched (→ key)")
    ax.barh(y_mid, non_key_sig, left=key_frac, height=0.35, color=_COLORS[0],
            edgecolor="white", linewidth=1.2, alpha=0.3, label=f"Signal, basis-mismatched (discarded)")
    ax.barh(y_mid, decoy_frac, left=src.P_mu, height=0.35, color=_COLORS[1],
            edgecolor="white", linewidth=1.2, alpha=0.7, label="Decoy (→ PE bounds)")
    ax.barh(y_mid, vac_frac, left=src.P_mu + decoy_frac, height=0.35, color=_COLORS[2],
            edgecolor="white", linewidth=1.2, alpha=0.7, label="Vacuum (→ Y₀ bound)")

    for val, start, label in [
        (key_frac, 0, f"q·P_µ={key_frac:.3f}"),
        (non_key_sig, key_frac, f"{non_key_sig:.3f}"),
    ]:
        if val > 0.03:
            ax.text(start + val / 2, y_mid, label, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white")

    ax.text(-0.01, y_mid, "After sifting", ha="right", va="center", fontsize=9)

    # Annotations
    ax.annotate("", xy=(key_frac / 2, y_mid + 0.18), xytext=(src.P_mu / 2, y_top - 0.18),
                arrowprops=dict(arrowstyle="->", color=_COLORS[0], lw=1.5))
    ax.text(src.P_mu / 2 + 0.02, 0.5, f"q={q:.3f}", fontsize=8, color=_COLORS[0])

    # Reference numbers
    f = src.f_clock
    info = (f"$f_{{clock}}$ = {f/1e6:.0f} MHz  |  "
            f"Signal rate = {f*src.P_mu/1e6:.0f} MHz  |  "
            f"Key-candidate rate = {f*key_frac/1e6:.1f} MHz  |  "
            f"$P_X$ = {src.P_X}")
    ax.set_title(f"Pulse Budget — Reference Configuration\n{info}",
                 fontsize=10.5, fontweight="bold")

    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel("Fraction of clock pulses", fontsize=11)
    ax.set_yticks([])
    ax.legend(fontsize=8.5, loc="lower right", ncol=2)
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def plot_all(save_dir: str | None = None) -> list[plt.Figure]:
    """Render all source figures. Saves to viz/out/source/ by default."""
    out = save_dir or _OUT_DIR
    os.makedirs(out, exist_ok=True)

    figs = [
        plot_sifting_factor(),
        plot_prep_probability(),
        plot_qrng_constraint(),
        plot_pulse_budget(),
    ]
    names = ["sifting_factor", "prep_probability", "qrng_constraint", "pulse_budget"]
    for fig, name in zip(figs, names):
        fig.savefig(os.path.join(out, f"{name}.png"), dpi=150)
    return figs


if __name__ == "__main__":
    plot_all()
    plt.show()
