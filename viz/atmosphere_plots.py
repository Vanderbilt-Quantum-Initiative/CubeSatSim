"""
viz/atmosphere_plots.py — Visualisations for physics/atmosphere.py outputs.

Plots:
    plot_cn2_profiles      Cn2(h) vs altitude for multiple Hufnagel-Valley configurations.
    plot_hv_sensitivity    Sensitivity of Cn2(h) to ground-level Cn2_0 and wind speed v.
    plot_attenuation       Beer-Lambert eta_atm vs slant range and elevation angle.
    plot_all               Convenience: render all three figures.
"""

from __future__ import annotations

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

from physics.atmosphere import hufnagel_valley, from_measurements, atmospheric_attenuation


# ---------------------------------------------------------------------------
# Colour palette (accessible, consistent across all plots)
# ---------------------------------------------------------------------------
_COLORS = plt.cm.tab10.colors


def _db(eta: float) -> float:
    return 10.0 * math.log10(eta) if eta > 0 else float("-inf")


# ---------------------------------------------------------------------------
# Figure 1 — Cn2 profiles
# ---------------------------------------------------------------------------

def plot_cn2_profiles(ax: plt.Axes | None = None) -> plt.Figure:
    """
    Cn²(h) vs altitude for three H-V configurations and one measurement-derived profile.

    Shows the three physical regimes: ground layer, stratospheric background,
    and jet-stream tropopause peak.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

    h = np.linspace(0, 20_000, 2000)   # 0–20 km

    configs = [
        ("Weak  (Cn2_0 = 1e-15, v=10)",  hufnagel_valley(1e-15, v=10.0),  _COLORS[0], "-"),
        ("Medium (Cn2_0 = 1e-14, v=21)", hufnagel_valley(1e-14, v=21.0),  _COLORS[1], "-"),
        ("Strong (Cn2_0 = 1e-13, v=21)", hufnagel_valley(1e-13, v=21.0),  _COLORS[2], "-"),
        ("Strong + high wind (v=40)",     hufnagel_valley(1e-13, v=40.0),  _COLORS[3], "--"),
    ]

    for label, profile, color, ls in configs:
        cn2 = np.array([profile(hi) for hi in h])
        ax.semilogy(cn2, h / 1e3, color=color, ls=ls, lw=1.8, label=label)

    # Measurement-derived overlay (synthetic but realistic sample points)
    sample_h = np.array([0, 500, 1000, 3000, 5000, 8000, 10000, 15000, 20000], dtype=float)
    sample_v = np.array([3e-14, 1e-14, 5e-15, 2e-15, 1e-15, 4e-15, 3e-15, 5e-16, 1e-16])
    meas = from_measurements(sample_h, sample_v)
    cn2_meas = np.array([meas(hi) for hi in h])
    ax.semilogy(cn2_meas, h / 1e3, color=_COLORS[4], ls=":", lw=2.0,
                label="Measured (synthetic sample)")
    ax.scatter(sample_v, sample_h / 1e3, color=_COLORS[4], s=30, zorder=5)

    ax.set_xlabel(r"$C_n^2(h)$ (m$^{-2/3}$)", fontsize=11)
    ax.set_ylabel("Altitude (km)", fontsize=11)
    ax.set_title("Hufnagel-Valley $C_n^2$ Profiles", fontsize=12, fontweight="bold")
    ax.set_ylim(0.01, 20)
    ax.set_xlim(left=1e-20)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # Annotate the three regimes
    ax.axhspan(0, 0.5, alpha=0.07, color=_COLORS[2], label="_ground layer")
    ax.axhspan(8, 12, alpha=0.07, color=_COLORS[0], label="_tropopause")
    ax.text(2e-19, 0.2, "Ground layer", fontsize=7.5, color=_COLORS[2])
    ax.text(2e-19, 9.5, "Jet stream\ntropopause", fontsize=7.5, color=_COLORS[0])

    if fig is not None:
        fig.tight_layout()
    return fig or ax.figure


# ---------------------------------------------------------------------------
# Figure 2 — H-V sensitivity to Cn2_0 and v_wind
# ---------------------------------------------------------------------------

def plot_hv_sensitivity(ax: plt.Axes | None = None) -> plt.Figure:
    """
    Integrated Cn²(h) path weight vs Cn2_0 and v_wind, for a vertical path.

    Integrating Cn²(h) dh from 0 to 20 km gives the total turbulence strength
    driving the Rytov variance. This surface shows which parameter dominates.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))

    h = np.linspace(0, 20_000, 500)
    dh = h[1] - h[0]

    cn2_0_vals = np.logspace(-16, -12, 40)
    v_vals     = np.array([10, 15, 21, 30, 40])

    for i, v in enumerate(v_vals):
        integrals = []
        for cn2_0 in cn2_0_vals:
            profile = hufnagel_valley(cn2_0, v=v)
            cn2 = np.array([profile(hi) for hi in h])
            integrals.append(np.trapz(cn2, h))
        ax.loglog(cn2_0_vals, integrals, color=_COLORS[i], lw=1.8,
                  label=f"v = {v} m/s")

    ax.set_xlabel(r"Ground-level $C_{n,0}^2$ (m$^{-2/3}$)", fontsize=11)
    ax.set_ylabel(r"$\int_0^{20\,\mathrm{km}} C_n^2(h)\,dh$ (m$^{1/3}$)", fontsize=10)
    ax.set_title(r"H-V Integrated Turbulence Strength vs $C_{n,0}^2$ and Wind", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=9, title="Wind speed")
    ax.grid(True, which="both", alpha=0.3)

    if fig is not None:
        fig.tight_layout()
    return fig or ax.figure


# ---------------------------------------------------------------------------
# Figure 3 — Beer-Lambert attenuation
# ---------------------------------------------------------------------------

def plot_attenuation(ax_left: plt.Axes | None = None,
                     ax_right: plt.Axes | None = None) -> plt.Figure:
    """
    Two panels:
      Left:  η_atm (dB) vs slant range for several extinction coefficients.
      Right: η_atm (dB) vs elevation angle for a 400 km orbit, same α values.
    """
    fig = None
    if ax_left is None or ax_right is None:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Effective path-averaged extinction coefficients for a LEO downlink.
    # α here is the Beer-Lambert coefficient applied over the FULL slant range L.
    # Because the real atmosphere is only ~20 km thick (vs 400-2000 km slant range),
    # α_eff ≈ α_volumetric × H_atm / L_slant — roughly 10–100× smaller than the
    # volumetric surface value (~3e-5 m⁻¹). Typical zenith losses at 850 nm:
    #   clear sky  ~0.3 dB  →  α_eff ≈ 2e-7 m⁻¹
    #   nominal    ~0.8 dB  →  α_eff ≈ 5e-7 m⁻¹
    #   light haze ~3 dB    →  α_eff ≈ 2e-6 m⁻¹
    #   hazy       ~9 dB    →  α_eff ≈ 5e-6 m⁻¹
    alphas = [
        (2e-7,  "Clear sky   (α_eff=2e-7 m⁻¹, ~0.3 dB zenith)",  _COLORS[0], "-"),
        (5e-7,  "Nominal     (α_eff=5e-7 m⁻¹, ~0.9 dB zenith)",  _COLORS[1], "-"),
        (2e-6,  "Light haze  (α_eff=2e-6 m⁻¹, ~3.5 dB zenith)",  _COLORS[2], "-"),
        (5e-6,  "Hazy        (α_eff=5e-6 m⁻¹, ~8.7 dB zenith)",  _COLORS[3], "--"),
    ]

    # --- Left: eta_atm vs slant range ---
    L_km = np.linspace(200, 2000, 400)
    for alpha, label, color, ls in alphas:
        eta = np.array([atmospheric_attenuation(alpha, L * 1e3) for L in L_km])
        db  = np.where(eta > 0, 10 * np.log10(eta), -300)
        ax_left.plot(L_km, db, color=color, ls=ls, lw=1.8, label=label)

    ax_left.set_xlabel("Slant range (km)", fontsize=11)
    ax_left.set_ylabel(r"$\eta_\mathrm{atm}$ (dB)", fontsize=11)
    ax_left.set_title("Beer-Lambert Attenuation vs Slant Range", fontsize=11, fontweight="bold")
    ax_left.legend(fontsize=8.5)
    ax_left.grid(True, alpha=0.3)
    ax_left.set_xlim(200, 2000)

    # Annotate the nominal curve at 500 km
    ref_L = 500
    ref_alpha = 5e-7
    ref_eta = atmospheric_attenuation(ref_alpha, ref_L * 1e3)
    ax_left.annotate(
        f"{10*math.log10(ref_eta):.2f} dB\n@ {ref_L} km (nominal)",
        xy=(ref_L, 10*math.log10(ref_eta)),
        xytext=(ref_L + 300, 10*math.log10(ref_eta) - 1.5),
        fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"),
    )

    # --- Right: eta_atm vs elevation angle (400 km orbit) ---
    h_orbit = 400e3   # m
    el_deg  = np.linspace(10, 90, 300)
    el_rad  = np.deg2rad(el_deg)

    # Slant range from ground for a spherical Earth (approximate):
    # L = -R_E * sin(el) + sqrt((R_E*sin(el))^2 + h*(h + 2*R_E))
    R_E = 6.371e6
    L_el = (-R_E * np.sin(el_rad)
            + np.sqrt((R_E * np.sin(el_rad))**2 + h_orbit * (h_orbit + 2 * R_E)))

    for alpha, label, color, ls in alphas:
        eta = np.array([atmospheric_attenuation(alpha, Li) for Li in L_el])
        db  = np.where(eta > 0, 10 * np.log10(eta), -300)
        ax_right.plot(el_deg, db, color=color, ls=ls, lw=1.8, label=label)

    ax_right.axvline(20, color="gray", lw=1, ls=":", alpha=0.8)
    ax_right.text(21, ax_right.get_ylim()[0] + 0.1 if ax_right.get_ylim()[0] != 0 else -0.5,
                  "min el\n20°", fontsize=7.5, color="gray")
    ax_right.set_xlabel("Elevation angle (deg)", fontsize=11)
    ax_right.set_ylabel(r"$\eta_\mathrm{atm}$ (dB)", fontsize=11)
    ax_right.set_title("Beer-Lambert Attenuation vs Elevation\n(h = 400 km orbit)", fontsize=11,
                        fontweight="bold")
    ax_right.legend(fontsize=8.5)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_xlim(10, 90)

    if fig is not None:
        fig.tight_layout()
    return fig or ax_left.figure


# ---------------------------------------------------------------------------
# Convenience: all figures
# ---------------------------------------------------------------------------

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "atmosphere")


def plot_all(save_dir: str | None = None) -> list[plt.Figure]:
    """Render all atmosphere figures. Saves to viz/out/atmosphere/ by default."""
    out = save_dir or _OUT_DIR
    os.makedirs(out, exist_ok=True)

    figs = [
        plot_cn2_profiles(),
        plot_hv_sensitivity(),
        plot_attenuation(),
    ]
    names = ["cn2_profiles", "hv_sensitivity", "attenuation"]
    for fig, name in zip(figs, names):
        fig.savefig(os.path.join(out, f"{name}.png"), dpi=150)
    return figs


if __name__ == "__main__":
    figs = plot_all()
    plt.show()
