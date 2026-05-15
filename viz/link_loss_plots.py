"""
viz/link_loss_plots.py — Visualisations for physics/link_loss.py outputs.

Plots:
    plot_diffraction      eta_diff vs range and aperture size; beam-radius overlay.
    plot_pointing         eta_pnt vs pointing error and jitter; divergence contours.
    plot_loss_budget      Waterfall chart of a full LossBudget at a reference geometry.
    plot_budget_vs_elev   All loss terms vs elevation angle for a 400 km pass.
    plot_all              Convenience: render all four figures.
"""

from __future__ import annotations

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

from physics.link_loss import diffraction_loss, pointing_loss, compute_loss_budget
from physics.atmosphere import atmospheric_attenuation

_COLORS = plt.cm.tab10.colors

# Reference parameters (representative 3U CubeSat downlink at 850 nm)
#
# alpha is the effective path-averaged Beer-Lambert coefficient for the FULL
# slant range.  The real atmosphere is ~20 km thick; applying the volumetric
# surface value (~3e-5 m⁻¹) over 400–600 km of slant range would give
# unphysical losses (-50 to -70 dB).  Instead we use the effective coefficient
# that reproduces the correct zenith optical depth (~0.5–1 dB for clear sky):
#   alpha_eff ≈ tau_zenith / L_zenith ≈ 0.115 / 400e3 ≈ 3e-7 m⁻¹
REF = dict(
    lambda_=850e-9,   # m
    w0=0.03,          # m  (3 cm beam waist)
    D_rx=0.4,         # m  (40 cm aperture)
    h_orbit=400e3,    # m
    eta_tx=0.80,
    eta_rx=0.70,
    alpha=3e-7,       # m^-1  effective path-averaged (clear sky, ~0.5 dB zenith)
    theta_pnt=3e-6,   # rad  (3 urad mean error)
    sigma_pnt=3e-6,   # rad  (3 urad jitter 1-sigma)
)
R_E = 6.371e6  # m, mean Earth radius


def _slant_range(el_rad: np.ndarray, h_orbit: float) -> np.ndarray:
    """Geometric slant range for a spherical Earth."""
    return (-R_E * np.sin(el_rad)
            + np.sqrt((R_E * np.sin(el_rad))**2 + h_orbit * (h_orbit + 2 * R_E)))


def _db(eta: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(eta, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(arr > 0, 10 * np.log10(arr), -300.0)


# ---------------------------------------------------------------------------
# Figure 1 — Diffraction loss
# ---------------------------------------------------------------------------

def plot_diffraction() -> plt.Figure:
    """
    Two panels:
      Left:  η_diff (dB) vs slant range for several aperture diameters (w0=3 cm).
      Right: η_diff (dB) as a 2-D heatmap over (D_rx, L) at fixed w0 and λ.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    lam = REF["lambda_"]
    w0  = REF["w0"]
    L_km = np.linspace(200, 1200, 400)

    # --- Left: curves for different aperture sizes ---
    diameters = [0.1, 0.2, 0.4, 0.6, 1.0]
    for i, D in enumerate(diameters):
        eta = np.array([diffraction_loss(lam, w0, Li * 1e3, D) for Li in L_km])
        ax_l.plot(L_km, _db(eta), color=_COLORS[i], lw=1.8, label=f"D = {D*100:.0f} cm")

    # Overlay beam radius W(L) on a secondary y-axis
    ax_l2 = ax_l.twinx()
    z_R = math.pi * w0**2 / lam
    W_L = w0 * np.sqrt(1 + (L_km * 1e3 / z_R)**2)
    ax_l2.plot(L_km, W_L, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax_l2.set_ylabel("Beam radius W(L) (m)", fontsize=9, color="gray")
    ax_l2.tick_params(axis="y", colors="gray")
    ax_l2.yaxis.label.set_color("gray")

    ax_l.set_xlabel("Slant range (km)", fontsize=11)
    ax_l.set_ylabel(r"$\eta_\mathrm{diff}$ (dB)", fontsize=11)
    ax_l.set_title(f"Diffraction Loss vs Range\n"
                   f"(λ={lam*1e9:.0f} nm, w₀={w0*100:.0f} cm)", fontsize=11, fontweight="bold")
    ax_l.legend(fontsize=9, loc="lower left")
    ax_l.grid(True, alpha=0.3)

    # --- Right: 2-D heatmap (D_rx vs L) ---
    D_vals = np.linspace(0.1, 1.0, 60)
    L_vals = np.linspace(200, 1200, 60)
    DD, LL = np.meshgrid(D_vals, L_vals)
    ETA = np.array([[diffraction_loss(lam, w0, L * 1e3, D)
                     for D in D_vals] for L in L_vals])
    DB = _db(ETA)

    cf = ax_r.contourf(D_vals * 100, L_vals, DB,
                       levels=np.linspace(-60, 0, 25), cmap="viridis")
    cs = ax_r.contour(D_vals * 100, L_vals, DB,
                      levels=[-50, -40, -30, -20, -10, -5], colors="white",
                      linewidths=0.8, alpha=0.6)
    ax_r.clabel(cs, fmt="%.0f dB", fontsize=7.5, inline=True)
    fig.colorbar(cf, ax=ax_r, label=r"$\eta_\mathrm{diff}$ (dB)")

    # Mark the reference point
    ax_r.plot(REF["D_rx"] * 100, 400, "r*", ms=12, label=f"Ref: D={REF['D_rx']*100:.0f}cm, L=400km")
    ax_r.legend(fontsize=8.5, loc="upper right")
    ax_r.set_xlabel("Receiver diameter D_rx (cm)", fontsize=11)
    ax_r.set_ylabel("Slant range (km)", fontsize=11)
    ax_r.set_title(f"Diffraction Loss Heatmap\n"
                   f"(λ={lam*1e9:.0f} nm, w₀={w0*100:.0f} cm)", fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Pointing loss
# ---------------------------------------------------------------------------

def plot_pointing() -> plt.Figure:
    """
    Two panels:
      Left:  η_pnt (dB) vs total RMS pointing error for several beam waists.
      Right: 2-D heatmap of η_pnt over (theta_pnt, sigma_pnt) for reference config.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    lam = REF["lambda_"]

    # --- Left: loss vs total RMS error for different w0 ---
    theta_total_urad = np.linspace(0, 20, 400)   # urad
    w0_vals = [0.01, 0.02, 0.03, 0.05, 0.10]

    for i, w0 in enumerate(w0_vals):
        theta_div_urad = (lam / (math.pi * w0)) * 1e6
        eta = np.array([pointing_loss(lam, w0, t * 1e-6, 0.0) for t in theta_total_urad])
        ax_l.plot(theta_total_urad, _db(eta), color=_COLORS[i], lw=1.8,
                  label=f"w₀={w0*100:.0f} cm  (θ_div={theta_div_urad:.1f} µrad)")

    ax_l.axvline(REF["theta_pnt"] * 1e6, color="gray", ls=":", lw=1.2)
    ax_l.axvline(REF["sigma_pnt"] * 1e6, color="gray", ls="--", lw=1.2)
    ax_l.text(REF["theta_pnt"] * 1e6 + 0.3, -2, "θ_pnt", fontsize=8, color="gray")
    ax_l.text(REF["sigma_pnt"] * 1e6 + 0.3, -5, "σ_pnt", fontsize=8, color="gray")

    ax_l.set_xlabel("Total RMS pointing error (µrad)", fontsize=11)
    ax_l.set_ylabel(r"$\eta_\mathrm{pnt}$ (dB)", fontsize=11)
    ax_l.set_title(f"Pointing Loss vs Error\n(λ={lam*1e9:.0f} nm)", fontsize=11, fontweight="bold")
    ax_l.legend(fontsize=8, loc="lower left")
    ax_l.set_ylim(-80, 1)
    ax_l.grid(True, alpha=0.3)

    # --- Right: 2-D heatmap (theta_pnt, sigma_pnt) ---
    w0 = REF["w0"]
    theta_vals = np.linspace(0, 10, 60) * 1e-6   # rad
    sigma_vals = np.linspace(0, 10, 60) * 1e-6   # rad

    ETA = np.array([[pointing_loss(lam, w0, th, sg)
                     for sg in sigma_vals] for th in theta_vals])
    DB = _db(ETA)

    cf = ax_r.contourf(sigma_vals * 1e6, theta_vals * 1e6, DB,
                       levels=np.linspace(-80, 0, 25), cmap="plasma")
    cs = ax_r.contour(sigma_vals * 1e6, theta_vals * 1e6, DB,
                      levels=[-60, -40, -20, -10, -5, -1], colors="white",
                      linewidths=0.8, alpha=0.7)
    ax_r.clabel(cs, fmt="%.0f dB", fontsize=7.5, inline=True)
    fig.colorbar(cf, ax=ax_r, label=r"$\eta_\mathrm{pnt}$ (dB)")

    theta_div_urad = (lam / (math.pi * w0)) * 1e6
    # Draw θ_div contour
    t_line = np.linspace(0, 10, 200)
    s_line = np.sqrt(np.clip(theta_div_urad**2 / 2 - t_line**2, 0, None))
    ax_r.plot(s_line, t_line, "w--", lw=1.5, alpha=0.6, label=r"$\theta_\mathrm{rms}=\theta_\mathrm{div}/\sqrt{2}$")

    ax_r.plot(REF["sigma_pnt"] * 1e6, REF["theta_pnt"] * 1e6, "r*", ms=12, label="Reference")
    ax_r.legend(fontsize=8.5, loc="upper right")
    ax_r.set_xlabel("Pointing jitter σ_pnt (µrad)", fontsize=11)
    ax_r.set_ylabel("Mean offset θ_pnt (µrad)", fontsize=11)
    ax_r.set_title(f"Pointing Loss Heatmap\n"
                   f"(λ={lam*1e9:.0f} nm, w₀={w0*100:.0f} cm, θ_div={theta_div_urad:.1f} µrad)",
                   fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Loss budget waterfall at a reference geometry
# ---------------------------------------------------------------------------

def plot_loss_budget(elevation_deg: float = 45.0) -> plt.Figure:
    """
    Horizontal waterfall chart decomposing η₀ into per-term contributions (dB).

    Uses the reference parameter set at a specified elevation angle.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    lam    = REF["lambda_"]
    w0     = REF["w0"]
    D_rx   = REF["D_rx"]
    h_orb  = REF["h_orbit"]

    el_rad = math.radians(elevation_deg)
    L = float(_slant_range(np.array([el_rad]), h_orb)[0])

    eta_tx   = REF["eta_tx"]
    eta_atm  = atmospheric_attenuation(REF["alpha"], L)
    eta_diff = diffraction_loss(lam, w0, L, D_rx)
    eta_pnt  = pointing_loss(lam, w0, REF["theta_pnt"], REF["sigma_pnt"])
    eta_rx   = REF["eta_rx"]

    budget = compute_loss_budget(eta_tx, eta_atm, eta_diff, eta_pnt, eta_rx)
    db_dict = budget.to_db_dict()

    labels = ["η_tx", "η_atm", "η_diff", "η_pnt", "η_rx"]
    values_db = [db_dict[k] for k in ["eta_tx", "eta_atm", "eta_diff", "eta_pnt", "eta_rx"]]

    # Waterfall: accumulate magnitude (positive) so bars build left-to-right
    cumulative = 0.0
    bar_starts = []
    for v in values_db:
        bar_starts.append(cumulative)
        cumulative += abs(v)

    bars = ax.barh(
        labels[::-1], [abs(v) for v in values_db[::-1]],
        left=[s for s in bar_starts[::-1]],
        color=[_COLORS[i] for i in range(len(labels))[::-1]],
        height=0.55, edgecolor="white", linewidth=0.8,
    )

    for bar, v in zip(bars, values_db[::-1]):
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} dB", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white")

    # Total η₀
    ax.axvline(abs(db_dict["eta_0"]), color="red", lw=2, ls="--", alpha=0.8)
    ax.text(abs(db_dict["eta_0"]) + 0.3, len(labels) - 0.5,
            f"η₀ = {db_dict['eta_0']:.1f} dB", color="red", fontsize=10, fontweight="bold")

    ax.set_xlabel("Cumulative loss (dB)", fontsize=11)
    ax.set_title(f"Link Budget Waterfall — Elevation {elevation_deg:.0f}°\n"
                 f"(L = {L/1e3:.0f} km, λ={lam*1e9:.0f} nm, "
                 f"w₀={w0*100:.0f} cm, D={D_rx*100:.0f} cm)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, abs(db_dict["eta_0"]) + 3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Loss terms vs elevation angle
# ---------------------------------------------------------------------------

def plot_budget_vs_elev() -> plt.Figure:
    """
    η_atm, η_diff, η_pnt, and η₀ (dB) vs elevation angle for a 400 km pass.

    η_tx and η_rx are static so omitted; they shift the η₀ curve by a constant.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

    lam   = REF["lambda_"]
    w0    = REF["w0"]
    D_rx  = REF["D_rx"]
    h_orb = REF["h_orbit"]

    el_deg = np.linspace(10, 90, 300)
    el_rad = np.deg2rad(el_deg)
    L_arr  = _slant_range(el_rad, h_orb)

    eta_atm  = np.array([atmospheric_attenuation(REF["alpha"], L) for L in L_arr])
    eta_diff = np.array([diffraction_loss(lam, w0, L, D_rx) for L in L_arr])
    eta_pnt  = pointing_loss(lam, w0, REF["theta_pnt"], REF["sigma_pnt"])  # scalar (static)
    eta_0    = REF["eta_tx"] * eta_atm * eta_diff * eta_pnt * REF["eta_rx"]

    # Top panel: individual terms
    ax_top.plot(el_deg, _db(eta_atm),  color=_COLORS[0], lw=2, label=r"$\eta_\mathrm{atm}$")
    ax_top.plot(el_deg, _db(eta_diff), color=_COLORS[1], lw=2, label=r"$\eta_\mathrm{diff}$")
    ax_top.axhline(_db(eta_pnt), color=_COLORS[2], lw=1.5, ls="--",
                   label=rf"$\eta_\mathrm{{pnt}}$ (static) = {_db(eta_pnt):.1f} dB")
    ax_top.axhline(_db(REF["eta_tx"]), color=_COLORS[3], lw=1.2, ls=":",
                   label=rf"$\eta_\mathrm{{tx}}$ = {_db(REF['eta_tx']):.1f} dB")
    ax_top.axhline(_db(REF["eta_rx"]), color=_COLORS[4], lw=1.2, ls="-.",
                   label=rf"$\eta_\mathrm{{rx}}$ = {_db(REF['eta_rx']):.1f} dB")

    ax_top.axvline(20, color="gray", lw=1, ls=":", alpha=0.7)
    ax_top.text(21, ax_top.get_ylim()[0] if ax_top.get_ylim()[0] != 0 else -5,
                "min el", fontsize=8, color="gray")
    ax_top.set_ylabel("Loss term (dB)", fontsize=11)
    ax_top.set_title(f"Link Loss Terms vs Elevation\n"
                     f"(h={h_orb/1e3:.0f} km, λ={lam*1e9:.0f} nm, "
                     f"w₀={w0*100:.0f} cm, D={D_rx*100:.0f} cm)",
                     fontsize=11, fontweight="bold")
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.grid(True, alpha=0.3)

    # Bottom panel: total η₀ and slant range
    ax_bot.plot(el_deg, _db(eta_0), color="black", lw=2.5, label=r"$\eta_0$ (total)")
    ax_bot.fill_between(el_deg, _db(eta_0), _db(eta_0).min(), alpha=0.12, color="black")
    ax_bot.set_xlabel("Elevation angle (deg)", fontsize=11)
    ax_bot.set_ylabel(r"$\eta_0$ (dB)", fontsize=11)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(fontsize=10, loc="upper left")

    # Secondary axis: slant range
    ax_bot2 = ax_bot.twinx()
    ax_bot2.plot(el_deg, L_arr / 1e3, color="steelblue", lw=1.5, ls="--", alpha=0.7)
    ax_bot2.set_ylabel("Slant range (km)", fontsize=10, color="steelblue")
    ax_bot2.tick_params(axis="y", colors="steelblue")

    # Annotate zenith and horizon
    eta0_zenith = float(_db(eta_0[-1]))
    eta0_10deg  = float(_db(eta_0[0]))
    ax_bot.annotate(f"{eta0_zenith:.1f} dB", xy=(90, eta0_zenith),
                    xytext=(80, eta0_zenith + 2), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="gray"))
    ax_bot.annotate(f"{eta0_10deg:.1f} dB", xy=(10, eta0_10deg),
                    xytext=(20, eta0_10deg + 2), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="gray"))

    ax_bot.axvline(20, color="gray", lw=1, ls=":", alpha=0.7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "link_loss")


def plot_all(save_dir: str | None = None) -> list[plt.Figure]:
    """Render all link_loss figures. Saves to viz/out/link_loss/ by default."""
    out = save_dir or _OUT_DIR
    os.makedirs(out, exist_ok=True)

    figs = [
        plot_diffraction(),
        plot_pointing(),
        plot_loss_budget(elevation_deg=45.0),
        plot_budget_vs_elev(),
    ]
    names = ["diffraction", "pointing", "loss_budget_45deg", "budget_vs_elev"]
    for fig, name in zip(figs, names):
        fig.savefig(os.path.join(out, f"{name}.png"), dpi=150)
    return figs


if __name__ == "__main__":
    figs = plot_all()
    plt.show()
