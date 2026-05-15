"""
viz/detector_plots.py — Visualisations for physics/detector.py.

Plots:
    plot_clock_vs_deadtime    Max f_clock vs tau_d; detector technology comparison.
    plot_dark_count_noise     Dark count and background contribution to Y_0.
    plot_gate_jitter          Valid gate-width / timing-jitter operating region.
    plot_technology_comparison Radar / bar chart of Si-SPAD vs SNSPD trade.
    plot_all                  Render all four figures → viz/out/detector/.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

from physics.detector import DetectorModel

_COLORS = plt.cm.tab10.colors
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "detector")

# Technology reference points
DETECTORS = {
    "Si-SPAD\n(slow, −20°C)":  DetectorModel(eta_det=0.40, p_d=5e-6, tau_d=100e-9, delta_t=2e-9),
    "Si-SPAD\n(fast, −20°C)":  DetectorModel(eta_det=0.50, p_d=2e-6, tau_d=20e-9,  delta_t=1e-9),
    "InGaAs SPAD\n(1550 nm)":  DetectorModel(eta_det=0.25, p_d=1e-4, tau_d=10e-6,  delta_t=1e-9),
    "SNSPD\n(cryo, 850 nm)":   DetectorModel(eta_det=0.90, p_d=1e-8, tau_d=10e-9,  delta_t=0.5e-9,
                                              sigma_t=50e-12),
}

# Reference detector for detailed plots
REF_DET = DETECTORS["Si-SPAD\n(fast, −20°C)"]


# ---------------------------------------------------------------------------
# Figure 1 — Max clock rate vs dead time
# ---------------------------------------------------------------------------

def plot_clock_vs_deadtime() -> plt.Figure:
    """
    f_max = 1/tau_d vs tau_d, with technology markers and clock-rate contours.

    tau_d is the hard upper bound on f_clock: driving the laser faster than
    1/tau_d causes pile-up that distorts Q_mu and E_mu.  The detector dead time
    must be communicated to Payload early — it sets the f_clock design space.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    tau_ns = np.logspace(0, 4, 500)      # 1 ns to 10 µs
    f_max_MHz = 1e3 / tau_ns

    ax.loglog(tau_ns, f_max_MHz, color=_COLORS[0], lw=2.5,
              label=r"$f_{max} = 1/\tau_d$")

    # Shade forbidden region (f > f_max is pile-up zone)
    ax.fill_between(tau_ns, f_max_MHz, 1e4, alpha=0.08, color="red",
                    label="Pile-up / saturation zone")

    # Clock rate reference lines
    for f_MHz, label, ls in [(100, "100 MHz",  "--"),
                               (50,  "50 MHz",   ":"),
                               (10,  "10 MHz",   "-.")]:
        ax.axhline(f_MHz, color="gray", lw=1.0, ls=ls, alpha=0.7)
        ax.text(1.2, f_MHz * 1.12, label, fontsize=7.5, color="gray")

    # Technology markers
    colors_det = [_COLORS[1], _COLORS[2], _COLORS[3], _COLORS[4]]
    for (name, det), color in zip(DETECTORS.items(), colors_det):
        tau_ns_val = det.tau_d * 1e9
        f_mhz_val = det.max_clock_rate() / 1e6
        ax.scatter([tau_ns_val], [f_mhz_val], color=color, s=120, zorder=6,
                   edgecolors="white", linewidths=1.2)
        offset_y = 1.4 if f_mhz_val > 20 else 0.5
        ax.annotate(name.replace("\n", " ") + f"\n({tau_ns_val:.0f} ns, {f_mhz_val:.0f} MHz)",
                    xy=(tau_ns_val, f_mhz_val),
                    xytext=(tau_ns_val * 1.8, f_mhz_val * offset_y),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1))

    ax.set_xlabel(r"Dead time $\tau_d$ (ns)", fontsize=11)
    ax.set_ylabel(r"Max clock rate $f_{max}$ (MHz)", fontsize=11)
    ax.set_title(r"Detector Dead-Time Constraint on $f_{clock}$"
                 "\n(operating above the curve causes pile-up distortion)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(0.1, 1e4)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Dark count and background contribution to Y_0
# ---------------------------------------------------------------------------

def plot_dark_count_noise() -> plt.Figure:
    """
    Two panels:
      Left:  Y_0 = p_d + p_bg vs gate width delta_t, for several p_d values.
             Shows crossover where dark counts dominate over background.
      Right: Y_0 heatmap over (p_d, delta_t) for a fixed sky background.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    # Background photon rate per gate (fixed sky + filter conditions)
    # p_bg = H_bg * Omega_FOV * A_rx * delta_lambda * delta_t * eta_rx
    # Representative values: H_bg=1e-3 W/m²/sr/nm, Ω=1e-8 sr, A=0.13m², Δλ=1nm, η_rx=0.7
    H_bg = 1e-3; Omega = 1e-8; A_rx = 0.13; dlam = 1e-9; eta_rx = 0.7
    bg_rate_per_s = H_bg * Omega * A_rx * dlam / (6.626e-34 * 3e8 / 850e-9) * eta_rx
    # photons/s: power/(hν) then × gate → probability per gate

    # Simpler: p_bg ∝ delta_t (proportional to gate width)
    # Use a representative spectral background rate Rbg [photons/s]
    R_bg = 1e4   # background photons/s reaching detector (after filtering)

    delta_t_ns = np.logspace(-1, 2, 400)  # 0.1 ns to 100 ns

    # --- Left: Y_0 vs delta_t for different p_d ---
    p_d_vals = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    labels   = ["1e-8 (SNSPD)", "1e-7", "1e-6 (Si-SPAD, good)", "1e-5", "1e-4 (InGaAs)"]

    for p_d, label, color in zip(p_d_vals, labels, _COLORS):
        p_bg = R_bg * delta_t_ns * 1e-9
        Y_0 = p_d + p_bg
        ax_l.loglog(delta_t_ns, Y_0, color=color, lw=1.8, label=f"$p_d$ = {label}")

    # Background-only line
    p_bg_line = R_bg * delta_t_ns * 1e-9
    ax_l.loglog(delta_t_ns, p_bg_line, color="gray", lw=1.2, ls="--",
                label=r"Background only ($p_{bg}$)")

    # Reference gate width
    ax_l.axvline(REF_DET.delta_t * 1e9, color="black", lw=1.2, ls=":",
                 label=f"Ref gate ({REF_DET.delta_t*1e9:.0f} ns)")

    ax_l.set_xlabel(r"Gate width $\Delta t$ (ns)", fontsize=11)
    ax_l.set_ylabel(r"Noise yield $Y_0 = p_d + p_{bg}$", fontsize=11)
    ax_l.set_title(f"Noise Yield vs Gate Width\n"
                   f"($R_{{bg}}$ = {R_bg:.0e} photons/s after filter)",
                   fontsize=11, fontweight="bold")
    ax_l.legend(fontsize=7.5, loc="upper left")
    ax_l.grid(True, which="both", alpha=0.3)

    # Shade: dark-count dominated vs background dominated
    cross_dt = {p_d: p_d / R_bg * 1e9 for p_d in p_d_vals}  # in ns

    # --- Right: Y_0 heatmap over (p_d, delta_t) ---
    p_d_arr = np.logspace(-9, -4, 80)
    dt_arr  = np.logspace(-1, 2, 80)
    PD, DT = np.meshgrid(p_d_arr, dt_arr)
    Y0_map  = PD + R_bg * DT * 1e-9

    cf = ax_r.pcolormesh(p_d_arr, dt_arr, Y0_map,
                         norm=LogNorm(vmin=1e-9, vmax=1e-3), cmap="plasma_r",
                         shading="auto")
    fig.colorbar(cf, ax=ax_r, label=r"$Y_0 = p_d + p_{bg}$")

    # Contours at security-relevant thresholds
    cs = ax_r.contour(p_d_arr, dt_arr, Y0_map,
                      levels=[1e-7, 1e-6, 1e-5, 1e-4],
                      colors="white", linewidths=0.9, alpha=0.7)
    ax_r.clabel(cs, fmt="%.0e", fontsize=7.5, inline=True)

    # Mark technology points
    for (name, det), color in zip(DETECTORS.items(), colors_det := _COLORS[1:]):
        ax_r.scatter([det.p_d], [det.delta_t * 1e9], color=color, s=80,
                     edgecolors="white", linewidths=1, zorder=6)

    # Background-equal-to-dark-count diagonal
    dt_diag = p_d_arr / R_bg * 1e9
    mask = (dt_diag >= 0.1) & (dt_diag <= 100)
    ax_r.plot(p_d_arr[mask], dt_diag[mask], "w--", lw=1.5, alpha=0.8,
              label=r"$p_d = p_{bg}$ (equal noise sources)")
    ax_r.legend(fontsize=7.5, loc="upper left")

    ax_r.set_xscale("log"); ax_r.set_yscale("log")
    ax_r.set_xlabel(r"Dark count probability per gate $p_d$", fontsize=11)
    ax_r.set_ylabel(r"Gate width $\Delta t$ (ns)", fontsize=11)
    ax_r.set_title(r"$Y_0$ Heatmap over $(p_d,\, \Delta t)$"
                   f"\n($R_{{bg}}$ = {R_bg:.0e} ph/s)",
                   fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Gate width vs timing jitter validity
# ---------------------------------------------------------------------------

def plot_gate_jitter() -> plt.Figure:
    """
    Valid operating region: delta_t ≥ 2·sigma_t.

    If the gate is narrower than ~2σ_t the detector clips its own IRF,
    reducing effective eta_det. The boundary delta_t = 2·sigma_t is the
    DetectorModel.validate() warning threshold.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    sigma_t_ps = np.logspace(1, 4, 300)   # 10 ps to 10 ns
    delta_t_2sig = 2 * sigma_t_ps          # boundary

    ax.loglog(sigma_t_ps, delta_t_2sig, color=_COLORS[0], lw=2.5,
              label=r"$\Delta t = 2\sigma_t$ (validate boundary)")
    ax.fill_between(sigma_t_ps, delta_t_2sig, 1e5, alpha=0.12, color=_COLORS[0],
                    label=r"Valid region ($\Delta t \geq 2\sigma_t$)")
    ax.fill_between(sigma_t_ps, 1, delta_t_2sig, alpha=0.12, color="red",
                    label=r"Invalid: gate clips IRF")

    ax.text(15, 5e3, "Valid\n" + r"($\Delta t \geq 2\sigma_t$)",
            fontsize=10, color=_COLORS[0])
    ax.text(1e3, 20, "Invalid: gate clips\ntiming distribution",
            fontsize=9, color="red")

    # Technology points
    for (name, det), color in zip(DETECTORS.items(), _COLORS[1:]):
        if det.sigma_t is None:
            sig_ps = det.delta_t * 1e12 / 4    # assume sigma_t ≈ delta_t/4 for plotting
            valid_str = "(assumed)"
        else:
            sig_ps = det.sigma_t * 1e12
            valid_str = "validate OK" if det.delta_t >= 2 * det.sigma_t else "clips IRF"
        dt_ps = det.delta_t * 1e12
        ax.scatter([sig_ps], [dt_ps], color=color, s=100,
                   edgecolors="white", linewidths=1.2, zorder=6)
        ax.annotate(name.replace("\n", " ") + f"\n{valid_str}",
                    xy=(sig_ps, dt_ps), xytext=(sig_ps * 1.6, dt_ps * 2.5),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.9))

    # Typical timing sync values
    for sync_ps, label in [(100, "σ_sync = 100 ps"), (500, "σ_sync = 500 ps"),
                            (1000, "σ_sync = 1 ns")]:
        ax.axvline(sync_ps, color="gray", lw=0.9, ls=":", alpha=0.6)
        ax.text(sync_ps * 1.05, 1.5e4, label, fontsize=7, color="gray", rotation=90, va="top")

    ax.set_xlabel(r"Timing jitter $\sigma_t$ (ps)", fontsize=11)
    ax.set_ylabel(r"Gate width $\Delta t$ (ps)", fontsize=11)
    ax.set_title("Gate Width vs Timing Jitter\n"
                 r"(valid region: $\Delta t \geq 2\sigma_t$)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(10, 1e4)
    ax.set_ylim(1, 1e4)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Technology comparison
# ---------------------------------------------------------------------------

def plot_technology_comparison() -> plt.Figure:
    """
    Side-by-side bar chart comparing detector technologies on four key axes:
    eta_det, p_d (inverted: lower is better), max f_clock, and delta_t.

    Shows the trade between Si-SPAD (warm, fast clocking, worse efficiency)
    and SNSPD (cryogenic, excellent efficiency and dark counts).
    """
    fig, axes = plt.subplots(1, 4, figsize=(13, 5))

    det_names = [n.replace("\n", "\n") for n in DETECTORS.keys()]
    dets = list(DETECTORS.values())
    colors = list(_COLORS[1:1 + len(dets)])
    x = np.arange(len(dets))
    w = 0.6

    metrics = [
        ("Detection efficiency $\\eta_{det}$", [d.eta_det for d in dets],
         "%", lambda v: f"{v*100:.0f}%", None, False),
        ("Dark count prob./gate $p_d$\n(lower is better)", [d.p_d for d in dets],
         "", lambda v: f"{v:.0e}", "log", True),
        ("Max clock rate $f_{max}$ (MHz)", [d.max_clock_rate()/1e6 for d in dets],
         "MHz", lambda v: f"{v:.0f}", "log", False),
        ("Gate width $\\Delta t$ (ns)", [d.delta_t*1e9 for d in dets],
         "ns", lambda v: f"{v:.1f}", None, False),
    ]

    for ax, (title, vals, unit, fmt, yscale, lower_better) in zip(axes, metrics):
        bars = ax.bar(x, vals, width=w, color=colors, edgecolor="white", linewidth=0.8)
        if yscale == "log":
            ax.set_yscale("log")

        for bar, val in zip(bars, vals):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y * (1.15 if yscale == "log" else 1.02),
                    fmt(val), ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(det_names, fontsize=7.5)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        if lower_better:
            # Green = lower is better — invert color hint via text
            ax.text(0.98, 0.98, "↓ better", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color="green", alpha=0.7)
        else:
            ax.text(0.98, 0.98, "↑ better", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color=_COLORS[0], alpha=0.7)

    fig.suptitle("Single-Photon Detector Technology Comparison",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def plot_all(save_dir: str | None = None) -> list[plt.Figure]:
    """Render all detector figures. Saves to viz/out/detector/ by default."""
    out = save_dir or _OUT_DIR
    os.makedirs(out, exist_ok=True)

    figs = [
        plot_clock_vs_deadtime(),
        plot_dark_count_noise(),
        plot_gate_jitter(),
        plot_technology_comparison(),
    ]
    names = ["clock_vs_deadtime", "dark_count_noise", "gate_jitter", "technology_comparison"]
    for fig, name in zip(figs, names):
        fig.savefig(os.path.join(out, f"{name}.png"), dpi=150, bbox_inches="tight")
    return figs


if __name__ == "__main__":
    plot_all()
    plt.show()
