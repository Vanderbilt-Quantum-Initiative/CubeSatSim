"""
params/definitions.py — Canonical parameter definitions for the VQI QKD link budget.

Every adjustable input to the simulation is registered here with metadata,
physical bounds, and a baseline (design-point) default.  Physics modules
never read this file — they only see the typed dataclasses constructed by
the registry from these values.

Status vocabulary
-----------------
TBD        Placeholder. Must not be used for simulation runs.
ESTIMATED  Engineering estimate; uncertainty ≥ 20%.
BASELINED  System-level agreement on value; uncertainty < 20%.
MEASURED   Lab or field measurement; uncertainty < 5%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Status(str, Enum):
    TBD        = "TBD"
    ESTIMATED  = "ESTIMATED"
    BASELINED  = "BASELINED"
    MEASURED   = "MEASURED"


@dataclass
class ParamDef:
    name: str
    symbol: str
    unit: str
    owner: str
    status: Status
    description: str
    default: float | None = None
    bounds: tuple[float, float] | None = None
    depends_on: list[str] = field(default_factory=list)
    derivation: Callable | None = None
    tier: int = 1   # 1 = fast (point eval), 2 = slow (pass sim)


# ---------------------------------------------------------------------------
# All parameter definitions.  Keys match the attribute names on ChannelConfig /
# SourceConfig / PostProcessingConfig / Geometry.
# ---------------------------------------------------------------------------

PARAM_DEFS: dict[str, ParamDef] = {

    # ── Block 1: Transmitter Optics (Payload) ────────────────────────────────

    "eta_tx": ParamDef(
        name="eta_tx", symbol="η_tx", unit="dimensionless",
        owner="Payload",
        status=Status.ESTIMATED,
        description="Transmitter optical chain efficiency (polariser, fibre coupler, output optics).",
        default=0.50,
        bounds=(0.1, 1.0),
        tier=1,
    ),
    "lambda_": ParamDef(
        name="lambda_", symbol="λ", unit="m",
        owner="Payload",
        status=Status.BASELINED,
        description="Centre wavelength of the WCP source.",
        default=850e-9,
        bounds=(600e-9, 1600e-9),
        tier=1,
    ),
    "w0": ParamDef(
        name="w0", symbol="w₀", unit="m",
        owner="Payload",
        status=Status.ESTIMATED,
        description="Transmitter beam waist at exit aperture (1/e² intensity radius).",
        default=0.08,
        bounds=(0.01, 0.30),
        tier=1,
    ),
    "mu": ParamDef(
        name="mu", symbol="μ", unit="photons/pulse",
        owner="Payload",
        status=Status.ESTIMATED,
        description="Mean photon number per signal-intensity pulse.",
        default=0.5,
        bounds=(0.01, 2.0),
        tier=1,
    ),
    "nu": ParamDef(
        name="nu", symbol="ν", unit="photons/pulse",
        owner="Payload",
        status=Status.ESTIMATED,
        description="Mean photon number per weak-decoy pulse.  Must satisfy ν < μ.",
        default=0.1,
        bounds=(0.001, 0.5),
        tier=1,
    ),
    "P_mu": ParamDef(
        name="P_mu", symbol="P_μ", unit="dimensionless",
        owner="Payload",
        status=Status.ESTIMATED,
        description="Probability of emitting a signal-intensity pulse.",
        default=0.6,
        bounds=(0.0, 1.0),
        tier=1,
    ),
    "P_nu": ParamDef(
        name="P_nu", symbol="P_ν", unit="dimensionless",
        owner="Payload",
        status=Status.ESTIMATED,
        description="Probability of emitting a weak-decoy pulse.",
        default=0.3,
        bounds=(0.0, 1.0),
        tier=1,
    ),
    "P_X": ParamDef(
        name="P_X", symbol="P_X", unit="dimensionless",
        owner="Payload",
        status=Status.ESTIMATED,
        description="Probability of choosing the X (key-generating) basis per pulse.",
        default=0.9,
        bounds=(0.5, 1.0),
        tier=1,
    ),
    "f_clock": ParamDef(
        name="f_clock", symbol="f_clock", unit="Hz",
        owner="Payload/Ground Station (joint trade)",
        status=Status.ESTIMATED,
        description=(
            "Pulse repetition rate.  Constrained from above by detector dead time: "
            "f_clock ≤ 1/τ_d.  Si-SPAD: ≤ 10–50 MHz.  SNSPD: ≤ 100–200 MHz."
        ),
        default=100e6,
        bounds=(1e6, 500e6),
        tier=1,
    ),
    "e_opt": ParamDef(
        name="e_opt", symbol="e_opt", unit="dimensionless",
        owner="Payload/ADCS/Ground Station",
        status=Status.ESTIMATED,
        description=(
            "Residual optical error rate after active polarisation compensation.  "
            "This is NOT a fixed constant — it varies within a pass as the satellite "
            "attitude changes.  Treat as a worst-case static bound for modelling."
        ),
        default=0.03,
        bounds=(0.0, 0.10),
        tier=1,
    ),

    # ── Block 2: Atmospheric Channel (Mission / Systems) ─────────────────────

    "alpha": ParamDef(
        name="alpha", symbol="α", unit="1/m",
        owner="Mission/Systems",
        status=Status.ESTIMATED,
        description=(
            "Beer-Lambert extinction coefficient.  Derived from zenith optical depth τ₀ "
            "and orbital altitude H: α = τ₀ / H.  Typical 850 nm zenith τ₀ ≈ 0.2–0.4."
        ),
        default=0.329 / 400e3,
        bounds=(0.0, 1e-4),
        tier=1,
    ),
    "Cn2_0": ParamDef(
        name="Cn2_0", symbol="C_n²(0)", unit="m^{-2/3}",
        owner="Mission/Systems",
        status=Status.ESTIMATED,
        description=(
            "Ground-level atmospheric refractive index structure parameter.  "
            "Good sites: 1e-15.  Moderate: 1e-14.  Poor: 1e-13.  "
            "Single largest uncertainty in the turbulence model."
        ),
        default=1.7e-14,
        bounds=(1e-16, 1e-12),
        tier=1,
    ),
    "v_wind": ParamDef(
        name="v_wind", symbol="v", unit="m/s",
        owner="Mission/Systems",
        status=Status.ESTIMATED,
        description="RMS wind speed for the Hufnagel-Valley C_n² model (standard value 21 m/s).",
        default=21.0,
        bounds=(5.0, 50.0),
        tier=1,
    ),
    "h0": ParamDef(
        name="h0", symbol="h₀", unit="m",
        owner="Mission/Systems",
        status=Status.BASELINED,
        description="Ground station altitude above mean sea level.",
        default=0.0,
        bounds=(0.0, 5000.0),
        tier=1,
    ),
    "H_max": ParamDef(
        name="H_max", symbol="H_max", unit="m",
        owner="Mission/Systems",
        status=Status.BASELINED,
        description="Effective top of turbulent atmosphere for Rytov integral (~20 km standard).",
        default=20e3,
        bounds=(10e3, 30e3),
        tier=1,
    ),

    # ── Block 3: Background Noise (Ground Station / Mission) ─────────────────

    "H_bg": ParamDef(
        name="H_bg", symbol="H_bg", unit="W/m²/sr/m",
        owner="Ground Station",
        status=Status.ESTIMATED,
        description=(
            "Sky spectral radiance (night-time: ~1e-8 W/m²/sr/m; "
            "bright twilight: ~1e-5 W/m²/sr/m)."
        ),
        default=1e-8,
        bounds=(0.0, 1e-3),
        tier=1,
    ),
    "Omega_FOV": ParamDef(
        name="Omega_FOV", symbol="Ω_FOV", unit="sr",
        owner="Ground Station",
        status=Status.ESTIMATED,
        description="Receiver field-of-view solid angle.  Smaller → less background.",
        default=1e-10,
        bounds=(0.0, 1e-6),
        tier=1,
    ),
    "delta_lambda": ParamDef(
        name="delta_lambda", symbol="Δλ", unit="m",
        owner="Ground Station",
        status=Status.ESTIMATED,
        description="Spectral filter bandwidth.  Must cover Doppler shift at maximum range rate.",
        default=1e-9,
        bounds=(0.1e-9, 10e-9),
        tier=1,
    ),

    # ── Block 4: Ground Station Receiver (Ground Station) ────────────────────

    "D_rx": ParamDef(
        name="D_rx", symbol="D_rx", unit="m",
        owner="Ground Station",
        status=Status.ESTIMATED,
        description=(
            "Receiver aperture diameter.  D_rx ≥ 0.7 m required for ℓ_finite > 0 "
            "at 400 km with 100 MHz Si-SPAD (AEP-limited).  D_rx = 1 m practical."
        ),
        default=1.0,
        bounds=(0.1, 4.0),
        tier=1,
    ),
    "eta_rx": ParamDef(
        name="eta_rx", symbol="η_rx", unit="dimensionless",
        owner="Ground Station",
        status=Status.ESTIMATED,
        description="Receiver optical chain efficiency (telescope, filters, fibre, coupling).",
        default=0.45,
        bounds=(0.1, 1.0),
        tier=1,
    ),

    # ── Block 5: Pointing (ADCS) ─────────────────────────────────────────────

    "theta_pnt": ParamDef(
        name="theta_pnt", symbol="θ_pnt", unit="rad",
        owner="ADCS",
        status=Status.ESTIMATED,
        description="Mean (bias) pointing error.  Usually zero by design; non-zero if misaligned.",
        default=0.0,
        bounds=(0.0, 1e-4),
        tier=1,
    ),
    "sigma_pnt": ParamDef(
        name="sigma_pnt", symbol="σ_pnt", unit="rad",
        owner="ADCS",
        status=Status.ESTIMATED,
        description=(
            "Pointing jitter 1-sigma.  Optimal beam waist satisfies "
            "w₀* = λ/(π·σ_pnt·√2).  Typical CubeSat: 2–10 µrad."
        ),
        default=2e-6,
        bounds=(0.1e-6, 100e-6),
        tier=1,
    ),

    # ── Block 6: Detector (Ground Station) ───────────────────────────────────

    "dark_count_rate": ParamDef(
        name="dark_count_rate", symbol="DCR", unit="counts/s",
        owner="Ground Station",
        status=Status.ESTIMATED,
        description="Detector dark count rate.  Si-SPAD typical: 100–1000 cps.",
        default=500.0,
        bounds=(0.0, 1e6),
        tier=1,
    ),
    "eta_det": ParamDef(
        name="eta_det", symbol="η_det", unit="dimensionless",
        owner="Ground Station",
        status=Status.ESTIMATED,
        description="Single-photon detection efficiency.",
        default=0.60,
        bounds=(0.0, 1.0),
        tier=1,
    ),
    "tau_dead": ParamDef(
        name="tau_dead", symbol="τ_d", unit="s",
        owner="Ground Station",
        status=Status.ESTIMATED,
        description=(
            "Detector dead time.  Sets upper limit on f_clock: f_clock ≤ 1/τ_d. "
            "Si-SPAD: 20–100 ns.  SNSPD: 5–10 ns."
        ),
        default=50e-9,
        bounds=(1e-9, 1e-6),
        tier=1,
    ),
    "delta_t": ParamDef(
        name="delta_t", symbol="Δt", unit="s",
        owner="Ground Station",
        status=Status.ESTIMATED,
        description="Detection gate window duration = 1/f_clock.",
        default=1.0 / 100e6,
        bounds=(1e-9, 1e-6),
        tier=1,
    ),

    # ── Block 7: Protocol and Post-Processing (Software / OBC) ───────────────

    "r_PE": ParamDef(
        name="r_PE", symbol="r_PE", unit="dimensionless",
        owner="Software/OBC",
        status=Status.ESTIMATED,
        description=(
            "Fraction of sifted bits reserved for parameter estimation.  "
            "Too small → loose decoy bounds.  Too large → fewer key bits.  "
            "Optimal typically 10–20% for realistic pass lengths."
        ),
        default=0.10,
        bounds=(0.01, 0.50),
        tier=2,
    ),
    "ec_algorithm": ParamDef(
        name="ec_algorithm", symbol="EC algorithm", unit="string",
        owner="Software/OBC",
        status=Status.ESTIMATED,
        description="Error correction algorithm: 'cascade' (interactive) or 'ldpc' (one-way).",
        default=None,   # stored as str; special handling in registry
        bounds=None,
        tier=2,
    ),
    "epsilon_PA": ParamDef(
        name="epsilon_PA", symbol="ε_PA", unit="dimensionless",
        owner="Mission/Systems",
        status=Status.ESTIMATED,
        description=(
            "Composable security parameter for privacy amplification.  "
            "Typical: 1e-9 to 1e-12 depending on mission security requirements."
        ),
        default=1e-10,
        bounds=(1e-15, 1e-6),
        tier=2,
    ),
    "rf_bandwidth": ParamDef(
        name="rf_bandwidth", symbol="B_RF", unit="bits/s",
        owner="Ground Station",
        status=Status.ESTIMATED,
        description="Classical channel bandwidth available for post-processing data exchange.",
        default=10e6,
        bounds=(1e3, 1e9),
        tier=2,
    ),

    # ── Block 8: Orbital and Pass Geometry (Mission / Systems) ───────────────

    "h_orbit": ParamDef(
        name="h_orbit", symbol="H", unit="m",
        owner="Mission/Systems",
        status=Status.BASELINED,
        description="Nominal orbital altitude above WGS84 ellipsoid.",
        default=400e3,
        bounds=(200e3, 2000e3),
        tier=2,
    ),
    "inclination": ParamDef(
        name="inclination", symbol="i", unit="deg",
        owner="Mission/Systems",
        status=Status.ESTIMATED,
        description="Orbital inclination.  Determines pass geometry and coverage.",
        default=51.6,
        bounds=(0.0, 98.0),
        tier=2,
    ),
    "gs_lat": ParamDef(
        name="gs_lat", symbol="φ_GS", unit="deg",
        owner="Mission/Systems",
        status=Status.BASELINED,
        description="Ground station geodetic latitude (positive = north).",
        default=36.1,   # generic mid-latitude site
        bounds=(-90.0, 90.0),
        tier=2,
    ),
    "gs_lon": ParamDef(
        name="gs_lon", symbol="λ_GS", unit="deg",
        owner="Mission/Systems",
        status=Status.BASELINED,
        description="Ground station geodetic longitude (positive = east).",
        default=-86.7,  # Nashville, TN (Vanderbilt)
        bounds=(-180.0, 180.0),
        tier=2,
    ),
    "gs_alt_m": ParamDef(
        name="gs_alt_m", symbol="h_GS", unit="m",
        owner="Mission/Systems",
        status=Status.BASELINED,
        description="Ground station altitude above WGS84 ellipsoid.",
        default=182.0,  # Nashville elevation
        bounds=(0.0, 5000.0),
        tier=2,
    ),
    "theta_el_min": ParamDef(
        name="theta_el_min", symbol="θ_el,min", unit="deg",
        owner="Mission/Systems",
        status=Status.BASELINED,
        description="Minimum elevation angle for link operation (zenith angle < 80° limit).",
        default=10.0,
        bounds=(5.0, 30.0),
        tier=2,
    ),
    "t_acq": ParamDef(
        name="t_acq", symbol="t_acq", unit="s",
        owner="Mission/Systems",
        status=Status.ESTIMATED,
        description="Time for pointing acquisition and lock after satellite rises above θ_el,min.",
        default=15.0,
        bounds=(0.0, 60.0),
        tier=2,
    ),
    "dt_sim": ParamDef(
        name="dt_sim", symbol="Δt_sim", unit="s",
        owner="Mission/Systems",
        status=Status.BASELINED,
        description="Pass simulation timestep.  1 s is adequate for all published LEO QKD scenarios.",
        default=1.0,
        bounds=(0.1, 10.0),
        tier=2,
    ),
}
