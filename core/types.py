"""
core/types.py — Shared data structures for the VQI QKD link budget simulation.

All dataclasses used across more than one module live here. Module-local
types (QRNGModel, DetectorModel, AccumulationResult) stay in their own files.

Organisation:
    1. Imports
    2. Geometry and configuration inputs  (Geometry, ChannelConfig, SourceConfig,
                                           PostProcessingConfig)
    3. Physics intermediate outputs        (LossBudget, DetectionResult,
                                           DecoyBounds, EURDecoyBounds,
                                           SecurityBudget, KeyRateResult,
                                           PostProcessingResult)
    4. Composite state containers          (LinkState, PassResult)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # FadingModel is a Protocol defined in physics/turbulence.py.
    # Imported under TYPE_CHECKING only to avoid a circular import at runtime;
    # LinkState holds a reference typed as 'FadingModel'.
    from physics.turbulence import FadingModel


# ---------------------------------------------------------------------------
# 1. Geometry and configuration inputs
# ---------------------------------------------------------------------------

@dataclass
class Geometry:
    """
    Satellite-ground geometry at a single instant.

    All angles in radians. Constructed by orbit/geometry.py (Skyfield wrapper)
    and consumed by physics modules and evaluate_point().
    """
    theta_el: float     # elevation angle above horizon (rad)
    L: float            # slant range from ground station to satellite (m)
    zeta: float         # zenith angle = π/2 − theta_el (rad)
    h_orbit: float      # orbital altitude above WGS84 ellipsoid (m)


@dataclass
class ChannelConfig:
    """
    Static channel and hardware parameters that do not change during a pass.

    Consumed by evaluate_point() and passed into the physics modules. The
    parameter registry constructs this object from its stored values; physics
    modules never read the registry directly.

    Fields with defaults are either physically standard values (v_wind, H_max)
    or zero-baseline assumptions (h0). All others must be set explicitly.
    """

    # --- Transmitter optics (Payload team) ---
    eta_tx: float       # transmitter optical chain efficiency (dimensionless)
    lambda_: float      # wavelength (m); underscore avoids shadowing builtin
    w0: float           # transmitter beam waist at exit aperture (m)

    # --- Receiver optics (Ground Station team) ---
    D_rx: float         # receiver aperture diameter (m)
    eta_rx: float       # receiver optical chain efficiency (dimensionless)

    # --- Atmospheric channel (Mission / Systems) ---
    alpha: float        # Beer-Lambert extinction coefficient (1/m)
    Cn2_0: float        # ground-level refractive index structure parameter (m^{-2/3})
    v_wind: float = 21.0    # RMS wind speed for Hufnagel-Valley model (m/s)
    h0: float = 0.0         # ground station altitude above sea level (m)
    H_max: float = 20e3     # effective top of turbulent atmosphere (m); ~20 km standard

    # --- Background noise (Ground Station / Mission) ---
    H_bg: float = 0.0           # sky spectral radiance (W/m²/sr/m)
    Omega_FOV: float = 0.0      # receiver field of view solid angle (sr)
    delta_lambda: float = 1e-9  # spectral filter bandwidth (m); must cover Doppler shift

    # --- Pointing (ADCS team) ---
    theta_pnt: float = 0.0      # mean pointing error (rad)
    sigma_pnt: float = 1e-6     # pointing jitter 1-sigma (rad)

    # --- Optical QBER floor (Payload / ADCS / Ground Station) ---
    e_opt: float = 0.03         # residual optical error rate after compensation


@dataclass
class SourceConfig:
    """
    Quantum source configuration: intensities, preparation probabilities, clock rate.

    Owned by Payload team. f_clock is a joint Payload/Ground Station trade —
    it cannot exceed 1 / tau_d of the detector without pile-up distortion.

    P_vac is always derived; do not set it directly.
    sifting_factor() returns the effective basis-matching probability q.
    """
    mu: float           # signal intensity (mean photons/pulse)
    nu: float           # weak decoy intensity (mean photons/pulse); must satisfy nu < mu
    P_mu: float         # probability of emitting a signal-intensity pulse
    P_nu: float         # probability of emitting a weak-decoy pulse
    P_X: float          # probability of choosing the X (key-generating) basis
    f_clock: float      # pulse repetition rate (Hz)

    @property
    def P_vac(self) -> float:
        """Probability of emitting a vacuum (zero-intensity) pulse."""
        return 1.0 - self.P_mu - self.P_nu

    def sifting_factor(self) -> float:
        """
        Effective sifting factor q: fraction of pulses surviving basis reconciliation.

        Standard BB84 (P_X = 0.5): q = 0.5.
        Efficient / asymmetric BB84 (P_X → 1): q → P_X² + (1-P_X)² → 1.

        Only signal-intensity pulses in matching bases contribute to the sifted key.
        """
        return self.P_X ** 2 + (1.0 - self.P_X) ** 2

    def validate(self) -> tuple[bool, str]:
        """Check internal consistency of preparation probabilities and intensities."""
        if not (0.0 < self.nu < self.mu):
            return False, f"Require 0 < nu < mu; got nu={self.nu}, mu={self.mu}"
        if not (0.0 <= self.P_mu <= 1.0 and 0.0 <= self.P_nu <= 1.0):
            return False, "P_mu and P_nu must be in [0, 1]"
        if self.P_vac < 0.0:
            return False, f"P_mu + P_nu > 1; P_vac={self.P_vac:.4f}"
        if not (0.0 < self.P_X < 1.0):
            return False, f"P_X must be in (0, 1); got {self.P_X}"
        if self.f_clock <= 0.0:
            return False, f"f_clock must be positive; got {self.f_clock}"
        return True, "ok"


@dataclass
class PostProcessingConfig:
    """
    Classical post-processing parameters.

    Owned by Software / Mission Operations. These are not free parameters:
    r_PE trades decoy bound tightness against key generation bits;
    ec_algorithm determines f_EC and classical bandwidth requirements;
    epsilon_PA is set by mission security requirements.
    """
    r_PE: float             # fraction of sifted bits reserved for parameter estimation
    ec_algorithm: str       # "cascade" (interactive) or "ldpc" (one-way)
    epsilon_PA: float       # composable security parameter for privacy amplification
    rf_bandwidth: float     # classical channel bandwidth (bits/s)

    def validate(self) -> tuple[bool, str]:
        if not (0.0 < self.r_PE < 1.0):
            return False, f"r_PE must be in (0, 1); got {self.r_PE}"
        if self.ec_algorithm not in ("cascade", "ldpc"):
            return False, f"ec_algorithm must be 'cascade' or 'ldpc'; got {self.ec_algorithm!r}"
        if self.epsilon_PA <= 0.0:
            return False, f"epsilon_PA must be positive; got {self.epsilon_PA}"
        if self.rf_bandwidth <= 0.0:
            return False, f"rf_bandwidth must be positive; got {self.rf_bandwidth}"
        return True, "ok"


# ---------------------------------------------------------------------------
# 2. Physics intermediate outputs
# ---------------------------------------------------------------------------

@dataclass
class LossBudget:
    """
    Per-term transmissivity breakdown for a single geometry.

    eta_0 = eta_tx * eta_atm * eta_diff * eta_pnt * eta_rx.

    Returned by physics/link_loss.py. Never collapsed to a scalar before
    this point — the per-term breakdown is required for waterfall plots and
    sensitivity analysis.
    """
    eta_tx: float       # transmitter optical chain efficiency
    eta_atm: float      # atmospheric (Beer-Lambert) attenuation
    eta_diff: float     # diffraction / beam-spreading loss
    eta_pnt: float      # pointing loss
    eta_rx: float       # receiver optical chain efficiency
    eta_0: float        # product of all terms above

    def to_db_dict(self) -> dict[str, float]:
        """Return each loss term in dB (negative = loss). Useful for waterfall plots."""
        terms = {
            "eta_tx":   self.eta_tx,
            "eta_atm":  self.eta_atm,
            "eta_diff": self.eta_diff,
            "eta_pnt":  self.eta_pnt,
            "eta_rx":   self.eta_rx,
            "eta_0":    self.eta_0,
        }
        return {
            k: 10.0 * math.log10(v) if v > 0.0 else float("-inf")
            for k, v in terms.items()
        }


@dataclass
class DetectionResult:
    """
    Fading-averaged detection observables for a single source intensity.

    Produced by physics/detection.py for each of the three intensities
    (mu, nu, vacuum) at a single geometry. All three share the same
    FadingModel instance (same channel, different intensity).

    n_counts is left at 0.0 by evaluate_point(); the pass simulator
    fills it during per-timestep accumulation:
        n_counts = f_clock * P_intensity * Q * sifting_factor * dt
    """
    Q: float            # fading-averaged gain (detection probability per pulse)
    E: float            # fading-averaged QBER
    intensity: float    # source intensity used (mu, nu, or 0.0)
    n_counts: float     # expected detections this timestep; filled by accumulator


@dataclass
class DecoyBounds:
    """
    Decoy-state estimates of single-photon gain and phase error rate.

    Produced by physics/decoy.py after full-pass accumulation.
    mode distinguishes whether statistical corrections have been applied.

    Never computed per-timestep — requires accumulated count statistics
    across the full pass for meaningful estimation.
    """
    Q1_lower: float     # lower bound on single-photon gain
    e1_upper: float     # upper bound on single-photon phase error rate
    mode: str           # "asymptotic" (design exploration) or "finite" (key extraction)


@dataclass
class EURDecoyBounds:
    """
    EUR-based decoy-state bounds for per-basis, per-photon-number analysis.

    Produced by physics/decoy.eur_decoy_bounds() using per-basis, per-intensity
    counts from the accumulator.  Inputs to eur_key_length().

    Reference: Lim, Curty, Walenta, Xu & Zbinden (2014), PRA 89, 022307;
               Wiesemann et al. (2026), Quantum 10, 2037.
    """
    s_Z0_lower: float   # lower bound on vacuum-source detections in key basis
    s_Z1_lower: float   # lower bound on single-photon detections in key basis
    phi_Z_upper: float  # upper bound on single-photon phase error rate in key basis

    # Diagnostic intermediates
    Y0_bound: float     # vacuum yield used (calibrated)
    Y1_lower: float     # single-photon yield lower bound (basis-independent)
    e1_upper: float     # test-basis single-photon QBER upper bound (= phi_Z via EUR)


@dataclass
class SecurityBudget:
    """
    Security parameter composition for EUR finite-key proof.

    The total composable security parameter epsilon_total is subdivided equally
    across n_terms failure sub-events (parameter estimation, EC, PA, smoothing).
    For Phase 1, equal subdivision is used; future work can optimise the split.

    Reference: Lim et al. (2014) §III; Wiesemann et al. (2026) §IV.
    """
    epsilon_total: float = 1e-10    # total composable security parameter
    n_terms: int = 6                # number of sub-events to cover

    @property
    def epsilon_sub(self) -> float:
        """Per-sub-event security parameter."""
        return self.epsilon_total / self.n_terms


@dataclass
class KeyRateResult:
    """
    Secret key rate outputs.

    Produced by physics/keyrate.py after decoy bounds are available.
    R may be negative (link infeasible); ell_finite is clamped to >= 0.
    """
    R: float            # asymptotic key fraction per pulse (bits/pulse; may be < 0)
    skbr: float         # secret key bit rate = f_clock * R (bits/s; may be < 0)
    ell_finite: float   # composably secure key bits per pass; clamped to max(0, ...)


@dataclass
class PostProcessingResult:
    """
    Classical post-processing chain outputs.

    Produced by physics/post_processing.py. ec_feasible flags whether the
    classical channel can carry the required syndrome data within T_pass.
    A False here means the key rate calculation is academic — the link
    cannot close even if ell_finite > 0.
    """
    n_sifted: float             # sifted bits after basis reconciliation
    n_PE: float                 # bits reserved for parameter estimation
    n_key: float                # bits available for key generation = n_sifted - n_PE
    f_EC: float                 # error correction efficiency (>= 1.0; function of n, E_mu)
    leak_EC: float              # bits leaked during error correction
    classical_data_volume: float    # total classical bits exchanged during post-processing
    classical_rounds: int           # number of interactive rounds (1 for LDPC, >1 for Cascade)
    ec_feasible: bool               # True if rf_bandwidth can support post-processing in T_pass


# ---------------------------------------------------------------------------
# 3. Composite state containers
# ---------------------------------------------------------------------------

@dataclass
class LinkState:
    """
    Complete evaluation state at a single geometry and instant in time.

    Returned by core/evaluate_point(). Consumed by the pass accumulator,
    which extracts DetectionResult values and discards the rest per timestep.

    decoy_bounds and key_rate are always None at the per-timestep level.
    They are populated only at the pass level, after accumulation, by
    physics/decoy.py and physics/keyrate.py respectively.
    """
    geometry: Geometry
    loss_budget: LossBudget
    fading: FadingModel                         # same instance used for all three intensities
    detections: dict[str, DetectionResult]      # keys: "signal", "decoy", "vacuum"
    decoy_bounds: DecoyBounds | None            # None until pass-level computation
    key_rate: KeyRateResult | None              # None until pass-level computation


@dataclass
class PassResult:
    """
    Complete output of a single satellite pass simulation.

    Returned by orbit/pass_sim.simulate_pass(). Time-series arrays cover the
    usable window only (above theta_el_min, after acquisition).

    Jensen gaps validate the fading integration:
        gain_gap  = Q_mu(mean_eta) - <Q_mu(eta)>  must be > 0
        qber_gap  = <QBER(eta)>    - QBER(mean_eta) must be > 0
    A negative gap in either indicates a bug in the fading integration.

    go is the mission-level binary: True iff ell_finite > 0.
    """

    # --- Time-series arrays (length = number of usable timesteps) ---
    time: np.ndarray            # elapsed time within usable window (s)
    elevation: np.ndarray       # elevation angle at each timestep (rad)
    eta_0: np.ndarray           # static transmissivity at each timestep
    qber_instant: np.ndarray    # fading-averaged QBER at each timestep
    R_instant: np.ndarray       # instantaneous key fraction estimate (informational only)
    cumulative_n: np.ndarray    # cumulative signal detections up to each timestep

    # --- Pass-level scalars ---
    n_sifted: float             # total sifted bits for the pass
    E_mu_weighted: float        # detection-weighted mean signal QBER across pass
    ell_finite: float           # composably secure key bits extracted (clamped >= 0)
    T_pass: float               # usable pass duration (s)
    go: bool                    # ell_finite > 0

    # --- Structured sub-results ---
    post_processing: PostProcessingResult
    decoy_bounds: DecoyBounds
    key_rate: KeyRateResult

    # --- Validation diagnostics ---
    jensen_gain_gap: float      # Q_mu(mean_eta) - <Q_mu(eta)>; must be > 0
    jensen_qber_gap: float      # <QBER(eta)> - QBER(mean_eta);  must be > 0

    # --- EUR-specific (None when proof_method="aep") ---
    eur_decoy_bounds: "EURDecoyBounds | None" = None