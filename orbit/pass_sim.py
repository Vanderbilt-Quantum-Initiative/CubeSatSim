"""
orbit/pass_sim.py — Full-pass simulation: geometry → accumulation → key bits.

Architecture
------------
The pass simulator is the only place that knows the full calling sequence:

    1. Build geometry profile via orbit/geometry.py (Skyfield).
    2. Determine usable window (elevation cut + acquisition time).
    3. Loop over timesteps: evaluate_point() → accumulate.
    4. Finalise accumulation → pass-level observables.
    5. Decoy bounds (finite or asymptotic) from pass-level counts.
    6. Post-processing chain → n_sifted, n_key, f_EC, ℓ_finite.
    7. Assemble PassResult.

The inner loop is evaluate_point() in core/evaluator.py.  The pass
simulator does not touch any physics equations directly.

AccumulationStrategy Protocol
-----------------------------
Any object implementing accumulate(dt, link_state, source) and finalise()
can be plugged in.  StandardAccumulation is the default.

Usage
-----
    from params.registry import ParameterRegistry
    from params.scenarios import get_scenario
    from orbit.pass_sim import simulate_pass

    reg = ParameterRegistry()
    reg.update(get_scenario("vqi_400km"))
    result = simulate_pass(reg, t_start=..., t_end=...)
    print(f"ℓ_finite = {result.ell_finite:.0f} bits  go={result.go}")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from core.types import (
    DecoyBounds,
    Geometry,
    KeyRateResult,
    LinkState,
    PassResult,
    PostProcessingResult,
    SourceConfig,
)
from core.evaluator import evaluate_point
from orbit.geometry import create_satellite, elevation_profile, usable_window
from physics.atmosphere import hufnagel_valley
from physics.decoy import asymptotic_bounds, finite_bounds
from physics.keyrate import gllp_asymptotic, finite_key_length, binary_entropy
from physics.post_processing import (
    ec_efficiency,
    classical_bandwidth_check,
    pe_split,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

class AccumulationStrategy(Protocol):
    def accumulate(self, dt: float, link_state: LinkState,
                   source: SourceConfig) -> None: ...
    def finalise(self) -> "AccumulationResult": ...


@dataclass
class AccumulationResult:
    n_signal: float          # total expected signal-intensity detections
    n_decoy: float           # total expected decoy-intensity detections
    n_vacuum: float          # total expected vacuum detections
    E_mu_weighted: float     # detection-weighted mean signal QBER
    E_nu_weighted: float     # detection-weighted mean decoy QBER
    Q_mu_avg: float          # time-averaged signal gain
    Q_nu_avg: float          # time-averaged decoy gain
    n_timesteps: int         # number of timesteps accumulated


class StandardAccumulation:
    """Default accumulation strategy: detection-weighted QBER, Poisson counts."""

    def __init__(self, f_clock: float) -> None:
        self._f_clock = f_clock
        self._n_sig    = 0.0
        self._n_dec    = 0.0
        self._n_vac    = 0.0
        self._wqber_sig = 0.0   # sum of Q_mu * E_mu * dt (numerator for weighted QBER)
        self._wqber_dec = 0.0
        self._sum_Q_mu  = 0.0
        self._sum_Q_nu  = 0.0
        self._steps     = 0

    def accumulate(self, dt: float, link_state: LinkState,
                   source: SourceConfig) -> None:
        sig = link_state.detections["signal"]
        dec = link_state.detections["decoy"]
        vac = link_state.detections["vacuum"]
        q   = source.sifting_factor()

        # Expected detections per timestep per intensity
        # n_mu = f_clock * P_mu * Q_mu * q * dt
        dn_sig = self._f_clock * source.P_mu  * sig.Q * q * dt
        dn_dec = self._f_clock * source.P_nu  * dec.Q     * dt  # decoy not sifted
        dn_vac = self._f_clock * source.P_vac * vac.Q     * dt

        self._n_sig += dn_sig
        self._n_dec += dn_dec
        self._n_vac += dn_vac

        # Detection-weighted QBER numerator
        self._wqber_sig += sig.E * dn_sig
        self._wqber_dec += dec.E * dn_dec

        self._sum_Q_mu += sig.Q
        self._sum_Q_nu += dec.Q
        self._steps    += 1

    def finalise(self) -> AccumulationResult:
        E_mu = self._wqber_sig / self._n_sig if self._n_sig > 0 else 0.5
        E_nu = self._wqber_dec / self._n_dec if self._n_dec > 0 else 0.5
        Q_mu_avg = self._sum_Q_mu / self._steps if self._steps > 0 else 0.0
        Q_nu_avg = self._sum_Q_nu / self._steps if self._steps > 0 else 0.0
        return AccumulationResult(
            n_signal=self._n_sig,
            n_decoy=self._n_dec,
            n_vacuum=self._n_vac,
            E_mu_weighted=E_mu,
            E_nu_weighted=E_nu,
            Q_mu_avg=Q_mu_avg,
            Q_nu_avg=Q_nu_avg,
            n_timesteps=self._steps,
        )


# ---------------------------------------------------------------------------
# Per-timestep arrays (for time-series outputs in PassResult)
# ---------------------------------------------------------------------------

@dataclass
class _TimeSeries:
    time:          list[float] = field(default_factory=list)
    elevation:     list[float] = field(default_factory=list)
    eta_0:         list[float] = field(default_factory=list)
    qber_instant:  list[float] = field(default_factory=list)
    R_instant:     list[float] = field(default_factory=list)
    cumulative_n:  list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def simulate_pass(
    registry,   # ParameterRegistry (typed import avoided to prevent circular)
    t_start: datetime | None = None,
    t_end: datetime | None = None,
    accumulation: AccumulationStrategy | None = None,
    decoy_mode: str = "finite",
) -> PassResult:
    """Run a full satellite pass simulation.

    Parameters
    ----------
    registry
        A ParameterRegistry with all parameters set.
    t_start, t_end
        UTC start and end of the window to simulate.  If None, a default
        90-minute window beginning at 2025-01-01 00:00 UTC is used —
        good for exploratory runs; replace with actual pass times for
        mission analysis.
    accumulation
        Accumulation strategy.  Defaults to StandardAccumulation.
    decoy_mode
        "finite" (composably secure, default) or "asymptotic" (design exploration).

    Returns
    -------
    PassResult
        Complete pass-level output.  go = True iff ell_finite > 0.

    Raises
    ------
    RuntimeError
        If no usable timesteps exist (satellite never above θ_el,min, or
        all timesteps rejected by the 80° zenith guard).
    """
    errors = registry.validate()
    if errors:
        for e in errors:
            logger.warning("Registry validation: %s", e)

    # ── 1. Extract typed configs ───────────────────────────────────────────
    channel  = registry.build_channel()
    source   = registry.build_source()
    pp_cfg   = registry.build_post_processing()
    detector = registry.build_detector()

    gs_lat      = registry.get("gs_lat")
    gs_lon      = registry.get("gs_lon")
    gs_alt_m    = registry.get("gs_alt_m")
    h_orbit     = registry.get("h_orbit")
    inclination = registry.get("inclination")
    theta_el_min= registry.get("theta_el_min")
    t_acq       = registry.get("t_acq")
    dt          = registry.get("dt_sim")

    # ── 2. Build geometry profile ─────────────────────────────────────────
    if t_start is None:
        t_start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    if t_end is None:
        t_end = datetime(2025, 1, 1, 1, 30, 0, tzinfo=timezone.utc)

    sat = create_satellite(h_orbit=h_orbit, inclination=inclination)
    profile = elevation_profile(sat, gs_lat, gs_lon, gs_alt_m,
                                t_start, t_end, dt=dt)

    usable, T_pass = usable_window(profile, theta_el_min, t_acq, dt=dt)

    if not usable:
        raise RuntimeError(
            "No usable timesteps in pass window.  "
            "Satellite may not pass above θ_el,min, or all steps rejected by "
            "the 80° zenith guard.  Check t_start/t_end and gs_lat/gs_lon."
        )

    logger.info("Pass: %d usable steps, T_pass=%.1f s", len(usable), T_pass)

    # ── 3. Pre-compute shared objects ─────────────────────────────────────
    cn2_profile = hufnagel_valley(Cn2_0=channel.Cn2_0, v=channel.v_wind)

    if accumulation is None:
        accumulation = StandardAccumulation(f_clock=source.f_clock)

    ts = _TimeSeries()
    cumulative_n = 0.0
    q = source.sifting_factor()

    # ── 4. Timestep loop ──────────────────────────────────────────────────
    jensen_gain_gaps: list[float] = []
    jensen_qber_gaps: list[float] = []

    for step_i, geom in enumerate(usable):
        try:
            link_state = evaluate_point(
                geometry=geom,
                channel=channel,
                source=source,
                detector=detector,
                cn2_profile=cn2_profile,
            )
        except ValueError as exc:
            logger.warning("Step %d skipped: %s", step_i, exc)
            continue

        accumulation.accumulate(dt, link_state, source)

        sig = link_state.detections["signal"]
        dn = source.f_clock * source.P_mu * sig.Q * q * dt
        cumulative_n += dn

        # Jensen gap tracking (informational)
        import math as _math
        mean_eta = link_state.fading.mean_eta()
        Y_0 = detector.dark_count_rate() / source.f_clock + 0.0  # approx
        Q_naive = 1.0 - (1.0 - Y_0) * _math.exp(-mean_eta * source.mu)
        jensen_gain_gaps.append(Q_naive - sig.Q)

        # Time-series arrays
        t_elapsed = step_i * dt
        ts.time.append(t_elapsed)
        ts.elevation.append(geom.theta_el)
        ts.eta_0.append(link_state.loss_budget.eta_0)
        ts.qber_instant.append(sig.E)
        ts.cumulative_n.append(cumulative_n)
        # Instantaneous key fraction (informational — requires pass-level decoy bounds)
        ts.R_instant.append(0.0)   # filled after decoy bounds computed

    # ── 5. Finalise accumulation ──────────────────────────────────────────
    acc = accumulation.finalise()

    if acc.n_signal <= 0:
        raise RuntimeError("Zero signal detections accumulated.  Check channel parameters.")

    logger.info(
        "Accumulated: n_sig=%.2e  n_dec=%.2e  E_mu=%.4f  E_nu=%.4f",
        acc.n_signal, acc.n_decoy, acc.E_mu_weighted, acc.E_nu_weighted,
    )

    # ── 6. Decoy bounds ───────────────────────────────────────────────────
    from core.types import DetectionResult
    sig_obs = DetectionResult(Q=acc.Q_mu_avg, E=acc.E_mu_weighted,
                              intensity=source.mu, n_counts=acc.n_signal)
    dec_obs = DetectionResult(Q=acc.Q_nu_avg, E=acc.E_nu_weighted,
                              intensity=source.nu, n_counts=acc.n_decoy)
    vac_obs = DetectionResult(Q=0.0, E=0.5, intensity=0.0, n_counts=acc.n_vacuum)

    # Y_0: dark count probability per gate
    Y_0 = detector.dark_count_rate() / source.f_clock

    # Pulse counts for Hoeffding bounds.
    # Q_mu is a frequency estimated from signal pulses; Q_nu from decoy pulses.
    # All decoy pulses go to PE (they never contribute to the sifted key).
    # The signal PE fraction r_PE is used here consistently with pe_split below.
    n_mu_pulses_PE = source.f_clock * source.P_mu * T_pass  # all signal pulses
    n_nu_pulses_PE = source.f_clock * source.P_nu * T_pass  # all decoy pulses → all PE

    if decoy_mode == "finite":
        decoy_b = finite_bounds(
            sig_obs, dec_obs, vac_obs,
            source.mu, source.nu, Y_0,
            n_PE=acc.n_signal * pp_cfg.r_PE,
            confidence=0.99,
            n_mu_pulses=n_mu_pulses_PE,
            n_nu_pulses=n_nu_pulses_PE,
            n_nu_detections=acc.n_decoy,
        )
    else:
        decoy_b = asymptotic_bounds(
            sig_obs, dec_obs, vac_obs,
            source.mu, source.nu, Y_0,
        )

    logger.info("Decoy bounds (%s): Q1_lower=%.4e  e1_upper=%.4f",
                decoy_mode, decoy_b.Q1_lower, decoy_b.e1_upper)

    # ── 7. Post-processing chain ──────────────────────────────────────────
    n_sifted = acc.n_signal   # already sifting-corrected in accumulation
    n_PE, n_key = pe_split(n_sifted, pp_cfg.r_PE)

    f_EC = ec_efficiency(n_key, acc.E_mu_weighted, pp_cfg.ec_algorithm)
    ec_feasible, classical_vol, ec_rounds = classical_bandwidth_check(
        n_sifted, acc.E_mu_weighted, f_EC,
        pp_cfg.ec_algorithm, T_pass, pp_cfg.rf_bandwidth,
    )
    leak_EC = n_key * f_EC * binary_entropy(acc.E_mu_weighted)

    pp_result = PostProcessingResult(
        n_sifted=n_sifted,
        n_PE=n_PE,
        n_key=n_key,
        f_EC=f_EC,
        leak_EC=leak_EC,
        classical_data_volume=classical_vol,
        classical_rounds=ec_rounds,
        ec_feasible=ec_feasible,
    )

    # ── 8. Key rate ───────────────────────────────────────────────────────
    R = gllp_asymptotic(decoy_b, sig_obs, q, f_EC)
    ell_finite = finite_key_length(n_key, R, acc.E_mu_weighted, f_EC, pp_cfg.epsilon_PA)
    skbr = source.f_clock * R

    kr_result = KeyRateResult(R=R, skbr=skbr, ell_finite=ell_finite)

    logger.info("Key rate: R=%.4e  ℓ_finite=%.2e bits  go=%s",
                R, ell_finite, ell_finite > 0)

    # ── 9. Assemble PassResult ────────────────────────────────────────────
    jensen_gain_gap = float(np.mean(jensen_gain_gaps)) if jensen_gain_gaps else 0.0
    jensen_qber_gap = 0.0  # tracked in evaluator; not re-computed here

    return PassResult(
        time=np.array(ts.time),
        elevation=np.array(ts.elevation),
        eta_0=np.array(ts.eta_0),
        qber_instant=np.array(ts.qber_instant),
        R_instant=np.zeros(len(ts.time)),
        cumulative_n=np.array(ts.cumulative_n),
        n_sifted=n_sifted,
        E_mu_weighted=acc.E_mu_weighted,
        ell_finite=ell_finite,
        T_pass=T_pass,
        go=ell_finite > 0,
        post_processing=pp_result,
        decoy_bounds=decoy_b,
        key_rate=kr_result,
        jensen_gain_gap=jensen_gain_gap,
        jensen_qber_gap=jensen_qber_gap,
    )
