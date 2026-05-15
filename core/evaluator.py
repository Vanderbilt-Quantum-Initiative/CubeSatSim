"""
core/evaluator.py — Composes the full physics chain at a single geometry.

This is the inner loop. The pass simulator calls evaluate_point() at each
timestep, accumulates DetectionResult values across the pass, then runs
decoy bounds and key rate extraction on the accumulated statistics.

Adding or reordering a step in the evaluation chain means modifying this
file, not pass_sim.py. Physics modules are pure functions; this module
is the only place that knows the calling sequence.

Typical call from pass_sim.py:

    for geom in pass_geometries:
        state = evaluate_point(geom, channel, source, detector, ...)
        accumulator.accumulate(dt, state, source)
"""

from __future__ import annotations

import logging
import math

from core.types import (
    ChannelConfig,
    DetectionResult,
    Geometry,
    LinkState,
)
from core.types import SourceConfig
from physics.atmosphere import Cn2Profile, atmospheric_attenuation, hufnagel_valley
from physics.detector import DetectorModel
from physics.link_loss import (
    compute_loss_budget,
    diffraction_loss,
    pointing_loss,
)
from physics.turbulence import FadingModel, rytov_variance, select_fading_model
from physics import detection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ZETA_MAX = math.radians(80.0)  # reject zenith angles beyond 80° (sec^{11/6} diverges)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_point(
    geometry: Geometry,
    channel: ChannelConfig,
    source: SourceConfig,
    detector: DetectorModel,
    fading_model: FadingModel | None = None,
    cn2_profile: Cn2Profile | None = None,
) -> LinkState:
    """Evaluate the full optical-to-detection chain at a single geometry.

    Parameters
    ----------
    geometry
        Satellite-ground geometry (elevation, slant range, zenith angle).
    channel
        Static channel and hardware parameters for this scenario.
    source
        Quantum source configuration (intensities, prep probabilities, clock).
    detector
        Detector model (efficiency, dark counts, dead time, gate width).
    fading_model
        Pre-computed fading model for this geometry. If None, one is built
        from the channel parameters and cn2_profile. Pass a pre-computed
        model when the same geometry is evaluated at multiple source configs,
        or when fading models are cached on a zenith-angle grid.
    cn2_profile
        Cn² altitude profile. Required if fading_model is None. If also None,
        a Hufnagel-Valley profile is constructed from channel.Cn2_0 and
        channel.v_wind.

    Returns
    -------
    LinkState
        Complete per-timestep state. detections dict contains results for
        "signal", "decoy", and "vacuum" intensities. decoy_bounds and
        key_rate are None — they require pass-level accumulated statistics.

    Raises
    ------
    ValueError
        If zenith angle exceeds 80° (Rytov integral diverges).
    """

    # ── 0. Guard: reject geometries where the turbulence model breaks ─────

    if geometry.zeta > _ZETA_MAX:
        raise ValueError(
            f"Zenith angle {math.degrees(geometry.zeta):.1f}° exceeds 80° limit; "
            f"sec^{{11/6}}(ζ) diverges. Raise theta_el_min or check geometry."
        )

    # ── 1. Static loss budget ─────────────────────────────────────────────
    #
    # Each loss term is computed independently and assembled into a
    # LossBudget. The per-term breakdown feeds waterfall visualisations;
    # the scalar product η_0 feeds the fading model.

    loss_budget = _compute_loss_budget(geometry, channel)

    # ── 2. Fading model ───────────────────────────────────────────────────
    #
    # If no pre-computed model was provided, build one from the Cn² profile
    # and the Rytov integral at this geometry.

    if fading_model is None:
        if cn2_profile is None:
            cn2_profile = hufnagel_valley(Cn2_0=channel.Cn2_0, v=channel.v_wind)
        sigma_R2 = rytov_variance(
            lambda_=channel.lambda_,
            zeta=geometry.zeta,
            h0=channel.h0,
            cn2_profile=cn2_profile,
            H_max=channel.H_max,
        )
        fading_model = select_fading_model(sigma_R2, loss_budget.eta_0)
        logger.debug("Built fading model: σ²_R=%.3f → %s", sigma_R2, type(fading_model).__name__)

    # ── 3. Noise yield ────────────────────────────────────────────────────
    #
    # Y_0 = dark counts + background photons per gate window.
    # Shared across all three intensity settings (same detector, same sky).

    A_rx = math.pi * (channel.D_rx / 2.0) ** 2
    Y_0 = detection.noise_yield(
        detector=detector,
        H_bg=channel.H_bg,
        Omega_FOV=channel.Omega_FOV,
        A_rx=A_rx,
        delta_lambda=channel.delta_lambda,
        eta_rx=channel.eta_rx,
    )

    # ── 4. Multi-intensity detection loop ─────────────────────────────────
    #
    # For each intensity (signal μ, weak decoy ν, vacuum 0), compute the
    # fading-averaged gain Q and QBER E. All three share the same fading
    # model (same channel) but different Poisson statistics.
    #
    # Both Q and E are integrals over P(η), NOT evaluations at ⟨η⟩.
    # Using the mean transmissivity violates Jensen's inequality for both.

    detections: dict[str, DetectionResult] = {
        role: detection.compute_detection(
            fading=fading_model,
            detector=detector,
            intensity=intensity,
            e_opt=channel.e_opt,
            Y_0=Y_0,
        )
        for role, intensity in (("signal", source.mu), ("decoy", source.nu), ("vacuum", 0.0))
    }

    # ── 5. Jensen gap validation ──────────────────────────────────────────
    #
    # Computes the "naive" values at mean transmissivity and verifies
    # that fading makes things worse (lower gain, higher QBER). Negative
    # gaps indicate a bug in the fading integration.

    _validate_jensen_gaps(fading_model, source.mu, channel.e_opt, Y_0, detections["signal"])

    # ── 6. Assemble LinkState ─────────────────────────────────────────────

    return LinkState(
        geometry=geometry,
        loss_budget=loss_budget,
        fading=fading_model,
        detections=detections,
        decoy_bounds=None,   # requires pass-level accumulation
        key_rate=None,       # requires pass-level accumulation
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_loss_budget(geometry: Geometry, channel: ChannelConfig):
    """Compute the deterministic (no-turbulence) transmissivity breakdown."""

    eta_atm = atmospheric_attenuation(channel.alpha, geometry.L)

    eta_diff = diffraction_loss(
        lambda_=channel.lambda_,
        w0=channel.w0,
        L=geometry.L,
        D_rx=channel.D_rx,
    )

    eta_pnt = pointing_loss(
        lambda_=channel.lambda_,
        w0=channel.w0,
        theta_pnt=channel.theta_pnt,
        sigma_pnt=channel.sigma_pnt,
    )

    return compute_loss_budget(
        eta_tx=channel.eta_tx,
        eta_atm=eta_atm,
        eta_diff=eta_diff,
        eta_pnt=eta_pnt,
        eta_rx=channel.eta_rx,
    )


def _validate_jensen_gaps(
    fading: FadingModel,
    mu: float,
    e_opt: float,
    Y_0: float,
    signal: DetectionResult,
) -> None:
    """Log a warning if Jensen's inequality is violated for gain or QBER."""

    mean_eta = fading.mean_eta()
    Q_naive = 1.0 - (1.0 - Y_0) * math.exp(-mean_eta * mu)
    gain_gap = Q_naive - signal.Q

    E_naive = detection.instantaneous_qber(mean_eta, mu, e_opt, Y_0)
    qber_gap = signal.E - E_naive

    if gain_gap < -1e-12:
        logger.error(
            "Jensen gain gap is negative (%.2e): fading-averaged gain exceeds "
            "gain at mean transmissivity. Bug in expected_gain() or integrate().",
            gain_gap,
        )
    if qber_gap < -1e-12:
        logger.error(
            "Jensen QBER gap is negative (%.2e): fading-averaged QBER is below "
            "QBER at mean transmissivity. Bug in expected_qber() or integrate().",
            qber_gap,
        )

    logger.debug(
        "Jensen gaps: gain=%.3e (Q_naive=%.4e, Q_fading=%.4e) "
        "qber=%.3e (E_fading=%.4e, E_naive=%.4e)",
        gain_gap, Q_naive, signal.Q,
        qber_gap, signal.E, E_naive,
    )
