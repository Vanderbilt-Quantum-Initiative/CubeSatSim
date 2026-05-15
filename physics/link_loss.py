"""
physics/link_loss.py — Optical link loss budget for the VQI QKD simulation.

All loss terms are dimensionless transmissivities in [0, 1].  Functions are
stateless and operate on scalars so they can be called per-timestep without
allocating intermediate arrays.

Provided:
    diffraction_loss     Gaussian beam clipping at a finite circular aperture.
    pointing_loss        Gaussian-beam pointing loss from mean offset + RMS jitter.
    compute_loss_budget  Assembles per-term transmissivities into a LossBudget.

Design note (from Plan.md §3.6):
    The loss budget is never collapsed to a scalar before compute_loss_budget.
    Per-term breakdown is required for waterfall plots and sensitivity analysis.
"""

from __future__ import annotations

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import LossBudget


# ---------------------------------------------------------------------------
# Diffraction loss
# ---------------------------------------------------------------------------

def diffraction_loss(lambda_: float, w0: float, L: float, D_rx: float) -> float:
    """
    Power fraction captured by a circular receiver aperture from a Gaussian beam.

    The Gaussian beam radius at propagation distance L from the transmitter waist:

        W(L) = w0 · √(1 + (λL / (π w0²))²)

    The fraction of total beam power collected by a hard-edged circular aperture
    of diameter D_rx (the Encircled Energy integral for a Gaussian profile):

        η_diff = 1 − exp(−2 r_rx² / W(L)²)     where r_rx = D_rx / 2

    This is exact for a TEM₀₀ Gaussian beam and a perfectly centred aperture.
    Pointing errors are applied separately in pointing_loss().

    Args:
        lambda_: Wavelength (m).
        w0:      Transmitter beam waist radius at exit aperture (m).
        L:       Slant-range from transmitter to receiver (m).
        D_rx:    Receiver aperture diameter (m).

    Returns:
        η_diff ∈ (0, 1].  Approaches 1 for large D_rx or short L.

    Raises:
        ValueError: on non-positive or physically implausible arguments.
    """
    if lambda_ <= 0.0:
        raise ValueError(f"lambda_ must be positive; got {lambda_}")
    if w0 <= 0.0:
        raise ValueError(f"w0 must be positive; got {w0}")
    if L < 0.0:
        raise ValueError(f"L must be non-negative; got {L}")
    if D_rx <= 0.0:
        raise ValueError(f"D_rx must be positive; got {D_rx}")

    if L == 0.0:
        return 1.0

    # Far-field Rayleigh range
    z_R = math.pi * w0 ** 2 / lambda_
    W_L = w0 * math.hypot(1.0, L / z_R)   # hypot avoids cancellation for L >> z_R

    r_rx = D_rx / 2.0
    return 1.0 - math.exp(-2.0 * r_rx ** 2 / W_L ** 2)


# ---------------------------------------------------------------------------
# Pointing loss
# ---------------------------------------------------------------------------

def pointing_loss(
    lambda_: float,
    w0: float,
    theta_pnt: float,
    sigma_pnt: float,
) -> float:
    """
    Pointing loss for a Gaussian beam with a static mean offset and Gaussian jitter.

    The far-field 1/e² half-angle divergence of a TEM₀₀ Gaussian beam:

        θ_div = λ / (π w0)

    For a pointing error that is the sum of a deterministic mean offset θ_pnt and
    zero-mean Gaussian jitter with 1-σ = σ_pnt, the time-averaged on-axis intensity
    fraction is (Andrews & Phillips 2005, §4.3):

        η_pnt = exp(−2 (θ_pnt² + σ_pnt²) / θ_div²)

    This is the "static Gaussian model": the jitter is treated as a fixed RMS
    contribution to the effective offset rather than a time-varying random process.
    For a more accurate fading treatment of pointing wander, use the FadingModel
    (deferred to second-order additions, Plan.md §10.12).

    Args:
        lambda_:   Wavelength (m).
        w0:        Transmitter beam waist radius (m).
        theta_pnt: Mean pointing error (rad); zero for perfect boresight.
        sigma_pnt: Pointing jitter 1-σ (rad).

    Returns:
        η_pnt ∈ (0, 1].  Returns 1.0 when both theta_pnt and sigma_pnt are zero.

    Raises:
        ValueError: on non-positive lambda_/w0 or negative angle arguments.
    """
    if lambda_ <= 0.0:
        raise ValueError(f"lambda_ must be positive; got {lambda_}")
    if w0 <= 0.0:
        raise ValueError(f"w0 must be positive; got {w0}")
    if theta_pnt < 0.0:
        raise ValueError(f"theta_pnt must be non-negative; got {theta_pnt}")
    if sigma_pnt < 0.0:
        raise ValueError(f"sigma_pnt must be non-negative; got {sigma_pnt}")

    theta_div = lambda_ / (math.pi * w0)       # 1/e² half-angle divergence (rad)
    theta_rms_sq = theta_pnt ** 2 + sigma_pnt ** 2
    return math.exp(-2.0 * theta_rms_sq / theta_div ** 2)


# ---------------------------------------------------------------------------
# Loss budget assembly
# ---------------------------------------------------------------------------

def compute_loss_budget(
    eta_tx: float,
    eta_atm: float,
    eta_diff: float,
    eta_pnt: float,
    eta_rx: float,
) -> LossBudget:
    """
    Assemble individual transmissivity terms into a LossBudget.

    The total static channel transmissivity is:

        η₀ = η_tx · η_atm · η_diff · η_pnt · η_rx

    Each term is stored individually so that waterfall plots and sensitivity
    analysis can decompose the budget without recomputation.

    This function does not call physics routines — it is a pure assembly step.
    The caller (evaluate_point or a test) supplies pre-computed per-term values.

    Args:
        eta_tx:   Transmitter optical chain efficiency (e.g. beam shaping, polariser).
        eta_atm:  Beer-Lambert atmospheric attenuation (from atmosphere.py).
        eta_diff: Diffraction / beam-spreading capture fraction (from diffraction_loss).
        eta_pnt:  Pointing loss factor (from pointing_loss).
        eta_rx:   Receiver optical chain efficiency (filters, coupling, detector window).

    Returns:
        LossBudget with all five terms and their product η₀.

    Raises:
        ValueError: if any term is outside (0, 1].
    """
    terms = {
        "eta_tx":   eta_tx,
        "eta_atm":  eta_atm,
        "eta_diff": eta_diff,
        "eta_pnt":  eta_pnt,
        "eta_rx":   eta_rx,
    }
    for name, val in terms.items():
        if not (0.0 < val <= 1.0):
            raise ValueError(
                f"{name} must be in (0, 1]; got {val:.6g}.  "
                "Transmissivities are dimensionless power fractions, not dB values."
            )

    eta_0 = eta_tx * eta_atm * eta_diff * eta_pnt * eta_rx
    return LossBudget(
        eta_tx=eta_tx,
        eta_atm=eta_atm,
        eta_diff=eta_diff,
        eta_pnt=eta_pnt,
        eta_rx=eta_rx,
        eta_0=eta_0,
    )
