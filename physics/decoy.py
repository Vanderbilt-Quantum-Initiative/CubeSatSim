"""
physics/decoy.py — Decoy-state single-photon bounds (Stage 4).

Implements the Lo-Ma-Chen / Ma et al. (2005) decoy-state estimation
for a three-intensity protocol: signal (μ), weak decoy (ν), vacuum (0).

Two modes:
    asymptotic_bounds   Closed-form bounds. No statistical correction.
                        For design-space exploration where n → ∞.
    finite_bounds       Hoeffding-corrected bounds. Adversarial worst-case
                        on all observed rates. Required for finite-key extraction.

The bounds estimate two quantities:
    Q1_lower    Lower bound on the single-photon gain (detection probability
                per single-photon pulse). Enters GLLP key fraction as positive term.
    e1_upper    Upper bound on the single-photon phase error rate. Enters
                H₂(e₁) — the privacy-amplification cost.

Key reference:
    Ma, Qi, Zhao & Lo (2005), PRA 72, 012326 — closed-form decoy bounds.
    Lo, Ma & Chen (2005), PRL 94, 230504 — original decoy-state security proof.
    Scarani & Renner (2008), PRL 100, 200501 — finite-key framework.

IMPORTANT: The asymptotic bounds MUST NOT be used for finite-key extraction.
They assume infinite statistics and will overestimate Q1 and underestimate e1
for finite pass data (n ~ 10⁵–10⁶), yielding falsely optimistic ℓ_finite.
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import DetectionResult, DecoyBounds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hoeffding_slack(n: float, delta: float) -> float:
    """
    Hoeffding one-sided confidence interval half-width.

        slack = sqrt(ln(1/δ) / (2n))

    For an observable estimated from n Bernoulli trials with failure
    probability δ, the true probability lies within ±slack of the sample
    estimate with probability 1−δ.

    Args:
        n:     Number of samples (trials).
        delta: Failure probability = 1 − confidence.

    Returns:
        Hoeffding slack (dimensionless).
    """
    if n <= 0.0:
        return 0.5  # no data: maximum uncertainty
    return math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def _count_split(n_PE: float, Q_mu: float, Q_nu: float) -> tuple[float, float]:
    """
    Split n_PE parameter-estimation events between signal and decoy intensities
    proportional to their expected detection rates.

    Y_0 (vacuum/dark count rate) is calibrated independently and is NOT
    split from n_PE — see finite_bounds() docstring.

    The fraction at each intensity is proportional to Q at that intensity,
    assuming roughly equal preparation probabilities (exact splitting requires
    P_mu, P_nu which are not part of the decoy-bounds interface).

    Returns:
        (n_mu, n_nu) — expected PE detection counts at each intensity.
    """
    total_Q = Q_mu + Q_nu
    if total_Q <= 0.0:
        return 0.0, 0.0
    n_mu = n_PE * Q_mu / total_Q
    n_nu = n_PE * Q_nu / total_Q
    return n_mu, n_nu


def _q1_lower(Q_mu: float, Q_nu: float, Y_0: float, mu: float, nu: float) -> float:
    """
    Closed-form lower bound on single-photon gain Q1.

    Ma et al. (2005) Eq. (5):

        Q1^L = μ² e^{−μ} / (μν − ν²) × [Q_ν e^ν − Q_μ e^μ ν²/μ² − (μ²−ν²)/μ² × Y_0]

    This is the tightest two-intensity lower bound on the single-photon
    contribution to total gain. It exploits the Poisson photon-number
    statistics at both intensities to isolate the n=1 term.

    Returns:
        max(0, Q1^L) — clamped to zero; a negative bound means the
        channel is too lossy to bound Q1 from below (set Q1_lower = 0).
    """
    denom = mu * nu - nu ** 2   # μν − ν² = ν(μ−ν) > 0 since μ > ν > 0
    prefactor = mu ** 2 * math.exp(-mu) / denom
    bracket = (
        Q_nu * math.exp(nu)
        - Q_mu * math.exp(mu) * (nu / mu) ** 2
        - (mu ** 2 - nu ** 2) / mu ** 2 * Y_0
    )
    return max(0.0, prefactor * bracket)


def _e1_upper(E_nu: float, Q_nu: float, Y_0: float, Q1_lower: float,
              mu: float, nu: float) -> float:
    """
    Closed-form upper bound on single-photon phase error rate e1.

    Ma et al. (2005) Eq. (6):

        e1^U = (E_ν Q_ν e^ν − e0 Y_0) / (Y1 ν)

    where:
        e0 = 0.5  (vacuum detections have uniformly random bit values)
        Y1 = Q1_lower × e^μ / μ  (single-photon yield from gain bound)

    If Q1_lower = 0 (completely unbounded) e1 defaults to 0.5.

    Returns:
        min(0.5, e1^U) — clamped to maximum physical value.
    """
    e0 = 0.5
    if Q1_lower <= 0.0:
        return 0.5
    Y1 = Q1_lower * math.exp(mu) / mu
    numerator = E_nu * Q_nu * math.exp(nu) - e0 * Y_0
    denominator = Y1 * nu
    if denominator <= 0.0:
        return 0.5
    return min(0.5, numerator / denominator)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def asymptotic_bounds(
    signal: DetectionResult,
    decoy: DetectionResult,
    vacuum: DetectionResult,
    mu: float,
    nu: float,
    Y_0: float,
) -> DecoyBounds:
    """
    Asymptotic (infinite-statistics) decoy-state bounds.

    Applies Ma et al. (2005) closed-form expressions directly to the
    fading-averaged observables. No statistical correction.

    Use for design exploration and sensitivity analysis. Do NOT use
    for extracting finite-key lengths from real or simulated pass data —
    finite_bounds() applies the mandatory Hoeffding correction.

    Args:
        signal:  DetectionResult for signal pulses (intensity μ).
        decoy:   DetectionResult for weak-decoy pulses (intensity ν).
        vacuum:  DetectionResult for vacuum pulses (intensity 0).
        mu:      Signal intensity (photons/pulse).
        nu:      Weak-decoy intensity (photons/pulse); must satisfy 0 < ν < μ.
        Y_0:     Noise yield per gate (dimensionless probability).

    Returns:
        DecoyBounds(Q1_lower, e1_upper, mode="asymptotic").

    Raises:
        ValueError: if μ ≤ ν or either intensity is non-positive.
    """
    if not (0.0 < nu < mu):
        raise ValueError(f"Require 0 < ν < μ; got ν={nu}, μ={mu}")

    Q1_L = _q1_lower(signal.Q, decoy.Q, Y_0, mu, nu)
    e1_U = _e1_upper(decoy.E, decoy.Q, Y_0, Q1_L, mu, nu)

    return DecoyBounds(Q1_lower=Q1_L, e1_upper=e1_U, mode="asymptotic")


def finite_bounds(
    signal: DetectionResult,
    decoy: DetectionResult,
    vacuum: DetectionResult,
    mu: float,
    nu: float,
    Y_0: float,
    n_PE: float,
    confidence: float = 0.99,
    n_mu_pulses: float | None = None,
    n_nu_pulses: float | None = None,
    n_nu_detections: float | None = None,
) -> DecoyBounds:
    """
    Finite-sample decoy-state bounds with Hoeffding statistical correction.

    Each observed rate (Q_μ, Q_ν, E_ν, Y_0) is subject to sampling noise.
    With n_PE parameter-estimation events split proportionally among the three
    intensities, the true rate lies within ±slack of the observed value with
    probability ≥ confidence (Hoeffding's inequality, one-sided).

    To obtain conservative (adversarial) bounds:
        - Q_ν is replaced by Q_ν − slack  (lower → smaller Q1^L)
        - Q_μ is replaced by Q_μ + slack  (higher → smaller Q1^L)
        - E_ν is replaced by E_ν + slack  (higher → larger e1^U)
        - Y_0 is replaced by Y_0 + slack  (higher noise → smaller Q1^L and larger e1^U)

    This is the standard approach in finite-key QKD implementations
    (Scarani & Renner 2008; Tomamichel et al. 2012). Using asymptotic bounds
    for finite n_PE will systematically overestimate Q1^L and underestimate
    e1^U, yielding falsely optimistic ℓ_finite.

    Args:
        signal:     DetectionResult for signal pulses (intensity μ).
        decoy:      DetectionResult for weak-decoy pulses (intensity ν).
        vacuum:     DetectionResult for vacuum pulses (intensity 0).
        mu:         Signal intensity (photons/pulse).
        nu:         Weak-decoy intensity (photons/pulse); 0 < ν < μ.
        Y_0:        Noise yield per gate (observed estimate, dimensionless).
        n_PE:       Total number of parameter-estimation events (bits). Split
                    proportionally among intensities by detection rate.
        confidence: Statistical confidence level ∈ (0, 1). Default 0.99.
                    δ = 1 − confidence is the per-bound failure probability.

    Returns:
        DecoyBounds(Q1_lower, e1_upper, mode="finite").

    Note on Y_0:
        Y_0 is the dark-count + background noise yield, calibrated from
        dedicated measurements (e.g., blocking the signal beam, recording
        dark counts over many gates). It is NOT estimated from n_PE — its
        statistical uncertainty is governed by the calibration run, which
        typically uses orders of magnitude more events than a single pass.
        Applying a Hoeffding correction based on n_PE would produce a
        pathologically large slack (n_vac = n_PE × Y_0 ≈ 10⁻⁶ × n_PE ≈ 0),
        collapsing all finite bounds to zero. Y_0 therefore enters unchanged.

    Raises:
        ValueError: if μ ≤ ν, confidence out of range, or n_PE < 0.
    """
    if not (0.0 < nu < mu):
        raise ValueError(f"Require 0 < ν < μ; got ν={nu}, μ={mu}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    if n_PE < 0.0:
        raise ValueError(f"n_PE must be non-negative; got {n_PE}")

    delta = 1.0 - confidence

    # Q_mu = detections/pulses  → Hoeffding trials = PULSES
    # Q_nu = detections/pulses  → Hoeffding trials = PULSES
    # E_nu = errors/detections  → Hoeffding trials = DETECTIONS
    #
    # If pulse counts are provided, use them for Q bounds (correct per Ma et al. 2005).
    # For E_nu use detection counts if provided, else derive from gain × pulse count.
    # Fall back to detection-count split (conservative) when pulse counts are absent.
    if n_mu_pulses is not None and n_nu_pulses is not None:
        n_for_Q_mu = n_mu_pulses
        n_for_Q_nu = n_nu_pulses
        # E_nu is estimated from detections; use provided count or derive from pulses × gain
        n_for_E_nu = (n_nu_detections if n_nu_detections is not None
                      else n_nu_pulses * decoy.Q)
    else:
        # Conservative fallback: use detection counts split by gain ratio
        n_for_Q_mu, n_for_Q_nu = _count_split(n_PE, signal.Q, decoy.Q)
        n_for_E_nu = n_for_Q_nu

    slack_mu   = _hoeffding_slack(n_for_Q_mu, delta)
    slack_Q_nu = _hoeffding_slack(n_for_Q_nu, delta)
    slack_E_nu = _hoeffding_slack(n_for_E_nu, delta)

    # Adversarial rates for Q1 lower bound: push to shrink Q1^L
    Q_mu_adv = min(1.0, signal.Q + slack_mu)    # more signal detections → smaller bracket
    Q_nu_adv = max(0.0, decoy.Q  - slack_Q_nu)  # fewer decoy detections → smaller bracket

    Q1_L = _q1_lower(Q_mu_adv, Q_nu_adv, Y_0, mu, nu)

    # Adversarial rates for e1 upper bound: push to raise e1^U
    E_nu_adv = min(0.5, decoy.E + slack_E_nu)   # higher decoy QBER → larger e1

    e1_U = _e1_upper(E_nu_adv, Q_nu_adv, Y_0, Q1_L, mu, nu)

    return DecoyBounds(Q1_lower=Q1_L, e1_upper=e1_U, mode="finite")
