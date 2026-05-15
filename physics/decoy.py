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

from core.types import DetectionResult, DecoyBounds, EURDecoyBounds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kato_slack(n: float, delta: float, p_est: float) -> float:
    """
    Kato (2020) / empirical Bernstein concentration bound.

    Tighter than Hoeffding for rare events (small p_est).  For Bernoulli(p)
    with n i.i.d. trials and observed frequency p_est, the true probability
    lies within ±slack of p_est with probability ≥ 1−delta.

        slack = √(2 · p_est · (1−p_est) · ln(1/δ) / n) + ln(1/δ) / (3n)

    The variance-scaled √ term shrinks as p_est→0, unlike the Hoeffding bound
    whose √(ln(1/δ)/(2n)) is constant.  For decoy/vacuum yields (p~10⁻⁴),
    Kato is 20–50× tighter than Hoeffding.

    Reference: Kato (2020), arXiv:2002.04357;
               Maurer & Pontil (2009) — empirical Bernstein inequality.

    Args:
        n:      Number of Bernoulli trials (must match the event type: pulses
                for gain estimation, detections for QBER estimation).
        delta:  Per-bound failure probability (1 − confidence).
        p_est:  Observed sample frequency (point estimate of p).

    Returns:
        Slack (additive; non-negative).
    """
    if n <= 0.0:
        return 0.5
    p_est = max(0.0, min(1.0, p_est))
    log_inv_delta = math.log(1.0 / delta)
    variance_term = math.sqrt(2.0 * p_est * (1.0 - p_est) * log_inv_delta / n)
    third_moment  = log_inv_delta / (3.0 * n)
    return variance_term + third_moment


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


def eur_decoy_bounds(
    n_Z: dict[str, float],       # key-basis detections: {"signal", "decoy", "vacuum"}
    n_X: dict[str, float],       # test-basis detections per intensity
    m_X: dict[str, float],       # test-basis errors per intensity
    N_pulses: dict[str, float],  # total pulses per intensity (for correct trial counts)
    mu: float,
    nu: float,
    P_X: float,                  # key-basis preparation probability
    Y_0: float,                  # calibrated vacuum yield (dark count probability per gate)
    epsilon_s: float = 1e-11,    # per-bound statistical failure probability
    concentration: str = "kato", # "kato" (recommended) or "hoeffding"
) -> EURDecoyBounds:
    """
    EUR-compatible decoy-state bounds from per-basis, per-intensity observations.

    Uses test-basis (minority, 1-P_X) observations for parameter estimation via
    Ma et al. (2005) decoy analysis with Kato concentration inequalities, then
    maps the resulting per-photon-number yields to key-basis counts to produce
    the EUR formula inputs: s_Z0^L, s_Z1^L, φ_Z^U.

    Per-basis split of detections:
        key basis: fraction P_X²   of matched pulses per intensity
        test basis: fraction (1-P_X)² of matched pulses per intensity

    The key-basis data is fully allocated to key generation (no PE split needed
    — parameter estimation is done entirely from the test-basis minority data).

    Statistical correction selection (Section 3.3 of FixedFiniteKeyRate.md):
        gain slack  → uses PULSE counts  (Bernoulli on pulses)
        QBER slack  → uses DETECTION counts (Bernoulli on detections)

    References:
        Lim, Curty, Walenta, Xu & Zbinden (2014), PRA 89, 022307
        Wiesemann et al. (2026), Quantum 10, 2037
        Kato (2020), arXiv:2002.04357

    Args:
        n_Z:         Key-basis detection counts per intensity.
        n_X:         Test-basis detection counts per intensity.
        m_X:         Test-basis error counts per intensity.
        N_pulses:    Total pulse counts per intensity across the full pass.
        mu:          Signal intensity (photons/pulse).
        nu:          Weak-decoy intensity (photons/pulse); 0 < ν < μ.
        P_X:         Key-basis preparation probability (high P_X → efficient BB84).
        Y_0:         Calibrated noise yield per gate (dark counts / f_clock).
        epsilon_s:   Per-bound statistical failure probability.
        concentration: Concentration inequality to use ("kato" or "hoeffding").

    Returns:
        EURDecoyBounds with s_Z0^L, s_Z1^L, φ_Z^U and diagnostic fields.
    """
    if not (0.0 < nu < mu):
        raise ValueError(f"Require 0 < ν < μ; got ν={nu}, μ={mu}")
    if not (0.0 < P_X < 1.0):
        raise ValueError(f"P_X must be in (0, 1); got {P_X}")

    P_test = 1.0 - P_X

    N_sig = N_pulses.get("signal", 0.0)
    N_dec = N_pulses.get("decoy", 0.0)

    # Test-basis effective "matched" pulse counts (both parties chose test basis)
    N_sig_test = N_sig * P_test ** 2
    N_dec_test = N_dec * P_test ** 2

    n_X_sig = n_X.get("signal", 0.0)
    n_X_dec = n_X.get("decoy", 0.0)

    # Observed test-basis gains (per matched test pulse)
    Q_mu_test = n_X_sig / N_sig_test if N_sig_test > 0 else 0.0
    Q_nu_test = n_X_dec / N_dec_test if N_dec_test > 0 else 0.0
    E_nu_test = m_X.get("decoy", 0.0) / n_X_dec if n_X_dec > 0 else 0.5

    # Statistical slack function (correct trial counts per §3.4)
    def slack(n_trials: float, p_est: float) -> float:
        if concentration == "kato":
            return _kato_slack(n_trials, epsilon_s, p_est)
        return _hoeffding_slack(n_trials, epsilon_s)

    # Gain slacks use PULSE counts (trials = pulses sent in test basis)
    # QBER slack uses DETECTION counts (trials = detections)
    slack_Q_mu = slack(N_sig_test, Q_mu_test)
    slack_Q_nu = slack(N_dec_test, Q_nu_test)
    slack_E_nu = slack(n_X_dec, E_nu_test)

    # Adversarial rates: push each bound in the direction that shrinks the key
    Q_mu_adv = min(1.0, Q_mu_test + slack_Q_mu)   # ↑ Q_mu → smaller Q1^L bracket
    Q_nu_adv = max(0.0, Q_nu_test - slack_Q_nu)   # ↓ Q_nu → smaller Q1^L bracket
    E_nu_adv = min(0.5, E_nu_test + slack_E_nu)   # ↑ E_nu → larger e1^U

    # Ma et al. (2005) single-photon gain lower bound from test-basis data
    Q1_L = _q1_lower(Q_mu_adv, Q_nu_adv, Y_0, mu, nu)

    # Single-photon yield lower bound: Y1^L = Q1^L / P(1|μ) = Q1^L / (μ e^{−μ})
    P1_mu = mu * math.exp(-mu)
    Y1_lower = Q1_L / P1_mu if P1_mu > 0 else 0.0

    # Phase error rate upper bound — EUR connection:
    #   φ_Z^U (phase error in key basis) = e1^U (bit error of single-photons in test basis)
    e1_U = _e1_upper(E_nu_adv, Q_nu_adv, Y_0, Q1_L, mu, nu)
    phi_Z_upper = e1_U

    # ── Key-basis single-photon count lower bound ─────────────────────────────
    n_Z_sig = n_Z.get("signal", 0.0)
    N_sig_key = N_sig * P_X ** 2   # key-basis matched signal pulses

    # Q_mu in key basis = n_Z_sig / N_sig_key (should equal Q_mu_test; use adversarial)
    Q_mu_key_adv = n_Z_sig / N_sig_key if N_sig_key > 0 else 0.0
    # Use max of adversarial Q_mu_adv and key-basis estimate for the denominator
    Q_mu_denom = max(Q_mu_key_adv, Q_mu_adv)

    if Q_mu_denom > 0 and n_Z_sig > 0:
        # s_Z1^L = n_Z_sig × Q1^L / Q_mu  — fraction of key detections from single photons
        s_Z1_lower = max(0.0, n_Z_sig * Q1_L / Q_mu_denom)
    else:
        s_Z1_lower = 0.0

    # ── Key-basis vacuum count lower bound ────────────────────────────────────
    # Vacuum-source detections in key basis are directly accumulated.
    s_Z0_lower = n_Z.get("vacuum", 0.0)

    return EURDecoyBounds(
        s_Z0_lower=s_Z0_lower,
        s_Z1_lower=s_Z1_lower,
        phi_Z_upper=phi_Z_upper,
        Y0_bound=Y_0,
        Y1_lower=Y1_lower,
        e1_upper=e1_U,
    )
