"""
physics/keyrate.py — Secret key rate computation (Stage 5).

Implements the GLLP-decoy key fraction and the Tomamichel et al. (2012)
finite-key correction.

Chain:
    binary_entropy(x)                             — H₂(x)
    gllp_asymptotic(bounds, signal, q, f_EC)      — R per pulse (may be negative)
    finite_key_length(n_key, R, E_mu, f_EC, ε_PA) — ℓ_finite (clamped ≥ 0)
    compute_key_rate(...)                          — KeyRateResult

Key references:
    Gottesman, Lo, Lütkenhaus & Preskill (2004), QIC 4, 325   — GLLP security proof
    Lo, Ma & Chen (2005), PRL 94, 230504                       — Decoy-state framework
    Tomamichel, Lim, Gisin & Renner (2012), Nat. Commun. 3    — Finite-key formula
    Scarani & Renner (2008), PRL 100, 200501                   — Composable security

GLLP formula (per sifted signal photon):

    R ≥ q · [Q₁(1 − H₂(e₁)) − Q_μ · f_EC · H₂(E_μ)]

where:
    Q₁    lower-bounded single-photon gain (from decoy bounds)
    e₁    upper-bounded single-photon phase error rate (from decoy bounds)
    Q_μ   fading-averaged signal gain (from detection.py)
    E_μ   fading-averaged signal QBER (from detection.py)
    q     sifting factor (0.5 for standard BB84; → 1 for efficient BB84)
    f_EC  error-correction efficiency (≥ 1; 1.0 = Shannon limit)

Physical interpretation:
    Q₁(1 − H₂(e₁)) — privacy-amplified key bits from single-photon events
    Q_μ·f_EC·H₂(E_μ) — bits consumed by error correction across all events
    Positive R requires the former to exceed the latter.

Finite-key correction (Tomamichel et al. 2012):

    ℓ_finite = max(0, n_key · R − Δ_AEP − log₂(2/ε_PA))

    Δ_AEP = 4 · √(n_key · log₂(6/ε_PA))

    The O(√n) AEP term dominates at the short block lengths produced by
    satellite passes (~10⁵–10⁶ sifted bits). Even with R > 0, Δ_AEP can
    consume the entire key yield if n_key is too small.
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import DetectionResult, DecoyBounds, EURDecoyBounds, KeyRateResult, SecurityBudget


# ---------------------------------------------------------------------------
# Binary entropy
# ---------------------------------------------------------------------------

def binary_entropy(x: float) -> float:
    """
    Binary Shannon entropy H₂(x) = −x log₂(x) − (1−x) log₂(1−x).

    Edge cases:
        x = 0.0 → 0.0  (no error → no uncertainty)
        x = 1.0 → 0.0  (certain error → no uncertainty)
        x < 0 or x > 1 → raises ValueError

    Args:
        x: Probability ∈ [0, 1].

    Returns:
        H₂(x) ∈ [0, 1] (bits).
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"binary_entropy: x must be in [0, 1]; got {x}")
    if x == 0.0 or x == 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


# ---------------------------------------------------------------------------
# GLLP asymptotic key fraction
# ---------------------------------------------------------------------------

def gllp_asymptotic(
    bounds: DecoyBounds,
    signal: DetectionResult,
    q: float,
    f_EC: float,
) -> float:
    """
    GLLP-decoy asymptotic key fraction R per sifted signal photon.

    R = q · [Q₁(1 − H₂(e₁)) − Q_μ · f_EC · H₂(E_μ)]

    R may be negative. A negative R means the link cannot generate a secret
    key even asymptotically — no amount of data will help. The caller should
    check R before proceeding to finite_key_length.

    Args:
        bounds:  DecoyBounds from decoy.py (Q1_lower, e1_upper).
        signal:  DetectionResult for signal intensity μ (Q_μ, E_μ).
        q:       Sifting factor — SourceConfig.sifting_factor().
                 0.5 for standard BB84; approaches 1 for efficient BB84.
        f_EC:    Error-correction efficiency (≥ 1.0). 1.0 = Shannon limit;
                 typical values: 1.10–1.16 depending on algorithm and block size.

    Returns:
        R (bits per sifted signal photon). May be negative.
    """
    pa_term = bounds.Q1_lower * (1.0 - binary_entropy(bounds.e1_upper))
    ec_term = signal.Q * f_EC * binary_entropy(signal.E)
    return q * (pa_term - ec_term)


# ---------------------------------------------------------------------------
# Tomamichel finite-key length
# ---------------------------------------------------------------------------

def finite_key_length(
    n_key: float,
    R: float,
    E_mu: float,
    f_EC: float,
    epsilon_PA: float,
) -> float:
    """
    Composably secure finite-key length from Tomamichel et al. (2012).

    ℓ_finite = max(0, n_key · R − Δ_AEP − log₂(2/ε_PA))

    where the smooth min-entropy AEP correction is:

        Δ_AEP = 4 · √(n_key · log₂(6/ε_PA))

    Physical interpretation:
        n_key · R       — asymptotic key bits (from GLLP)
        Δ_AEP           — O(√n) penalty from smooth min-entropy fluctuations
        log₂(2/ε_PA)    — privacy amplification hash length

    The Δ_AEP term is the binding constraint at short block lengths. For a
    typical satellite pass with n_key = 10⁵ and ε_PA = 10⁻¹⁰:
        Δ_AEP ≈ 4 · √(10⁵ · 33) ≈ 7, 300 bits
    This means the link needs R ≥ 0.073 just to break even — a tight
    constraint that can eliminate all key even when R > 0.

    Args:
        n_key:      Sifted bits available for key generation (after PE split).
        R:          GLLP asymptotic key fraction (from gllp_asymptotic).
                    If R ≤ 0 the result is immediately 0 (no key possible).
        E_mu:       Fading-averaged signal QBER (used for EC leak bookkeeping
                    in the caller; not consumed directly by this formula since
                    EC cost is already reflected in R via gllp_asymptotic).
        f_EC:       EC efficiency (same note as E_mu — passed for logging).
        epsilon_PA: Composable security parameter ε_PA ∈ (0, 1).
                    Typical: 1e-10 for mission-grade security.

    Returns:
        ℓ_finite ≥ 0 (clamped; negative values indicate an infeasible link).
    """
    if R <= 0.0 or n_key <= 0.0:
        return 0.0

    gross = n_key * R
    delta_aep = 4.0 * math.sqrt(n_key * math.log2(6.0 / epsilon_PA))
    hash_cost = math.log2(2.0 / epsilon_PA)

    return max(0.0, gross - delta_aep - hash_cost)


# ---------------------------------------------------------------------------
# EUR finite-key length (Lim et al. 2014 / Wiesemann et al. 2026)
# ---------------------------------------------------------------------------

def eur_key_length(
    s_Z0_lower: float,
    s_Z1_lower: float,
    phi_Z_upper: float,
    leak_EC: float,
    epsilon: float,
    epsilon_PA: float,
) -> float:
    """
    EUR-based composable finite-key length (Lim et al. 2014, Wiesemann et al. 2026).

    ℓ = s_Z0^L + s_Z1^L · [1 − H₂(φ_Z^U)] − leak_EC
          − 6 · log₂(2/ε̃) − log₂(2/ε_PA)

    The finite-size corrections are O(log(1/ε)), not O(√n) as in the AEP approach.
    At ε = 10⁻¹¹, 6·log₂(2/ε) ≈ 220 bits — versus thousands of bits from AEP
    at the block sizes produced by a CubeSat pass (n ~ 10⁵–10⁸).

    Physical interpretation:
        s_Z0^L                       — key bits certified from vacuum events
                                       (random, contribute 1 bit each)
        s_Z1^L · (1 − H₂(φ_Z^U))   — privacy-amplified key bits from single-photon events
        leak_EC                      — bits sacrificed to error correction
        6·log₂(2/ε̃)                  — O(log) correction for statistical fluctuations
                                       in parameter estimation and smooth min-entropy
        log₂(2/ε_PA)                 — privacy amplification hash cost

    Difference from AEP: The Δ_AEP = 4·√(n·log₂(6/ε)) term (O(√n)) is replaced by
    the O(log 1/ε) correction. The √n penalty has moved into the parameter estimation
    step (bounded via Kato/Hoeffding in decoy.eur_decoy_bounds).

    Args:
        s_Z0_lower:  Lower bound on vacuum-source detections in key basis.
        s_Z1_lower:  Lower bound on single-photon detections in key basis.
        phi_Z_upper: Upper bound on single-photon phase error rate in key basis.
        leak_EC:     Bits leaked during error correction (n_Z · f_EC · H₂(E_Z)).
        epsilon:     Smoothing / parameter-estimation failure probability (ε̃).
                     Typically epsilon_total / n_terms from SecurityBudget.
        epsilon_PA:  Privacy amplification security parameter.
                     Typically epsilon_total / n_terms from SecurityBudget.

    Returns:
        ℓ ≥ 0 (clamped; negative values indicate an infeasible link).
    """
    if phi_Z_upper >= 0.5:
        return 0.0
    if s_Z1_lower <= 0.0 and s_Z0_lower <= 0.0:
        return 0.0

    key_bits = (
        s_Z0_lower
        + s_Z1_lower * (1.0 - binary_entropy(phi_Z_upper))
        - leak_EC
        - 6.0 * math.log2(2.0 / epsilon)
        - math.log2(2.0 / epsilon_PA)
    )
    return max(0.0, key_bits)


# ---------------------------------------------------------------------------
# Top-level computation
# ---------------------------------------------------------------------------

def compute_key_rate(
    bounds: "DecoyBounds | EURDecoyBounds",
    signal: DetectionResult,
    n_key: float,
    q: float,
    f_EC: float,
    f_clock: float,
    epsilon_PA: float,
    proof_method: str = "eur",
    leak_EC: float | None = None,
    budget: "SecurityBudget | None" = None,
) -> KeyRateResult:
    """
    Full key rate computation: GLLP + finite-key correction.

    Two proof methods:
        "eur" (default) — EUR-based formula (Lim et al. 2014 / Wiesemann et al. 2026).
                          Requires bounds to be EURDecoyBounds.  Finite-size
                          corrections are O(log 1/ε) — much tighter at CubeSat
                          block sizes.  Use leak_EC and budget for full control.
        "aep"           — Tomamichel et al. (2012) AEP formula (legacy).
                          Corrections are O(√n) — pessimistic at n ~ 10⁵.

    Args:
        bounds:       EURDecoyBounds (for "eur") or DecoyBounds (for "aep").
        signal:       DetectionResult for signal intensity μ.
        n_key:        Sifted bits available for key generation.
        q:            Sifting factor (SourceConfig.sifting_factor()).
        f_EC:         Error-correction efficiency (≥ 1.0).
        f_clock:      Pulse repetition rate (Hz). Used to convert R to SKBR.
        epsilon_PA:   Composable security parameter (total or per-sub-event).
        proof_method: "eur" or "aep".
        leak_EC:      Pre-computed EC leakage (bits). Required for "eur".
                      If None under "eur", derived from n_key * f_EC * H₂(E_mu).
        budget:       SecurityBudget for sub-event epsilon allocation.
                      If None, uses SecurityBudget(epsilon_total=epsilon_PA, n_terms=6).

    Returns:
        KeyRateResult(R, skbr, ell_finite).
    """
    R = gllp_asymptotic(
        bounds if isinstance(bounds, DecoyBounds) else
        DecoyBounds(Q1_lower=getattr(bounds, "Y1_lower", 0.0) * signal.intensity * math.exp(-signal.intensity),
                    e1_upper=bounds.e1_upper, mode="eur"),
        signal, q, f_EC
    )
    skbr = f_clock * R

    if proof_method == "aep":
        ell = finite_key_length(n_key, R, signal.E, f_EC, epsilon_PA)
    else:
        # EUR path
        if not isinstance(bounds, EURDecoyBounds):
            raise TypeError("proof_method='eur' requires EURDecoyBounds, not DecoyBounds.")
        if budget is None:
            budget = SecurityBudget(epsilon_total=epsilon_PA, n_terms=6)
        eps_sub = budget.epsilon_sub
        _leak = leak_EC if leak_EC is not None else n_key * f_EC * binary_entropy(signal.E)
        ell = eur_key_length(
            s_Z0_lower=bounds.s_Z0_lower,
            s_Z1_lower=bounds.s_Z1_lower,
            phi_Z_upper=bounds.phi_Z_upper,
            leak_EC=_leak,
            epsilon=eps_sub,
            epsilon_PA=eps_sub,
        )

    return KeyRateResult(R=R, skbr=skbr, ell_finite=ell)
