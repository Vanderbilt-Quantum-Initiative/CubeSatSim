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

from core.types import DetectionResult, DecoyBounds, KeyRateResult


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
# Top-level computation
# ---------------------------------------------------------------------------

def compute_key_rate(
    bounds: DecoyBounds,
    signal: DetectionResult,
    n_key: float,
    q: float,
    f_EC: float,
    f_clock: float,
    epsilon_PA: float,
) -> KeyRateResult:
    """
    Full key rate computation: GLLP + Tomamichel finite-key correction.

    Computes:
        R       — asymptotic key fraction per sifted signal photon (may be < 0)
        skbr    — secret key bit rate = f_clock · R (may be < 0)
        ell_finite — composably secure key bits; clamped ≥ 0

    Args:
        bounds:     DecoyBounds from decoy.py (Q1_lower, e1_upper).
        signal:     DetectionResult for signal intensity μ.
        n_key:      Sifted bits available for key generation.
        q:          Sifting factor (SourceConfig.sifting_factor()).
        f_EC:       Error-correction efficiency (≥ 1.0).
        f_clock:    Pulse repetition rate (Hz). Used to convert R to SKBR.
        epsilon_PA: Composable security parameter ε_PA.

    Returns:
        KeyRateResult(R, skbr, ell_finite).
    """
    R = gllp_asymptotic(bounds, signal, q, f_EC)
    skbr = f_clock * R
    ell = finite_key_length(n_key, R, signal.E, f_EC, epsilon_PA)
    return KeyRateResult(R=R, skbr=skbr, ell_finite=ell)
