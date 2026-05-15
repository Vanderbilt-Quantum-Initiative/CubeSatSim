"""
physics/post_processing.py — Classical post-processing chain (Stage 6).

Owns everything after quantum measurement: sifting, parameter estimation
split, error correction modelling, and classical bandwidth feasibility check.

Chain:
    sifting_yield(N_total, source, Q_mu)          — n_sifted
    pe_split(n_sifted, r_PE)                       — (n_PE, n_key)
    ec_efficiency(n_key, E_mu, algorithm)           — f_EC  (not a constant)
    classical_bandwidth_check(...)                  — (feasible, volume, rounds)
    compute_post_processing(...)                    → PostProcessingResult

The EC efficiency f_EC depends on block size n_key and error rate E_mu.
It is NOT a free parameter — using a constant f_EC overestimates the key
yield at short block lengths typical of satellite passes.

Cascade (interactive):
    f_EC ≈ 1.16 at n=10⁵, E=5%    (multiple-round interactive reconciliation)
    Cascade requires >1 classical round trip, which strains low-bandwidth links.

LDPC (one-way):
    f_EC ≈ 1.10 at n=10⁵, E=5%    (designed code; one-way transmission)
    f_EC → 1.05 at n≫10⁶           (approaches capacity only at large block size)
    LDPC requires exactly 1 round trip regardless of pass geometry.

Classical data volume:
    Cascade: ceil(log₂(1/E_μ)) round trips × n × H₂(E_μ) bits/round
    LDPC:    1 round trip  × n × f_EC × H₂(E_μ) bits
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import SourceConfig, PostProcessingConfig, PostProcessingResult
from physics.keyrate import binary_entropy


# ---------------------------------------------------------------------------
# Sifting
# ---------------------------------------------------------------------------

def sifting_yield(N_total: float, source: SourceConfig, Q_mu: float) -> float:
    """
    Expected sifted bit count from signal pulses surviving basis reconciliation.

        n_sifted = N_total × P_μ × Q_μ × q

    Only signal-intensity pulses in matching bases contribute to the sifted key.
    Decoy and vacuum pulses are consumed by parameter estimation, not key bits.

    Args:
        N_total:  Total pulses sent during the pass (f_clock × T_pass).
        source:   SourceConfig providing P_mu and sifting_factor().
        Q_mu:     Fading-averaged signal gain (from detection.py).

    Returns:
        Expected number of sifted bits (float; ceil before use as integer).
    """
    return N_total * source.P_mu * Q_mu * source.sifting_factor()


# ---------------------------------------------------------------------------
# Parameter estimation split
# ---------------------------------------------------------------------------

def pe_split(n_sifted: float, r_PE: float) -> tuple[float, float]:
    """
    Split sifted bits into parameter estimation and key generation.

        n_PE  = n_sifted × r_PE
        n_key = n_sifted × (1 − r_PE)

    The PE fraction r_PE trades decoy bound tightness against key bits.
    Tighter bounds (larger r_PE) improve finite-key yield only when the
    resulting n_PE gives enough statistical power to close the bounds —
    otherwise the extra PE bits are wasted.

    Args:
        n_sifted:  Total sifted bits.
        r_PE:      Fraction reserved for parameter estimation ∈ (0, 1).

    Returns:
        (n_PE, n_key).
    """
    n_PE  = n_sifted * r_PE
    n_key = n_sifted * (1.0 - r_PE)
    return n_PE, n_key


# ---------------------------------------------------------------------------
# Error correction efficiency
# ---------------------------------------------------------------------------

def ec_efficiency(n_key: float, E_mu: float, algorithm: str) -> float:
    """
    Error-correction efficiency f_EC as a function of block size and QBER.

    f_EC ≥ 1.0; f_EC = 1.0 is the Shannon limit (theoretical minimum leakage).
    Real codes operate above 1.0; the gap closes as n_key grows.

    Model (empirically calibrated against published LDPC/Cascade results):

        Cascade (interactive, base at short blocks):
            f_EC = 1.16 + 0.04 × exp(−n_key / 2×10⁵)
            Weakly dependent on E_mu at typical error rates (3–10%).

        LDPC (one-way):
            f_EC = 1.05 + 0.10 × exp(−n_key / 5×10⁵) + 0.04 × max(0, E_mu − 0.03)
            Stronger E_mu dependence: LDPC codes degrade faster at high error rates.

    The exponential terms model the approach to the asymptotic efficiency as
    block length increases. At n_key → ∞ both converge toward their floor.

    Args:
        n_key:     Number of bits in the key-generation block.
        E_mu:      Fading-averaged signal QBER.
        algorithm: "cascade" or "ldpc".

    Returns:
        f_EC ≥ 1.0.

    Raises:
        ValueError: if algorithm is not "cascade" or "ldpc".
    """
    if algorithm == "cascade":
        return 1.16 + 0.04 * math.exp(-n_key / 2e5)
    elif algorithm == "ldpc":
        e_penalty = 0.04 * max(0.0, E_mu - 0.03)
        return 1.05 + 0.10 * math.exp(-n_key / 5e5) + e_penalty
    else:
        raise ValueError(f"ec_algorithm must be 'cascade' or 'ldpc'; got {algorithm!r}")


# ---------------------------------------------------------------------------
# Classical bandwidth check
# ---------------------------------------------------------------------------

def classical_bandwidth_check(
    n_sifted: float,
    E_mu: float,
    f_EC: float,
    algorithm: str,
    T_pass: float,
    rf_bandwidth: float,
) -> tuple[bool, float, int]:
    """
    Check whether the classical channel can support post-processing within T_pass.

    Data volume model:
        Cascade: ceil(log₂(1/E_μ)) round trips.
                 Each trip exchanges ~n_sifted × H₂(E_μ) syndrome bits.
                 Total = rounds × n_sifted × H₂(E_μ) bits.

        LDPC:    1 round trip.
                 Total = n_sifted × f_EC × H₂(E_μ) bits (one-way syndrome).

    Feasibility: total_data / T_pass ≤ rf_bandwidth.

    Args:
        n_sifted:     Total sifted bits (before PE split).
        E_mu:         Fading-averaged signal QBER.
        f_EC:         Error-correction efficiency.
        algorithm:    "cascade" or "ldpc".
        T_pass:       Usable pass duration (s).
        rf_bandwidth: Classical channel bandwidth (bits/s).

    Returns:
        (feasible, total_data_volume_bits, classical_rounds).

    Raises:
        ValueError: if algorithm is not "cascade" or "ldpc".
    """
    H2_E = binary_entropy(E_mu)

    if algorithm == "cascade":
        if E_mu <= 0.0:
            rounds = 1
        else:
            rounds = max(1, math.ceil(math.log2(1.0 / max(E_mu, 1e-9))))
        total_bits = rounds * n_sifted * H2_E
    elif algorithm == "ldpc":
        rounds = 1
        total_bits = n_sifted * f_EC * H2_E
    else:
        raise ValueError(f"ec_algorithm must be 'cascade' or 'ldpc'; got {algorithm!r}")

    required_rate = total_bits / T_pass if T_pass > 0.0 else float("inf")
    feasible = required_rate <= rf_bandwidth

    return feasible, total_bits, rounds


# ---------------------------------------------------------------------------
# Top-level computation
# ---------------------------------------------------------------------------

def compute_post_processing(
    n_sifted: float,
    E_mu: float,
    source: SourceConfig,
    config: PostProcessingConfig,
    T_pass: float,
) -> PostProcessingResult:
    """
    Full post-processing chain: PE split → EC model → bandwidth check.

    Args:
        n_sifted:   Total sifted bits from the pass (signal pulses only).
        E_mu:       Fading-averaged signal QBER for the pass.
        source:     SourceConfig (not consumed here; available for callers
                    that need sifting_factor() alongside the result).
        config:     PostProcessingConfig (r_PE, ec_algorithm, epsilon_PA,
                    rf_bandwidth).
        T_pass:     Usable pass duration (s).

    Returns:
        PostProcessingResult with all classical chain outputs.
    """
    n_PE, n_key = pe_split(n_sifted, config.r_PE)
    f_EC = ec_efficiency(n_key, E_mu, config.ec_algorithm)
    leak_EC = n_key * f_EC * binary_entropy(E_mu)

    feasible, data_vol, rounds = classical_bandwidth_check(
        n_sifted, E_mu, f_EC, config.ec_algorithm, T_pass, config.rf_bandwidth
    )

    return PostProcessingResult(
        n_sifted=n_sifted,
        n_PE=n_PE,
        n_key=n_key,
        f_EC=f_EC,
        leak_EC=leak_EC,
        classical_data_volume=data_vol,
        classical_rounds=rounds,
        ec_feasible=feasible,
    )
