"""
physics/source.py — Quantum source model and QRNG constraint.

SourceConfig lives in core/types.py (shared across many modules). This file
re-exports it so callers can treat physics/source.py as the canonical import
for all source-related types, and adds QRNGModel.

Provided:
    SourceConfig   Re-exported from core.types (intensities, probabilities, clock rate).
    QRNGModel      QRNG hardware model: validates that the RNG can sustain f_clock.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass

# Re-export so callers can `from physics.source import SourceConfig, QRNGModel`
from core.types import SourceConfig

__all__ = ["SourceConfig", "QRNGModel"]


# ---------------------------------------------------------------------------
# QRNG model
# ---------------------------------------------------------------------------

@dataclass
class QRNGModel:
    """
    QRNG hardware constraint: validates that the random-number generator can
    sustain the requested pulse rate.

    Each pulse consumes bits_per_pulse random bits:
        • 1 bit  — basis choice (X or Z)
        • 1 bit  — intensity choice (signal / decoy / vacuum from P_mu, P_nu)
        • ~1 bit — additional randomness for preparation

    At f_clock = 100 MHz with 3 bits/pulse the QRNG must produce ≥ 300 Mbps.
    This is a hardware selection constraint set by Payload, not a free parameter.

    Attributes:
        rate:           QRNG output rate (bits/second).
        bits_per_pulse: Random bits consumed per emitted pulse (typically 3).
    """

    rate: float           # QRNG output rate (bits/s)
    bits_per_pulse: int   # bits consumed per pulse

    def __post_init__(self) -> None:
        if self.rate <= 0.0:
            raise ValueError(f"QRNG rate must be positive; got {self.rate}")
        if self.bits_per_pulse < 1:
            raise ValueError(f"bits_per_pulse must be ≥ 1; got {self.bits_per_pulse}")

    def max_clock_rate(self) -> float:
        """
        Maximum sustainable pulse rate (Hz) given the QRNG output rate.

            f_max = rate / bits_per_pulse

        Setting f_clock > f_max means the QRNG cannot supply enough entropy
        to randomise every pulse independently — a security violation.
        """
        return self.rate / self.bits_per_pulse

    def validate(self, f_clock: float) -> tuple[bool, str]:
        """
        Check whether f_clock is within the QRNG's sustainable rate.

        Args:
            f_clock: Requested pulse repetition rate (Hz).

        Returns:
            (True, "ok") if the QRNG can sustain f_clock.
            (False, message) with a human-readable explanation if not.
        """
        f_max = self.max_clock_rate()
        if f_clock <= f_max:
            return True, "ok"
        return (
            False,
            f"f_clock {f_clock / 1e6:.1f} MHz exceeds QRNG limit "
            f"{f_max / 1e6:.1f} MHz "
            f"(rate={self.rate / 1e6:.0f} Mbps, bits_per_pulse={self.bits_per_pulse}). "
            f"Reduce f_clock or increase QRNG rate.",
        )
