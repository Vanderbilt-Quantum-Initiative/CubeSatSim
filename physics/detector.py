"""
physics/detector.py — Single-photon detector model.

Provided:
    DetectorModel   Encapsulates detector hardware parameters and derived quantities.

The detector is always ground-based in a downlink configuration. Dark count rate
p_d is governed by cooling temperature, not radiation dose (radiation damage
applies only to uplink detectors — see Full Budget v1.3 §Architecture note).

All time quantities use SI units (seconds). The budget document sometimes
expresses tau_d in nanoseconds; convert before constructing DetectorModel.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DetectorModel:
    """
    Single-photon detector hardware model.

    Encodes the four quantities that directly enter the link budget:
        eta_det  — quantum efficiency (detection probability per arriving photon)
        p_d      — dark count probability per gate (dimensionless)
        tau_d    — dead time (s); sets the maximum sustainable clock rate
        delta_t  — gate width (s); sets the background photon collection window

    The optional sigma_t (timing jitter, s) is stored for future use in
    timing-resolution analysis and Doppler gate widening but is not consumed
    by any current physics module.

    Second-order effects deferred to Plan §10:
        • Pile-up / dead-time correction to effective_efficiency (§10.5)
        • After-pulsing (§10.4)

    Attributes:
        eta_det:  Detector quantum efficiency ∈ (0, 1].
                  Si-SPAD: 0.20–0.65; SNSPD: 0.80–0.95.
        p_d:      Dark count probability per gate (dimensionless).
                  Typical Si-SPAD at −20 °C: ~1e-6 per gate.
        tau_d:    Dead time (s). Sets pile-up threshold: f_clock must stay
                  below 1/tau_d to avoid count-rate saturation.
                  Typical Si-SPAD: 20–100 ns = 20e-9 to 100e-9 s.
        delta_t:  Gate width (s). Set by timing synchronisation precision
                  sigma_sync; not a free parameter. Enters the background
                  photon rate: p_bg ∝ H_bg * Omega_FOV * A_rx * delta_lambda
                  * delta_t * eta_rx.
        sigma_t:  Timing jitter 1-sigma (s), optional. IRF half-width.
    """

    def __init__(
        self,
        eta_det: float,
        p_d: float,
        tau_d: float,
        delta_t: float,
        sigma_t: float | None = None,
    ) -> None:
        """
        Args:
            eta_det:  Quantum efficiency ∈ (0, 1].
            p_d:      Dark count probability per gate ∈ [0, 1).
            tau_d:    Dead time (s). Must be positive.
            delta_t:  Gate width (s). Must be positive and ≤ tau_d.
            sigma_t:  Timing jitter 1-sigma (s), optional.

        Raises:
            ValueError: on out-of-range parameters.
        """
        if not (0.0 < eta_det <= 1.0):
            raise ValueError(f"eta_det must be in (0, 1]; got {eta_det}")
        if not (0.0 <= p_d < 1.0):
            raise ValueError(f"p_d must be in [0, 1); got {p_d}")
        if tau_d <= 0.0:
            raise ValueError(f"tau_d must be positive (s); got {tau_d}")
        if delta_t <= 0.0:
            raise ValueError(f"delta_t must be positive (s); got {delta_t}")
        if delta_t > tau_d:
            raise ValueError(
                f"delta_t ({delta_t * 1e9:.1f} ns) exceeds tau_d "
                f"({tau_d * 1e9:.1f} ns); gate must fit within dead time."
            )
        if sigma_t is not None and sigma_t < 0.0:
            raise ValueError(f"sigma_t must be non-negative (s); got {sigma_t}")

        self.eta_det = eta_det
        self.p_d = p_d
        self.tau_d = tau_d
        self.delta_t = delta_t
        self.sigma_t = sigma_t

    # ------------------------------------------------------------------
    # Primary interface used by detection.py and pass_sim.py
    # ------------------------------------------------------------------

    def dark_count_rate(self) -> float:
        """
        Dark count probability per gate (dimensionless).

        Named "rate" for API symmetry with background photon rate; both are
        probabilities per gate window and are added directly in noise_yield().

        Returns p_d as stored — temperature-dependent, not radiation-driven.
        """
        return self.p_d

    def effective_efficiency(self, count_rate: float = 0.0) -> float:
        """
        Effective detection efficiency, optionally corrected for pile-up.

        At count rates well below 1/tau_d the correction is negligible and
        eta_det is returned unchanged.  Pile-up correction (dead-time modified
        Poisson) is deferred to Plan §10.5; count_rate is accepted now so the
        signature is stable when the correction is added.

        Args:
            count_rate: Expected detection rate (counts/s). Unused currently.

        Returns:
            eta_det (dimensionless).
        """
        return self.eta_det

    def max_clock_rate(self) -> float:
        """
        Maximum sustainable pulse rate before dead-time saturation (Hz).

            f_max = 1 / tau_d

        Driving f_clock above this limit causes pile-up: detections from one
        pulse prevent detection of the next, distorting Q_mu and E_mu. This
        bound must be checked against SourceConfig.f_clock before each pass.

        Returns:
            Maximum clock rate in Hz.
        """
        return 1.0 / self.tau_d

    def validate_clock_rate(self, f_clock: float) -> tuple[bool, str]:
        """
        Check whether f_clock is within the detector's dead-time limit.

        Args:
            f_clock: Requested pulse repetition rate (Hz).

        Returns:
            (True, "ok") if f_clock ≤ 1/tau_d.
            (False, message) describing the violation and headroom if not.
        """
        f_max = self.max_clock_rate()
        if f_clock <= f_max:
            return True, "ok"
        ratio = f_clock / f_max
        return (
            False,
            f"f_clock {f_clock / 1e6:.1f} MHz exceeds detector limit "
            f"{f_max / 1e6:.1f} MHz (tau_d={self.tau_d * 1e9:.1f} ns). "
            f"Clock is {ratio:.2f}× the dead-time bound; pile-up will "
            f"distort Q_mu and E_mu. Reduce f_clock or use a faster detector.",
        )

    def validate(self) -> tuple[bool, str]:
        """
        Full parameter consistency check.

        Checks that delta_t is consistent with the timing jitter if provided:
        a gate narrower than ~2*sigma_t will clip a significant fraction of
        genuine detections, reducing effective eta_det.

        Returns:
            (True, "ok") or (False, explanation).
        """
        if self.sigma_t is not None and self.delta_t < 2.0 * self.sigma_t:
            return (
                False,
                f"delta_t ({self.delta_t * 1e12:.0f} ps) < 2 * sigma_t "
                f"({2 * self.sigma_t * 1e12:.0f} ps): gate clips timing "
                f"distribution, reducing effective eta_det.",
            )
        return True, "ok"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        jitter = f", sigma_t={self.sigma_t * 1e12:.0f}ps" if self.sigma_t else ""
        return (
            f"DetectorModel("
            f"eta_det={self.eta_det:.3f}, "
            f"p_d={self.p_d:.2e}, "
            f"tau_d={self.tau_d * 1e9:.1f}ns, "
            f"delta_t={self.delta_t * 1e9:.1f}ns"
            f"{jitter})"
        )
