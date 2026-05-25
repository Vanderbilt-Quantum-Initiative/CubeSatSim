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
        tau_d:    Dead time (s). Enters the Rogers P_{0,0} correction to the
                  sifted bit rate. At satellite link losses (Q_mu ~ 1e-5),
                  the correction is negligible; the binding clock-rate
                  constraint is timing jitter. Typical Si-SPAD: 20–100 ns.
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

    def rogers_p00(self, p_total: float, f_clock: float) -> float:
        """
        Rogers (2007) steady-state probability that both detectors in a basis
        are simultaneously active — the fraction of detections that produce a
        valid, uncompromised sifted bit.

            k = tau_d * f_clock          (dead-time periods per clock cycle)
            P_{0,0} = [1 + 2k*(2p/(1-2p)) + (k²-k)*(2p/(1-2p))²]⁻¹

        where p = p_total is the per-detector sifted-bit probability per clock
        cycle (signal + dark, pre-divided by 8 for basis/state factors).

        At satellite link losses (p ~ 1e-6), P_{0,0} ≈ 1 regardless of k.
        Dead time has negligible effect on sifted bit rate in this regime;
        the binding clock-rate constraint is timing jitter, not dead time.

        Args:
            p_total: Per-detector sifted-bit probability per clock cycle.
            f_clock: Clock rate (Hz).

        Returns:
            P_{0,0} ∈ (0, 1].
        """
        k = self.tau_d * f_clock
        if k == 0.0 or p_total == 0.0:
            return 1.0
        two_p = 2.0 * p_total
        if two_p >= 1.0:
            return 0.0
        ratio = two_p / (1.0 - two_p)
        denom = 1.0 + 2.0 * k * ratio + (k * k - k) * ratio * ratio
        return 1.0 / denom

    def dead_time_regime(self, f_clock: float, Q_mu: float) -> dict:
        """
        Classify the dead-time operating regime for diagnostics.

        Computes rho_rx * tau_d — the fractional detector occupancy. Values
        above 0.01 indicate meaningful dead-time suppression; below 0.01 the
        effect is negligible (P_{0,0} ≈ 1).

        The binding clock-rate constraint at satellite losses is timing jitter:
            f_clock_max_jitter ≈ 1 / (3 * sigma_t)
        Dead time only becomes binding when link loss is very low (< ~20 dB).

        Args:
            f_clock: Clock rate (Hz).
            Q_mu:    Signal gain (detection probability per pulse).

        Returns:
            Dict with keys: k, rho_rx, occupancy, negligible (bool), warning (str|None).
        """
        k = self.tau_d * f_clock
        rho_rx = 4.0 * f_clock * Q_mu   # total detection rate across 4 detectors
        occupancy = rho_rx * self.tau_d
        negligible = occupancy < 0.01
        warning = None
        if not negligible:
            warning = (
                f"Dead-time occupancy {occupancy:.3f} > 0.01 — "
                f"Rogers P_{{0,0}} correction is non-negligible. "
                f"k={k:.2f}, rho_rx={rho_rx:.1f} cps."
            )
        return dict(k=k, rho_rx=rho_rx, occupancy=occupancy,
                    negligible=negligible, warning=warning)

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
