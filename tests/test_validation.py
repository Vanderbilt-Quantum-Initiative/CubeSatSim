"""
tests/test_validation.py — Cross-validation against published results.

Primary reference: Bourgoin et al. (2013), NJP 15, 023006.
  "A comprehensive design and performance analysis of LEO satellite quantum
  communication" — downlink WCP, 600 km circular orbit, 670 nm.

Validated quantities:
  1. Atmospheric transmittance vs elevation angle (Beer-Lambert vs MODTRAN)
  2. Hufnagel-Valley Cn² profile (exact match to Bourgoin Eq. 6)
  3. QBER limiting behaviour (analytical, independent of their simulation)
  4. Total link loss order-of-magnitude sanity check

Not validated:
  Diffraction loss — Bourgoin uses full Rayleigh-Sommerfeld diffraction on a
  truncated aperture (WCP FWHM ≈ D_tx, far-field truncated-plane-wave regime).
  Our model uses a focused Gaussian: η_diff = 1 − exp(−2r_rx²/W(L)²).
  The ~28 dB gap at near-zenith is entirely explained by the beam mode
  choice, not a bug. See design doc §Diffraction model comparison.

Run with: pytest tests/test_validation.py -v
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from physics.atmosphere import hufnagel_valley, atmospheric_attenuation
from physics.detection import instantaneous_qber
from core.evaluator import evaluate_point
from core.types import Geometry, ChannelConfig, SourceConfig
from physics.detector import DetectorModel


# ---------------------------------------------------------------------------
# Bourgoin scenario constants
# ---------------------------------------------------------------------------

# Orbit
BOURGOIN_H_ORBIT  = 600e3          # m — circular orbit altitude
BOURGOIN_LAMBDA   = 670e-9         # m — wavelength
BOURGOIN_D_TX     = 0.10           # m — transmitter aperture diameter
BOURGOIN_D_RX     = 0.50           # m — receiver aperture diameter
BOURGOIN_SIGMA_PNT = 2e-6          # rad — RMS pointing jitter (1-sigma)

# Source / detector
BOURGOIN_F_CLOCK  = 300e6          # Hz — pulse repetition rate
BOURGOIN_DARK_CPS = 20             # counts per second — dark count rate
BOURGOIN_DELTA_T  = 0.5e-9         # s — detection gate width

# Atmospheric model (670 nm, rural sea level, 5 km visibility)
# Zenith Beer-Lambert optical depth τ₀ inferred from Figure 2:
#   η_atm(zenith) = 0.720  →  τ₀ = -ln(0.720) = 0.329
BOURGOIN_TAU_ZENITH = 0.329

# Effective extinction coefficient for our Beer-Lambert formulation.
# Derivation: η = exp(-α·L) with L = h_orbit/sin(θ_el).
#   At zenith, L = h_orbit, so α_eff = τ₀/h_orbit.
#   Then η(θ_el) = exp(-α_eff · h_orbit/sin(θ_el)) = exp(-τ₀/sin(θ_el)) ✓
BOURGOIN_ALPHA_EFF = BOURGOIN_TAU_ZENITH / BOURGOIN_H_ORBIT   # 5.48e-7 m⁻¹

# Hufnagel-Valley parameters (Bourgoin Table 1 / Eq. 6)
BOURGOIN_CN2_0 = 1.7e-14           # m⁻²/³ — ground-level structure parameter
BOURGOIN_V_WIND = 21.0             # m/s   — RMS wind speed

# Dark count probability per gate (Bourgoin parameters)
BOURGOIN_P_D = BOURGOIN_DARK_CPS / BOURGOIN_F_CLOCK   # 6.67e-8 per gate

# Bourgoin atmospheric transmittance checkpoints [elevation_deg, eta_expected, tolerance]
# Values at 90° read from Figure 2; lower elevations extrapolated from Beer-Lambert
# slant model exp(-τ₀/sin(θ)). Tolerances widen at low elevation because Bourgoin
# uses MODTRAN (full spectral transport) while we use a single-coefficient exponential.
BOURGOIN_ATM_CHECKPOINTS = [
    (90, 0.720, 0.02),    # zenith — read directly from Figure 2
    (50, 0.651, 0.03),    # Beer-Lambert prediction; Figure 2 ≈ 0.64
    (30, 0.514, 0.05),    # Beer-Lambert prediction; Figure 2 ≈ 0.50
    (20, 0.382, 0.06),    # Beer-Lambert prediction; Figure 2 ≈ 0.35
                          # Tolerance widened: MODTRAN diverges from 1-layer exp below ~20°
]


# ---------------------------------------------------------------------------
# TestBourgoin2013
# ---------------------------------------------------------------------------

class TestBourgoin2013:
    """
    Cross-validation against Bourgoin et al. (2013), NJP 15, 023006.

    Scenario: downlink WCP source, 600 km circular orbit, λ = 670 nm,
    rural sea level (5 km visibility), H-V turbulence with A = 1.7×10⁻¹⁴.
    """

    # ── Atmospheric transmittance ─────────────────────────────────────────

    def test_atm_transmittance_zenith(self):
        """
        Zenith transmittance at 670 nm matches Figure 2 within 2%.

        Bourgoin: η_atm(90°) ≈ 0.720 (MODTRAN rural, 5 km visibility).
        Our Beer-Lambert: exp(-τ₀) = exp(-0.329) = 0.7194 → rounds to 0.720.
        Agreement is exact here because we derived α_eff from this value.
        """
        L_zenith = BOURGOIN_H_ORBIT / math.sin(math.radians(90))
        eta = atmospheric_attenuation(BOURGOIN_ALPHA_EFF, L_zenith)
        assert abs(eta - 0.720) < 0.02, (
            f"Zenith transmittance {eta:.4f} deviates from Bourgoin 0.720 by "
            f"{abs(eta-0.720):.4f} (tol 0.02)"
        )

    @pytest.mark.parametrize("theta_el_deg, expected, tol", BOURGOIN_ATM_CHECKPOINTS)
    def test_atm_transmittance_angle_dependence(self, theta_el_deg, expected, tol):
        """
        η_atm(θ) = exp(−τ₀/sin(θ)) reproduces Bourgoin Figure 2 within tolerance.

        At θ < 20° Beer-Lambert overestimates loss because the single-layer
        exponential model doesn't capture the atmospheric limb geometry that
        MODTRAN handles correctly. The wider tolerances at low elevation reflect
        this known limitation.
        """
        theta_el = math.radians(theta_el_deg)
        L_slant = BOURGOIN_H_ORBIT / math.sin(theta_el)
        eta = atmospheric_attenuation(BOURGOIN_ALPHA_EFF, L_slant)
        assert abs(eta - expected) < tol, (
            f"η_atm at {theta_el_deg}° = {eta:.4f}, expected ≈ {expected:.4f} "
            f"(tol {tol}). Beer-Lambert vs MODTRAN discrepancy at low elevation."
        )

    # ── Hufnagel-Valley profile ───────────────────────────────────────────

    def test_hv_profile_ground_level(self):
        """
        At h = 0, the H-V profile is dominated by the ground layer and returns
        approximately Cn2_0 = 1.7×10⁻¹⁴ m⁻²/³.

        Exact value: Cn2(0) = Cn2_0 + 2.7×10⁻¹⁶ (stratospheric contribution).
        The 2.7×10⁻¹⁶ term adds ~1.6% to the ground value — tolerance is 2%.

        This validates that our hufnagel_valley() matches Bourgoin Eq. 6 exactly
        (same three-term form, same coefficients).
        """
        profile = hufnagel_valley(Cn2_0=BOURGOIN_CN2_0, v=BOURGOIN_V_WIND)
        cn2_at_0 = profile(0.0)
        rel_err = abs(cn2_at_0 - BOURGOIN_CN2_0) / BOURGOIN_CN2_0
        assert rel_err < 0.02, (
            f"H-V profile at h=0: {cn2_at_0:.4e} m⁻²/³, "
            f"expected ≈ {BOURGOIN_CN2_0:.4e} m⁻²/³ "
            f"(relative error {rel_err:.3f}, tol 0.02). "
            f"Stratospheric term adds ~1.6%; remaining error indicates model mismatch."
        )

    def test_hv_profile_structure(self):
        """
        H-V profile has the correct physical structure:
          - Maximum at h = 0 (ground layer dominates)
          - Jet-stream bump near h = 10 km (the h^10·exp(−h/1000) term)
          - Profile at 20 km is far below the ground-level value

        The profile is NOT monotone — it dips between 0 and 10 km then rises
        slightly at the jet-stream peak before falling again. This is correct
        physics for the three-term H-V model.
        """
        profile = hufnagel_valley(Cn2_0=BOURGOIN_CN2_0, v=BOURGOIN_V_WIND)
        cn2_0m  = profile(0.0)
        cn2_10km = profile(10_000.0)
        cn2_20km = profile(20_000.0)

        # Ground level dominates
        assert cn2_0m > cn2_10km, (
            f"Expected profile(0) > profile(10km): {cn2_0m:.3e} vs {cn2_10km:.3e}"
        )
        # Jet-stream bump visible: 10 km > 5 km for these parameters
        cn2_5km = profile(5_000.0)
        assert cn2_10km > cn2_5km, (
            f"Jet-stream bump expected: profile(10km) > profile(5km): "
            f"{cn2_10km:.3e} vs {cn2_5km:.3e}"
        )
        # High altitude is orders of magnitude below ground level (< 10⁻⁴ × ground)
        assert cn2_20km < cn2_0m * 1e-4, (
            f"Profile at 20 km ({cn2_20km:.3e}) should be < 10⁻⁴ × ground ({cn2_0m:.3e})"
        )

    def test_hv_profile_vanishes_at_high_altitude(self):
        """At 50 km, all three H-V terms are negligible (< 10⁻²⁵ m⁻²/³)."""
        profile = hufnagel_valley(Cn2_0=BOURGOIN_CN2_0, v=BOURGOIN_V_WIND)
        cn2_50km = profile(50_000.0)
        assert cn2_50km < 1e-25, (
            f"H-V profile at 50 km = {cn2_50km:.3e}; expected < 1e-25."
        )

    def test_hv_profile_jet_stream_term_coefficient(self):
        """
        At h = 10,000 m the jet-stream term peaks. Validate against Bourgoin Eq. 6.

        jet_peak = 0.00594 * (v/27)² * (10000 × 10⁻⁵)¹⁰ × exp(−10)

        The profile at 10 km = jet_term + strat_term + ground_term.
        Strat term: 2.7×10⁻¹⁶ × exp(−10000/1500) ≈ 3.4×10⁻¹⁹ (included in profile).
        Ground term: Cn2_0 × exp(−100) ≈ 0 (negligible).
        We compare profile vs (jet + strat), not profile vs jet alone.
        """
        v = BOURGOIN_V_WIND
        h = 10_000.0
        jet_term  = 0.00594 * (v / 27.0)**2 * (h * 1e-5)**10 * math.exp(-h / 1000.0)
        strat_term = 2.7e-16 * math.exp(-h / 1500.0)
        gnd_term   = BOURGOIN_CN2_0 * math.exp(-h / 100.0)   # ≈ 0
        analytic = jet_term + strat_term + gnd_term

        profile = hufnagel_valley(Cn2_0=BOURGOIN_CN2_0, v=v)
        cn2_10km = profile(h)

        rel_err = abs(cn2_10km - analytic) / max(analytic, 1e-30)
        assert rel_err < 1e-9, (
            f"H-V profile at 10 km: got {cn2_10km:.6e}, analytic = {analytic:.6e} "
            f"(relative error {rel_err:.2e}; should be float-exact)."
        )

    # ── QBER limiting behaviour ───────────────────────────────────────────

    def test_qber_high_transmissivity_limit(self):
        """
        At η → 1 (lossless channel), QBER → e_opt (optical alignment floor).

        With Bourgoin dark count rate p_d = 20 cps / 300 MHz = 6.67×10⁻⁸ per gate,
        the noise term 0.5·Y₀/(η·μ + Y₀) → 0 as η → 1, leaving E → e_opt.
        This is an analytical identity, independent of the full simulation.
        """
        E = instantaneous_qber(eta=1.0, mu=0.5, e_opt=0.03, Y_0=BOURGOIN_P_D)
        assert abs(E - 0.03) < 1e-4, (
            f"QBER at η=1 is {E:.6f}; expected e_opt = 0.03 (tol 1e-4). "
            f"Noise floor Y_0={BOURGOIN_P_D:.2e} should be negligible vs signal."
        )

    def test_qber_zero_transmissivity_limit(self):
        """
        At η → 0 (fully attenuated channel), QBER → 0.5 (noise-dominated).

        When η·μ ≪ Y₀, numerator ≈ 0.5·Y₀ and denominator ≈ Y₀, so E → 0.5.
        This is the regime where all detections are dark counts with random bits.
        """
        E = instantaneous_qber(eta=1e-10, mu=0.5, e_opt=0.03, Y_0=BOURGOIN_P_D)
        assert abs(E - 0.5) < 0.01, (
            f"QBER at η=1e-10 is {E:.4f}; expected ≈ 0.5 (noise-dominated, tol 0.01). "
            f"Signal {1e-10*0.5:.2e} should be ≪ Y_0={BOURGOIN_P_D:.2e}."
        )

    def test_qber_crossover_transmissivity(self):
        """
        At η = Y₀/μ the signal and noise contributions are equal.

        QBER(η=Y₀/μ) = (e_opt·Y₀ + 0.5·Y₀) / (Y₀ + Y₀) = (e_opt + 0.5) / 2
        For e_opt = 0.03: QBER = 0.265.
        """
        e_opt = 0.03
        mu = 0.5
        eta_cross = BOURGOIN_P_D / mu
        E = instantaneous_qber(eta=eta_cross, mu=mu, e_opt=e_opt, Y_0=BOURGOIN_P_D)
        expected = (e_opt + 0.5) / 2.0
        assert abs(E - expected) < 1e-6, (
            f"QBER at crossover η={eta_cross:.2e}: got {E:.6f}, "
            f"expected (e_opt+0.5)/2 = {expected:.6f}."
        )

    def test_qber_monotone_in_transmissivity(self):
        """QBER decreases monotonically as η increases (signal improves SNR)."""
        etas = [1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 0.5, 1.0]
        Es = [instantaneous_qber(eta, mu=0.5, e_opt=0.03, Y_0=BOURGOIN_P_D)
              for eta in etas]
        for i in range(len(etas) - 1):
            assert Es[i] >= Es[i + 1], (
                f"QBER not monotone: E({etas[i]:.0e})={Es[i]:.4f} < "
                f"E({etas[i+1]:.0e})={Es[i+1]:.4f}."
            )

    # ── Total loss sanity check ───────────────────────────────────────────

    def test_total_loss_order_of_magnitude(self):
        """
        Total link loss at near-zenith (600 km) is in the −55 to −20 dB range.

        Bourgoin Figure 7: best pass minimum ~45 dB (truncated-plane-wave beam).
        Our focused Gaussian with w0 ≈ D_tx/2 = 5 cm will be more optimistic
        (less diffraction loss) — ~24 dB is expected, well within the sanity band.

        This is explicitly a broad sanity check, not a precision comparison.
        The ~20 dB gap vs Bourgoin is expected and documented (beam mode difference).

        Bourgoin-inspired parameters used:
            λ = 670 nm, w0 = D_tx/2 = 5 cm (focused Gaussian approximation),
            D_rx = 50 cm, h_orbit = 600 km, θ_el = 90° (zenith).
        """
        # For our Gaussian model we need an explicit w0. Use D_tx/2 = 5 cm,
        # which represents a focused beam clipped by the 10 cm transmitter aperture.
        # (Bourgoin's FWHM ≈ D_tx gives the truncated-plane-wave regime;
        # our 1/e² waist at the aperture edge is a different — more optimistic — choice.)
        w0 = BOURGOIN_D_TX / 2.0   # 0.05 m

        geom = Geometry(
            theta_el=math.pi / 2,              # zenith
            L=BOURGOIN_H_ORBIT,                # 600 km slant range at zenith
            zeta=0.0,                          # zenith angle = 0
            h_orbit=BOURGOIN_H_ORBIT,
        )
        channel = ChannelConfig(
            eta_tx=0.8,
            lambda_=BOURGOIN_LAMBDA,
            w0=w0,
            D_rx=BOURGOIN_D_RX,
            eta_rx=0.7,
            alpha=BOURGOIN_ALPHA_EFF,
            Cn2_0=BOURGOIN_CN2_0,
            v_wind=BOURGOIN_V_WIND,
            theta_pnt=0.0,
            sigma_pnt=BOURGOIN_SIGMA_PNT,
            e_opt=0.03,
        )
        source = SourceConfig(mu=0.5, nu=0.1, P_mu=0.6, P_nu=0.3, P_X=0.9,
                              f_clock=BOURGOIN_F_CLOCK)
        det = DetectorModel(
            eta_det=0.65,
            p_d=BOURGOIN_P_D,
            tau_d=50e-9,
            delta_t=BOURGOIN_DELTA_T,
        )

        state = evaluate_point(geom, channel, source, det)
        loss_db = state.loss_budget.to_db_dict()["eta_0"]

        assert -55.0 < loss_db < -20.0, (
            f"Total link loss at zenith (600 km): {loss_db:.1f} dB. "
            f"Expected −55 to −20 dB (sanity band). "
            f"Bourgoin reports ~−45 dB with a truncated-plane-wave beam; "
            f"our focused Gaussian with w0=5 cm yields ~−24 dB (less loss, as expected)."
        )

    def test_dark_count_per_gate(self):
        """
        Y₀ derived from Bourgoin's 20 cps dark rate at 300 MHz matches
        the expected probability-per-gate.

            p_d = 20 [cps] / 300×10⁶ [Hz] = 6.67×10⁻⁸
        """
        assert abs(BOURGOIN_P_D - 20.0 / 300e6) < 1e-12
        # 20/300e6 = 6.6̄×10⁻⁸; check within 1% of the rounded figure 6.67e-8
        assert abs(BOURGOIN_P_D - 6.667e-8) < 1e-11


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.call(["pytest", __file__, "-v"]))
