"""
physics/turbulence.py — Turbulence models and fading distributions.

Provided:
    FadingModel         Protocol: pdf / mean_eta / integrate.
    LogNormalFading     Weak-turbulence log-normal fading (σ_R² < threshold).
    GammaGammaFading    Moderate/strong-turbulence Gamma-Gamma fading.
    rytov_variance      Slant-path Rytov variance via adaptive quadrature.
    select_fading_model Choose fading model from σ_R² and η₀.

Physics references:
    Andrews & Phillips (2005) — Laser Beam Propagation through Random Media, 2nd ed.
    Bourgoin et al. (2013)   — A comprehensive design and performance analysis of LEO
                               satellite quantum communication.
"""

from __future__ import annotations

import math
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from scipy import integrate, special

from physics.atmosphere import Cn2Profile


# ---------------------------------------------------------------------------
# FadingModel protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FadingModel(Protocol):
    """
    Distribution of channel transmissivity η due to atmospheric turbulence.

    The distribution already encodes the mean transmissivity η₀ (all static
    losses). Callers never separate the fading component from η₀ — they just
    call integrate(f) to get E[f(η)] averaged over the full fading distribution.

    Required invariants (checked in tests):
        integrate(lambda eta: 1.0) ≈ 1.0          # normalisation
        integrate(lambda eta: eta) ≈ mean_eta()    # first moment
    """

    def pdf(self, eta: float) -> float:
        """Probability density at transmissivity η ∈ (0, 1]."""
        ...

    def mean_eta(self) -> float:
        """Mean transmissivity E[η] = η₀."""
        ...

    def integrate(self, f: Callable[[float], float]) -> float:
        """
        Return E[f(η)] = ∫ f(η) p(η) dη.

        Used by detection.py for fading-averaged gain and QBER:
            E[Q_μ(η)] = integrate(lambda eta: 1 - (1 - Y_0) * exp(-eta * mu))
            E[E_μ(η)] = integrate(lambda eta: instantaneous_qber(eta, mu, e_opt, Y_0))
        """
        ...


# ---------------------------------------------------------------------------
# Log-normal fading (weak turbulence)
# ---------------------------------------------------------------------------

class LogNormalFading:
    """
    Log-normal fading for weak turbulence (σ_R² ≲ 0.75).

    If η is log-normally distributed with mean η₀:
        ln(η) ~ N(μ_ln, σ_R²)
        μ_ln = ln(η₀) − σ_R² / 2     ← ensures E[η] = η₀ exactly

    The variance of ln(η) equals σ_R² (Rytov variance), which is the standard
    weak-turbulence identification (Andrews & Phillips 2005, §8.2).

    All integrals are evaluated by substituting u = ln(η), transforming the
    lognormal density into a Gaussian density in u, then applying scipy.quad.
    The substitution avoids evaluating the integrand at η = 0.
    """

    def __init__(self, eta_0: float, sigma_R2: float) -> None:
        """
        Args:
            eta_0:    Mean transmissivity (all static losses already folded in).
            sigma_R2: Rytov variance.  Zero gives a delta-function at η₀.
        """
        if not (0.0 < eta_0 <= 1.0):
            raise ValueError(f"eta_0 must be in (0, 1]; got {eta_0}")
        if sigma_R2 < 0.0:
            raise ValueError(f"sigma_R2 must be ≥ 0; got {sigma_R2}")

        self._eta_0 = eta_0
        self._sigma_R2 = sigma_R2
        self._mu_ln = math.log(eta_0) - sigma_R2 / 2.0
        self._sigma_ln = math.sqrt(sigma_R2) if sigma_R2 > 0.0 else 0.0

    # -- FadingModel interface --

    def pdf(self, eta: float) -> float:
        """Log-normal PDF.  Returns 0 for η ≤ 0; ∞ at η₀ when σ_R² = 0."""
        if eta <= 0.0:
            return 0.0
        if self._sigma_ln == 0.0:
            return float("inf") if eta == self._eta_0 else 0.0
        u = math.log(eta)
        return (
            math.exp(-0.5 * ((u - self._mu_ln) / self._sigma_ln) ** 2)
            / (eta * self._sigma_ln * math.sqrt(2.0 * math.pi))
        )

    def mean_eta(self) -> float:
        return self._eta_0

    def integrate(self, f: Callable[[float], float]) -> float:
        """
        E[f(η)] via Gaussian quadrature in log-space.

        Substitution u = ln(η):
            ∫ f(η) p_LN(η) dη  →  ∫ f(e^u) · N(u; μ_ln, σ_ln²) du

        Limits: μ_ln ± 8σ_ln (captures > 1 − 10⁻¹⁵ of the Gaussian mass).
        """
        if self._sigma_ln == 0.0:
            return f(self._eta_0)

        mu, sig = self._mu_ln, self._sigma_ln
        inv_norm = 1.0 / (sig * math.sqrt(2.0 * math.pi))

        def integrand(u: float) -> float:
            gaussian = inv_norm * math.exp(-0.5 * ((u - mu) / sig) ** 2)
            return f(math.exp(u)) * gaussian

        result, _ = integrate.quad(
            integrand, mu - 8.0 * sig, mu + 8.0 * sig, limit=200
        )
        return result

    # -- Diagnostics --

    @property
    def sigma_R2(self) -> float:
        return self._sigma_R2

    @property
    def mu_ln(self) -> float:
        return self._mu_ln

    @property
    def sigma_ln(self) -> float:
        return self._sigma_ln

    def __repr__(self) -> str:
        return (f"LogNormalFading(eta_0={self._eta_0:.4g}, "
                f"sigma_R2={self._sigma_R2:.4g})")


# ---------------------------------------------------------------------------
# Gamma-Gamma fading (moderate / strong turbulence)
# ---------------------------------------------------------------------------

class GammaGammaFading:
    """
    Gamma-Gamma fading for moderate/strong turbulence (σ_R² ≳ 0.75).

    The Gamma-Gamma distribution for normalised irradiance I = η / η₀:

        p(I) = 2(αβ)^((α+β)/2) / [Γ(α)Γ(β)] · I^((α+β)/2−1) · K_{|α−β|}(2√(αβI))

    where K_ν is the modified Bessel function of the second kind.

    Parameters α, β are derived from σ_R² via the spherical-wave approximations
    of Andrews & Phillips (2005), Eqs. 9.67–9.70:

        α = {exp[0.49 σ_R² / (1 + 0.56 σ_R^(12/5))^(7/6)] − 1}⁻¹
        β = {exp[0.51 σ_R² / (1 + 0.69 σ_R^(12/5))^(5/6)] − 1}⁻¹

    Limiting behaviour:
        σ_R² → 0:  α, β → ∞  (approaches log-normal / delta function)
        σ_R² → ∞:  α → 1, β → 1  (saturated turbulence / exponential)

    Numerical stability:
        The PDF has a K_ν-driven singularity near η = 0 (η ≈ 0, I ≈ 0).
        integrate() applies the substitution u = ln(η/η₀), which replaces
        the singularity with a smooth exponential decay as u → −∞.
    """

    def __init__(self, eta_0: float, sigma_R2: float) -> None:
        """
        Args:
            eta_0:    Mean transmissivity η₀ ∈ (0, 1].
            sigma_R2: Rytov variance (must be > 0 for GammaGamma to be meaningful).
        """
        if not (0.0 < eta_0 <= 1.0):
            raise ValueError(f"eta_0 must be in (0, 1]; got {eta_0}")
        if sigma_R2 <= 0.0:
            raise ValueError(f"sigma_R2 must be positive for GammaGamma; got {sigma_R2}")

        self._eta_0 = eta_0
        self._sigma_R2 = sigma_R2
        self._alpha, self._beta = _gg_alpha_beta(sigma_R2)

        a, b = self._alpha, self._beta
        self._nu = abs(a - b)
        # log of 2(αβ)^((α+β)/2) / (Γ(α) Γ(β))
        self._log_C = (
            math.log(2.0)
            + 0.5 * (a + b) * math.log(a * b)
            - special.gammaln(a)
            - special.gammaln(b)
        )

    # -- FadingModel interface --

    def pdf(self, eta: float) -> float:
        """Gamma-Gamma PDF at transmissivity η."""
        if eta <= 0.0:
            return 0.0
        x = eta / self._eta_0                        # normalised irradiance I
        a, b = self._alpha, self._beta
        arg = 2.0 * math.sqrt(a * b * x)
        if arg < 1e-300:
            return 0.0
        log_pdf = (
            self._log_C
            + ((a + b) / 2.0 - 1.0) * math.log(x)
            - math.log(self._eta_0)                  # Jacobian for η → I
            + math.log(max(special.kv(self._nu, arg), 1e-300))
        )
        return math.exp(log_pdf)

    def mean_eta(self) -> float:
        return self._eta_0

    def integrate(self, f: Callable[[float], float]) -> float:
        """
        E[f(η)] via adaptive quadrature in log-space.

        Substitution u = ln(η/η₀), so I = e^u and dI = e^u du:

            ∫₀^∞ f(η₀ I) p(I) dI  →  ∫_{-∞}^{+∞} f(η₀ e^u) · g(e^u) du

        where g(x) = C · x^((α+β)/2) · K_{|α-β|}(2√(αβ x)).

        As u → −∞: x → 0, x^((α+β)/2) → 0 faster than K diverges → integrand → 0.
        As u → +∞: K decays exponentially → integrand → 0.
        The transformed integrand is smooth everywhere.
        """
        a, b = self._alpha, self._beta
        ab = a * b
        eta_0 = self._eta_0
        log_C = self._log_C
        nu = self._nu

        def integrand(u: float) -> float:
            x = math.exp(u)          # normalised irradiance
            eta = eta_0 * x
            arg = 2.0 * math.sqrt(ab * x)
            if arg < 1e-300:
                return 0.0
            kv_val = special.kv(nu, arg)
            if kv_val <= 0.0:
                return 0.0
            log_g = log_C + (a + b) / 2.0 * u + math.log(kv_val)
            return f(eta) * math.exp(log_g)

        # Limits chosen to capture the full distribution.
        # Lower: u = -40 → I = e^{-40} ≈ 4e-18 (deep tail, g ≈ 0)
        # Upper: u = +10  → I = e^{10}  ≈ 22000 (far above peak, K kills it)
        result, _ = integrate.quad(integrand, -40.0, 10.0, limit=300)
        return result

    # -- Diagnostics --

    @property
    def sigma_R2(self) -> float:
        return self._sigma_R2

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    def __repr__(self) -> str:
        return (f"GammaGammaFading(eta_0={self._eta_0:.4g}, "
                f"sigma_R2={self._sigma_R2:.4g}, "
                f"alpha={self._alpha:.4g}, beta={self._beta:.4g})")


def _gg_alpha_beta(sigma_R2: float) -> tuple[float, float]:
    """
    Spherical-wave Gamma-Gamma parameters from Andrews & Phillips (2005), Eq. 9.70.

    Uses the spherical-wave form appropriate for a satellite downlink.
    """
    sr_12_5 = sigma_R2 ** (6.0 / 5.0)    # σ_R^(12/5)
    exp_alpha = math.exp(0.49 * sigma_R2 / (1.0 + 0.56 * sr_12_5) ** (7.0 / 6.0))
    exp_beta  = math.exp(0.51 * sigma_R2 / (1.0 + 0.69 * sr_12_5) ** (5.0 / 6.0))
    alpha = 1.0 / (exp_alpha - 1.0)
    beta  = 1.0 / (exp_beta  - 1.0)
    return alpha, beta


# ---------------------------------------------------------------------------
# Rytov variance — slant-path integral
# ---------------------------------------------------------------------------

def rytov_variance(
    lambda_: float,
    zeta: float,
    h0: float,
    cn2_profile: Cn2Profile,
    H_max: float = 20e3,
) -> float:
    """
    Slant-path Rytov variance σ_R² for a downlink (spherical wave).

    Formula (Andrews & Phillips 2005; Bourgoin et al. 2013, Eq. A1):

        σ_R² = 2.25 k^(7/6) sec^(11/6)(ζ)
               × ∫_{h₀}^{H_max} Cn²(h) (h − h₀)^(5/6) dh

    where k = 2π/λ is the optical wave number.

    Derivation sketch:
        For a spherical-wave downlink with turbulence concentrated in the first
        ~20 km and LEO altitude ≫ H_max, the factor (1 − z/L)^(5/6) ≈ 1
        throughout the integration range. The path-length element dz = dh sec(ζ)
        and z^(5/6) = (h−h₀)^(5/6) sec^(5/6)(ζ) combine to give sec^(11/6)(ζ).

    The integrand is dominated by the H-V ground-layer term (∝ e^{−h/100}), so
    nearly all of σ_R² accumulates in the first ~500 m. scipy.quad's adaptive
    algorithm handles this concentration automatically.

    Args:
        lambda_:     Wavelength (m).
        zeta:        Zenith angle (rad).  Must satisfy 0 ≤ ζ < π/2.
        h0:          Ground station altitude above sea level (m).
        cn2_profile: Cn2Profile callable, e.g. from hufnagel_valley().
        H_max:       Upper integration limit (m). Turbulence is negligible above
                     ~20 km; the default captures > 99.9% of the integral.

    Returns:
        σ_R² ≥ 0 (dimensionless).  Zero when Cn² ≡ 0 over [h₀, H_max].

    Raises:
        ValueError: if ζ is outside [0, π/2) or H_max ≤ h0.
    """
    if not (0.0 <= zeta < math.pi / 2.0):
        raise ValueError(f"zeta must be in [0, π/2); got {zeta:.4f} rad")
    if H_max <= h0:
        raise ValueError(f"H_max ({H_max}) must be greater than h0 ({h0})")

    k = 2.0 * math.pi / lambda_
    prefactor = 2.25 * k ** (7.0 / 6.0) * math.cos(zeta) ** (-11.0 / 6.0)

    def integrand(h: float) -> float:
        dh = h - h0
        if dh <= 0.0:
            return 0.0
        return cn2_profile(h) * dh ** (5.0 / 6.0)

    # The integrand peaks near h0 + a few hundred metres (H-V ground layer).
    # Provide a hint to quad so it can concentrate quadrature points there.
    h_peak = h0 + 200.0
    result, _ = integrate.quad(
        integrand, h0, H_max,
        points=[h_peak, h0 + 1e3, h0 + 5e3],
        limit=200,
    )
    return prefactor * result


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_fading_model(
    sigma_R2: float,
    eta_0: float,
    threshold: float = 0.75,
) -> FadingModel:
    """
    Return the appropriate fading model for the given Rytov variance.

    Selection rule:
        σ_R² = 0              → LogNormalFading (degenerate: delta function at η₀)
        σ_R² < threshold      → LogNormalFading  (weak turbulence)
        σ_R² ≥ threshold      → GammaGammaFading (moderate / strong turbulence)

    The default threshold of 0.75 is the conventional weak/moderate boundary
    (Andrews & Phillips 2005, §9.4). It is intentionally configurable: some
    analyses use 0.5, others 1.0.

    Args:
        sigma_R2:  Rytov variance from rytov_variance().
        eta_0:     Mean transmissivity (static losses already included).
        threshold: σ_R² value above which GammaGamma is used.

    Returns:
        A FadingModel instance satisfying the Protocol.
    """
    if sigma_R2 < 0.0:
        raise ValueError(f"sigma_R2 must be ≥ 0; got {sigma_R2}")
    if not (0.0 < eta_0 <= 1.0):
        raise ValueError(f"eta_0 must be in (0, 1]; got {eta_0}")
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive; got {threshold}")

    if sigma_R2 < threshold:
        return LogNormalFading(eta_0, sigma_R2)
    return GammaGammaFading(eta_0, sigma_R2)
