"""
physics/atmosphere.py — Atmospheric models for the VQI QKD link budget simulation.

Provides:
    Cn2Profile       Protocol for altitude-dependent refractive index structure parameter.
    hufnagel_valley  Hufnagel-Valley Cn2 profile factory.
    from_measurements Cn2 profile from tabulated altitude/value pairs (log-interpolated).
    atmospheric_attenuation  Beer-Lambert transmission over a slant path.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Cn2Profile protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Cn2Profile(Protocol):
    """
    Altitude-dependent refractive index structure parameter Cn²(h).

    Callable: takes altitude h above sea level (m), returns Cn² (m^{-2/3}).
    Ground station altitude offsets are applied externally (Rytov integral
    shifts the lower integration limit, not the profile).
    """

    def __call__(self, h: float) -> float:
        """Return Cn²(h) in m^{-2/3} at altitude h (m)."""
        ...


# ---------------------------------------------------------------------------
# Hufnagel-Valley profile
# ---------------------------------------------------------------------------

def hufnagel_valley(Cn2_0: float, v: float = 21.0) -> Cn2Profile:
    """
    Return a Hufnagel-Valley Cn² profile.

    The standard HV model (Andrews & Phillips 2005, Ch. 3):

        Cn²(h) = 0.00594 · (v/27)² · (h · 10⁻⁵)¹⁰ · exp(−h/1000)
               + 2.7 × 10⁻¹⁶ · exp(−h/1500)
               + Cn2_0 · exp(−h/100)

    The three terms model:
      • Tropopause jet-stream turbulence  (first term, peaks ~10 km)
      • Background stratospheric layer    (second term)
      • Ground-layer turbulence           (third term, dominant in first ~500 m)

    Args:
        Cn2_0: Ground-level Cn² (m^{-2/3}). Typical: 1e-14 (weak) – 1e-13 (strong).
        v:     RMS wind speed (m/s), default 21 m/s (standard HV 5/7 model).

    Returns:
        Cn2Profile callable: h (m) → Cn² (m^{-2/3}).
    """
    _v_factor = (v / 27.0) ** 2

    def _profile(h: float) -> float:
        h = max(h, 0.0)
        h_km = h * 1e-5          # h in units of 10^5 m for the exponent
        jet = 0.00594 * _v_factor * (h_km ** 10) * math.exp(-h / 1000.0)
        strat = 2.7e-16 * math.exp(-h / 1500.0)
        ground = Cn2_0 * math.exp(-h / 100.0)
        return jet + strat + ground

    return _profile


# ---------------------------------------------------------------------------
# Measurement-derived profile
# ---------------------------------------------------------------------------

def from_measurements(
    altitudes: np.ndarray,
    values: np.ndarray,
) -> Cn2Profile:
    """
    Return a Cn² profile interpolated from tabulated altitude/value pairs.

    Interpolation is performed in log-space (log10) so that the result
    respects the order-of-magnitude variation typical of Cn² profiles.
    Outside the measured range the profile is clamped to the nearest
    boundary value (no extrapolation).

    Args:
        altitudes: 1-D array of altitudes (m), strictly increasing.
        values:    1-D array of Cn² values (m^{-2/3}), same length, all > 0.

    Returns:
        Cn2Profile callable: h (m) → Cn² (m^{-2/3}).

    Raises:
        ValueError: if inputs are inconsistent or contain non-positive values.
    """
    altitudes = np.asarray(altitudes, dtype=float)
    values = np.asarray(values, dtype=float)

    if altitudes.ndim != 1 or values.ndim != 1:
        raise ValueError("altitudes and values must be 1-D arrays.")
    if altitudes.shape != values.shape:
        raise ValueError(
            f"altitudes and values must have the same length; "
            f"got {altitudes.shape[0]} and {values.shape[0]}."
        )
    if np.any(values <= 0.0):
        raise ValueError("All Cn² values must be strictly positive.")
    if not np.all(np.diff(altitudes) > 0):
        raise ValueError("altitudes must be strictly increasing.")

    log_values = np.log10(values)
    _interp = interp1d(
        altitudes, log_values,
        kind="linear",
        bounds_error=False,
        fill_value=(log_values[0], log_values[-1]),  # clamp at boundaries
    )

    def _profile(h: float) -> float:
        return float(10.0 ** _interp(max(h, 0.0)))

    return _profile


# ---------------------------------------------------------------------------
# Beer-Lambert atmospheric attenuation
# ---------------------------------------------------------------------------

def atmospheric_attenuation(alpha: float, L: float) -> float:
    """
    Atmospheric transmissivity over a slant path via Beer-Lambert law.

        η_atm = exp(−α · L)

    Args:
        alpha: Extinction coefficient (m⁻¹). Includes absorption + scattering.
               Typical clear-sky value at 850 nm: ~3e-5 m⁻¹ (≈ 0.13 dB/km).
        L:     Slant-path length from ground station to satellite (m).

    Returns:
        η_atm (dimensionless, in [0, 1]).

    Notes:
        α is assumed wavelength-integrated and vertically homogeneous. A more
        accurate treatment would integrate the altitude-dependent extinction
        along the slant path; that is deferred to Section 10.11 (lowtran).
        For zenith and moderate elevations (> 20°) the homogeneous approximation
        introduces < 5% error in η_atm.
    """
    if alpha < 0.0:
        raise ValueError(f"Extinction coefficient alpha must be >= 0; got {alpha}.")
    if L < 0.0:
        raise ValueError(f"Slant range L must be >= 0; got {L}.")
    return math.exp(-alpha * L)
