"""
physics/detection.py — Fading-averaged detection observables (Stage 3).

Both gain Q_μ and QBER E_μ are integrated over P(η), the channel fading
distribution. Computing either at the mean transmissivity ⟨η⟩ produces
incorrect results due to Jensen's inequality:

    Gain: Q_μ(η) = 1 − (1−Y_0)e^{−ημ} is concave in η.
          Jensen ⟹ ⟨Q_μ(η)⟩ < Q_μ(⟨η⟩): fading reduces mean gain.

    QBER: e(η) = (e_opt·ημ + 0.5·Y_0)/(ημ + Y_0) is convex in η.
          Jensen ⟹ ⟨e(η)⟩ > e(⟨η⟩): fading raises mean QBER.

Both corrections must be positive. Negative Jensen gap = integration bug.

Provided:
    noise_yield          Dark + background count probability per gate.
    instantaneous_qber   QBER at a fixed transmissivity (no fading).
    expected_gain        Fading-averaged gain ⟨Q_μ⟩.
    expected_qber        Fading-averaged QBER ⟨E_μ⟩.
    jensen_gaps          Compute both gaps and check for integration errors.
    compute_detection    Full DetectionResult for one intensity level.
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import DetectionResult
from physics.detector import DetectorModel
from physics.turbulence import FadingModel


# ---------------------------------------------------------------------------
# Noise yield
# ---------------------------------------------------------------------------

def noise_yield(
    detector: DetectorModel,
    H_bg: float,
    Omega_FOV: float,
    A_rx: float,
    delta_lambda: float,
    eta_rx: float,
) -> float:
    """
    Total noise detection probability per gate: Y_0 = p_d + p_bg.

    Both terms are dimensionless probabilities per gate window. They add
    directly because both dark counts and background photons are Poisson
    processes that produce a detection with a given probability per gate.

    Background photon count per gate:
        p_bg = H_bg × Omega_FOV × A_rx × delta_lambda × delta_t × eta_rx

    where:
        H_bg        sky spectral radiance (W/m²/sr/m)
        Omega_FOV   receiver field-of-view solid angle (sr)
        A_rx        receiver aperture area (m²)
        delta_lambda spectral filter bandwidth (m)
        delta_t     detector gate width (s) — taken from detector.delta_t
        eta_rx      receiver optical chain efficiency

    Args:
        detector:     DetectorModel providing p_d and delta_t.
        H_bg:         Sky spectral radiance (W/m²/sr/m). Use 0.0 for night.
        Omega_FOV:    Receiver FOV solid angle (sr).
        A_rx:         Receiver aperture area (m²); π(D_rx/2)².
        delta_lambda: Spectral filter bandwidth (m).
        eta_rx:       Receiver optical efficiency (dimensionless).

    Returns:
        Y_0 = p_d + p_bg (dimensionless probability per gate).
    """
    p_bg = H_bg * Omega_FOV * A_rx * delta_lambda * detector.delta_t * eta_rx
    return detector.dark_count_rate() + p_bg


# ---------------------------------------------------------------------------
# Per-photon detection formulas
# ---------------------------------------------------------------------------

def instantaneous_qber(eta: float, mu: float, e_opt: float, Y_0: float) -> float:
    """
    QBER at a fixed (non-fluctuating) transmissivity η.

        E(η) = (e_opt · η·μ + 0.5 · Y_0) / (η·μ + Y_0)

    Numerator: signal errors at rate e_opt (optical misalignment) plus half
    the noise detections (uniformly random bit values).
    Denominator: total detections per gate.

    Special case: if η·μ + Y_0 = 0 (no photons, no noise) the gate yields no
    detection, so E is undefined; we return 0.5 (maximum uncertainty) as a
    conservative placeholder that will not contribute to any weighted average
    because Q = 0 in this regime.

    Args:
        eta:   Channel transmissivity for this pulse (dimensionless).
        mu:    Source intensity (mean photons/pulse).
        e_opt: Optical QBER floor (residual polarisation/phase error).
        Y_0:   Noise yield per gate (dark + background).

    Returns:
        QBER in [0, 1].
    """
    signal = eta * mu
    denom = signal + Y_0
    if denom <= 0.0:
        return 0.5
    return (e_opt * signal + 0.5 * Y_0) / denom


def _gain_integrand(eta: float, mu: float, Y_0: float) -> float:
    """Q_μ(η) = 1 − (1−Y_0)·exp(−η·μ)."""
    return 1.0 - (1.0 - Y_0) * math.exp(-eta * mu)


# ---------------------------------------------------------------------------
# Fading-averaged observables
# ---------------------------------------------------------------------------

def expected_gain(fading: FadingModel, mu: float, Y_0: float) -> float:
    """
    Fading-averaged gain ⟨Q_μ⟩ = ∫ Q_μ(η) P(η) dη.

    Q_μ(η) is the probability that at least one photon from a WCP pulse of
    mean intensity μ is detected, given transmissivity η and noise floor Y_0:

        Q_μ(η) = 1 − (1−Y_0)·e^{−ημ}

    In a fading channel η is a random variable drawn from P(η); the observed
    gain is the expectation of Q_μ over this distribution.

    Jensen's inequality: Q_μ(η) is concave in η, so
        ⟨Q_μ(η)⟩ ≤ Q_μ(⟨η⟩).
    Computing gain at mean η overestimates the true fading-averaged gain.

    Args:
        fading: FadingModel instance (LogNormal or GammaGamma).
        mu:     Source intensity (mean photons/pulse).
        Y_0:    Noise yield per gate (dimensionless).

    Returns:
        Fading-averaged gain ⟨Q_μ⟩ ∈ (0, 1].
    """
    return fading.integrate(lambda eta: _gain_integrand(eta, mu, Y_0))


def expected_qber(
    fading: FadingModel, mu: float, e_opt: float, Y_0: float
) -> float:
    """
    Fading-averaged QBER ⟨E_μ⟩ = ∫ E(η) · Q_μ(η) · P(η) dη / ⟨Q_μ⟩.

    The QBER must be detection-weighted: a gate with near-zero transmissivity
    produces almost no detections and its (undefined or extreme) QBER should
    not dominate the average. The correct estimator is:

        ⟨E_μ⟩ = ⟨E(η) · Q_μ(η)⟩ / ⟨Q_μ⟩

    Numerator = ∫ [(e_opt·η·μ + 0.5·Y_0)] · P(η) dη
    Denominator = ⟨Q_μ⟩

    This is equivalent to the ratio of total error counts to total detection
    counts, which is the operationally measured QBER.

    Jensen: E(η) is convex in η near the noise floor, so ⟨E(η)⟩ > E(⟨η⟩).

    Args:
        fading: FadingModel instance.
        mu:     Source intensity (mean photons/pulse).
        e_opt:  Optical QBER floor.
        Y_0:    Noise yield per gate.

    Returns:
        Fading-averaged detection-weighted QBER ∈ [0, 0.5].
    """
    def _error_rate_integrand(eta: float) -> float:
        signal = eta * mu
        denom = signal + Y_0
        if denom <= 0.0:
            return 0.0
        return (e_opt * signal + 0.5 * Y_0)

    numerator = fading.integrate(_error_rate_integrand)
    denominator = expected_gain(fading, mu, Y_0)
    if denominator <= 0.0:
        return 0.5
    return numerator / denominator


# ---------------------------------------------------------------------------
# Jensen gap diagnostics
# ---------------------------------------------------------------------------

def jensen_gaps(
    fading: FadingModel, mu: float, e_opt: float, Y_0: float
) -> tuple[float, float]:
    """
    Compute both Jensen correction gaps for validation.

    Both gaps must be positive for a physically correct integration.

    Gain gap:  Q_μ(⟨η⟩) − ⟨Q_μ(η)⟩  > 0  (gain at mean η > fading-averaged gain)
    QBER gap:  ⟨E_μ(η)⟩ − E_μ(⟨η⟩)  > 0  (fading-averaged QBER > QBER at mean η)

    A negative gap indicates either:
        • A bug in fading.integrate() (wrong normalisation).
        • σ_R² ≈ 0: LogNormal degenerates to a delta function; gaps → 0 (acceptable).

    Args:
        fading: FadingModel instance.
        mu:     Source intensity.
        e_opt:  Optical QBER floor.
        Y_0:    Noise yield per gate.

    Returns:
        (gain_gap, qber_gap) — both should be ≥ 0.
    """
    mean_eta = fading.mean_eta()

    gain_at_mean = _gain_integrand(mean_eta, mu, Y_0)
    avg_gain = expected_gain(fading, mu, Y_0)
    gain_gap = gain_at_mean - avg_gain

    qber_at_mean = instantaneous_qber(mean_eta, mu, e_opt, Y_0)
    avg_qber = expected_qber(fading, mu, e_opt, Y_0)
    qber_gap = avg_qber - qber_at_mean

    return gain_gap, qber_gap


# ---------------------------------------------------------------------------
# Top-level computation
# ---------------------------------------------------------------------------

def compute_detection(
    fading: FadingModel,
    detector: DetectorModel,
    intensity: float,
    e_opt: float,
    Y_0: float,
) -> DetectionResult:
    """
    Fading-averaged gain and QBER for a single source intensity.

    Computes ⟨Q⟩ and ⟨E⟩ by integrating over the fading distribution.
    n_counts is left at 0.0; the pass accumulator fills it per timestep:
        n_counts = f_clock × P_intensity × Q × sifting_factor × dt

    For vacuum pulses (intensity = 0.0):
        Q_vac = Y_0  (only noise detections)
        E_vac = 0.5  (random noise — no signal to correct)

    Args:
        fading:    FadingModel for the channel at this geometry.
        detector:  DetectorModel (used only for parameter access here;
                   Y_0 should be pre-computed via noise_yield()).
        intensity: Source intensity μ, ν, or 0.0 (photons/pulse).
        e_opt:     Optical QBER floor (residual alignment error).
        Y_0:       Noise yield per gate (dark + background, dimensionless).

    Returns:
        DetectionResult with Q, E, intensity, n_counts=0.0.
    """
    if intensity == 0.0:
        return DetectionResult(Q=Y_0, E=0.5, intensity=0.0, n_counts=0.0)

    Q = expected_gain(fading, intensity, Y_0)
    E = expected_qber(fading, intensity, e_opt, Y_0)
    return DetectionResult(Q=Q, E=E, intensity=intensity, n_counts=0.0)
