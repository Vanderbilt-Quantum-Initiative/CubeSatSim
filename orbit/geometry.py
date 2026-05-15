"""
orbit/geometry.py — Pass geometry via Skyfield + sgp4.

Thin wrapper around Skyfield for satellite-ground geometry.  No custom
orbital mechanics — SGP4 propagation, WGS84 ellipsoid, and coordinate
transforms are all delegated to the library.

For hypothetical (unlaunched) satellites, sgp4.Satrec.sgp4init constructs
a propagator from raw orbital elements without needing a TLE.

Public API
----------
    create_satellite(h_orbit, inclination, raan, epoch) → EarthSatellite
    elevation_profile(satellite, gs_lat, gs_lon, gs_alt_m,
                      t_start, t_end, dt) → list[Geometry]
    usable_window(profile, theta_el_min_deg, t_acq_s)
        → (usable_profile, T_pass)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84
from sgp4.api import Satrec, WGS84

from core.types import Geometry


# Skyfield timescale — loaded once (downloads leap-second data on first use)
_TS = load.timescale()


# ---------------------------------------------------------------------------
# Satellite construction
# ---------------------------------------------------------------------------

def create_satellite(
    h_orbit: float,
    inclination: float,
    raan: float = 0.0,
    epoch: datetime | None = None,
) -> EarthSatellite:
    """Construct a Skyfield EarthSatellite from Keplerian elements.

    Parameters
    ----------
    h_orbit
        Orbital altitude above WGS84 ellipsoid (m).
    inclination
        Orbital inclination (degrees).
    raan
        Right ascension of the ascending node (degrees).
    epoch
        UTC epoch for the TLE.  Defaults to 2025-01-01 if not given.

    Returns
    -------
    EarthSatellite
        Skyfield satellite object suitable for topocentric computations.
    """
    if epoch is None:
        epoch = datetime(2025, 1, 1, tzinfo=timezone.utc)

    R_earth_km = 6378.137               # WGS84 semi-major axis (km)
    h_km = h_orbit / 1e3
    a_km = R_earth_km + h_km           # semi-major axis
    mu_km3_s2 = 398600.4418            # GM (km³/s²)
    n_rad_per_min = math.sqrt(mu_km3_s2 / a_km**3) * 60.0  # mean motion (rad/min)

    # Epoch expressed as days from 1949-12-31 (sgp4 convention)
    epoch_j2000 = (epoch - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds()
    epoch_jd = 2451545.0 + epoch_j2000 / 86400.0

    sat = Satrec()
    sat.sgp4init(
        WGS84,
        "i",                   # 'i' = improved mode
        0,                     # satnum
        epoch_jd - 2433281.5,  # epoch (days from 1949-12-31 00:00 UT)
        0.0,                   # bstar drag term (negligible for short passes)
        0.0,                   # ndot (not used in SGP4)
        0.0,                   # nddot (not used in SGP4)
        0.0,                   # eccentricity (circular orbit)
        0.0,                   # argument of perigee (irrelevant for circular)
        math.radians(inclination),
        0.0,                   # mean anomaly (arbitrary initial phase)
        n_rad_per_min,
        math.radians(raan),
    )
    return EarthSatellite.from_satrec(sat, _TS)


# ---------------------------------------------------------------------------
# Elevation profile
# ---------------------------------------------------------------------------

def elevation_profile(
    satellite: EarthSatellite,
    gs_lat: float,
    gs_lon: float,
    gs_alt_m: float,
    t_start: datetime,
    t_end: datetime,
    dt: float = 1.0,
) -> list[Geometry]:
    """Compute a time-series of Geometry objects for a satellite pass.

    Parameters
    ----------
    satellite
        Skyfield EarthSatellite (from create_satellite or a TLE).
    gs_lat, gs_lon
        Ground station geodetic coordinates (degrees; lon positive = east).
    gs_alt_m
        Ground station altitude above WGS84 ellipsoid (m).
    t_start, t_end
        UTC start and end of the window to evaluate.
    dt
        Timestep (seconds).  1 s is adequate for all LEO QKD scenarios.

    Returns
    -------
    list[Geometry]
        One Geometry per timestep.  Negative-elevation entries are included —
        filter them with usable_window() if needed.
    """
    gs = wgs84.latlon(gs_lat, gs_lon, elevation_m=gs_alt_m)

    total_s = (t_end - t_start).total_seconds()
    n_steps = max(1, int(round(total_s / dt)) + 1)
    offsets_s = np.linspace(0.0, total_s, n_steps)

    # Build Skyfield time array
    t0_sf = _TS.from_datetime(t_start.replace(tzinfo=timezone.utc)
                              if t_start.tzinfo is None else t_start)
    times = _TS.tt_jd(t0_sf.tt + offsets_s / 86400.0)

    diff = satellite - gs
    topo = diff.at(times)
    alt_obj, _az, dist_obj = topo.altaz()

    alt_deg = alt_obj.degrees        # ndarray
    dist_m  = dist_obj.m             # ndarray

    geometries: list[Geometry] = []
    for i in range(n_steps):
        theta_el = math.radians(float(alt_deg[i]))
        zeta     = math.pi / 2.0 - theta_el
        L        = float(dist_m[i])

        # Approximate orbital altitude: slant range projected to radial component.
        # For a circular orbit this is consistent with h_orbit used at construction.
        # Exact value isn't critical — only used for alpha*L in Beer-Lambert.
        h_orb = L * math.cos(max(0.0, zeta)) if zeta < math.pi / 2.0 else 400e3

        geometries.append(Geometry(
            theta_el=theta_el,
            L=L,
            zeta=zeta,
            h_orbit=h_orb,
        ))

    return geometries


# ---------------------------------------------------------------------------
# Usable window extraction
# ---------------------------------------------------------------------------

def usable_window(
    profile: list[Geometry],
    theta_el_min_deg: float,
    t_acq_s: float,
    dt: float = 1.0,
) -> tuple[list[Geometry], float]:
    """Extract the usable sub-window from an elevation profile.

    Applies two cuts:
        1. Elevation ≥ θ_el,min (link geometry constraint).
        2. First t_acq seconds after rising above θ_el,min are excluded
           (acquisition time before quantum transmission begins).

    Also rejects timesteps where ζ > 80° (Rytov integral diverges in
    evaluate_point).

    Parameters
    ----------
    profile
        Full elevation profile from elevation_profile().
    theta_el_min_deg
        Minimum elevation angle (degrees).
    t_acq_s
        Acquisition time to skip at the start of the visible window (seconds).
    dt
        Timestep used when building the profile (seconds).

    Returns
    -------
    usable_profile
        Sub-list of Geometry objects within the usable window.
    T_pass
        Usable pass duration in seconds.
    """
    theta_el_min_rad = math.radians(theta_el_min_deg)
    zeta_max = math.radians(80.0)
    acq_steps = int(math.ceil(t_acq_s / dt))

    # Find contiguous window above elevation limit
    above = [g for g in profile if g.theta_el >= theta_el_min_rad]
    if not above:
        return [], 0.0

    # Skip acquisition period at start
    usable = above[acq_steps:]
    if not usable:
        return [], 0.0

    # Further filter: zenith angle < 80° (evaluator guard)
    usable = [g for g in usable if g.zeta < zeta_max]

    T_pass = len(usable) * dt
    return usable, T_pass
