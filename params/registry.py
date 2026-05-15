"""
params/registry.py — Parameter storage, validation, and typed-config construction.

The registry holds a flat key→value store augmented with metadata from
definitions.py.  It provides:

    set / get           — typed read/write with bounds checking
    validate            — check all non-None values are in bounds
    downstream          — which parameters depend on a given one (propagation)
    build_channel       — construct ChannelConfig from stored values
    build_source        — construct SourceConfig
    build_post_processing — construct PostProcessingConfig
    to_dict             — snapshot for logging / serialisation

Physics modules are always called with the typed dataclasses, never with the
registry directly.  This separation means you can swap the registry without
touching any physics code.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any

from params.definitions import PARAM_DEFS, ParamDef, Status
from core.types import ChannelConfig, PostProcessingConfig, SourceConfig
from physics.detector import DetectorModel

logger = logging.getLogger(__name__)


class ParameterRegistry:
    """Flat key→value store with metadata, bounds checking, and config construction."""

    def __init__(self, scenario: str = "baseline") -> None:
        self._scenario = scenario
        # Initialise from defaults in definitions (skipping None-default params)
        self._values: dict[str, Any] = {}
        for name, pdef in PARAM_DEFS.items():
            if pdef.default is not None:
                self._values[name] = pdef.default

        # String parameters stored separately (bounds don't apply)
        self._str_values: dict[str, str] = {
            "ec_algorithm": "ldpc",
        }

    # ── Core accessors ────────────────────────────────────────────────────────

    def set(self, name: str, value: Any) -> None:
        """Set a parameter value with bounds checking."""
        if name not in PARAM_DEFS:
            raise KeyError(f"Unknown parameter: {name!r}.  Check params/definitions.py.")

        pdef = PARAM_DEFS[name]

        if isinstance(value, str):
            # PyYAML parses e.g. "20.0e6" (no sign) as a string, not a float.
            # Try numeric coercion for parameters that have numeric bounds.
            if pdef.bounds is not None:
                try:
                    value = float(value)
                except ValueError:
                    pass
            if isinstance(value, str):
                self._str_values[name] = value
                return

        if pdef.bounds is not None:
            lo, hi = pdef.bounds
            if not (lo <= value <= hi):
                raise ValueError(
                    f"Parameter {name!r} = {value} is outside bounds [{lo}, {hi}]."
                )
        self._values[name] = value

    def get(self, name: str) -> Any:
        """Return the current value of a parameter."""
        if name in self._str_values:
            return self._str_values[name]
        if name in self._values:
            return self._values[name]
        pdef = PARAM_DEFS.get(name)
        if pdef is None:
            raise KeyError(f"Unknown parameter: {name!r}.")
        if pdef.default is not None:
            return pdef.default
        raise KeyError(f"Parameter {name!r} has no value and no default.")

    def update(self, overrides: dict[str, Any]) -> None:
        """Apply a dict of overrides (used by scenario system)."""
        for k, v in overrides.items():
            self.set(k, v)

    # ── Validation ───────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = ok)."""
        errors: list[str] = []
        for name, pdef in PARAM_DEFS.items():
            if name not in self._values:
                continue
            val = self._values[name]
            if pdef.status == Status.TBD:
                errors.append(f"{name}: status is TBD — must be resolved before simulation.")
            if pdef.bounds is not None:
                lo, hi = pdef.bounds
                if not (lo <= val <= hi):
                    errors.append(f"{name} = {val} outside bounds [{lo}, {hi}].")

        # Cross-parameter checks
        try:
            if self._values.get("nu", 0.0) >= self._values.get("mu", 1.0):
                errors.append("nu must be strictly less than mu.")
            if self._values.get("P_mu", 0.0) + self._values.get("P_nu", 0.0) > 1.0:
                errors.append("P_mu + P_nu must be ≤ 1 (P_vac would be negative).")
            fc = self._values.get("f_clock", 0.0)
            td = self._values.get("tau_dead", 1.0)
            if fc > 1.0 / td:
                errors.append(
                    f"f_clock={fc:.2e} Hz exceeds dead-time limit 1/τ_d={1/td:.2e} Hz."
                )
        except Exception:
            pass

        return errors

    # ── Dependency queries ───────────────────────────────────────────────────

    def downstream(self, name: str) -> list[str]:
        """Return parameters that list *name* in their depends_on field."""
        return [
            k for k, pdef in PARAM_DEFS.items()
            if name in pdef.depends_on
        ]

    def upstream(self, name: str) -> list[str]:
        """Return the direct dependencies of *name*."""
        return PARAM_DEFS[name].depends_on if name in PARAM_DEFS else []

    # ── Typed config builders ─────────────────────────────────────────────────

    def build_channel(self) -> ChannelConfig:
        """Construct a ChannelConfig from current registry values."""
        g = self._values
        return ChannelConfig(
            eta_tx=g["eta_tx"],
            lambda_=g["lambda_"],
            w0=g["w0"],
            D_rx=g["D_rx"],
            eta_rx=g["eta_rx"],
            alpha=g["alpha"],
            Cn2_0=g["Cn2_0"],
            v_wind=g.get("v_wind", 21.0),
            h0=g.get("h0", 0.0),
            H_max=g.get("H_max", 20e3),
            H_bg=g.get("H_bg", 0.0),
            Omega_FOV=g.get("Omega_FOV", 0.0),
            delta_lambda=g.get("delta_lambda", 1e-9),
            theta_pnt=g.get("theta_pnt", 0.0),
            sigma_pnt=g.get("sigma_pnt", 1e-6),
            e_opt=g.get("e_opt", 0.03),
        )

    def build_source(self) -> SourceConfig:
        """Construct a SourceConfig from current registry values."""
        g = self._values
        return SourceConfig(
            mu=g["mu"],
            nu=g["nu"],
            P_mu=g["P_mu"],
            P_nu=g["P_nu"],
            P_X=g["P_X"],
            f_clock=g["f_clock"],
        )

    def build_post_processing(self) -> PostProcessingConfig:
        """Construct a PostProcessingConfig from current registry values."""
        g = self._values
        return PostProcessingConfig(
            r_PE=g.get("r_PE", 0.10),
            ec_algorithm=self._str_values.get("ec_algorithm", "ldpc"),
            epsilon_PA=g.get("epsilon_PA", 1e-10),
            rf_bandwidth=g.get("rf_bandwidth", 10e6),
        )

    def build_detector(self) -> DetectorModel:
        """Construct a DetectorModel from current registry values."""
        g = self._values
        f_clock = g.get("f_clock", 100e6)
        delta_t = g.get("delta_t", 1.0 / f_clock)
        dcr     = g.get("dark_count_rate", 500.0)
        p_d     = dcr / f_clock   # dark count probability per gate
        return DetectorModel(
            eta_det=g.get("eta_det", 0.60),
            p_d=p_d,
            tau_d=g.get("tau_dead", 50e-9),
            delta_t=delta_t,
        )

    # ── Snapshot / serialisation ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return all current values as a plain dict (for logging, JSON export)."""
        out = dict(self._values)
        out.update(self._str_values)
        return out

    def __repr__(self) -> str:
        return f"ParameterRegistry(scenario={self._scenario!r}, n_params={len(self._values)})"
