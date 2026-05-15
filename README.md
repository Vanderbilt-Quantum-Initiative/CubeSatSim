# CubeSatSim — VQI QKD Link Budget Simulation

A first-principles simulation framework for a CubeSat quantum key distribution (QKD) downlink. Models the complete chain from hardware parameters to secret key bits per pass, built for design-space exploration and parameter sensitivity analysis.

**Protocol:** Decoy-state BB84, weak coherent pulse downlink (CubeSat = Alice/transmitter, OGS = Bob/receiver)  
**Mission:** VQI CubeSat QKD Demonstrator

---

## What it does

Given a set of hardware and environmental parameters, the simulator:

1. Computes the optical link budget (diffraction, pointing, atmospheric, turbulence losses)
2. Models atmospheric fading using log-normal or Gamma-Gamma statistics
3. Calculates fading-averaged detection observables (gain Q, QBER E) for signal, decoy, and vacuum intensities
4. Applies Lo-Ma-Chen decoy-state bounds (asymptotic or finite-key Hoeffding-corrected)
5. Extracts the GLLP secret key fraction R
6. Applies Tomamichel 2012 finite-key correction to get composably secure ℓ_finite per pass
7. Integrates all of the above over a real satellite pass geometry via Skyfield/SGP4

The binding mission success criterion is **ℓ_finite > 0** — not QBER alone.

---

## Quick start

```bash
pip install numpy scipy matplotlib skyfield sgp4
```

```python
from datetime import datetime, timezone
from params.registry import ParameterRegistry
from params.scenarios import get_scenario
from orbit.pass_sim import simulate_pass

reg = ParameterRegistry()
reg.update(get_scenario("optimistic"))   # or "vqi_400km", "conservative", "bourgoin_2013"

result = simulate_pass(
    reg,
    t_start=datetime(2025, 1, 1, 7, 0, tzinfo=timezone.utc),
    t_end=datetime(2025, 1, 1, 8, 30, tzinfo=timezone.utc),
    decoy_mode="finite",   # composably secure; use "asymptotic" for design exploration
)

print(f"T_pass    = {result.T_pass:.0f} s")
print(f"n_sifted  = {result.n_sifted:.2e} bits")
print(f"E_mu      = {result.E_mu_weighted*100:.2f} %")
print(f"R         = {result.key_rate.R:.4e} bits/pulse")
print(f"ℓ_finite  = {result.ell_finite:.0f} bits")
print(f"go        = {result.go}")
```

### Generate all plots

```bash
python viz/pass_sim_plots.py        # pass profile, loss waterfall, key budget, scenario comparison
python analysis/sensitivity.py      # tornado plots (ℓ_finite elasticity per parameter)
python analysis/sweep.py            # 1D/2D parameter sweeps
python viz/atmosphere_plots.py      # Hufnagel-Valley profiles, Rytov variance
python viz/decoy_plots.py           # decoy bound tightness vs PE sample size
python viz/keyrate_plots.py         # binary entropy, key fraction, finite-key length
python viz/post_processing_plots.py # EC efficiency, sifting funnel, PE tradeoff
```

Plots are saved to `viz/out/<module>/`.

### Run validation tests

```bash
python -m pytest tests/test_validation.py -v
```

15 tests cross-validating against Bourgoin et al. (2013), *New J. Phys.* 15, 023006.

---

## Repository structure

```
CubeSatSim/
├── core/
│   ├── types.py            Shared dataclasses (Geometry, ChannelConfig, PassResult, …)
│   └── evaluator.py        Point evaluator — composes full physics chain at one geometry
│
├── physics/
│   ├── atmosphere.py       Hufnagel-Valley Cn² profile, Beer-Lambert attenuation
│   ├── turbulence.py       Rytov variance, log-normal and Gamma-Gamma fading models
│   ├── link_loss.py        Diffraction loss, pointing loss, loss budget assembly
│   ├── source.py           WCP source model, QRNG
│   ├── detector.py         Single-photon detector model (efficiency, dark counts, dead time)
│   ├── detection.py        Fading-averaged gain Q and QBER E (Jensen-correct integrals)
│   ├── decoy.py            Lo-Ma-Chen decoy bounds — asymptotic and finite-key Hoeffding
│   ├── keyrate.py          GLLP key fraction, finite-key length (Tomamichel 2012)
│   └── post_processing.py  Sifting, PE split, EC efficiency, classical bandwidth check
│
├── orbit/
│   ├── geometry.py         Skyfield/SGP4 wrapper — elevation profile, usable window
│   └── pass_sim.py         Pass integrator, StandardAccumulation, simulate_pass()
│
├── params/
│   ├── definitions.py      All parameter definitions (symbol, unit, owner, bounds, default)
│   ├── registry.py         Parameter store with bounds checking and typed config builders
│   └── scenarios.py        Named scenario overrides (vqi_400km, optimistic, conservative, …)
│
├── analysis/
│   ├── sensitivity.py      Normalised elasticity (central finite differences) + tornado plot
│   └── sweep.py            1D/2D parameter sweeps + contour heatmaps
│
├── viz/
│   ├── atmosphere_plots.py
│   ├── link_loss_plots.py
│   ├── detection_plots.py
│   ├── decoy_plots.py
│   ├── keyrate_plots.py
│   ├── post_processing_plots.py
│   ├── source_plots.py
│   ├── detector_plots.py
│   └── pass_sim_plots.py
│
├── tests/
│   └── test_validation.py  15-test Bourgoin 2013 cross-validation suite
│
└── docs/
    ├── Plan.md             Architecture design document
    └── Full Budget v1.3.md Physics reference (equations, model derivations)
```

---

## Physics chain

```
Hardware & environment parameters
        │
        ▼
Stage 1 — Static transmissivity η₀
        η₀ = η_tx · η_atm · η_diff · η_pnt · η_rx
        │
        ▼
Stage 2 — Fading distribution P(η)
        Slant-path Rytov variance → log-normal (σ²_R < 0.75) or Gamma-Gamma
        │
        ▼
Stage 3 — Fading-averaged observables
        Q_μ = ∫[1−(1−Y₀)e^{−ημ}] P(η) dη        (Jensen-correct: Q < Q(⟨η⟩))
        E_μ = ∫ QBER(η) · P(η) dη / Q_μ
        │
        ▼
Stage 4 — Decoy-state bounds (Ma et al. 2005)
        Q₁^L, e₁^U  from signal + weak-decoy + vacuum observables
        Finite-key: Hoeffding correction on pulse counts (not detection counts)
        │
        ▼
Stage 5 — GLLP key fraction
        R = q · [Q₁(1 − H₂(e₁)) − Q_μ · f_EC · H₂(E_μ)]
        │
        ▼
Stage 6 — Post-processing & finite-key correction
        ℓ_finite = max(0,  n·R − 4√(n·log₂(6/ε)) − log₂(2/ε))
```

---

## Scenarios

| Scenario | h_orbit | D_rx | f_clock | ℓ_finite (typical good pass) |
|----------|---------|------|---------|-------------------------------|
| `conservative` | 400 km | 0.5 m | 50 MHz | 0 (AEP-limited) |
| `vqi_400km` | 400 km | 1.0 m | 100 MHz | 0 (margin) |
| `optimistic` | 400 km | 1.5 m | 100 MHz | ~1.6 Mbit |
| `bourgoin_2013` | 600 km | 0.5 m | 300 MHz | validation only |

"Typical good pass" = 54° maximum elevation, 07:00 UTC, Nashville OGS (36.1°N, 86.7°W).

The `vqi_400km` scenario sits right at the AEP breakeven boundary — it needs either a slightly larger aperture, a higher-elevation pass, or an SNSPD (to avoid the 100 MHz > 1/τ_d dead-time violation flagged at runtime).

---

## Key design decisions

**Fading-averaged integrals, not mean-transmissivity approximations.**  
Computing Q and E at ⟨η⟩ violates Jensen's inequality. Both are integrated over P(η) numerically.

**Finite decoy bounds use pulse counts for Hoeffding, detection counts for QBER.**  
Q_μ = detections/pulses → Bernoulli trials are pulses. E_ν = errors/detections → trials are detections. Using detection counts for Q inflates the slack by ~√(1/Q) ≈ 45×, collapsing Q₁^L to zero.

**f_EC is block-size-dependent, not a constant.**  
Cascade: ~1.16 at n=10⁵; LDPC: ~1.05 at n≫10⁶. Using a fixed f_EC overestimates key yield at short block lengths typical of satellite passes.

**No custom orbital mechanics.**  
Skyfield + SGP4 handle propagation, WGS84 ellipsoid, and coordinate transforms. For unlaunched satellites, `sgp4.Satrec.sgp4init` constructs a propagator from orbital elements without a TLE.

---

## Key sensitivity results

From the tornado analysis (optimistic scenario, ±5% central finite differences):

| Parameter | Elasticity ε(ℓ) | Interpretation |
|-----------|----------------|----------------|
| D_rx (OGS aperture) | **+3.9** | Dominant lever — ground telescope, not satellite |
| w₀ (beam waist) | +2.6 | Optimal at w₀* = λ/(π·σ_pnt·√2) ≈ 13–19 cm |
| η_tx, η_rx | +2.4 each | Optical chain efficiency matters equally on both ends |
| σ_pnt (pointing jitter) | −1.3 | CubeSat ADCS is a hard constraint |
| μ (signal intensity) | +1.4 | Modest; optimised by decoy protocol |
| α (extinction) | −0.8 | Site selection matters |
| C_n²(0) (turbulence) | ~0 | Turbulence regime doesn't matter much at low σ²_R |

The beam waist optimum is real physics (Andrews & Phillips 2005 §4.3), confirmed by the Micius satellite (30 cm transmit aperture at 500 km).

---

## References

- Bourgoin et al. (2013), *New J. Phys.* 15, 023006 — feasibility analysis; primary validation target
- Ma, Qi, Zhao & Lo (2005), *PRA* 72, 012326 — closed-form decoy-state bounds
- Lo, Ma & Chen (2005), *PRL* 94, 230504 — decoy-state security proof
- Tomamichel, Lim, Gisin & Renner (2012) — composable finite-key length
- Andrews & Phillips (2005), *Laser Beam Propagation through Random Media* — fading models
- Liao et al. (2017), *Nature* 549, 43 — Micius satellite QKD demonstration
