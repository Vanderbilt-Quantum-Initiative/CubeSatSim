# VQI QKD Link Budget Simulation — Design Document

**Version:** 0.3 (final initial draft)
**Author:** Zach Smith
**Protocol:** Decoy-state BB84, weak coherent pulse downlink
**Companion documents:** Full Budget v2.1, Parameter Registry v2.1

---

## 1. Purpose and Scope

This document specifies the architecture for a simulation framework that:

1. Evaluates the complete equation chain from hardware/environmental parameters to secret key bits per pass ($\ell_{finite}$), modelling from first principles.
2. Integrates over realistic satellite pass geometry (time-varying elevation, slant range, pointing).
3. Exposes a parameter dependency graph so that changes to any input propagate correctly to all downstream outputs.
4. Supports comparative scenario analysis (altitude, form factor, site selection).
5. Provides a foundation for a future interactive interface where parameter updates produce immediate visual feedback on mission feasibility.

The simulation models the full physical chain: source → channel → detection → classical post-processing → secret key. Every stage is built from first principles and validated against published results. No empirical loss-vs-elevation fits are used; the channel model decomposes loss into physically distinct terms whose parameters are independently measurable.

### What This Is Not

- Not a real-time control system or flight software component.
- Not a replacement for SatQuMA or other validated finite-key toolchains. It is a design-space exploration and parameter sensitivity tool. Flight-grade key extraction requires a validated, peer-reviewed implementation.

### Abstraction Principle

The architecture splits physics into modules whose upgrade paths are independent. The governing question for every file boundary is: "Can I replace the model in this file without touching any other file?" Some files start with simple functions because the upgrade path (not the current complexity) determines the boundary.

---

## 2. Core Design Decisions

### 2.1 Point Evaluator + Pass Integrator

The binding mission criterion is $\ell_{finite} > 0$ per pass. The architecture has two computational layers:

- **Point evaluator:** Given a fixed geometry and fixed parameters → compute $\eta_0$, $P(\eta)$, detection observables, decoy bounds, $R$. Stateless, deterministic, fast (~10 ms). Implemented as an explicit `evaluate_point()` in `core/evaluator.py`.
- **Pass integrator:** Given an orbital trajectory and ground station location → call point evaluator at each timestep → accumulate statistics → apply post-processing and finite-key correction → output $\ell_{finite}$.

Adding a step to the evaluation chain modifies `evaluator.py`, not `pass_sim.py`.

### 2.2 Parameter Dependency DAG

Parameters fall into four categories:

| Category | Examples | Behaviour |
|----------|----------|-----------|
| Independent inputs | $D_{rx}$, $\lambda$, $w_0$, $p_d$ | Set by user or team decision |
| Derived (static) | $A_{rx} = \pi(D_{rx}/2)^2$, $W(L)$ | Recompute on change |
| Coupled | $f_{clock}$ ↔ $\tau_d$ ↔ QRNG rate | Constrained jointly |
| Time-varying | $L(t)$, $\zeta(t)$, $\eta_{pnt}(t)$, $e_{opt}(t)$ | Functions of pass geometry |

The DAG enables ripple analysis ("if I change $w_0$, what downstream quantities change?") and sensitivity analysis ("which inputs does $\ell_{finite}$ depend on most?").

#### Two-Tier Analysis Model

- **Tier 1 (DAG, fast):** Point evaluation at a reference geometry (e.g., zenith). Instant feedback for interactive sliders.
- **Tier 2 (simulation, slow):** Full pass integration. Required for accurate $\ell_{finite}$. Triggered explicitly.

Pass-level outputs ($n$, $\ell_{finite}$) are registered as Tier 2 derived parameters in the DAG — reachable from any input, but recomputation requires running the simulation.

### 2.3 Scenario System

Scenarios are deltas from a baseline parameter set: 400 km/3U, 400 km/6U, 550 km/3U, 550 km/6U, plus custom configurations. Unoverridden parameters inherit from the baseline.

### 2.4 Language and Stack

**Core simulation:** Python. scipy (quadrature, Bessel functions, optimisation), numpy, matplotlib.

**Orbital mechanics:** `skyfield` + `sgp4` (both MIT-licensed). These handle SGP4 propagation, coordinate frame transforms (TEME → ITRF → topocentric), WGS84 ellipsoid, and satellite-ground station geometry. No custom orbital mechanics code.

**Interface (future):** React frontend via FastAPI. First deliverable is the simulation library with CLI/notebook interface.

**Performance:** Point evaluator <10 ms. Main bottleneck: numerical integration of QBER and gain over $P(\eta)$ in `detection.py`.

---

## 3. Module Architecture

```
vqi_sim/
├── core/
│   ├── types.py           # All shared data structures
│   └── evaluator.py       # Composes full chain for a single geometry
├── params/
│   ├── registry.py        # Parameter DAG, storage, invalidation
│   ├── definitions.py     # All parameter definitions (metadata, bounds, owners)
│   └── scenarios.py       # Scenario management (baseline + overrides)
├── physics/
│   ├── atmosphere.py      # Cn2 profiles (H-V + injectable), atmospheric attenuation
│   ├── turbulence.py      # Rytov integral, fading model construction (FadingModel)
│   ├── link_loss.py       # Diffraction, pointing loss, transmissivity → LossBudget
│   ├── source.py          # Source model, preparation probabilities, QRNG constraints
│   ├── detector.py        # Detector model (dark counts, efficiency, pile-up interface)
│   ├── detection.py       # Fading-averaged gain and QBER (both integrated over P(η))
│   ├── decoy.py           # Decoy-state bounds (asymptotic + finite-sample)
│   ├── keyrate.py         # GLLP key fraction, finite-key correction
│   └── post_processing.py # Sifting, parameter estimation, EC model, bandwidth check
├── orbit/
│   ├── geometry.py        # Thin wrapper around Skyfield for pass geometry
│   └── pass_sim.py        # Pass integrator + AccumulationStrategy
├── analysis/
│   ├── sensitivity.py     # Parameter sensitivity / Jacobian
│   ├── sweep.py           # Parameter sweeps (1D, 2D)
│   └── compare.py         # Multi-scenario comparison
├── viz/
│   ├── budget.py          # Link budget waterfall plots
│   ├── pass_profile.py    # Per-pass time series
│   └── sensitivity.py     # Tornado / spider plots
├── cli.py
└── tests/
    ├── test_atmosphere.py
    ├── test_turbulence.py
    ├── test_link_loss.py
    ├── test_source.py
    ├── test_detector.py
    ├── test_detection.py
    ├── test_decoy.py
    ├── test_keyrate.py
    ├── test_post_processing.py
    ├── test_evaluator.py
    ├── test_pass_sim.py
    └── test_validation.py
```

### 3.1 `core/types.py` — Shared Data Structures

```python
@dataclass
class Geometry:
    theta_el: float    # elevation angle (rad)
    L: float           # slant range (m)
    zeta: float        # zenith angle (rad)
    h_orbit: float     # orbital altitude (m)

@dataclass
class LossBudget:
    eta_tx: float      # transmitter optics
    eta_atm: float     # atmospheric attenuation
    eta_diff: float    # diffraction / beam spreading
    eta_pnt: float     # pointing loss
    eta_rx: float      # receiver optics
    eta_0: float       # product of above
    def to_db_dict(self) -> dict[str, float]: ...

@dataclass
class DetectionResult:
    Q: float           # fading-averaged gain
    E: float           # fading-averaged QBER
    intensity: float   # which intensity (μ, ν, or 0)
    n_counts: float    # expected counts in this timestep (for accumulation)

@dataclass
class DecoyBounds:
    Q1_lower: float
    e1_upper: float
    mode: str          # "asymptotic" or "finite"

@dataclass
class KeyRateResult:
    R: float           # key fraction per pulse (may be negative)
    skbr: float        # bits/second
    ell_finite: float  # finite-key bits (clamped ≥ 0)

@dataclass
class PostProcessingResult:
    n_sifted: float
    n_PE: float
    n_key: float
    f_EC: float
    leak_EC: float
    classical_data_volume: float  # total classical bits exchanged
    classical_rounds: int
    ec_feasible: bool

@dataclass
class LinkState:
    """Complete evaluation state at a single geometry."""
    geometry: Geometry
    loss_budget: LossBudget
    fading: 'FadingModel'
    detections: dict[str, DetectionResult]  # "signal", "decoy", "vacuum"
    decoy_bounds: DecoyBounds | None        # None at per-timestep level
    key_rate: KeyRateResult | None          # None at per-timestep level

@dataclass
class PassResult:
    time: np.ndarray
    elevation: np.ndarray
    eta_0: np.ndarray
    qber_instant: np.ndarray
    R_instant: np.ndarray
    cumulative_n: np.ndarray

    n_sifted: float
    E_mu_weighted: float
    ell_finite: float
    T_pass: float
    go: bool                  # ell_finite > 0

    post_processing: PostProcessingResult
    jensen_gap: float         # ⟨QBER(η)⟩ − QBER(⟨η⟩), should be > 0
    gain_gap: float           # Q_μ(⟨η⟩) − ⟨Q_μ(η)⟩, should be > 0
```

### 3.2 `core/evaluator.py` — Point Evaluator

```python
def evaluate_point(
    geometry: Geometry,
    params: dict,
    source: SourceConfig,
    detector: DetectorModel,
    fading_model: FadingModel | None = None,
    cn2_profile: Cn2Profile | None = None,
) -> LinkState:
    """
    Full chain at a single geometry:
    1. link_loss → LossBudget
    2. atmosphere + turbulence → FadingModel (if not precomputed)
    3. For each intensity (μ, ν, vacuum):
         detection → DetectionResult (gain AND QBER integrated over fading)
    4. Assemble → LinkState

    Decoy bounds and key rate are NOT computed here — they require
    accumulated statistics across the full pass. The pass simulator
    calls this per-timestep, accumulates, then runs decoy + keyrate.
    """
```

The multi-intensity detection loop is explicit. Each intensity from `SourceConfig` ($\mu$, $\nu$, vacuum) gets its own fading-averaged $Q$ and $E$. All three use the same `FadingModel` (same channel), different intensity parameter.

Note: decoy bounds and key rate move out of per-timestep evaluation and into the pass-level computation. This is a structural change from v0.2. Per-timestep, we only need detection observables. Decoy analysis and key extraction operate on accumulated pass statistics.

### 3.3 `params/` — Parameter Registry

**`registry.py`**

```
ParamDef:
    name: str
    symbol: str
    unit: str
    owner: str
    status: TBD | ESTIMATED | BASELINED | MEASURED
    bounds: (min, max) | None
    default: float | None
    derivation: Callable | None
    depends_on: list[str]
    tier: 1 | 2
    description: str
```

Key methods: `set`, `get`, `downstream`, `upstream`, `stale_tier2`, `to_dict`, `validate`.

**`definitions.py`** — All parameters from the Registry document, now including: preparation probabilities ($P_{\mu}$, $P_{\nu}$), basis choice probability ($P_X$), parameter estimation fraction ($r_{PE}$), EC algorithm choice, QRNG rate, and classical channel bandwidth.

**`scenarios.py`** — Scenarios as deltas from baseline.

### 3.4 `physics/atmosphere.py` — Atmospheric Models

**`Cn2Profile` Protocol:**

```python
class Cn2Profile(Protocol):
    def __call__(self, h: float) -> float: ...

def hufnagel_valley(Cn2_0: float, v: float = 21.0) -> Cn2Profile: ...
def from_measurements(altitudes: np.ndarray, values: np.ndarray) -> Cn2Profile: ...
```

`atmospheric_attenuation(alpha, L) → eta_atm` — Beer-Lambert.

### 3.5 `physics/turbulence.py` — Turbulence and Fading

**`FadingModel` Protocol:**

```python
class FadingModel(Protocol):
    def pdf(self, eta: float) -> float: ...
    def mean_eta(self) -> float: ...
    def integrate(self, f: Callable[[float], float]) -> float: ...
```

Implementations: `LogNormalFading` (standard quadrature), `GammaGammaFading` (log-space transform for Bessel stability).

`rytov_variance(lambda_, zeta, h0, cn2_profile: Cn2Profile, H_max=20e3) → sigma_R2` — Slant-path integral via adaptive quadrature. Takes `Cn2Profile`, not raw parameters.

`select_fading_model(sigma_R2, eta_0, threshold=0.75) → FadingModel` — Configurable threshold.

### 3.6 `physics/link_loss.py` — Loss Budget

`diffraction_loss(lambda_, w0, L, D_rx) → eta_diff` — Gaussian beam clipping.

`pointing_loss(theta_pnt, sigma_pnt) → eta_pnt` — Static Gaussian model.

`compute_loss_budget(eta_tx, eta_atm, eta_diff, eta_pnt, eta_rx) → LossBudget` — Returns per-term breakdown, not scalar.

### 3.7 `physics/source.py` — Source Model and QRNG

```python
@dataclass
class SourceConfig:
    mu: float           # signal intensity (photons/pulse)
    nu: float           # weak decoy intensity
    # vacuum is implicit (mu_3 = 0)
    P_mu: float         # probability of sending signal pulse
    P_nu: float         # probability of sending weak decoy
    # P_vac = 1 - P_mu - P_nu
    P_X: float          # basis choice probability (0.5 for standard BB84, ~0.9 for efficient)
    f_clock: float      # pulse rate (Hz)

    @property
    def P_vac(self) -> float:
        return 1.0 - self.P_mu - self.P_nu

    def sifting_factor(self) -> float:
        """Effective sifting factor for efficient BB84."""
        # For standard BB84: q = 0.5
        # For efficient BB84 with asymmetric basis: q ≈ P_X
        ...

@dataclass
class QRNGModel:
    rate: float             # random bits per second
    bits_per_pulse: int     # bits consumed per pulse (~3: basis + intensity)

    def max_clock_rate(self) -> float:
        return self.rate / self.bits_per_pulse

    def validate(self, f_clock: float) -> tuple[bool, str]: ...
```

The source model makes preparation probabilities explicit throughout the chain. The total pulses sent per second is $f_{clock}$. Of those, a fraction $P_\mu$ are signal pulses. Only signal pulses in matching bases contribute to the sifted key.

QRNG validates that the random number generator can sustain the requested pulse rate. At $f_{clock} = 100$ MHz with 3 bits per pulse, the QRNG must produce $\geq 300$ Mbps. This is a hardware selection constraint.

### 3.8 `physics/detector.py` — Detector Model

```python
class DetectorModel:
    def __init__(self, eta_det, p_d, tau_d, delta_t, sigma_t=None):
        self.eta_det = eta_det
        self.p_d = p_d
        self.tau_d = tau_d
        self.delta_t = delta_t
        self.sigma_t = sigma_t

    def dark_count_rate(self) -> float: return self.p_d
    def effective_efficiency(self, count_rate: float = 0.0) -> float: return self.eta_det
    def max_clock_rate(self) -> float: return 1e9 / self.tau_d
    def validate_clock_rate(self, f_clock: float) -> tuple[bool, str]: ...
```

### 3.9 `physics/detection.py` — Fading-Averaged Observables (Stage 3)

Both gain and QBER integrate over the fading distribution. This corrects the v0.2 design where gain used the mean transmissivity.

**Why both must integrate:** A WCP source with intensity $\mu$ emits $n$ photons with Poisson probability $P(n|\mu) = \mu^n e^{-\mu}/n!$. Summing the per-photon-number yields over the Poisson weights gives $Q_\mu(\eta) = 1 - (1-Y_0)e^{-\eta\mu}$. In a fading channel, $\eta$ fluctuates and the observed gain is:

$$\langle Q_\mu \rangle = 1 - (1-Y_0)\int e^{-\eta\mu} P(\eta)\,d\eta$$

Since $e^{-\eta\mu}$ is convex in $\eta$, Jensen's inequality gives $\langle e^{-\eta\mu}\rangle > e^{-\langle\eta\rangle\mu}$, so the fading-averaged gain is *lower* than the gain computed at mean transmissivity. Using $Q_\mu = 1 - (1-Y_0)e^{-\langle\eta\rangle\mu}$ overestimates gain, which propagates errors into decoy bounds and sifted bit count.

```python
def expected_gain(fading: FadingModel, mu: float, Y_0: float) -> float:
    """Fading-averaged gain. Integrates over P(η)."""
    return fading.integrate(lambda eta: 1 - (1 - Y_0) * np.exp(-eta * mu))

def expected_qber(fading: FadingModel, mu: float, e_opt: float, Y_0: float) -> float:
    """Fading-averaged QBER. Integrates over P(η)."""
    return fading.integrate(lambda eta: instantaneous_qber(eta, mu, e_opt, Y_0))

def instantaneous_qber(eta: float, mu: float, e_opt: float, Y_0: float) -> float:
    """QBER at a fixed transmissivity η."""
    signal = eta * mu
    return (e_opt * signal + 0.5 * Y_0) / (signal + Y_0)

def noise_yield(detector: DetectorModel, H_bg, Omega_FOV, A_rx, 
                delta_lambda, eta_rx) -> float:
    return detector.dark_count_rate() + H_bg * Omega_FOV * A_rx * delta_lambda * detector.delta_t * eta_rx

def compute_detection(fading: FadingModel, detector: DetectorModel,
                      intensity: float, e_opt: float, Y_0: float) -> DetectionResult:
    Q = expected_gain(fading, intensity, Y_0)
    E = expected_qber(fading, intensity, e_opt, Y_0)
    return DetectionResult(Q=Q, E=E, intensity=intensity, n_counts=0.0)
```

**Validation built-in:** Every evaluation computes the Jensen gap for both gain and QBER. Both gaps must be positive (fading worsens both). Negative gap = bug.

### 3.10 `physics/decoy.py` — Decoy-State Bounds (Stage 4)

Two separate functions. Both consume `DetectionResult` objects.

`asymptotic_bounds(signal, decoy, vacuum, mu, nu, Y_0) → DecoyBounds`
: Lo-Ma-Chen / Ma et al. (2005) closed-form bounds. For design exploration.

`finite_bounds(signal, decoy, vacuum, mu, nu, Y_0, n_PE, confidence) → DecoyBounds`
: Statistically corrected bounds using Hoeffding/Chernoff. `n_PE` is the number of parameter estimation bits, not total sifted bits. Required for finite-key calculations.

The preparation probabilities affect the decoy bounds indirectly: the number of counts at each intensity available for parameter estimation is $n_{\mu_i}^{PE} = n_{PE} \cdot P_{\mu_i} \cdot Q_{\mu_i} / \sum_j P_{\mu_j} Q_{\mu_j}$ (proportional to the detection rate at each intensity). Tighter bounds at each intensity require more PE counts, which come at the cost of key generation bits.

### 3.11 `physics/keyrate.py` — Secret Key Rate (Stage 5)

`gllp_asymptotic(bounds, signal, q, f_EC) → float` — Key fraction $R$.

`finite_key_length(n_key, R, E_mu, f_EC, epsilon_PA) → float` — Tomamichel et al. (2012). Returns $\max(0, \cdot)$.

`binary_entropy(x) → float` — With edge case handling.

`compute_key_rate(bounds, signal, n_key, q, f_EC, f_clock, epsilon_PA) → KeyRateResult`

### 3.12 `physics/post_processing.py` — Classical Post-Processing

Owns everything after quantum measurement: sifting, parameter estimation, error correction, privacy amplification. Models the classical communication requirements.

```python
@dataclass
class PostProcessingConfig:
    r_PE: float                 # fraction of sifted bits for parameter estimation
    ec_algorithm: str           # "cascade" or "ldpc"
    epsilon_PA: float           # security parameter
    rf_bandwidth: float         # classical channel bandwidth (bits/s)

def sifting_yield(N_total: float, source: SourceConfig) -> float:
    """Sifted key bits from signal pulses only.
    n_sifted = N_total × P_μ × Q_μ × q
    Only signal-intensity pulses in matching bases contribute."""
    ...

def pe_split(n_sifted: float, r_PE: float) -> tuple[float, float]:
    """Split sifted bits into PE and key generation.
    Returns (n_PE, n_key) where n_key = n_sifted × (1 - r_PE)."""
    return n_sifted * r_PE, n_sifted * (1 - r_PE)

def ec_efficiency(n_key: float, E_mu: float, algorithm: str) -> float:
    """f_EC as a function of block size and error rate.
    NOT a constant. At n=10^5 with E_μ=5%:
      Cascade: ~1.16 (interactive, multiple rounds)
      LDPC: ~1.10 (one-way, but code design matters at short blocks)
    Asymptotic LDPC: ~1.05 (only valid at n >> 10^6)"""
    ...

def classical_bandwidth_check(
    n_sifted: float, E_mu: float, f_EC: float,
    algorithm: str, T_pass: float, rf_bandwidth: float
) -> tuple[bool, float, int]:
    """Check whether classical channel supports post-processing.
    Cascade: ~ceil(log2(1/E_μ)) round trips, each exchanging
             ~n × H_2(E_μ) bits of syndrome data.
    LDPC:    one-way syndrome of ~n × f_EC × H_2(E_μ) bits.
    Returns (feasible, total_data_volume_bits, round_trips)."""
    ...

def compute_post_processing(
    n_sifted: float, E_mu: float, source: SourceConfig,
    config: PostProcessingConfig, T_pass: float
) -> PostProcessingResult:
    """Full post-processing chain."""
    n_PE, n_key = pe_split(n_sifted, config.r_PE)
    f_EC = ec_efficiency(n_key, E_mu, config.ec_algorithm)
    leak = n_key * f_EC * binary_entropy(E_mu)
    feasible, data_vol, rounds = classical_bandwidth_check(
        n_sifted, E_mu, f_EC, config.ec_algorithm, T_pass, config.rf_bandwidth
    )
    return PostProcessingResult(
        n_sifted=n_sifted, n_PE=n_PE, n_key=n_key,
        f_EC=f_EC, leak_EC=leak,
        classical_data_volume=data_vol, classical_rounds=rounds,
        ec_feasible=feasible
    )
```

### 3.13 `orbit/geometry.py` — Pass Geometry via Skyfield

Thin wrapper around `skyfield` and `sgp4`. No custom orbital mechanics.

```python
from skyfield.api import load, EarthSatellite, wgs84
from sgp4.api import Satrec, WGS84

def create_satellite(h_orbit, inclination, raan=0.0, epoch=None) -> EarthSatellite:
    """Construct a satellite from orbital elements (for hypothetical designs).
    Uses sgp4.Satrec.sgp4init for direct element construction."""
    ...

def elevation_profile(satellite: EarthSatellite, gs_lat, gs_lon, gs_alt,
                      t_start, t_end, dt=1.0) -> list[Geometry]:
    """Compute time-series of Geometry objects using Skyfield.
    Handles coordinate transforms, Earth oblateness, and proper
    slant range computation automatically."""
    gs = wgs84.latlon(gs_lat, gs_lon, elevation_m=gs_alt)
    ts = load.timescale()
    times = ts.utc(...)  # generate time array
    diff = satellite - gs
    topo = diff.at(times)
    alt, az, distance = topo.altaz()
    # Convert to Geometry objects
    ...

def usable_window(profile, theta_el_min, t_acq) -> (t_start, t_end, T_pass): ...
```

Skyfield gives us J2-corrected SGP4 propagation, proper WGS84 ellipsoid geometry, and validated coordinate transforms. For hypothetical satellites (not yet launched), `sgp4.Satrec.sgp4init` constructs a satellite model from raw orbital elements without needing a TLE.

### 3.14 `orbit/pass_sim.py` — Pass Integration

**Accumulation Strategy:**

```python
class AccumulationStrategy(Protocol):
    def accumulate(self, dt: float, link_state: LinkState,
                   source: SourceConfig) -> None: ...
    def finalise(self) -> AccumulationResult: ...

@dataclass
class AccumulationResult:
    n_signal: float          # total signal-intensity detections
    n_decoy: float           # total decoy-intensity detections
    n_vacuum: float          # total vacuum detections
    E_mu_weighted: float     # detection-weighted signal QBER
    E_nu_weighted: float     # detection-weighted decoy QBER
    Q_mu_avg: float
    Q_nu_avg: float
```

**`StandardAccumulation`:** Per-timestep signal contribution is $\Delta n_\mu = f_{clock} \cdot P_\mu \cdot Q_\mu(t) \cdot q \cdot dt$. QBER is detection-weighted: $\bar{E}_\mu = \sum_t E_\mu(t) \cdot \Delta n_\mu(t) / n_\mu$. Decoy and vacuum counts accumulate similarly for parameter estimation.

**`simulate_pass`:**

```python
def simulate_pass(
    registry: ParameterRegistry,
    scenario: str | None = None,
    dt: float = 1.0,
    accumulation: AccumulationStrategy | None = None,
    decoy_mode: str = "finite",
) -> PassResult:
```

Steps:
1. Extract parameters; construct `SourceConfig`, `DetectorModel`, `Cn2Profile`, `PostProcessingConfig`.
2. Validate constraints: `detector.validate_clock_rate(source.f_clock)`, `qrng.validate(source.f_clock)`.
3. Compute elevation profile via Skyfield (`geometry.py`).
4. Determine usable window (after $t_{acq}$, above $\theta_{el,min}$).
5. At each timestep: compute `Geometry` → `evaluate_point()` → feed to accumulation.
6. Finalise accumulation → `AccumulationResult`.
7. Post-processing: `compute_post_processing()` → `PostProcessingResult`.
8. Decoy bounds using PE counts from accumulation result.
9. Key rate using $n_{key}$ from post-processing.
10. Assemble `PassResult`.

---

## 4. Data Flow

```
              ┌──────────────────────┐
              │  Parameter Registry  │
              │  (DAG + scenarios)   │
              └──────────┬───────────┘
                         │ extract params, construct models
                         ▼
              ┌──────────────────────┐
              │  Constraint Checks   │
              │  • f_clock vs τ_d    │
              │  • QRNG rate         │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  Pass Simulator      │
              │  (Skyfield geometry)  │
              └──────────┬───────────┘
                         │ for each timestep:
                         ▼
              ┌──────────────────────┐
              │  evaluate_point()    │
              │                      │
              │  1. link_loss.py     │
              │     → LossBudget     │
              │                      │
              │  2. atmosphere.py    │
              │     → Cn2Profile     │
              │                      │
              │  3. turbulence.py    │
              │     → FadingModel    │
              │                      │
              │  4. detection.py     │  ← called 3×: μ, ν, vacuum
              │     × detector.py    │     gain AND QBER integrated
              │     → DetectionResult│     over fading for each
              │                      │
              │  → LinkState         │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │ AccumulationStrategy │
              │ (weighted by P_μᵢ)   │
              └──────────┬───────────┘
                         │ finalise
                         ▼
              ┌──────────────────────┐
              │ post_processing.py   │
              │ • sifting yield      │
              │ • PE / key split     │
              │ • f_EC(n, E_μ)       │
              │ • bandwidth check    │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │ decoy.py (on PE data)│
              │ → DecoyBounds        │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │ keyrate.py           │
              │ → ℓ_finite           │
              └──────────┬───────────┘
                         │
                    PassResult
```

---

## 5. Numerical Considerations

### 5.1 Slant-Path Rytov Integral

Use `scipy.integrate.quad` (adaptive Gauss-Kronrod). The H-V profile's ground-level term ($e^{-h/100}$) concentrates almost all contribution in the first ~500 m. Validate against Andrews & Phillips (2005) and Bourgoin et al. (2013) Table 1.

### 5.2 Gamma-Gamma Fading Integration

`GammaGammaFading.integrate()` handles the Bessel-function divergence near $\eta = 0$ internally (log-space transform). Callers use `fading.integrate(f)`.

### 5.3 Jensen's Inequality — Gain and QBER

Both $Q_\mu$ and $E_\mu$ must be computed as integrals over $P(\eta)$, not evaluated at $\langle\eta\rangle$. The Jensen gaps (both must be positive) are logged on every evaluation:

- **QBER gap:** $\langle\text{QBER}(\eta)\rangle - \text{QBER}(\langle\eta\rangle) > 0$ (fading increases QBER)
- **Gain gap:** $Q_\mu(\langle\eta\rangle) - \langle Q_\mu(\eta)\rangle > 0$ (fading decreases gain)

A negative gap in either indicates a bug.

### 5.4 Finite-Key and Entropy Edge Cases

$\ell_{finite}$ clamped to $\max(0, \cdot)$. $H_2(x)$ uses `scipy.special.xlogy` for $x \in \{0, 1\}$.

---

## 6. Testing Strategy

### 6.1 Unit Tests

- **Limiting cases:** $\eta = 1$, $\eta = 0$, $\mu = 0$, zero turbulence.
- **Monotonicity:** $\eta_{diff} \uparrow$ with $D_{rx}$; QBER $\downarrow$ with $\eta$; $R \downarrow$ with $E_\mu$; $\ell_{finite} \uparrow$ with $n$.
- **Normalisation:** `fading.integrate(lambda eta: 1.0) ≈ 1.0` for all fading models.
- **Jensen consistency:** Both gain and QBER gaps positive across all test configurations.
- **Poisson consistency:** At $\mu = 0$, gain equals $Y_0$ and QBER equals $0.5$ (vacuum).
- **FadingModel conformance:** All implementations pass the same property-based suite.
- **Post-processing:** $f_{EC}(n, E_\mu) \geq 1$ always; feasibility check fails at known limits.
- **Source constraints:** QRNG validation fails when $f_{clock} > \text{rate}/\text{bits\_per\_pulse}$.

### 6.2 Cross-Validation

- **Bourgoin et al. (2013) Table 1:** Reproduce LEO downlink link budgets. Tests channel/loss modules.
- **SatQuMA comparison:** Run identical parameter sets (matching their deterministic channel model, i.e., no fading). Results should match when our fading is disabled. Differences when fading is enabled quantify the scintillation effect.
- **Analytical limits:** At zero turbulence ($\sigma_R^2 = 0$), fading-averaged and deterministic results must agree exactly.

### 6.3 Integration Tests

- Full pass: $\ell_{finite} > 0$ for feasible scenarios, $= 0$ for infeasible.
- 6U outperforms 3U; 400 km outperforms 550 km at matched parameters.
- Parameter DAG: correct downstream updates; Tier 2 marked stale.
- Classical bandwidth check: Cascade fails at high $f_{clock}$ with low RF bandwidth.
- PE fraction sweep: $\ell_{finite}$ has a maximum at some intermediate $r_{PE}$ (too low → loose decoy bounds, too high → insufficient key bits).

### 6.4 Regression Tests

Freeze baseline numbers. >1% shift in $\ell_{finite}$ on benchmark scenarios triggers review.

---

## 7. Interface Foundation

### What the Interface Needs

1. **Serialisable state:** Registry → JSON with parameter values, status, owner, dependencies, tier.
2. **Tier-aware recomputation:** Tier 1 instant, Tier 2 on explicit request.
3. **Ripple tracing:** (parameter, old, new, Δ%). Tier 2 entries show "stale."
4. **Loss budget breakdown:** `LossBudget.to_db_dict()` for waterfalls.
5. **Constraint flags:** QRNG, detector dead time, classical bandwidth — surfaced as warnings.

### Interface Sketch (not this deliverable)

- Left: parameter table by block/owner, status indicators, constraint warnings.
- Centre: visualisation (waterfall, pass profile, sensitivity tornado).
- Right: dependency graph (Tier 1 green, Tier 2 amber).
- Bottom: scenario selector and comparison.

---

## 8. Build Sequence

### Phase 1 — Core Simulation (end-to-end point evaluation)

1. `core/types.py`
2. `params/definitions.py` — all parameters including preparation probabilities, PE fraction, EC params, QRNG rate.
3. `params/registry.py` — DAG with tier metadata.
4. `physics/atmosphere.py` — H-V, attenuation, `Cn2Profile`.
5. `physics/turbulence.py` — Rytov, `FadingModel`, log-normal, Gamma-Gamma.
6. `physics/link_loss.py` — diffraction, pointing, `LossBudget`.
7. `physics/source.py` — `SourceConfig`, `QRNGModel`.
8. `physics/detector.py` — `DetectorModel`.
9. `physics/detection.py` — fading-averaged gain AND QBER, Jensen gap logging.
10. `physics/decoy.py` — asymptotic, then finite bounds.
11. `physics/keyrate.py` — GLLP, finite-key.
12. `physics/post_processing.py` — sifting, PE split, EC model, bandwidth check.
13. `core/evaluator.py` — `evaluate_point()`.
14. Unit tests for all physics modules.
15. Cross-validation against Bourgoin et al. (2013).

### Phase 2 — Pass Integration

16. `orbit/geometry.py` — Skyfield wrapper.
17. `orbit/pass_sim.py` — `AccumulationStrategy`, `StandardAccumulation`, `simulate_pass()`.
18. Integration tests.
19. `params/scenarios.py`.

### Phase 3 — Analysis and Visualisation

20. `analysis/sensitivity.py`, `analysis/sweep.py`, `analysis/compare.py`.
21. `viz/` — all modules.
22. `cli.py`.

### Phase 4 — Interface Foundation

23. JSON serialisation with tier metadata.
24. API endpoint (FastAPI).
25. Interface prototype.

---

## 9. Dependencies

| Package | Version | License | Purpose |
|---------|---------|---------|---------|
| numpy | ≥1.24 | BSD | Array operations |
| scipy | ≥1.10 | BSD | Quadrature, Bessel functions, optimisation |
| matplotlib | ≥3.7 | PSF | Validation and analysis plots |
| skyfield | ≥1.46 | MIT | Satellite propagation and ground station geometry |
| sgp4 | ≥2.22 | MIT | SGP4 propagation (used by Skyfield) |

All dependencies are permissively licensed (MIT/BSD).

---

## 10. Second-Order Additions (deferred)

Ordered roughly by expected impact on $\ell_{finite}$ accuracy:

1. **Protocol parameter optimisation** — scipy.optimize over ($\mu$, $\nu$, $P_\mu$, $P_\nu$, $P_X$, $r_{PE}$) to maximise $\ell_{finite}$. Required for meaningful parameter sweeps.
2. **Improved finite-key sampling bounds** — Kato's inequality / tighter hypergeometric bounds (Sidhu et al. 2023) replacing Hoeffding/Chernoff.
3. **$f_{EC}(n, E_\mu)$ empirical fit** — published efficiency curves for Cascade (Martinez-Mateo 2015) and LDPC (Elkouss 2009) rather than interpolated estimates.
4. **Detector afterpulsing** — correlated noise following a detection. Probability depends on time since last event and count rate. Modifies effective $Y_0$.
5. **Pile-up / deadtime correction** — `DetectorModel.effective_efficiency(count_rate)` via deadtime-modified Poisson.
6. **Source intensity uncertainty** — $\mu \to \mu \pm \delta\mu$. Propagates through decoy bounds (Sidhu et al. 2023).
7. **Turbulence regime blending** — weighted log-normal/Gamma-Gamma mixture for $\sigma_R^2 \sim 0.5$–$1.5$.
8. **Time-varying $e_{opt}$** — sinusoidal roll-induced polarisation rotation with compensation residual.
9. **Spectral filter Doppler accommodation** — $\Delta\lambda$ must cover full Doppler shift (~0.1–0.3 nm at 850 nm).
10. **Multi-pass key accumulation** — campaign-level simulation. Net yield per day/week including weather and authentication overhead.
11. **`lowtran` integration** — spectral atmospheric transmission replacing Beer-Lambert. For wavelength trade studies or twilight operations.
12. **Beam wander** — large-scale turbulence-induced centroid displacement. Modifies pointing error distribution.
13. **Finite aperture averaging** — receiver aperture smooths scintillation for $D_{rx} > r_0$. Reduces effective $\sigma_R^2$.
14. **Authentication key consumption** — pre-shared key overhead for classical channel authentication.

---

## 11. Known Architectural Decisions and Rationale

1. **First principles only.** No empirical loss-vs-elevation fits. The value is understanding *which* physical parameters drive loss and how subsystem-level choices affect the budget.
2. **Decoy bounds and key rate at pass level, not per-timestep.** The decoy method estimates $Q_1$ and $e_1$ from accumulated statistics across the pass. Computing per-timestep decoy bounds is physically meaningless (too few counts per timestep for statistical estimation).
3. **Fading-averaged gain.** The $Q_\mu$ formula implicitly uses Poisson source statistics. In a fading channel, the exponential factor must be averaged over $P(\eta)$, not evaluated at $\langle\eta\rangle$. This is a Jensen's inequality correction analogous to the QBER correction, but applied to gain.
4. **Skyfield over custom orbital mechanics.** Validated SGP4 implementation, proper coordinate transforms, WGS84 ellipsoid. Supports hypothetical satellites via `sgp4init`. No reason to reimplement.
5. **Post-processing as physics, not bookkeeping.** Error correction efficiency depends on block size and error rate. Classical bandwidth is a real constraint for CubeSat missions. These are not free parameters — they are modelled.