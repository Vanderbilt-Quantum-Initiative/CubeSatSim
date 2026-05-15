# Finite-Key Analysis Upgrade: AEP → EUR

**Scope:** `physics/keyrate.py`, `physics/decoy.py`
**Priority:** High — the AEP correction is the dominant penalty killing key yield at CubeSat block sizes
**Primary references:**
- Lim, Curty, Walenta, Xu & Zbinden (2014), PRA 89, 022307 — EUR-based finite-key bounds for decoy-state BB84
- Wiesemann, Krause, Tupkary, Lütkenhaus, Rusca & Walenta (2026), Quantum 10, 2037 — consolidated EUR proof resolving technical flaws in earlier analyses; the cleanest reference implementation
- Curràs-Lorenzo, Navarrete, Pereira & Tamaki (2021), arXiv:2101.12603 — random sampling theory vs Azuma's inequality
- Zapatero & Curty (2025), PRL, arXiv:2410.04095 — sharp finite statistics via tight hypergeometric bounds
- Kato (2020), arXiv:2002.04357 — concentration inequality for rare events

---

## 1. The Problem

The current `keyrate.py` uses the Tomamichel et al. (2012) finite-key formula:

$$\ell \leq n \cdot R - \Delta_{AEP} - \text{leak}_{EC} - \log_2(2/\varepsilon_{PA})$$

The $\Delta_{AEP}$ term scales as $O(\sqrt{n})$. It comes from bounding the smooth min-entropy $H_{min}^\epsilon(X^n|E^n)$ via the quantum Asymptotic Equipartition Property:

$$H_{min}^\epsilon \geq n \cdot H(X|E) - \sqrt{n} \cdot \Delta(\epsilon)$$

At $n = 10^5$ (plausible for a CubeSat pass), $\sqrt{n} \approx 316$. Multiplied by $\Delta(\epsilon)$ (typically several bits depending on the smoothing parameter), this correction consumes hundreds to thousands of key bits — often exceeding the asymptotic yield entirely, producing $\ell_{finite} = 0$ on passes where the physics would otherwise support key generation.

The AEP bound is asymptotically tight ($\Delta_{AEP}/n \to 0$ as $n \to \infty$) but at satellite-scale block sizes it is provably too pessimistic. Staffieri, Scala & Lupo (2026) demonstrate that the EUR-based approach gives strictly tighter key rates than the AEP approach for BB84 at all practically relevant block lengths, and that the AEP approach can fail to certify a positive key in regimes where the EUR approach certifies nonzero key.

---

## 2. The EUR Alternative

The entropic uncertainty relation (EUR) approach avoids the AEP entirely. Instead of converting between min-entropy and von Neumann entropy (which costs $\sqrt{n}$), it bounds Eve's information directly via the uncertainty relation for conjugate BB84 bases.

The EUR-based secure key length for decoy-state BB84 (Lim et al. 2014, as consolidated in Wiesemann et al. 2026) is:

$$\ell \leq s_{Z,0}^L + s_{Z,1}^L \left[1 - H_2(\bar{\phi}_Z^U)\right] - \text{leak}_{EC} - 6\log_2(2/\tilde{\epsilon}) - \log_2(2/\varepsilon_{PA})$$

where:

- $s_{Z,0}^L$ = lower bound on vacuum-state detections in the Z (key-generation) basis
- $s_{Z,1}^L$ = lower bound on single-photon detections in the Z basis
- $\bar{\phi}_Z^U$ = upper bound on the single-photon phase error rate in the Z basis
- $\text{leak}_{EC} = \lambda_{EC}$ = bits leaked during error correction (same as before: $n_Z \cdot f_{EC} \cdot H_2(E_Z)$)
- $\tilde{\epsilon}$ = smoothing / failure probability parameter (composed from sub-protocol security parameters)
- $\varepsilon_{PA}$ = privacy amplification security parameter

The critical difference: the finite-size corrections are logarithmic ($\log_2(2/\tilde{\epsilon})$), not $\sqrt{n}$. At $\tilde{\epsilon} = 10^{-10}$, $6\log_2(2/10^{-10}) \approx 200$ bits, compared to the AEP correction which is typically thousands of bits at $n = 10^5$.

The $\sqrt{n}$-scaling penalty has moved from the key length formula into the parameter estimation step — bounding $s_{Z,0}^L$, $s_{Z,1}^L$, and $\bar{\phi}_Z^U$ from observed data. This is where improved concentration inequalities (Kato, tight Chernoff, exact hypergeometric) make a difference.

---

## 3. Changes Required

### 3.1 `physics/keyrate.py` — Replace the key length formula

**Remove:** `finite_key_length()` using the AEP correction ($\Delta_{AEP} \sim \sqrt{n}$).

**Add:** `eur_key_length()` using the EUR formula.

```python
def eur_key_length(
    s_Z0_lower: float,      # vacuum contribution in Z basis (lower bound)
    s_Z1_lower: float,      # single-photon contribution in Z basis (lower bound)
    phi_Z_upper: float,     # phase error rate in Z basis (upper bound)
    leak_EC: float,         # error correction leakage
    epsilon: float,         # smoothing parameter
    epsilon_PA: float,      # privacy amplification parameter
) -> float:
    """EUR-based composable finite-key length.

    Finite-size corrections are O(log(1/ε)), not O(√n).
    """
    if phi_Z_upper >= 0.5:
        return 0.0

    key_bits = (
        s_Z0_lower
        + s_Z1_lower * (1.0 - binary_entropy(phi_Z_upper))
        - leak_EC
        - 6.0 * math.log2(2.0 / epsilon)
        - math.log2(2.0 / epsilon_PA)
    )
    return max(0.0, key_bits)
```

**Keep:** `gllp_asymptotic()` unchanged — it's still useful for quick design-space exploration without finite-key effects. The `compute_key_rate()` convenience function gains a `proof_method` parameter: `"eur"` (default) or `"aep"` (legacy).

### 3.2 `physics/decoy.py` — Restructure around basis-specific counts

The EUR formula needs per-basis, per-photon-number statistics, not the basis-averaged quantities the current decoy module provides. Specifically, from the observed data in the X (test) basis, we need to bound quantities in the Z (key) basis.

**Current interface (basis-averaged):**

```python
def finite_bounds(signal, decoy, vacuum, mu, nu, Y_0, n_PE, confidence):
    → DecoyBounds(Q1_lower, e1_upper)
```

**New interface (basis-specific, EUR-compatible):**

```python
@dataclass
class EURDecoyBounds:
    s_Z0_lower: float       # vacuum detections in Z basis (lower bound)
    s_Z1_lower: float       # single-photon detections in Z basis (lower bound)
    phi_Z_upper: float      # phase error rate in Z basis (upper bound)

    # Diagnostic: the intermediate quantities used in derivation
    Y0_bound: float
    Y1_lower: float
    e1_upper: float

def eur_decoy_bounds(
    # Observed counts by basis and intensity
    n_Z: dict[str, float],      # {"signal": ..., "decoy": ..., "vacuum": ...} counts in Z
    n_X: dict[str, float],      # counts in X (test) basis
    m_X: dict[str, float],      # errors in X basis per intensity
    # Source parameters
    mu: float, nu: float,
    # Preparation probabilities (needed to weight contributions)
    P_mu: float, P_nu: float,
    # Security failure probabilities for each sub-estimation
    epsilon_s: float,           # parameter estimation failure probability
) -> EURDecoyBounds:
```

The function must:

1. From X-basis observations, bound $Y_0$, $Y_1$, $e_1$ using decoy analysis with statistical corrections.
2. From these bounds and the Z-basis observation counts, compute $s_{Z,0}^L$, $s_{Z,1}^L$, $\bar{\phi}_Z^U$.

The second step uses the relationship: $s_{Z,1}^L = n_Z \cdot P_1(\mu) \cdot Y_1^L / Q_\mu$ (proportional to single-photon events in Z), and the phase error rate is estimated from the X-basis error statistics via the EUR connection.

### 3.3 Parameter estimation: concentration inequalities

The statistical corrections applied when bounding $Y_1$, $e_1$, and the phase error rate from finite samples are where the second major improvement lives. Three options, in order of tightness:

**Option A — Hoeffding/Chernoff (current, loosest):**

Standard additive Chernoff bound. Slack scales as $\sqrt{\ln(1/\delta)/(2n)}$ where $n$ is the number of trials. Well-understood but loose, especially for rare events (decoy/vacuum counts are much rarer than signal counts).

**Option B — Kato's inequality (recommended):**

Kato's concentration inequality (2020) provides tighter bounds when the quantity being estimated involves events with very low probability of occurrence. For decoy-state QKD, the multi-photon contributions ($n \geq 2$) are precisely such rare events. Kato's inequality gives significantly tighter bounds than Azuma/Hoeffding for these terms.

This is what SatQuMA and several recent finite-key analyses use. The improvement is most pronounced for vacuum and weak-decoy count statistics, which is exactly where the standard Hoeffding bound is most over-conservative.

**Option C — Exact hypergeometric CDF (tightest, new):**

Zapatero & Curty (2025, PRL) show that for a broad parameter regime, the cumulative mass function of the hypergeometric distribution is accurately computable rather than being bounded by a tail inequality. This eliminates the tail bound entirely and gives the sharpest possible statistical estimate. They demonstrate that this "sharply decreases the minimum block sizes necessary for QKD." This is the state of the art as of 2025.

**Recommendation:** Implement Option B (Kato) as the default. Add Option C as an alternative for when maximum tightness matters. Keep Option A for comparison and regression testing.

Implementation in `decoy.py`:

```python
def _hoeffding_slack(n: float, delta: float) -> float:
    """Additive Hoeffding bound. Loosest, simplest."""
    return math.sqrt(math.log(1.0 / delta) / (2.0 * n))

def _kato_slack(n: float, delta: float, p_est: float) -> float:
    """Kato's inequality. Tighter for rare events (small p_est).
    p_est is the estimated probability of the event being bounded."""
    # Kato (2020), arXiv:2002.04357
    ...

def _hypergeometric_bound(k_obs: int, n_sample: int, n_pop: int, delta: float) -> float:
    """Exact hypergeometric CDF inversion. Tightest.
    Zapatero & Curty (2025)."""
    # Uses scipy.stats.hypergeom.ppf or direct CDF computation
    ...
```

### 3.4 Correct trial counts for each bound

As previously identified: the Hoeffding/Kato slack for gain estimation ($Q_\nu$) must use **pulse counts** (Bernoulli trials = pulses), while the slack for QBER estimation ($E_\nu$) must use **detection counts** (Bernoulli trials = detections). Two separate slack computations:

```python
slack_Q = concentration_bound(n_pulses_nu, delta_Q)
slack_E = concentration_bound(n_detections_nu, delta_E)
```

Using detection counts for both (the common mistake) inflates the gain slack by $\sqrt{1/Q_\nu} \approx 45\times$ and collapses $Q_1^L$ to zero.

### 3.5 Updated `AccumulationResult`

The accumulator in `pass_sim.py` must now track per-basis, per-intensity counts separately, not just basis-averaged totals.

```python
@dataclass
class AccumulationResult:
    # Per-basis, per-intensity counts
    n_Z: dict[str, float]       # Z-basis detections: {"signal", "decoy", "vacuum"}
    n_X: dict[str, float]       # X-basis detections
    m_X: dict[str, float]       # X-basis errors per intensity

    # Pulse counts (for correct Hoeffding/Kato trial denominators)
    N_pulses: dict[str, float]  # total pulses sent per intensity

    # Convenience
    n_Z_total: float            # total Z-basis signal detections
    E_Z_weighted: float         # detection-weighted QBER in Z basis
```

The per-timestep accumulation splits detections by basis using $P_X$: at each timestep, a fraction $P_X^2$ of matching-basis detections go to X, and $(1-P_X)^2$ go to Z (for efficient BB84). The total pulse count per intensity is $N_{\mu_i} = f_{clock} \cdot P_{\mu_i} \cdot T_{pass}$.

### 3.6 Updated types

Add `EURDecoyBounds` to `core/types.py`. Keep the existing `DecoyBounds` for the asymptotic path.

### 3.7 Security parameter composition

The EUR proof has multiple failure sub-events, each with its own probability bound. The total security parameter $\varepsilon$ must be composed:

$$\varepsilon = \varepsilon_{PE} + \varepsilon_{EC} + \varepsilon_{PA} + \bar{\varepsilon}$$

where $\varepsilon_{PE}$ covers parameter estimation failures (subdivided further across each decoy bound), $\varepsilon_{EC}$ is error correction failure, $\varepsilon_{PA}$ is privacy amplification, and $\bar{\varepsilon}$ is the smoothing parameter.

A clean approach: accept a total $\varepsilon$ (e.g., $10^{-10}$) and subdivide it. Lim et al. (2014) use equal subdivision. More sophisticated: optimise the subdivision (each sub-$\varepsilon$ affects a different term; the optimal allocation is not equal). For Phase 1, equal subdivision is fine.

```python
@dataclass
class SecurityBudget:
    epsilon_total: float        # e.g. 1e-10
    n_terms: int                # number of sub-events to cover

    @property
    def epsilon_pe(self) -> float: return self.epsilon_total / self.n_terms
    @property
    def epsilon_ec(self) -> float: return self.epsilon_total / self.n_terms
    # etc.
```

---

## 4. What Does Not Change

- `gllp_asymptotic()` — same GLLP formula, still useful for fast exploration.
- `binary_entropy()` — same utility.
- `compute_key_rate()` — same interface, adds `proof_method` parameter.
- The evaluator (`core/evaluator.py`) — unchanged. It produces per-timestep detection results; it doesn't touch key extraction.
- The fading model, channel model, detection model — all upstream physics unchanged.
- `skbr()` — still $f_{clock} \times R$, but $R$ is now computed differently.

---

## 5. Validation

### Against the AEP version
Run both EUR and AEP at the same parameters. EUR must give $\ell_{EUR} \geq \ell_{AEP}$ everywhere (it's a strictly tighter bound). If EUR ever gives a lower key than AEP, something is wrong.

### Asymptotic convergence
At very large $n$ ($10^{10}+$), both methods should converge to the same asymptotic rate $R$. Verify $|\ell_{EUR}/n - \ell_{AEP}/n| \to 0$.

### Crossover block size
Find the minimum $n$ at which $\ell > 0$ for both methods. The EUR threshold should be significantly lower than the AEP threshold (by roughly an order of magnitude in $n$).

### Boundary cases
- $E_\mu > 11\%$: both methods return zero.
- $n = 0$: both return zero.
- Zero turbulence (deterministic channel): results should match analytical predictions.

### SatQuMA comparison
If available, run identical parameters through SatQuMA (which uses EUR + improved sampling). Our results should agree to within differences from the concentration inequality choice (Hoeffding vs Kato vs exact hypergeometric).

---

## 6. Build Order

1. Add `EURDecoyBounds` and `SecurityBudget` to `types.py`.
2. Implement `_kato_slack()` in `decoy.py` (alongside existing Hoeffding).
3. Implement `eur_decoy_bounds()` in `decoy.py` with per-basis, per-intensity logic.
4. Implement `eur_key_length()` in `keyrate.py`.
5. Update `AccumulationResult` in `pass_sim.py` to track per-basis counts.
6. Update `simulate_pass()` to call the EUR path.
7. Validate against AEP version (EUR ≥ AEP everywhere).
8. Validate asymptotic convergence.

---

## 7. References

| Reference | Role |
|-----------|------|
| Lim, Curty, Walenta, Xu & Zbinden (2014), PRA 89, 022307 | EUR-based finite-key formula for decoy BB84 |
| Wiesemann, Krause, Tupkary, Lütkenhaus, Rusca & Walenta (2026), Quantum 10, 2037 | Consolidated EUR proof with technical corrections; primary implementation reference |
| Tomamichel, Lim, Gisin & Renner (2012), Nat. Commun. 3, 634 | Original AEP-based finite-key bounds (being replaced) |
| Staffieri, Scala & Lupo (2026), arXiv:2601.03829 | Comparison of AEP, EUR, and FME approaches; demonstrates EUR superiority |
| Kato (2020), arXiv:2002.04357 | Concentration inequality for rare events; tighter than Hoeffding |
| Zapatero & Curty (2025), PRL, arXiv:2410.04095 | Sharp finite statistics via exact hypergeometric CDF; state of the art |
| Sidhu et al. (2022), npj Quantum Inf. | Finite-key satellite QKD with improved sampling bounds |
| Islam et al. (2023), Commun. Phys. 6, 210 | Small satellite finite-key performance using SatQuMA |