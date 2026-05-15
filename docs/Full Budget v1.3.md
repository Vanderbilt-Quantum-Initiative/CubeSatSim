## From Optical Link Parameters to Secret Key Bit Rate

**Document type:** Reference budget — equations, model description, subsystem decomposition  
**Mission:** VQI CubeSat QKD Demonstrator  
**Protocol:** Decoy-state BB84, weak coherent pulse **downlink** (CubeSat = Alice/transmitter; OGS = Bob/receiver)  
**Companion document:** [[Param Registry v1.3|VQI Parameter Registry]] (living values table)  
**Version:** 2.1 — corrects slant-path turbulence model, radiation fallacy, $e_{opt}$ ownership, $f_{clock}$ scaling limit; adds Gamma-Gamma parameter derivation; symbol consistency pass

---

## Purpose

This document defines the complete mathematical chain from physical hardware and environmental parameters to the mission's ultimate success criterion: **secret key bits per pass** ($\ell_{finite}$). It serves as the equation reference. Current parameter values, ownership, and status are tracked in the companion Parameter Registry.

QBER is a necessary intermediate result but not the final criterion. A link may produce acceptable QBER yet yield zero secret key if the finite-key penalty exceeds the asymptotic gain. $\ell_{finite} > 0$ is the binding condition.

**Architecture note:** This is a downlink. The CubeSat transmits single photons to the ground. All single-photon detectors are at the Optical Ground Station (OGS). They are not subject to space radiation. Parameters relating to radiation damage apply only to uplink configurations and are not relevant here.

> **Related documents:**
> - [[QBER formulations]] — full derivation of the fading-channel QBER model (all stages)
> - [[KeyBit Error Rate Budget]] — GLLP-decoy key rate derivation and finite-key correction
> - [[Param Registry v1.3]] — companion living values table (all symbols, owners, status)
> - [[VQI Top-Level Requirements]] — mission-level "shall" statements these budgets serve
> - [[Concept of Operations (CONOPS)|CONOPS]] — operational states that define $T_{pass}$ and mode boundaries
> - [[Subsystem Budget]] — mass and power allocations by subsystem
> - [[Subsystem-Requirements]] - specific responsibilities for each subsystem

---

## Section 1: The Equation Chain

---

### Stage 1 — Static Transmissivity $\eta_0$

> **Model detail:** [[QBER formulations#2. Transmissivity and the Fading Channel Model|QBER formulations §2]] · [[Param Registry v1.3#Block 1 — Transmitter Optics (Owner: Payload / QKD)|Params Block 1]] · [[Param Registry v1.3#Block 4 — Ground Station Receiver (Owner: Ground Station)|Params Block 4]]

The geometric and hardware loss budget under deterministic (no turbulence, no pointing error) conditions:

$$\eta_0 = \eta_{tx} \cdot \eta_{atm} \cdot \eta_{diff} \cdot \eta_{pnt} \cdot \eta_{rx}$$

Component terms:

$$\eta_{atm} = \exp(-\alpha L)$$

$$\eta_{diff} = 1 - \exp!\left(-\frac{2D_{rx}^2}{W(L)^2}\right), \quad W(L) = \frac{\lambda L}{\pi w_0}$$

$$\eta_{pnt} = \exp!\left(-\frac{\theta_{pnt}^2}{2\sigma_{pnt}^2}\right)$$

**Parameters introduced:** $\eta_{tx}$, $\alpha$, $L$, $D_{rx}$, $\lambda$, $w_0$, $\theta_{pnt}$, $\sigma_{pnt}$, $\eta_{rx}$

**Sources:** Bourgoin et al. (2013), New J. Phys. 15, 023006; Ma et al. (2005), PRA 72, 012326

---

### Stage 2 — Fading Distribution $P(\eta)$

> **Model detail:** [[QBER formulations#2. Transmissivity and the Fading Channel Model|QBER formulations §2 — Fading Channel]] · [[QBER formulations#Why Static Analysis Underestimates QBER|Jensen's inequality caveat]] · [[Param Registry v1.3#Block 2 — Atmospheric Channel (Owner: Mission / Systems)|Params Block 2]]

Atmospheric turbulence causes the instantaneous transmissivity $\eta$ to fluctuate around $\eta_0$. The distribution $P(\eta)$ captures this stochastic behaviour.

#### Slant-Path Rytov Variance

For a LEO-to-ground slant path, turbulence is concentrated in the lowest ~20 km of atmosphere. The horizontal-path Rytov formula assumes constant $C_n^2$ along the full path length $L$ and must not be used — it overestimates scintillation significantly for satellite links. The correct formulation integrates the height-dependent structure parameter along the slant path:

$$\sigma_R^2 = 2.25, k^{7/6}, \sec^{11/6}(\zeta) \int_{h_0}^{H} C_n^2(h),(h - h_0)^{5/6}, dh$$

where $k = 2\pi/\lambda$, $\zeta$ is the zenith angle, $h_0$ is the ground station altitude, and $H \approx 20$ km is the effective top of the turbulent atmosphere.

The standard **Hufnagel-Valley model** for $C_n^2(h)$ is:

$$C_n^2(h) = 0.00594!\left(\frac{v}{27}\right)^{!2} (10^{-5}h)^{10} e^{-h/1000} + 2.7 \times 10^{-16} e^{-h/1500} + C_n^2(0), e^{-h/100}$$

where $v$ is the RMS wind speed (standard value 21 m/s) and $C_n^2(0)$ is the site-dependent ground-level structure parameter — the largest single uncertainty in the channel model.

**Parameters introduced:** $\sigma_R^2$, $\zeta$, $h_0$, $C_n^2(h)$, $v$, $C_n^2(0)$

#### Turbulence Regime and Fading Model

The value of $\sigma_R^2$ determines which fading model applies:

**Log-normal model** (weak turbulence, $\sigma_R^2 \ll 1$):

$$P(\eta) = \frac{1}{\eta, \sigma_{ln} \sqrt{2\pi}} \exp!\left(-\frac{(\ln\eta - \mu_{ln})^2}{2\sigma_{ln}^2}\right)$$

Appropriate for most zenith-to-moderate elevation satellite passes over good sites.

**Gamma-Gamma model** (moderate-to-strong turbulence, $\sigma_R^2 \gtrsim 1$):

$$P(\eta) \propto \eta^{(\alpha_{GG}+\beta_{GG})/2 - 1} K_{\alpha_{GG}-\beta_{GG}}!\left(2\sqrt{\alpha_{GG}\beta_{GG},\eta}\right)$$

where $K_\nu$ is the modified Bessel function of the second kind, and the shape parameters are derived from $\sigma_R^2$:

$$\alpha_{GG} = \left[\exp!\left(\frac{0.49,\sigma_R^2}{(1+1.11,\sigma_R^{12/5})^{7/6}}\right) - 1\right]^{-1}$$

$$\beta_{GG} = \left[\exp!\left(\frac{0.51,\sigma_R^2}{(1+0.69,\sigma_R^{12/5})^{5/6}}\right) - 1\right]^{-1}$$

The Gamma-Gamma model applies at low elevation angles where the slant path through the turbulent layer is longest. $\alpha_{GG}$ and $\beta_{GG}$ are fully determined by $\sigma_R^2$ and are not independent inputs.

**Parameters introduced:** $\alpha_{GG}$, $\beta_{GG}$ (derived from $\sigma_R^2$)

**Sources:** Rytov (1978); Andrews & Phillips, _Laser Beam Propagation through Random Media_ (2005); Vasylyev et al. (2016), PRL 117, 090501; Pirandola (2021), Phys. Rev. Research 3, 013279

---

### Stage 3 — Expected QBER

> **Model detail:** [[QBER formulations#1. Fundamental QBER Formulation|QBER formulations §1 — Fundamental QBER]] · [[QBER formulations#3. Noise Characterization ($Y_0$)|QBER formulations §3 — Noise]] · [[QBER formulations#4. Information-Theoretic Bounds on QBER|QBER formulations §4 — Security Thresholds]] · [[Param Registry v1.3#Block 5 — Optical QBER Budget (Shared: Payload / ADCS / Ground Station)|Params Block 5]] · [[Param Registry v1.3#Block 7 — Background Noise (Owner: Mission / Systems)|Params Block 7]]

The instantaneous QBER at transmissivity $\eta$ is:

$$\text{QBER}(\eta) = \frac{e_{opt},\eta\mu + \tfrac{1}{2}Y_0}{\eta\mu + Y_0}$$

where the total noise yield combines dark counts and sky background:

$$Y_0 = p_d + p_{bg}, \quad p_{bg} = H_{bg} \cdot \Omega_{FOV} \cdot A_{rx} \cdot \Delta\lambda \cdot \Delta t \cdot \eta_{rx}$$

Note that $e_{opt}$ is not a static constant. During a satellite pass, the polarization reference frame between the rotating satellite and the stationary ground station drifts continuously. This causes $e_{opt}$ to vary pass-by-pass and within a pass. The effective $e_{opt}$ entering this formula is the residual after ground station compensation (see Parameter Registry Block 5). A worst-case static bound is used for modelling unless compensation performance is explicitly characterised.

The **expected QBER** averaged over the fading channel is:

$$E_\mu = \langle\text{QBER}\rangle = \int_0^1 \frac{e_{opt},\eta\mu + \tfrac{1}{2}Y_0}{\eta\mu + Y_0} \cdot P(\eta),d\eta$$

The symbol $E_\mu$ is used for the expected QBER throughout the key rate formula below, consistent with its role as the measured signal-intensity error rate.

**Jensen's inequality caveat:** Because QBER($\eta$) is convex in $\eta$ at high loss, $\langle\text{QBER}(\eta)\rangle > \text{QBER}(\langle\eta\rangle)$. Computing QBER at the mean transmissivity systematically underestimates the true expected QBER. The integral must be evaluated numerically.

**Security threshold:** $E_\mu < 11\%$ for any key to survive privacy amplification (BB84, practical one-way EC). Necessary but not sufficient for mission success.

**Parameters introduced:** $Y_0$, $e_{opt}$, $\mu$, $p_d$, $H_{bg}$, $\Omega_{FOV}$, $A_{rx}$, $\Delta\lambda$, $\Delta t$

**Sources:** Gobby, Yuan & Shields (2004), Appl. Phys. Lett. 84(19); Miao et al. (2005), New J. Phys. 7, 215; Pirandola (2026), arXiv:2602.22319; Shor & Preskill (2000), PRL 85, 441

---

### Stage 4 — Overall Signal Gain and Decoy Bounds

> **Model detail:** [[KeyBit Error Rate Budget#Eq. 1 — Overall Signal Gain $Q_\mu$|KeyBit Budget Eq. 1 — $Q_\mu$]] · [[KeyBit Error Rate Budget#Eq. 2 — Single-Photon Gain $Q_1$ and Single-Photon QBER $e_1$ (Decoy State Estimation)|KeyBit Budget Eq. 2 — Decoy bounds]] · [[Param Registry v1.3#Block 11 — Observables (Post-Processing Inputs)|Params Block 11 — Observables]]

The **overall signal gain** (detection probability per pulse) is exactly:

$$Q_\mu = 1 - (1 - Y_0),e^{-\langle\eta\rangle\mu}$$

The approximation $Q_\mu \approx \langle\eta\rangle\mu + Y_0$ is valid at high loss ($\langle\eta\rangle\mu \ll 1$) but the exact form should be used in simulation to avoid truncation error in tight finite-key calculations.

The decoy-state method uses auxiliary pulses at intensity $\nu < \mu$ to bound the single-photon gain $Q_1$ and phase error rate $e_1$. Asymptotic bounds:

$$Q_1 \geq \frac{\mu^2 e^{-\mu}}{\mu\nu - \nu^2} \left(Q_\nu e^\nu - Q_\mu e^\mu \frac{\nu^2}{\mu^2} - \frac{\mu^2 - \nu^2}{\mu^2}Y_0\right)$$

$$e_1 \leq \frac{E_\nu Q_\nu e^\nu - e_0 Y_0}{Y_1 \nu}, \quad e_0 = \tfrac{1}{2}, \quad Y_1 = \frac{Q_1}{e^{-\mu}\mu}$$

**Important:** These asymptotic bounds must not be used for finite-key extraction. The software team must apply statistically corrected versions using Hoeffding or Chernoff inequalities, taking lower bounds on $Q_1$ and upper bounds on $e_1$ from the observed sample of size $n$. Using asymptotic bounds for finite-key calculations will overestimate the extractable key.

$Q_\nu$ and $E_\nu$ are observables measured from the decoy pulses during the pass. See Parameter Registry Block 11.

**Parameters introduced:** $\nu$, $Q_\nu$, $E_\nu$

**Sources:** Lo, Ma & Chen (2005), PRL 94, 230504; Ma et al. (2005), PRA 72, 012326

---

### Stage 5 — Secret Key Rate

> **Model detail:** [[KeyBit Error Rate Budget#Eq. 3 — The GLLP-Decoy Secret Key Rate Formula (Asymptotic)|KeyBit Budget Eq. 3 — GLLP]] · [[KeyBit Error Rate Budget#Eq. 4 — Conversion to Bits per Second|KeyBit Budget Eq. 4 — SKBR]] · [[KeyBit Error Rate Budget#Eq. 5 — Finite-Key Correction (Tomamichel et al. 2012)|KeyBit Budget Eq. 5 — Finite-key]] · [[Param Registry v1.3#Block 8 — Protocol and Post-Processing (Owner: Software / OBC)|Params Block 8]] · [[Param Registry v1.3#Block 9 — Orbital and Pass Geometry (Owner: Mission / Systems)|Params Block 9]] · [[Param Registry v1.3#Block 12 — Derived Performance Metrics|Params Block 12 — Outputs]]

#### 5a. Asymptotic Key Fraction (GLLP-Decoy)

$$R \geq q \left[Q_1\bigl(1 - H_2(e_1)\bigr) - Q_\mu, f_{EC}, H_2(E_\mu)\right]$$

where:

- $q$ = sifting factor: $\tfrac{1}{2}$ for standard BB84; $\approx 1$ for efficient/asymmetric BB84
- $H_2(x) = -x\log_2 x - (1-x)\log_2(1-x)$ is the binary entropy function
- $f_{EC} \geq 1$ is error correction efficiency relative to the Shannon limit
- $E_\mu$ = overall observed QBER from Stage 3

Physical interpretation: $Q_1(1 - H_2(e_1))$ is the information surviving privacy amplification from single-photon events; $Q_\mu f_{EC} H_2(E_\mu)$ is bits consumed by error correction. Positive $R$ requires the former to exceed the latter.

#### 5b. Secret Key Bit Rate

$$\text{SKBR} = f_{clock} \cdot R \quad \text{[bits/second]}$$

$$\ell_{asym} = f_{clock} \cdot T_{pass} \cdot R \quad \text{[bits/pass, asymptotic]}$$

**$f_{clock}$ constraint:** $f_{clock}$ is not a free parameter. It is coupled to the ground station detector dead time $\tau_d$. A Si-SPAD with $\tau_d \sim 20$–100 ns saturates at pulse rates above $\sim 1/\tau_d \sim 10$–50 MHz. Above this rate, pile-up and missed detections degrade the effective count rate and distort $Q_\mu$ and $E_\mu$ estimates. Supporting $f_{clock}$ above ~100 MHz without SKR collapse requires either multiplexed SPAD arrays or SNSPDs — a significant cost and complexity driver. $f_{clock}$ must be treated as a joint Payload/Ground Station trade.

**Parameters introduced:** $q$, $f_{EC}$, $f_{clock}$, $T_{pass}$

#### 5c. Finite-Key Correction

The composably secure finite-key length (Tomamichel, Lim, Gisin & Renner 2012):

$$\ell_{finite} \leq n \cdot R - \Delta_{AEP} - \text{leak}_{EC} - \log_2\frac{2}{\varepsilon_{PA}}$$

where:

- $n$ = sifted key bits available (after parameter estimation sample removed)
- $\Delta_{AEP} \sim O(\sqrt{n})$ = smooth min-entropy correction; dominant penalty for short blocks
- $\text{leak}_{EC} = n, f_{EC}, H_2(E_\mu)$
- $\varepsilon_{PA}$ = composable security parameter

For a CubeSat pass (~4–6 min usable at 400–550 km), $n$ may be $10^5$–$10^6$ bits. The $\Delta_{AEP}$ correction at this scale can eliminate all extractable key even when asymptotic $R > 0$. The $O(\sqrt{n})$ scaling means halving the usable pass window costs disproportionately more than half the key yield.

**Parameters introduced:** $n$, $\varepsilon_{PA}$

**Sources:** Tomamichel et al. (2012), Nat. Commun. 3, 634; Scarani & Renner (2008), PRL 100, 200501; Sidhu et al. (2022), npj Quantum Inf.; Islam et al. (2023), Commun. Phys. 6, 210; Gottesman, Lo, Lütkenhaus & Preskill (2004), QIC 4, 325

---

## Section 2: Subsystem Decomposition

> **See:** [[Param Registry v1.3]] for all current values · [[Subsystem Budget]] for mass and power allocations · [[VQI Top-Level Requirements]] for traceability and more info in [[Subsystem-Requirements]]

What follows is a summary of each team's parameter ownership and key design couplings. The definitive values table is in the Parameter Registry. This section explains the physics behind each ownership assignment.

---

### 2.1 Payload Team (QKD)

Owns the quantum source, polarization encoding, and onboard optical train.

- **$\lambda$** — wavelength. Sets the atmospheric window, detector compatibility, and all diffraction geometry. Primary driver of component selection chain-wide.
- **$w_0$** — transmitter beam waist. Determines $W(L)$ and therefore $\eta_{diff}$. Larger $w_0$ reduces divergence but demands larger CubeSat aperture and tighter pointing.
- **$\eta_{tx}$** — transmitter optical chain efficiency. Product of every loss element before free space.
- **$\mu$, $\nu$** — signal and decoy intensities. Security-constrained; $\mu$ bounded above by PNS attack threshold; $\nu < \mu$ required; optimised numerically per scenario.
- **$e_{opt,tx}$** — the static hardware floor for polarization QBER. Payload owns this component. The full $e_{opt}$ entering the model also depends on ADCS roll (Block 5).
- **$f_{clock}$** — pulse rate. **Cannot be set unilaterally by Payload.** Upper bound is the ground detector dead time $\tau_d$ (Ground Station). Must be resolved as a joint trade.

---

### 2.2 ADCS Team

> **See:** [[VQI Top-Level Requirements#Pointing|VQI Requirements — Pointing]] · [[Concept of Operations (CONOPS)#Nominal Orbit Cycle (~90 min repeat)|CONOPS — Orbit States]] · [[Param Registry v1.3#Block 3 — Pointing, Acquisition and Tracking (Owner: ADCS)|Params Block 3]]

Owns satellite attitude control and fine-pointing architecture.

- **$\theta_{pnt}$, $\sigma_{pnt}^2$** — pointing error and jitter. Enter $\eta_{pnt}$ directly. Sub-μrad final pointing is required; two-stage coarse/fine architecture (body + FSM) is standard for sub-μrad performance at CubeSat scale.
- **$\dot{\phi}_{roll}$** — satellite roll rate during pass. Drives the dynamic component of $e_{opt}$. ADCS must provide roll knowledge to the ground station compensation system. This is a shared interface with Ground Station.
- **$t_{acq}$, $P_{fail}$** — acquisition time and failure probability. Not in the QBER/SKR model but they directly reduce usable $T_{pass}$ and the probability of any key per pass. Must be characterised separately.

The static $\eta_{pnt}$ term does not capture acquisition dynamics or lock-loss events. The ADCS team must model APT behaviour as a separate pass-level analysis.

---

### 2.3 Ground Station Team

Owns the receiver telescope, single-photon detectors, timing electronics, and polarization compensation.

**Architecture note:** Detectors are ground-based. $p_d$ is governed by temperature, not radiation.

- **$D_{rx}$** — receiver aperture. Highest-leverage parameter with no SWaP constraint. Collection efficiency scales as $D_{rx}^2$. Should be baselined large immediately — this decision does not wait on form factor selection.
- **$\eta_{rx}$** — receiver optical efficiency. Enters both signal and background independently; must not be collapsed into $\eta_{det}$.
- **$\eta_{det}$** — detector quantum efficiency. Current DV-QKD ceiling. SNSPD achieves >90% but is cryogenic; Si-SPAD operates at 20–65%.
- **$\tau_d$** — detector dead time. The coupling that bounds $f_{clock}$. Must be provided to Payload early.
- **$p_d$** — dark count rate. Set by detector cooling temperature. Stable over mission life in a ground-based downlink configuration.
- **$\Delta\lambda$** — spectral filter. Primary background lever. Must accommodate full Doppler shift range across a pass.
- **$\Delta t$** — gate width. Set by system timing precision $\sigma_{sync}$, not a free choice.
- **$e_{opt,comp}$** — residual QBER after polarization compensation. Ground Station owns the compensation hardware. This is the dominant controllable contribution to in-pass $e_{opt}$.

---

### 2.4 OBC Team

> **See:** [[Param Registry v1.3#Block 6 — Timing Synchronisation (Owner: OBC / Ground Station)|Params Block 6]] · [[Concept of Operations (CONOPS)#Nominal Orbit Cycle (~90 min repeat)|CONOPS — QKD Mission State]]

Owns onboard timing, clock discipline, and Doppler pre-compensation.

- Onboard clock type — GNSS-disciplined timing is the standard solution for achieving sub-nanosecond synchronisation from LEO.
- $\sigma_{sync}$ — end-to-end timing synchronisation precision. Gates achievable $\Delta t$.
- Doppler pre-compensation — ~1–3 GHz shift at 850 nm for LEO orbital velocity. Must be computed from ephemeris and applied before each pass.

---

### 2.5 Structure / Thermal Team

Owns mechanical and thermal stability of the optical bench and bus.

- Orbital thermal cycling warps the optical bench, shifting $\theta_{pnt}$ and laser wavelength $\lambda$ (cavity length sensitivity). Passive design or active heaters are required.
- Temperature-driven fiber birefringence changes contribute to $e_{opt}$ independently of frame rotation — a joint Payload/Thermal interface.

---

### 2.6 Communications Team (Classical)

> **See:** [[Concept of Operations (CONOPS)#Nominal Orbit Cycle (~90 min repeat)|CONOPS — Classical Downlink State]] · [[VQI Top-Level Requirements#Operational|VQI Requirements — Operational]]

Owns the classical uplink/downlink channel for post-processing data exchange.

- Classical channel bandwidth must support sifting announcements, EC syndromes, and PA within $T_{pass}$. At high $f_{clock}$, data volume scales proportionally and RF becomes the bottleneck on CubeSat platforms.
- The classical channel must be authenticated. Without authentication, man-in-the-middle attacks are possible regardless of quantum link performance.

---

### 2.7 Software / Mission Operations

> **See:** [[KeyBit Error Rate Budget#Eq. 5 — Finite-Key Correction (Tomamichel et al. 2012)|KeyBit Budget Eq. 5 — Finite-key]] · [[KeyBit Error Rate Budget#Key Structural Observations|KeyBit Budget — Key Structural Observations]] · [[Param Registry v1.3#Block 8 — Protocol and Post-Processing (Owner: Software / OBC)|Params Block 8]]

Owns post-processing pipeline and protocol parameter choices.

- **$q$** — sifting factor. Efficient BB84 recovers the standard BB84 factor-of-2 penalty at no hardware cost. Should be the default.
- **$f_{EC}$** — error correction efficiency. Cascade (~1.16) is interactive; LDPC (~1.05) is one-way but computationally heavier.
- **Finite-key implementation** — the asymptotic GLLP formula and asymptotic decoy bounds must not be used for key extraction. Statistically corrected bounds (Hoeffding/Chernoff for decoy estimation; Tomamichel et al. 2012 for key length) are required. SatQuMA (Strathclyde group) is the standard open-source reference implementation.
- **$\varepsilon_{PA}$** — security parameter. Set by mission security requirements.

---

## Section 3: Mission Success Criterion

> **Traces to:** [[VQI Top-Level Requirements#QKD Performance|VQI Requirements — SYS-03 and SYS-04]] · [[KeyBit Error Rate Budget#Eq. 5 — Finite-Key Correction (Tomamichel et al. 2012)|KeyBit Budget Eq. 5]] · [[Param Registry v1.3#Block 12 — Derived Performance Metrics|Params Block 12 — Go/No-Go]]

The mission produces a non-zero secret key per pass if and only if all three conditions hold simultaneously:

1. **QBER ceiling:** $E_\mu < 11%$. Necessary but not sufficient.
2. **Positive asymptotic key fraction:** $R > 0$. Necessary but not sufficient.
3. **Positive finite-key yield:** $\ell_{finite} > 0$ under Tomamichel et al. (2012) bounds given actual $n$.

Conditions (1) and (2) can both hold while (3) fails. This is the most likely failure mode for short CubeSat passes at low clock rates or high loss. $\ell_{finite} > 0$ is the binding criterion.

---

## Section 4: Key References

| Reference                                                                | Role                                   |
| ------------------------------------------------------------------------ | -------------------------------------- |
| Bourgoin et al. (2013), New J. Phys. 15, 023006                          | Full LEO link budget framework         |
| Andrews & Phillips, _Laser Beam Propagation through Random Media_ (2005) | Slant-path turbulence model            |
| Vasylyev et al. (2016), PRL 117, 090501                                  | Fading PDT, Jensen argument            |
| Gobby, Yuan & Shields (2004), Appl. Phys. Lett. 84(19)                   | QBER formula                           |
| Miao et al. (2005), New J. Phys. 7, 215                                  | Background noise model                 |
| Lo, Ma & Chen (2005), PRL 94, 230504                                     | Decoy-state bounds                     |
| Ma et al. (2005), PRA 72, 012326                                         | Practical decoy bounds                 |
| Gottesman, Lo, Lütkenhaus & Preskill (2004), QIC 4, 325                  | GLLP secret key rate formula           |
| Shor & Preskill (2000), PRL 85, 441                                      | BB84 security proof, QBER threshold    |
| Pirandola (2021), Phys. Rev. Research 3, 013279                          | Fading channel framework               |
| Pirandola (2026), arXiv:2602.22319                                       | Fundamental limits                     |
| Scarani & Renner (2008), PRL 100, 200501                                 | Finite-key (smooth min-entropy)        |
| Tomamichel et al. (2012), Nat. Commun. 3, 634                            | Tight finite-key bounds                |
| Sidhu et al. (2022), npj Quantum Inf.                                    | Finite-key satellite QKD               |
| Islam et al. (2023), Commun. Phys. 6, 210                                | Small satellite finite-key performance |
| Liao et al. (2017), Nature 549, 43                                       | Micius: ~1.1 kbit/s at 500 km          |