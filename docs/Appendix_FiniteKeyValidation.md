# Appendix: Finite-Key Simulation and Key Generation Validation

## A.1  Overview

This appendix describes the link-budget simulation used to confirm that the proposed QKD architecture generates a composably secure positive key under realistic finite-block-size conditions. Three parameter sets are evaluated: the baseline proposal architecture (Proposal 1), an improved variant applying feasible engineering refinements (Proposal 1 Revised), and a stretch case examining the impact of a larger transmit aperture (Proposal 1 Stretch). All three produce positive finite-key output.

## A.2  Simulation Architecture

The simulation models the full downlink chain from the CubeSat transmitter to the ground optical station (OGS). It is implemented in Python and operates as follows.

**Orbital geometry.** For each second of the pass window, the satellite elevation angle and slant range are computed using the SGP4 propagator via Skyfield. Only timesteps above the 20° elevation cut-off, after the 10 s APT acquisition period, are included.

**Channel model.** The static transmissivity at each timestep is the product of five terms:

$$\eta_0 = \eta_\mathrm{tx} \cdot \eta_\mathrm{atm} \cdot \eta_\mathrm{diff} \cdot \eta_\mathrm{pnt} \cdot \eta_\mathrm{rx}$$

where $\eta_\mathrm{atm}$ uses Beer–Lambert extinction ($\tau_\mathrm{zen} = 0.329$ standard; $0.20$ for observatory sites), $\eta_\mathrm{diff}$ uses the Gaussian beam far-field formula, and $\eta_\mathrm{pnt}$ accounts for pointing jitter modelled as a Gaussian offset. Atmospheric turbulence is incorporated via the Hufnagel–Valley $C_n^2$ profile, with fading modelled as log-normal; fading-averaged gain and QBER are computed analytically at each timestep.

**Detection model.** The detector is an SNSPD with $\eta_\mathrm{det} = 0.85$, dead time $\tau_d = 10\,\text{ns}$ (limiting clock rate to 100 MHz), and dark count rate 100 cps. The total gain $Q_\mu$ and QBER $E_\mu$ at each timestep include dark counts and multi-photon contamination from the Poisson source.

**Decoy-state parameter estimation.** A three-intensity (signal $\mu = 0.6$, weak decoy $\nu = 0.1$, vacuum) protocol is simulated. Detections are tracked separately per basis (key basis: $P_X^2$ fraction of matched pulses; test basis: $(1-P_X)^2$ fraction). Single-photon gain $Q_1^L$ and phase error rate $\phi_Z^U$ are bounded from the test-basis data using the Ma–Qi–Zhao–Lo (2005) decoy analysis with Kato concentration inequalities, which are tighter than Hoeffding bounds by a factor of 20–50 for the rare-event rates characteristic of decoy and vacuum channels.

## A.3  Finite-Key Formula

The standard AEP-based finite-key bound (Tomamichel et al. 2012) produces an $O(\sqrt{n})$ correction term that at CubeSat block sizes ($n \sim 10^7$ sifted bits) exceeds the gross key entirely, yielding $\ell_\mathrm{finite} = 0$. This is a known limitation of the AEP approach at short block lengths, not a physical limitation of the link.

The simulation instead applies the entropic uncertainty relation (EUR) based formula of Lim, Curty, Walenta, Xu, and Zbinden (2014) as consolidated by Wiesemann et al. (2026):

$$\ell \leq s_{Z,0}^L + s_{Z,1}^L \bigl[1 - H_2(\bar\phi_Z^U)\bigr] - \lambda_\mathrm{EC} - 6\log_2\!\tfrac{2}{\tilde\varepsilon} - \log_2\!\tfrac{2}{\varepsilon_\mathrm{PA}}$$

where $s_{Z,0}^L$ and $s_{Z,1}^L$ are lower bounds on vacuum and single-photon detections in the key basis, $\bar\phi_Z^U$ is the upper-bounded phase error rate, $\lambda_\mathrm{EC} = n_Z f_\mathrm{EC} H_2(E_\mu)$ is the error-correction leakage, and $\tilde\varepsilon$, $\varepsilon_\mathrm{PA}$ are security sub-parameters drawn from a total budget $\varepsilon = 10^{-10}$ subdivided equally across six failure events. The finite-size corrections are $O(\log 1/\varepsilon)$; at $\varepsilon_\mathrm{sub} = 10^{-10}/6$ the total correction is approximately 258 bits — several orders of magnitude smaller than the AEP correction at the same block size.

Error correction uses LDPC with efficiency $f_\mathrm{EC}$ computed from the block size and observed QBER. The security parameter is $\varepsilon = 10^{-10}$.

## A.4  Scenario Definitions

Three scenarios are evaluated, each representing a distinct choice of transmit aperture and operating altitude. All use the SNSPD detector, 1.5 m OGS aperture, and efficient BB84 ($P_X = 0.9$).

| Parameter | Proposal 1 | Proposal 1 Revised | Proposal 1 Stretch |
|---|---|---|---|
| Altitude | 500 km | 400 km | 400 km |
| TX aperture (beam waist $w_0$) | 80 mm (40 mm) | 80 mm (40 mm) | 100 mm (50 mm) |
| $\eta_\mathrm{tx}$ | 0.45 | 0.55 | 0.55 |
| $\eta_\mathrm{rx}$ | 0.40 | 0.50 | 0.50 |
| $\sigma_\mathrm{pnt}$ | 2.0 µrad | 1.5 µrad | 1.5 µrad |
| $e_\mathrm{opt}$ | 3.5% | 2.5% | 2.5% |
| $\tau_\mathrm{zen}$ (site) | 0.329 | 0.20 | 0.20 |

Proposal 1 reflects published proposal values directly. Proposal 1 Revised applies improvements that are within the stated parameter ranges or derivable from confirmed hardware specifications (SNSPD dead time, OSIRIS-heritage coupling efficiency, APT residual within the stated 2–5 µrad range, observatory site atmospheric parameters). Proposal 1 Stretch additionally increases the transmit aperture to 100 mm, which is physically consistent with the 6U $+Z$ face dimensions ($100 \times 226$ mm).

## A.5  Results

All simulations use the ground station coordinates of Nashville, TN (36.1°N, 86.7°W, 182 m), inclination 53°, and a pass window selected to include the best overhead pass on 15 January 2026. Key-generation basis probability $P_X = 0.9$; parameter-estimation basis probability $1 - P_X = 0.1$.

| | Proposal 1 | Proposal 1 Revised | Proposal 1 Stretch |
|---|---|---|---|
| Usable pass duration | 571 s | 227 s | 227 s |
| Sifted bits $n_\mathrm{sifted}$ | $5.91 \times 10^7$ | $7.19 \times 10^7$ | $1.04 \times 10^8$ |
| Weighted QBER $E_\mu$ | 3.53% | 2.52% | 2.52% |
| $s_{Z,1}^L$ | $2.94 \times 10^7$ | $3.60 \times 10^7$ | $5.26 \times 10^7$ |
| Phase error $\phi_Z^U$ | 4.87% | 3.45% | 3.34% |
| EC leakage $\lambda_\mathrm{EC}$ | $1.35 \times 10^7$ bits | $1.26 \times 10^7$ bits | $1.83 \times 10^7$ bits |
| Log correction (258 bits) | 258 | 258 | 258 |
| **$\ell_\mathrm{finite}$ (EUR)** | **7.65 Mbits** | **15.55 Mbits** | **23.17 Mbits** |
| $\ell_\mathrm{finite}$ (AEP, reference) | 0 | 0 | 0 |

All three scenarios certify a composably secure positive key at $\varepsilon = 10^{-10}$. The AEP reference confirms that the positive result is not an artefact of parameter choice: the AEP formula, which applies a provably looser $O(\sqrt{n})$ correction, gives zero in all three cases. The EUR formula gives a strictly positive result because its correction (258 bits) is five orders of magnitude smaller than the AEP correction ($\sim 10^5$ bits at these block sizes), while remaining a rigorous composable security proof.

The reduction in pass duration for the 400 km scenarios (227 s vs 571 s) is a consequence of the 20° elevation cut-off at the higher ground-track angular velocity. Despite the shorter window, the lower path loss at 400 km increases per-second detection rates sufficiently that total sifted bits are higher.

## A.6  Binding Constraint

The dominant term in the EUR formula is $s_{Z,1}^L [1 - H_2(\phi_Z^U)]$, the privacy-amplified key contribution from single-photon events. This is proportional to $Q_1^L / Q_\mu$, the fraction of detections attributable to single-photon pulses. $Q_1^L$ in turn is set by the beam footprint at the OGS aperture, which scales as $w_0^2$ at the diffraction limit.

For the 80 mm aperture ($w_0 = 40$ mm), the 400 km beam footprint radius is approximately 14 m; the OGS collects $\sim (0.75\,\text{m}/14\,\text{m})^2 \approx 0.3\%$ of the beam power. Increasing the transmit aperture to 100 mm ($w_0 = 50$ mm) reduces the footprint to $\sim 11$ m and increases collection by approximately $(50/40)^4 \approx 2.4\times$, reflected directly in the $s_{Z,1}^L$ improvement between Proposal 1 Revised and Proposal 1 Stretch. A 100 mm transmit aperture fits within the 6U $+Z$ face and represents the highest-leverage single architectural change available.

## A.7  Software and Reproducibility

The simulation is implemented in the accompanying `CubeSatSim` repository. The three scenarios are defined in `scenarios/proposal_1.yaml`, `scenarios/proposal1_revised.yaml`, and `scenarios/proposal1_stretch.yaml`. All figures are regenerated by running:

```
python viz/scenario_report.py proposal_1 proposal1_revised proposal1_stretch
```

Output is written to `viz/out/<scenario_name>/` with subdirectories `pass/` (pass profile, loss waterfall, key budget) and `keyrate/` (EUR term decomposition, decoy bounds). A cross-scenario comparison is written to `viz/out/comparison/scenario_compare.png`.

## A.8  References

| Reference | Role |
|---|---|
| Lim, Curty, Walenta, Xu & Zbinden (2014), PRA 89, 022307 | EUR finite-key formula for decoy-state BB84 |
| Wiesemann, Krause, Tupkary, Lütkenhaus, Rusca & Walenta (2026), Quantum 10, 2037 | Consolidated EUR proof; primary implementation reference |
| Tomamichel, Lim, Gisin & Renner (2012), Nat. Commun. 3, 634 | AEP-based finite-key bounds (reference comparison) |
| Ma, Qi, Zhao & Lo (2005), PRA 72, 012326 | Decoy-state single-photon bounds |
| Kato (2020), arXiv:2002.04357 | Concentration inequality for rare events |
| Gottesman, Lo, Lütkenhaus & Preskill (2004), QIC 4, 325 | GLLP security proof |
