# OMEGA feasibility: can SIMBA support a nebular-line analysis of the z≈6–8 OMEGA galaxies?

**Verdict: GO WITH CAVEATS, and only for a narrow target.** Run the Synthesizer nebular-line analysis on
**m25n512 only**, using galaxies **above log M\* ≈ 8.75** (or the ones with at least 5 star particles younger than 10 Myr).
Compare them with the **metal-rich half of OMEGA (O/H ≳ 7.8)**. Do not expect SIMBA to reproduce:
- OMEGA's low-O/H galaxies;
- its high SFR at fixed mass;
- its high C/O and N/O.
Those are where SIMBA systematically misses. For OMEGA, m100n1024 is unusable: its resolution mass is above every OMEGA galaxy.
The progenitor question (do compact and disky quiescent galaxies differ already at z≈6–8?) is **not answerable at OMEGA masses**:
- m25n512 has 3–12 resolved progenitors per redshift;
- m100n1024 has 65–161, but only at log M\* ≥ 8.7, and they show no difference (|Cliff's δ| ≤ 0.13).

Inputs: 9 OMEGA galaxies from `omega_obs_z6-8.csv` (z 6.05–8.28, log M\* 7.89–9.14, log SFR_Hα 0.59–1.46,
12+log(O/H) 7.14–8.18; C/O for 9, N/O for 6 of which 3 are upper limits). SIMBA run on 2026-11-17
(`omega_feasibility/submit.sh`), numbers in `output/omega_feasibility/feasibility_numbers.json` and
`summary_tables.md`; figures in `output/omega_feasibility/figures/`.

## Definitions

- **Snapshots** (from the headers, both boxes): 026 z=7.963, 030 z=7.054, 036 z=5.930, 129 z=0.406. Each OMEGA galaxy is
  compared with the snapshot nearest its redshift.
- **Analogues**: central galaxies with SFR > 0. **Resolved**: ≥ 30 star particles **and** ≥ 5 star particles younger than 10 Myr. A
  "resolution mass" is the lowest 0.25-dex bin of log M\* above which ≥ 80 % of the galaxies in every bin are resolved.
- **Abundances**: true gas-phase number ratios of the **star-forming gas** (SFR > 0 particles), computed from SIMBA's tracked mass
  fractions as n_X/n_H = (X_X/A_X)/(X_H/A_H), with X_H = 1 − Z − Y per particle. The weighted sums of number densities are SFR-weighted
  (quoted) or mass-weighted (in the catalogue). Solar reference: Asplund+09 (12+log(O/H)⊙ = 8.69, log(C/O)⊙ = −0.26,
  log(N/O)⊙ = −0.86).
- **SFR10, SFR100**: current star-particle mass formed in the last 10 / 100 Myr. Checks: the gas SFR equals caesar's `sfr`, and SFR100 is
  within ~3 % of caesar's `sfr_100`.
- **Overlap**: an OMEGA point is "inside" when it lies within the 95 % KDE contour of the resolved analogues (an N/O upper limit counts
  when any value below it is inside). The offset is OMEGA minus the SIMBA median at fixed x (±0.25 dex).
- **Compact / disky**: median split of the stellar r80/r20 (caesar) among the z≈0.4 quiescent centrals (log M\* > 10,
  sSFR < 0.2/t_H). Splits: 3.06 (m25n512, 28 + 28 galaxies) and 2.98 (m100n1024, 1640 + 1639 galaxies).

## 1. Resolution: the young particles set the limit

| | m25n512 | m100n1024 |
|---|---|---|
| SFR10 carried by one young particle | 0.23 M⊙/yr | 1.86 M⊙/yr |
| ≥ 5 young particles needs SFR10 ≳ | 1.2 M⊙/yr | 9.3 M⊙/yr |
| resolution mass, ≥ 30 star particles | log M\* 7.75 | 8.75 |
| resolution mass, ≥ 30 star + ≥ 5 young (z = 8 / 7 / 6) | 8.75 / 8.75 / 8.75 | 9.5 / 9.5 / 9.75 |
| resolved analogues (z = 8 / 7 / 6) | 56 / 80 / 87 | 195 / 330 / 537 |
| their mass range | log M\* 7.8–9.8 | 8.7–10.7 |
| median young particles in a resolved analogue | 7–8 | 7 |

- **Sampling noise.** Anything that scales with the young stellar mass (SFR10, and later Q_H and the recombination lines) carries a Poisson
  error of 1/√N_young. That is 45 % (0.19 dex) at the threshold of 5, and ~38 % (0.16 dex) at the typical 7. For comparison, OMEGA's
  Hα SFR errors are 0.01–0.29 dex.
- **Masses OMEGA covers.** Only **2 of 9** OMEGA galaxies (log M\* 8.75 and 9.14) lie above the m25n512 resolution mass, and **0 of 9** above the m100n1024 one.
  At OMEGA's SFRs (4–29 M⊙/yr) an m25n512 galaxy would carry 17–125 young particles. Resolution is not the problem in that box; the
  problem is that SIMBA does not make galaxies that star-forming at those masses (section 2).

## 2. Overlap with OMEGA

Fraction of OMEGA points inside the 95 % locus of the resolved analogues:

| plane | m25n512 | m100n1024 | median offset OMEGA − SIMBA (m25 / m100) |
|---|---|---|---|
| M\*–SFR | 3/9 | 2/9 | +0.57 / −0.22 dex |
| M\*–O/H | 2/9 | 2/9 | −0.28 / −0.18 dex |
| O/H–C/O | 1/9 | 1/9 | +0.15 / +0.09 dex |
| O/H–N/O | 4/6 (3 are upper limits) | 1/6 | +0.45 / +0.35 dex |
| **inside in all four planes** | **0/9** | **1/9** (ID 115) | |

Where SIMBA misses:
- **Low O/H.** The four OMEGA galaxies with O/H ≤ 7.68 (IDs 1, 22, 24, 131) lie 0.3–1.0 dex below the SIMBA relation at their mass, and
  no resolved analogue at any of the three redshifts reaches them. At log M\* ≈ 8–8.5, SIMBA's star-forming gas sits at 12+log(O/H) ≈ 7.9–8.1.
- **SFR at fixed mass.** In m25n512, OMEGA's Hα SFRs are 0.4–0.8 dex above SIMBA's SFR10. The m100n1024 offset (−0.22 dex) is not a
  better match: the young-particle cut only lets through m100n1024 galaxies with SFR ≳ 9 M⊙/yr.
- **C/O.** SIMBA's C/O is a tight function of O/H (log C/O ≈ −1.1 to −0.8 over the resolved range), while OMEGA's spans −1.17 to −0.52.
  Only ID 115 falls in the locus; where a comparison exists OMEGA lies higher (IDs 6, 76, 77 by +0.06 to +0.41 dex).
- **N/O.** The N/O detections lie 0.25–0.54 dex above SIMBA at the same O/H (IDs 6 and 115). ID 1 (log N/O = −0.69 at
  O/H = 7.54) has no SIMBA counterpart at all. The m25n512 "4/6" is mostly upper limits, which are easy to satisfy.
- **The galaxy that overlaps everywhere** is ID 115: the most massive (log M\* 9.14) and most metal-rich (8.18) OMEGA galaxy, and inside the
  m100n1024 locus in all four planes. That locus extends below the m100n1024 80 % resolution mass, because it includes the resolved high-SFR tail
  down to log M\* 8.7.

## 3. Merger trees

**Backward** (the main progenitors of the z≈0.4 quiescent galaxies; resolved = ≥ 30 star particles and ≥ 5 young):

| | z | progenitors (compact / disky) | resolved | median log M\* | median 12+log(O/H) | median log SFR10 |
|---|---|---|---|---|---|---|
| m25n512 (28 + 28) | 8 | 5 / 9 | 1 / 2 | 8.13 / 8.14 | 8.07 / 8.12 | −0.31 / −0.16 |
| | 7 | 8 / 14 | 4 / 5 | 8.08 / 8.27 | 8.07 / 8.14 | +0.06 / −0.15 |
| | 6 | 12 / 20 | 5 / 7 | 8.21 / 8.40 | 8.07 / 8.11 | −0.14 / −0.16 |
| m100n1024 (1640 + 1639) | 8 | 96 / 132 | 25 / 40 | 8.86 / 8.83 | 8.24 / 8.25 | +0.75 / +0.77 |
| | 7 | 192 / 288 | 47 / 51 | 8.89 / 8.87 | 8.21 / 8.21 | +0.74 / +0.58 |
| | 6 | 437 / 605 | 76 / 85 | 8.90 / 8.90 | 8.15 / 8.15 | +0.59 / +0.58 |

- **m25n512.** The progenitors sit inside OMEGA's mass range and in the upper half of its O/H range, but **0.6–1.1 dex below its SFRs**. Most of them are
  below that box's resolution mass, and only 14–32 of the 56 quiescent galaxies still have a catalogued progenitor at z≈6–8.
- **m100n1024.** Only 7–32 % of the quiescent galaxies have a progenitor in the catalogue at z≈6–8: the rest are below caesar's detection limit there.
  Those that exist sit at log M\* ≈ 8.9 with OMEGA-like SFRs and O/H ≈ 8.2, at the massive, metal-rich edge of OMEGA (next to ID 115).
- Figure (a) shows the median tracks from z≈8 to 0.4, with 16–84 % bars at z = 8, 6, 2, 1, 0.4.

**Forward** (resolved z≈6–8 analogues that are the main progenitor of a z≈0.4 quiescent galaxy; 68 % Wilson intervals):

| | z = 8 | z = 7 | z = 6 |
|---|---|---|---|
| m25n512 | 3/56 = 5 % (3–9 %) | 8/80 = 10 % (7–14 %) | 11/87 = 13 % (10–17 %) |
| m100n1024 | 65/195 = 33 % (30–37 %) | 97/330 = 29 % (27–32 %) | 158/537 = 29 % (28–31 %) |

- **Split by class:** in m100n1024, 13–14 % end compact and 16–21 % end disky; m25n512 splits roughly evenly.
- **Main branch only:** an analogue that merges into another branch counts as "not". Of the resolved analogues, 18/56, 32/80 and 44/87 (m25n512)
  and 107/195, 164/330 and 291/537 (m100n1024) are on the main branch of any z≈0.4 galaxy.

## 4. Do compact and disky progenitors already differ at z≈6–8?

The test uses progenitors with ≥ 30 star particles, with two-sample KS and Anderson–Darling statistics; effect sizes are Cliff's δ (compact vs disky) and the median
difference, with 68 % bootstrap intervals.

- **m100n1024 (74–519 per class): no.**
  - Every variable has |δ| ≤ 0.13. Median differences are ≤ 0.025 dex in O/H, C/O and N/O, ≤ 0.15 in SFR10/SFR100 and ≤ 0.02 in f_gas.
  - The largest effect is the stellar mass itself at z≈8: compact progenitors are 0.07 dex more massive (δ = 0.13).
  - The smallest p-values (C/O at z≈6, KS p = 0.06, AD p = 0.05) go with δ = −0.08, i.e. 0.02 dex.
- **m25n512 (5–19 per class): moderate effects, none significant (all KS p ≥ 0.067), and not reproduced by m100n1024.**
  - Compact progenitors are lower in O/H (δ = −0.43 at z≈7, −0.35 at z≈6, i.e. 0.03–0.10 dex) and burstier in SFR10/SFR100
    (δ = +0.31 to +0.33 at all three redshifts).
  - At z≈7 the compact ones are also 0.37 dex less massive, which alone can explain the O/H offset.
  - Read these as small-sample noise until a larger high-resolution volume confirms them.

## 5. Caveats

1. **Resolution.** Nebular lines come from the < 10 Myr star particles. A resolved m25n512 analogue holds a median of 7, so per-galaxy line
   luminosities carry ~0.16–0.19 dex Poisson noise before any modelling error. In m100n1024 the 10 Myr window is resolved only above
   ~9 M⊙/yr. Quote ensemble statistics, not individual galaxies; consider extending the window (with the matching change in the line
   ages) or resampling the young populations.
2. **Abundance pattern and yields.** SIMBA's C/O and N/O are nearly deterministic functions of O/H, set by its fixed type II, Ia and AGB
   yield tables (Davé+19). It has no channel for the N-enhanced, C-rich young systems OMEGA finds (e.g. ID 1). Its N/O sits 0.25–0.54 dex below
   the OMEGA detections, so predicted N lines will be systematically weak.
3. **Metallicity normalisation.** SIMBA is 0.2–0.3 dex metal-rich at fixed mass relative to OMEGA, and never reaches 12+log(O/H) < 7.8 among
   resolved analogues. Part of this may be on the observational side: direct-T_e abundances can be biased low by temperature fluctuations. The
   comparison is in the true abundance, not a line-ratio diagnostic.
4. **Star-formation variability.** SIMBA's star formation follows a smooth, pressure-regulated sub-grid law. Its SFR10/SFR100 is limited by particle
   sampling, not by resolved bursts. OMEGA's Hα SFRs sit 0.4–0.8 dex above SIMBA at fixed mass, the bursty regime SIMBA does not produce.
   Matching by Hα luminosity will therefore pick different SIMBA galaxies than matching by M\*.
5. **Volume vs resolution.** The only box that resolves OMEGA masses (25 Mpc/h) holds just 56 quiescent galaxies at z≈0.4 and 3–12 resolved
   z≈6–8 progenitors of them. The box with the statistics (100 Mpc/h) does not resolve OMEGA's masses or SFRs.
6. **Selection effects in the trees.** m100n1024 progenitors are catalogued only above caesar's mass limit, and the class split uses
   main branches. Both bias the progenitor samples to the most massive, earliest-forming progenitors.

## Reproduce

```bash
sbatch omega_feasibility/submit.sh           # build (~1 h) + analysis, from the repository root
# or open omega_feasibility/omega_feasibility.ipynb (RUN_BUILD = True to build interactively)
```
