# obs_data/literature — published comparison samples used by the paper notebook

All files are copies (or column subsets) of the tables built in `~/Desktop/Projects/first_spectral_analysis/pilot_specphot/`
(the ALMA-C11 pilot analysis); nothing here is derived. Refresh by re-copying after the scripts named below are re-run.

`av_re_literature.csv` — LEGA-C + 3D-HST galaxies with BOTH an SED-fit A_V and an HST Sérsic half-light radius, built by
`pilot_specphot/scripts/fetch_av_re_literature.py` (`data/av_re_literature/av_re_literature.csv`, file of 2026-08-29; the
provenance columns `ra`, `dec`, `d4000n`, `re_err_arcsec` are dropped here, 6972 rows, log M* ≥ 10.2):
- **LEGA-C** (0.6 < z < 1.0, COSMOS, spectroscopic): `re_kpc` = LEGA-C DR3 (van der Wel+21) GALFIT semi-major half-light radius on
  ACS F814W; `av` = de Graaff+21 MAGPHYS total V-band attenuation (Charlot & Fall 2000 law, `av_code`); `av_fast` = the UltraVISTA
  FAST (Calzetti) A_V; `logM`, `logssfr`, `logage_lw` = `logage_fit` (MAGPHYS light-weighted age, log Gyr); `uv`, `vj`,
  `quiescent_uvj` from the Muzzin+13 EAZY rest-frame fluxes.
- **3D-HST** (0.2 < z < 6, five CANDELS fields): `re_kpc` = van der Wel+14 GALFIT semi-major radius on F125W (z < 1.5) or F160W;
  `av`, `logM`, `logssfr`, `logage_fit` (FAST age since the onset of the exponential SFH, log Gyr) = Momcheva+16 master
  catalogue (use_phot = 1, GALFIT flag 0); UVJ from the EAZY rest-frame fluxes.
- `re_5000_kpc` = `re_kpc` × (λ_rest / 5000 Å)^0.25 (van der Wel+14 quiescent size–wavelength correction), FlatLambdaCDM(70, 0.3);
  `q`, `n` = GALFIT axis ratio and Sérsic index. The ages of the two codes are NOT homogeneous with each other nor with the
  mass-weighted simulated ages — the paper notebook only uses them for its log(age/yr) > 9 cut.

`spilker18_legac.csv` — the eight Spilker+18 (ApJ 860, 103) LEGA-C passive galaxies, the `ref == "Spilker18"` rows of
`pilot_specphot/results/ks_comparison_samples.csv` (`scripts/build_ks_compilation.py`, columns verbatim): CO(2–1) M_H2 with
α_CO = 4.4 (`logMH2_h`, `mh2_ul` = 3σ upper limit), UV+IR SFR, no dust continuum and no tabulated sizes. The paper notebook joins
them by the `galaxy` id ("LEGA-C <id>" = the UltraVISTA / LEGA-C id) to the LEGA-C rows of `av_re_literature.csv` for A_V, M*, R_e.

Consumer: `paper_figures_quenched_m25.ipynb` Part 6 (figure G: Σ_e vs A_V coloured by M_dust/M_H2) — LEGA-C / 3D-HST running
medians under the cuts log M* > 10.25 and log(age/yr) > 9, Spilker+18 as hollow diamonds. ADF22-QG1 (Umehata+25) is typed
into that cell (`P6_UMEHATA`), with the paper tables it comes from.
