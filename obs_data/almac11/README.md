# obs_data/almac11 — observed ALMA-C11 tables used by the cluster notebooks

`ks_table.csv` — Kennicutt–Schmidt placement of the ALMA-C11 quiescent galaxies (z ≈ 0.34–0.43),
copied verbatim from `~/Desktop/Projects/first_spectral_analysis/pilot_specphot/results/ks_table.csv`
(written by `scripts/plot_ks.py`, 2026-08-19 refit, file regenerated 2026-08-27 11:46 with beam-based size limits; copied 2026-08-27). Refresh by re-copying the
file after a `plot_ks.py` rerun — nothing here is derived.

Conventions of that table (see the `plot_ks.py` docstring):
- M_H2 from CO(3–2) with α_CO = 4.36 Msun/(K km/s pc²) **including helium**, R31 = 0.5
  (`MH2_fid`, `logMH2`); SFR = fiducial CIGALE `bayes.sfh.sfr` (`SFR`, `logSFR`, asymmetric log errors
  `logSFR_elo/ehi`); `tdep_Gyr = M_H2/SFR`; `tdep_MS_Gyr` = Tacconi+18 main-sequence value at the source z.
- Surface densities for CO-detected sources with a uv size: `Σ = 0.5 M / (π R²)` for gas AND SFR with the
  SAME radius, `r_kpc` = CO(3–2) uv-Gaussian FWHM_maj/2 (face-on half-light radius);
  `logSigmaH2` [Msun/pc²], `logSigmaSFR` [Msun/yr/kpc²].
- Censoring: `is_ul` = the size is an upper limit (unresolved) → `sigma_is_ll` = both Σ are LOWER limits
  (the point moves along a slope-1 arrow). `co_det = False` rows (jw8, ctrl*) have no Σ; hc11's line
  shape is unconstrained → no Σ either.
- `dlog_<rel>` / `nsig_<rel>` = offset from Kennicutt 98 / Bigiel+08 / de los Reyes & Kennicutt 19 in
  dex and in units of sqrt(scatter² + own errors²).

`almac11_gas_dust.csv` — the per-source ALMA-C11 measurement / SED table, copied verbatim from the same
`pilot_specphot/results/` directory (file of 2026-08-27 13:06; copied 2026-08-29). Per source: the ALMA dust-continuum
and CO(3–2) fluxes with their detection flags (`det_dust`, `det_co`, S/N, 3σ limits, method / engine / QA columns), the
observed Dn4000 / age, and the CIGALE fits of the `noagn` and `agn` runs (`logage_*`, `logMstar_*`, `logMdust_*`,
`AV_*`, SFR, …) with the fiducial choice `fid_run` and its `logMstar`, `logMdust`, `logage` (log yr), `fracAGN_fid`,
`MH2_fid`. Refresh by re-copying — nothing here is derived.

`age_sersic_sigma.csv` — the per-source ALMA-C11 structure table, copied verbatim from `pilot_specphot/results/age_sersic_sigma.csv`
(written by `scripts/plot_age_sersic_sigma.py`, file of 2026-08-29 13:58; copied 2026-08-29). Per source: the optical half-light radius
`re_kpc` with its origin `re_source` (COSMOS-Web JWST semi-major Sérsic radius `re_jwst_maj_kpc`, the circularised `re_cosmosweb_kpc`,
ACS/COSMOS `re_acs_kpc` where JWST is missing: hc5, hc9, hc10, hc12), the Sérsic index `n` (`n_source`), the CIGALE no-AGN age /
A_V / dust fraction, the ALMA dust and CO sizes (`r_dust_kpc`, `r_co_kpc`, their limit flags) and Σ_dust. Refresh by re-copying.
Consumer: `paper_figures_quenched_m25.ipynb` Part 6 (Σ_e = M*/(2π R_e²) of every ALMA-C11 source, `re_kpc`).

Consumers: `ks_tracks_quenched_m25.ipynb` (Part 4) via `OBS_CSV` (ks_table.csv); `paper_figures_quenched_m25.ipynb`
Part 5 via `OBS_GD_CSV` (almac11_gas_dust.csv: the age / M_dust/M* range of the `det_dust` sources, drawn as the band
of the observed parameter space) and Part 6 (every source as a point: fiducial-run `AV_<fid_run>`, `logMstar`, `logDGR` + `dgr_censor`). The simulated Σ_H2 there is corrected
by ×1.36 (He) before comparison because SIMBA's per-particle H2 mass is hydrogen-only.
