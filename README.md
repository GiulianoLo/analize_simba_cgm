# simbanator

A Python toolkit for analysing SIMBA cosmological simulations.  
Covers the full pipeline from raw Caesar HDF5 catalogs to photometric SEDs: progenitor tracking, merger detection, particle extraction, SED modelling with Powderday, and flux extraction through telescope filters.

---

## Installation

```bash
# Editable install — recommended for active development on the cluster
git clone https://github.com/GiulianoLo/analize_simba_cgm.git
cd analize_simba_cgm
pip install -e .

# SED modelling (requires hyperion + caesar)
pip install -e ".[sed]"

# All optional features (yt, sphviewer, fsps, svo_filters)
pip install -e ".[full]"
```

> **Cluster note:** always use `pip install -e .` so source changes are picked up immediately.  
> `pip install .` copies files to site-packages and will serve stale code after edits.  
> If you hit import errors pointing to a wrong version, add this cell at the top of your notebook:
> ```python
> import sys; sys.path.insert(0, '/path/to/analize_simba_cgm')
> ```

> **Heavy dependencies** (`yt`, `caesar`, `py-sphviewer`) are easiest to install via conda.  
> Install them in your conda environment first, then `pip install -e .` for the rest.

---

## Quick start

```python
import simbanator as sb

# Register a simulation once per machine (writes ~/.simbanator/config.json)
sb.add_simulation(
    "cis100",
    data_dir    = "/mnt/share/simbas/SIMBA_100",
    catalog_dir = "/mnt/share/simbas/SIMBA_100/Groups",
    file_format = "m100n1024_{snap:03d}.hdf5",
)

# Then use from any script or notebook
sim = sb.Simulation("cis100")
out = sb.OutputPaths(sim.name)   # output/cis100/<task>/ created on first access
```

---

## Package layout

```
simbanator/
├── io/
│   ├── simba.py          # Simulation – path resolution, Caesar/snapshot loading
│   ├── paths.py          # OutputPaths – structured output directory manager
│   └── config.py         # ~/.simbanator/config.json read/write helpers
├── analysis/
│   ├── progenitors.py    # caesar_read_progen, read_progen – merger-tree track FITS
│   ├── mergers.py        # Progenitor, Galaxy, process_galaxies_with_tracks,
│   │                     #   analyze_mergers – companion detection & classification
│   │                     #   Units: positions Mpc/h, r_half kpc/h, masses M☉
│   ├── particles.py      # extract_particles – per-galaxy/halo/aperture HDF5 subsets
│   ├── sfh_caesar.py     # HDF5BuildHistory – property histories from Caesar catalogs
│   ├── sfh_fsps.py       # compute_sfh, bin_sfh, load_sfh_file – FSPS SFHs
│   ├── sfh_utils.py      # smooth_resample_sfh, recent_sfr – de-burst snapshot-cadence
│   │                     #   SFR tracks (Gaussian kernel + uniform resample);
│   │                     #   sfr_delayed_bq, fit_delayed_bq – CIGALE sfhdelayedbq
│   │                     #   form + bounded fit (shared by 7b′ and aperture truth);
│   │                     #   build_mfrac_lookup/mfrac_of – powderday-consistent FSPS
│   │                     #   surviving-mass fraction (linear-Z snap, cached .npz);
│   │                     #   archaeological_sfh – formed-mass SFH on a fixed 0→t_obs
│   │                     #   grid (linear in weights: disjoint subsets sum exactly);
│   │                     #   projected_region_sfh – SFH of r_in<R≤r_out along a
│   │                     #   sightline (Hyperion aperture geometry, nstar_min skip)
│   ├── profiles.py       # radial_profile – surface-density / mean radial profiles
│   ├── quenching.py      # find_quenching_times, load_quenching_events
│   ├── history.py        # deprecated shim → sfh_caesar
│   └── sfh.py            # deprecated shim → sfh_fsps
├── sed/
│   ├── makesed.py        # MakeSED – Powderday setup + flux extraction (needs hyperion)
│   │                     #   extract_flux_* take aperture= (Hyperion SED aperture index) and
│   │                     #   uncertainties= (MC errors → <filter>_err); list_sed_apertures QC;
│   │                     #   read_selection_centers – Stage-1 selection h5 → RT-grid centres
│   ├── flux_extraction.py# flux_extraction, get_svo_filters – SED → photometry
│   │                     #   flux_unc= propagates an SED uncertainty → mJy_err / mag_err
│   │                     #   annular_flux_table – F(<r_out)−F(<r_in) between two
│   │                     #   cumulative-aperture flux tables (non-positive → NaN)
│   │                     #   attenuation_mag – A_λ = −2.5 log10(F_on/F_off), NaN-safe
│   │                     #   dust_luminosity – L_dust [W] of a raw powderday SED
│   │                     #   (∫νLν dlnν beyond 3 µm; optional dust_on−dust_off)
│   ├── cigale.py         # CIGALE 2025.0 end-to-end: write_cigale_input (flux table →
│   │                     #   data_file, DB band names incl. Subaru Suprime-Cam — SVO
│   │                     #   broad g/r/i/z map to CIGALE g+/r+/i+/z+),
│   │                     #   stack_cigale_inputs/parse_stacked_id (many catalogs → ONE
│   │                     #   data_file, ids re-keyed '<id>__<tag>': pcigale builds its
│   │                     #   model grid once per RUN and shares it across sources, so
│   │                     #   every SED sharing a pinned grid is a row, not a run;
│   │                     #   absent bands filled NaN, not the 0.0 FITS mask fill),
│   │                     #   prepare_run
│   │                     #   (pcigale.ini/.spec,
│   │                     #   replaces init+genconf; genconf-style docs as ini comments;
│   │                     #   fit_bands= manual fitted-band list, '<band>'/'<band>_err'
│   │                     #   mix ok — errors auto-paired, unselected bands predicted),
│   │                     #   MODULE_REGISTRY (configobj types per module: sfhdelayed(bq),
│   │                     #   sfhfromfile, bc03, nebular, dustatt_modified_CF00, dl2014,
│   │                     #   skirtor2016 — AGN torus, goes after the dust-emission module;
│   │                     #   free fracAGN incl. 0 for a no-AGN null test — restframe_
│   │                     #   parameters, redshifting),
│   │                     #   describe_run (module/param/variable reminder, full or compact),
│   │                     #   check/run/read_results/plot_seds wrappers, compare_results
│   │                     #   (truth vs bayes.*: per-galaxy print, offset/NMAD stats,
│   │                     #   one-to-one panels → <run_dir>/out/simba_vs_cigale.fits+.png),
│   │                     #   plot_parameter_priors (per gridded param: fitted distribution
│   │                     #   vs prior nodes + extend/refine/trim advice → param_priors.*),
│   │                     #   write_slurm_array (SLURM array over prepared run dirs; one task
│   │                     #   per dir by default, runs_per_task=N/max_array_tasks=M bundle a
│   │                     #   slice per task for thousands of small per-object runs — each run
│   │                     #   in its own subshell so one failure does not abort the slice;
│   │                     #   default re-fits with CIGALE-native timestamped out/ backups,
│   │                     #   skip_if_done=True for cheap resubmits),
│   │                     #   collect_results (vstack every <run_dir>/out/results.fits into
│   │                     #   one table + id_map= extra columns from the run name, broadcast
│   │                     #   over however many rows a run holds; missing/empty/truncated
│   │                     #   runs counted in .meta, never raised),
│   │                     #   validate_sfh_file (sfhfromfile contract: time from 0 in strict
│   │                     #   1 Myr steps, and the all-zero column that normalise=True turns
│   │                     #   into an all-NaN results.fits; called from prepare_run),
│   │                     #   write_sfhfromfile (smoothed SFH → run_dir/sfh.fits on the
│   │                     #   strict 1 Myr grid; shift_myr drops the OLDEST Myr so the
│   │                     #   recent end survives the age cap; refuses all-zero columns),
│   │                     #   pin_umin (dl2014 umin node whose emissivity matches a known
│   │                     #   L_dust/M_dust — anchors the fitted dust.mass to a truth
│   │                     #   value; returns node, target ε, offset_dex),
│   │                     #   grid_options/nearest_option/split_by_metallicity (per-galaxy
│   │                     #   metallicity priors: snap SIMBA Z to the strict CIGALE grids
│   │                     #   in log space, one sub-catalog+run per bc03 node),
│   │                     #   optical_only_run (clone a prepared run dir into its decoupled
│   │                     #   optical-only twin: no dl2014, IR bands unfitted but still
│   │                     #   predicted — the L_abs side of the energy-balance test),
│   │                     #   find_pcigale (executable from the dedicated conda env),
│   │                     #   sanitize_input_errors (abs() legacy negative errors in place),
│   │                     #   nmad (robust sigma).
│   │                     #   No simbanator imports — loadable standalone in a CIGALE env.
│   ├── dl2014_fit.py     # standalone Draine & Li (2014) fit of the IR residual (free
│   │                     #   normalization, no energy balance): subtracts the optical-only
│   │                     #   run's bayes.<band> stellar continuum, fits L_dust analytically
│   │                     #   per (qpah, umin, gamma) → <run_dir>/out/dl2014_results.fits.
│   │                     #   emissivity_table + --emissivity-out CLI export the DL2014
│   │                     #   ε(qpah, umin, gamma) [W/kg] table cigale.pin_umin consumes.
│   │                     #   Run under the CIGALE env python (needs pcigale.data).
│   └── parameters_master.py / parameters_master-nodust.py / parameters_master-agn.py / parameters_master-agn-nenkova.py
│       parameters_master-nenkova-i90.py / -i60.py / -i30.py
│                         #   SED_APERTURE_NAP/MIN_KPC/MAX_KPC – multi-aperture SEDs
│                         #   (read by the powderday patch documented in
│                         #    powderday_flux_quenched_m25.ipynb; stock powderday ignores them)
│                         #   -agn: dust_on + BH_SED=True (Hopkins+2007, BH_var=False;
│                         #    needs PartType5 in the Stage-0 cutouts)
│                         #   -agn-nenkova: as -agn but CLUMPY i=90 torus, 4 sightlines
│                         #    (provenance of the CEERS dusty_simdust_chab_agn_nenkova tree)
│                         #   -nenkova-i{90,60,30}: almac11 torus-inclination arms — CLUMPY
│                         #    i=90/60/30, SINGLE sightline THETA=[0] (findx=0 only); staged
│                         #    by analogues_specphot_almac11_rt.ipynb into
│                         #    output/cis100/sed_almac11/{dust_on,dust_off,nenkova_i90,i60,i30}
│                         #    (selection tables/maps in output/cis100/almac11_specphot/)
├── utils/
│   ├── geometry.py       # shrink_center, principal_axes, rotate_to_frame,
│   │                     #   sightline_unit_vectors, projected_radius (image-plane R)
│   ├── svo_filters.py    # download_svo_filters – fetch filter curves from SVO
│   ├── conversions.py    # Z_to_OH12, Dust_to_Metal
│   ├── search.py         # findsatellites
│   └── debug.py          # print_ram_usage
├── visualization/
│   ├── plots.py          # HistoryPlots – generic multi-panel history figure
│   │                     # plot_merger_rate_by_phase – bar chart of mergers by phase
│   │                     # plot_main_galaxy_track – single-galaxy trajectory (unwrapped)
│   │                     # plot_neighborhood_track – track + nearby galaxies + merger events
│   │                     # plot_all_galaxy_tracks – all galaxy tracks overlaid (unwrapped)
│   │                     #   All track plots: positions in Mpc/h, radius param in Mpc/h
│   ├── animation.py      # create_animation – GIF from x/y frame sequence
│   └── rendering.py      # ParticleProjectionRender, RenderRGB, SingleRender
│                         #   (requires yt + py-sphviewer)
└── data/
    ├── convert.py        # convert_hdf5_fits – Caesar HDF5 → FITS (legacy)
    └── snap_z_maps/      # bundled snapshot → redshift tables per simulation box
```

Repository-root cluster jobs and notebooks (the reduced-particle → Σ-profile pipeline):

```
├── build_profiles_job.py           # SLURM worker: fixed-binning Σ profiles from a shared plan;
│                                   #   exports the unit/field recipes reused by the reduced job
│                                   #   (header_units, _to_kpc, _to_msun, _detect, _components,
│                                   #    _nH, _temperature, _halo_of); _components = caesar HI/H2
│                                   #   split (H2 = 0.76·m·fH2 with n_H ≥ 0.13 cut, HI = 0.76·m·
│                                   #   min(nh, 1−fH2)), stamped H2_RECIPE — fixed 2026-08-27
├── build_reduced_particles_job.py  # SLURM worker: lean 100 kpc reduced particle files (ISM+CGM);
│                                   #   files carry attr h2_recipe; a stale/absent stamp gets
│                                   #   m_HI/m_H2 recomputed on the next run (backfill path)
│                                   #   star fields m_star/member/tform (tform = formation scale
│                                   #   factor, added 2026-08-27; older files are backfilled);
│                                   #   `vel` (n,3) km/s peculiar = Velocities x sqrt(a) in BOTH
│                                   #   groups (added 2026-08-28 for the per-zone kappa_rot of
│                                   #   the KS notebook; re-running a plan's sbatch backfills it)
│                                   #   Batched snapshot I/O: _catalog_pass (per-galaxy candidate
│                                   #   lists, halo-cached), _gather (slab-streamed reads at the
│                                   #   sorted union of indices; skips unneeded slabs), _Ctx (lazy
│                                   #   per-snapshot column store serving galaxies from memory).
│                                   #   Extensible field producers backfill new fields from the
│                                   #   stored idx without redoing geometry; datasets are lzf.
│                                   #   Env: DUST_PLAN, REDUCED_RMAX_KPC, REDUCED_PREFIX,
│                                   #        REDUCED_OVERWRITE, REDUCED_GATHER_MB
├── submit_reduced_particles.sh     # sbatch wrapper (array over snapshots; plan = 1st script arg
│                                   #   or DUST_PLAN env, per anchor)
├── submit_find_progen_m25.sh       # sbatch wrapper (4-task array over disjoint sidecar ranges)
├── find_progen_m25_job.py          # one-time cluster job: merger-tree links for the SIMBA_25
│                                   #   catalogs (they ship WITHOUT tree_data, unlike cis50/100).
│                                   #   The share is file-level read-only -> progen_finder(save=
│                                   #   False) + sidecar files output/cis25/progen_links/*.hdf5
│                                   #   ((ngal,2) progen_galaxy_star; readers fall back via
│                                   #   progen_tree_file). Needs both snapshots of each pair
│                                   #   (slist indices -> particle IDs); resumable per sidecar
├── quench_mode_vs_sigma_gas.ipynb  # multi-z quench-mode analysis. Part 2 defines the shared
│                                   #   helpers every stage plot rides on: _stage_key/_stage_stack/
│                                   #   _cached_rows (record → row_<stage> → progenitor → cached
│                                   #   profile resolution, incl. the sft_p500/../qt_p1000 offset
│                                   #   pseudo-stages), _med_band_floor (floored log-median bands),
│                                   #   load_reduced(keys=) selective reads, and the parallel
│                                   #   profile cache build (CACHE_WORKERS fork pool)
├── powderday_flux_quenched_m25.ipynb # Powderday flux catalogs for quenched (0.2/τ, logM*>10,
│                                   #   ngas>20) cis25 galaxies at z≈0.3/0.6/0.7/1/2, split by
│                                   #   weak/strong AGN coupling (AGN_CLASSIFIER: pre_threshold =
│                                   #   ungated <w_jet> over [SFT-1 Gyr, SFT] < 0.1 / >= 0.5,
│                                   #   adopted 2026-08-28 | quench_window | pre_jet; all three
│                                   #   labels in the selection FITS; switching archives the old
│                                   #   tables as *_<tag>.fits + plots to powderday_quenched_<tag>/);
│                                   #   per-anchor gated
│                                   #   history+BH builds → sample stats (Part 2b: the class rule
│                                   #   vs plain terciles / quench_mode post-t_AGN terciles / the
│                                   #   8p6a epoch class along (M_BH, f_Edd, f_gas) ->
│                                   #   agn_classifier_axes.png, agn_classifiers_compare.fits;
│                                   #   Part 2c: candidate default AGN_CLASSIFIER="pre_jet" =
│                                   #   terciles of the UNGATED jet weight over [SFT-1 Gyr, SFT]
│                                   #   (coupling_pre_jet) + coupling onset lead/concurrent/late,
│                                   #   validity ledger + agn_classifier_prejet_{axes,clock}.png;
│                                   #   both labels + w_pre/agn_onset stored in the selection FITS;
│                                   #   Part 2d: the feedback-event sequence (ignition, M_BH
│                                   #   threshold, jet onset, SFT, gate, QT) + coupling AT those
│                                   #   points per class, and the pre-SFT vs [SFT,QT] selection
│                                   #   scorecard (terciles / thresholds / jet lead) ->
│                                   #   agn_feedback_evolution.png, agn_selection_{windows,
│                                   #   scorecard}.png, agn_classifier_{windows,tracks}.fits
│                                   #   (per-galaxy critical times + the tracks behind the stacks:
│                                   #   the inputs of paper_figures_quenched_m25.ipynb Part 1);
│                                   #   Part 2e: paper versions of the 2d figures for the two
│                                   #   rules compared, pre-SFT threshold vs [SFT,QT] terciles
│                                   #   (P2E_RULES) -> paper_agn_{feedback_sequence,
│                                   #   selection_windows,selection_scorecard}.{png,pdf};
│                                   #   coupling_pre_threshold = the adopted driver rule)
│                                   #   → dust_on/off RT over
│                                   #   5 log-spaced apertures (10–160 kpc, powderday patch) →
│                                   #   per-aperture flux catalogs with MC errors
├── ks_tracks_lib.py                # pure numpy/h5py helpers for the KS-track notebook (no
│                                   #   simbanator import -> unit-testable off the cluster:
│                                   #   tests/test_ks_tracks.py): face_on_R (pos @ evecs, columns =
│                                   #   axes), half_mass_radius, sfr_window (archaeological SFR from
│                                   #   the reduced `tform`), measure_ks (fixed + R50 apertures,
│                                   #   member-only), ks_columns (He x1.36, 0.5 M/pi R50^2
│                                   #   convention, SFR=0 upper limits), build_stage_records
│                                   #   (critical epochs; H2 trough after QT) + attach_bh_stages /
│                                   #   nearest_row (STAGES_BH agn_ign / jet_on from the m25 Part 2d
│                                   #   windows table, times relative to SFT), RELATIONS (K98/B08/RK19),
│                                   #   interp_track / grid_stats / ecdf (histories on the t - t_QT clock),
│                                   #   kappa_rot (Sales+12 K_rot/K, the m25 8j0 estimator) +
│                                   #   measure_zone_kinematics (kappa of gas / H2 / stars + stellar
│                                   #   age in the spherical KIN_ZONES ap3kpc / ann10kpc / ap10kpc;
│                                   #   needs `vel` in the reduced files, else NaN kappas)
├── ks_tracks_quenched_m25.ipynb    # Kennicutt-Schmidt EVOLUTION TRACKS of the m25 quenched sample:
│                                   #   critical epochs (sSFR peak/SFT/QT/post-quench/H2 trough/anchor
│                                   #   + AGN ignition / jet onset attached from
│                                   #   agn_classifier_windows.fits, 2026-08-28) via the per-anchor
│                                   #   histories + progen links -> plan for
│                                   #   build_reduced_particles_job.py (prof_kstracks) -> Sigma_H2 /
│                                   #   Sigma_SFR from the CAESAR member particles per epoch (Part 3;
│                                   #   Part 3b: kappa_rot of gas / H2 / stars + stellar age in the
│                                   #   core sphere / outskirt shell / 10 kpc rung at every epoch ->
│                                   #   ks_stage_kinematics.fits, incremental, re-measures files once
│                                   #   their `vel` is backfilled; anchor QC vs annulus_kinematics) ->
│                                   #   binned AVERAGE tracks (terciles of anchor SFR / M* / M_dust/M* /
│                                   #   Sigma_dust, own Sigma_H2, M_H2/M*, KS region, AGN class; 3 z
│                                   #   panels + all-z, R50(H2) only) vs the observed ALMA-C11 QGs
│                                   #   (obs_data/almac11/); ks_binned_tracks.csv; Part 5 = the KS
│                                   #   regions (below/on/above B08 at the track end) followed on the
│                                   #   t - t_QT clock: ks_track_histories.fits (history + BH history +
│                                   #   caesar rotation via progen), quench timing, AGN, fdust-age plane,
│                                   #   kappa_rot, H2 extent -> ks_regions_*.png, ks_region_properties.csv,
│                                   #   ks_galaxies.fits (one row per Q galaxy: region, class, clock,
│                                   #   stage snapshots; read by paper_figures_quenched_m25.ipynb).
│                                   #   AGN class = KS_AGN_CLASSIFIER (pre_threshold since 2026-08-28):
│                                   #   agn_class_<tag> of the selection table, applied to the cached
│                                   #   epochs on load (refresh_agn_classes; no rebuild); strength
│                                   #   variable COUP_VAR (w_pre | xstr_quench); non-legacy rules write
│                                   #   to ks_tracks_<tag>/ + plots/ks_tracks_<tag>/ (caches stay put)
├── paper_figures_quenched_m25.ipynb # PAPER FIGURES of the m25 quenched sample: pure reads of the
│                                   #   caches of the two notebooks above (no simulation access, no
│                                   #   simbanator; runs wherever output/cis25 is visible). Part 1 =
│                                   #   the AGN classifier figures of m25 Part 2e from
│                                   #   agn_classifier_{windows,tracks}.fits (feedback-event sequence,
│                                   #   pre-SFT threshold vs [SFT,QT] terciles tracks, scorecard +
│                                   #   sweeps); Part 2 = the KS Part 5c presentation figures (AGN
│                                   #   composition + strength, dust, kinematics of the KS regions on
│                                   #   the t - t_QT clock) from ks_tracks/ks_galaxies/ks_track_histories;
│                                   #   Part 3 = NEW test: 10.5 < log M* < 11.2 quenched galaxies
│                                   #   binned by their anchor A_V (annulus_av_allincl, core ap3kpc,
│                                   #   median over 4 sightlines; terciles, same edges everywhere)
│                                   #   followed SFT -> QT -> track end on the KS plane, one column
│                                   #   per AGN class (+ all classes) + the A_V ECDFs; Part 4 = m25
│                                   #   Part 8 (T4/T8) re-drawn without scatter / rank stats from
│                                   #   annulus_ism_truth + annulus_av_allincl + aperture_truth +
│                                   #   annulus_kinematics: figure D = A_V, M_dust/M*, Sigma_dust
│                                   #   (log axes) of each annulus vs the stellar age of the same
│                                   #   annulus (one colour per annulus, one column per class + SF),
│                                   #   figure E = radial ladder core (0-3.2) / outskirt (3.2-10) /
│                                   #   10-32 kpc down the y axis, medians + galaxy-bootstrap CIs
│                                   #   along x in A_V, M_dust/M*, Sigma_dust, stellar age,
│                                   #   kappa_rot^gas per quenched class, figure F (Part 4d) = the
│                                   #   radial trend across the 4 annuli (A_V, M_dust/M*, M_gas/M*,
│                                   #   M_H2/M*, age, kappa_rot gas/H2/stars) per class + SF controls
│                                   #   for log M* >/< 10.25; Part 5 = the quench sequence in the
│                                   #   (kappa_rot, stellar age) plane from ks_stage_kinematics at the
│                                   #   critical points AGN ignition -> jet onset -> SFT -> QT -> end
│                                   #   (the first two need the KS notebook of 2026-08-28 evening+;
│                                   #   fallback SFT -> QT -> end): figure A = kappa_rot^H2 / ^* of the
│                                   #   core sphere and the outskirt shell vs the zone's stellar age,
│                                   #   per anchor-A_V tercile, one column per class; figures B = the
│                                   #   critical points on a categorical axis, classes side by side
│                                   #   per A_V bin, core / outskirt rows (zone-coloured frames):
│                                   #   kinematics_sequence (kappa H2 0-1, kappa stars zoomed),
│                                   #   stellar_sequence (sSFR, age, dt / t_cosmic per interval),
│                                   #   ism_sequence (M_dust/M*, M_H2/M*); sSFR and the fractions
│                                   #   from the tracks' projected apertures; the ALMA-C11 dust- /
│                                   #   CO-detection ranges of obs_data/almac11/almac11_gas_dust.csv
│                                   #   shaded; figure C = paired per-galaxy changes (d kappa, dt)
│                                   #   over each interval between consecutive points; for log M*
│                                   #   >/< 10.25, dashed kappa = 0.3, class palette teal / amber /
│                                   #   dark red (CLASS_COLOR_PRES); AGN_CLASSIFIER must be the rule both
│                                   #   notebooks ran with; Part 6 = figures G and H (P6_FIGS), no
│                                   #   fitted plane: G = paper_sigma_age, Sigma_e vs stellar age, the
│                                   #   observed points coloured by A_V, the strong-coupling quenched
│                                   #   models split at log(M_dust/M*) = -3.75 (P6_SPLIT) into a dusty
│                                   #   (N=5: older, rotating H2 disc kappa_H2 0.8 vs 0.3) and a dust-poor
│                                   #   running median; H = paper_ism_prediction, the ISM content an
│                                   #   observer cannot cheaply measure (rows M_dust/M*, M_H2/M*) as the
│                                   #   models' running median against what they can (columns age,
│                                   #   Sigma_e, SED sSFR; weak / intermediate / strong, no age cut on
│                                   #   the models — per-figure options classes / logage_min / model —
│                                   #   + all quenched dotted), drawn from the CIGALE fits of the mock
│                                   #   CORE photometry (model="cigale": tables/cigale_region_results
│                                   #   of m25 Part 7f, P6_CIG_REGION core 0-3.2 kpc, dust_on arm:
│                                   #   M*, M_dust, bayes.sfh.sfr, pinned age of the fit; M_H2 = the
│                                   #   region's SIMBA H2 over the CIGALE M*; Sigma_e = the sim
│                                   #   half-mass radius with the core M* aperture-corrected to 0-32
│                                   #   kpc; 172/178 galaxies fitted, closure -0.06 / -0.08 dex in M* /
│                                   #   M_dust), the measured ALMA-C11 / Spilker+18 / ADF22-QG1 values
│                                   #   as the check (sSFR orders the dust content in every class, age
│                                   #   in weak / intermediate; detections 0.7-1.6 dex above the
│                                   #   tracks, the ALMA-C11 limits on them) — every ALMA-C11
│                                   #   source (fiducial CIGALE values, COSMOS-Web / ACS R_e and Sersic n
│                                   #   of obs_data/almac11/age_sersic_sigma.csv, censors as arrows), the
│                                   #   Spilker+18 LEGA-C passive galaxies (CO only) and ADF22-QG1
│                                   #   (Umehata+25, typed in: P6_UMEHATA), the LEGA-C and 3D-HST running
│                                   #   medians of obs_data/literature/av_re_literature.csv where both
│                                   #   axes exist; other axes on offer: A_V, R_e, kappa_H2 (8j0, models
│                                   #   only), Sersic n (observed only); model="sim" (figure G, the
│                                   #   default P6_MODEL) = 0-10 kpc dust / 1.36 M_H2 over the 0-32 kpc
│                                   #   stars, Sigma_e from the projected half-mass radius of the
│                                   #   1/3.2/10/32 kpc curve of growth, sSFR = SFR100/M* of the 0-32
│                                   #   kpc stars; models + literature cut at log M* > 10.25 and
│                                   #   log(age/yr) > 9; H_sim = paper_ism_prediction_sim, the grid of H with
│                                   #   model="sim" (the truth in the 0-10 / 0-32 kpc discs: the estimator and
│                                   #   the core at once; against the caesar catalogue of the same galaxies
│                                   #   the 0-10 kpc disc holds +0.26 dex more dust, +0.34 dex more gas, the
│                                   #   0-100 kpc rung +1.3 dex: no rung is the catalogue galaxy) ->
│                                   #   plots/paper_m25[_<tag>]/paper_*.{png,pdf},
│                                   #   paper_{ks_av_bins,dust_vs_age_annuli,core_vs_outskirt,radial_profiles,
│                                   #   kappa_vs_age,kappa_sequence,kappa_intervals,sigma_age,
│                                   #   ism_prediction,ism_prediction_sim}{,_points}.csv
├── paper_ism_prediction_boxes.ipynb # figure H from the caesar CATALOGUES alone (no particles, RT or
│                                   #   CIGALE): the 100 / 50 / 25 Mpc boxes at the m25 anchors (z=0.3-2,
│                                   #   ANCHORS; BOXES = catalogue dir + pattern under SHARE, default
│                                   #   /mnt/home/share/simbas, env SIMBA_SHARE for a local run through
│                                   #   the mount). Part 1 reads every galaxy with log M* > 9.5 per
│                                   #   (box, anchor) with h5py (masses, sfr/sfr_100, ngas/nstar, central,
│                                   #   ages.mass_weighted, half-mass radii kpccm -> physical, kappa_rot,
│                                   #   M_BH, f_Edd) into output/box_resolution/ism_prediction/
│                                   #   ism_prediction_catalogue.fits (incremental: only the pairs the
│                                   #   cache lacks are read), applies the m25 Part 1 rule on the
│                                   #   catalogue columns (log M* > 10, sSFR < 0.2/t_H, >= 21 gas / >= 20
│                                   #   star particles, M_dust >= 1e-4 M_H2; funnel per (box, anchor)) ->
│                                   #   Q, QM (log M* > 10.25); Part 2 = the figure-H grid (M_dust/M*
│                                   #   and the gas row: SIM_GAS_ROW 'fgas' (default) = the whole-galaxy
│                                   #   M_gas/M* for the simulation, 'fh2' = 1.36 M_H2/M*; the observed
│                                   #   points stay the alpha_CO H2 either way; not nested: caesar sums
│                                   #   HI / H2 over the halo gas assigned to the galaxy, masses.HI >
│                                   #   masses.gas, median M_gas / 1.36 M_H2 = +0.08 dex in cis100 / cis50,
│                                   #   +0.26 in cis25; lfh2 stays the H2 in every table; vs age, Sigma_e =
│                                   #   M*/2 pi R_e^2 with R_e = 0.75 x the 3-D stellar half-mass radius,
│                                   #   sSFR) with ONE running median
│                                   #   per box (bootstrap band, Spearman per panel), the m25 REFERENCE
│                                   #   track overlaid (M25_REF_MODEL: 'catalogue' = the quenched galaxies
│                                   #   of the m25 sample, tables/powderday_quenched_selection_pt.fits, read
│                                   #   from the cis25 catalogue rows - the same whole-galaxy columns, the
│                                   #   m25 classes, binned like figure H (4 bins); 'sim' = figure H_sim's
│                                   #   paper_ism_prediction_sim.csv (the truth in the 0-10 / 0-32 kpc
│                                   #   discs); 'cigale' = figure H's CIGALE-core paper_ism_prediction.csv;
│                                   #   no particle rung is the catalogue galaxy, see H_sim above;
│                                   #   SHOW_M25_CLASSES adds the class tracks), the same
│                                   #   observed points (ALMA-C11 / Spilker+18 / ADF22-QG1, A_V colour,
│                                   #   obs_vs_track offsets per box) -> paper_ism_prediction_boxes
│                                   #   .{png,pdf,csv,_points.csv}; Part 3 = the catalogue's only AGN
│                                   #   state: the same grid for P3_BOX split by SIMBA's jet criterion at
│                                   #   the anchor (log M_BH > 7.5, f_Edd < 0.2; Mann-Whitney contrasts,
│                                   #   jet fraction per box / anchor; the m25 reference weak / strong tracks as
│                                   #   the reference) -> paper_ism_prediction_agnstate_<box>.*; Part 4 =
│                                   #   the 50 Mpc feedback VARIANTS (VARIANTS: cis50 / cis50nox /
│                                   #   cis50noagn; cis50nojet configured) on the same grid, one track
│                                   #   per variant with finer bins (P4_NBIN 8, P4_NMIN 3), + the printed
│                                   #   per-panel median offsets of the detections from each variant's
│                                   #   track and the median |offset| over the dust / H2 panels (which
│                                   #   physics puts the quenched galaxies where the data are) ->
│                                   #   paper_ism_prediction_variants_m50.*; the variant catalogues live
│                                   #   in SHARE/SIMBA_50/<variant>/Groups (ROE_CATALOG_URL, wget -c);
│                                   #   Part 5 = the AGN state INSIDE each variant (agn_state_contrasts
│                                   #   per variant, grid split by state -> paper_ism_prediction_agnstate
│                                   #   _m50_<variant>.*), the same state across the variants (colour =
│                                   #   state, line = variant; Mann-Whitney vs the fiducial run's galaxies
│                                   #   in that state -> paper_ism_prediction_agnstate_variants_m50.*) and
│                                   #   the joint scorecard (every population's median offsets over the
│                                   #   dust / H2 panels, log M_dust/M_H2 vs the observed, ranked by the
│                                   #   larger |offset| -> ..._variants_m50_scorecard.csv; bands only for
│                                   #   >= P5_BAND_MIN galaxies). Result: at the same BH state the X-ray
│                                   #   channel is what removes the dust (s50 jet -4.7 vs s50nox jet -3.1),
│                                   #   every dust-rich population is 0.4 dex under the observed H2.
│                                   #   Part 6 = the m25 AGN COUPLING CLASSES for the other boxes: main-
│                                   #   branch histories of mstar / sfr / mgas / mbh / bh_fedd from the
│                                   #   in-catalogue trees (cis25: the progen_links sidecars) for every
│                                   #   anchor whose catalogue ladder END_SNAP..anchor is on disk (one pass,
│                                   #   each catalogue read once -> ism_prediction_histories_<box>.hdf5),
│                                   #   SFT / QT with find_quenching_times, w_pre over [SFT - 1 Gyr, SFT],
│                                   #   pre_threshold cuts 0.10 / 0.50 (agn_class3) and the class SCHEME
│                                   #   used everywhere (CLASS_SCHEME 'two' default: weak / strong only,
│                                   #   the rule's intermediate galaxies redistributed at TWO_CLASS_THR =
│                                   #   0.30 on w_pre; 'three' = the rule) -> ism_prediction_agn_classes.csv,
│                                   #   merged into Q / QM (agn_class, w_pre, t_sft, t_qt, agn_onset);
│                                   #   cis25 labels checked against powderday_quenched_selection_pt.fits;
│                                   #   grid per box split by class (the m25 reference class tracks)
│                                   #   -> paper_ism_prediction_agnclass_<box>.*, weak / strong across
│                                   #   the boxes -> paper_ism_prediction_agnclass_boxes.* + the scorecard
│                                   #   over every population (offset_scorecard). ISM_H_BOXES env narrows
│                                   #   a test run; the m50 z < 1.15 anchors need ROE snaps 101-133.
│                                   #   Part 7 = the DUSTY TAIL, the galaxies behind the medians: per
│                                   #   population (box, AGN state, class) the percentiles of log M_dust/M*
│                                   #   and the number / fraction at or above P7_THR (-3.5 / -3.2 / -3.0;
│                                   #   the ALMA-C11 detections span -3.8 to -2.6) -> paper_ism_prediction
│                                   #   _dusty_tail.csv; the tail (>= P7_TAIL -3.2) vs the rest per box
│                                   #   (Part 6b contrasts + state / class / anchor mix); figure = the
│                                   #   M_dust/M* row per box with EVERY galaxy as a point, the tail in the
│                                   #   box colour, the running median + 84th / 95th percentiles, the
│                                   #   observed points -> paper_ism_prediction_dusty_tail.{png,pdf}
│                                   #   (P7_Y = 'fh2' for the H2 fraction). Result: 8 % of the cis100 and
│                                   #   11 % of the cis50 quenched galaxies (4 % in cis25) sit at or above
│                                   #   log f_dust = -3.2, 4-6 % above -3.0 (the tail: younger, lower-mass,
│                                   #   more satellites, less often in jet mode, 2x the gas particles);
│                                   #   in s50nox / s50noagn the majority does (60 / 77 %) - the medians
│                                   #   hide a real minority.
│                                   #   Part 8 = THREE DUST SIDES at M_dust/M* = 1e-5 and 1e-3.5 (P8_EDGES:
│                                   #   dust-rich >= 1e-3.5, undetected [1e-5, 1e-3.5), no-dust < 1e-5 incl.
│                                   #   zero dust) x the class: the side fractions per class per box (Wilson
│                                   #   68 %), the class x side table with chi-square p + Cramer's V, the
│                                   #   best class -> side rule vs the majority baseline, Fisher's p + the
│                                   #   weak-vs-strong odds at each end (dust-rich vs rest, no-dust vs rest),
│                                   #   the AUC of w_pre against age / sSFR / M_gas/M* / M* / Sigma_e / kappa
│                                   #   / jet / central / z / tau_q at each end + Spearman rho with the
│                                   #   ordinal side -> paper_ism_prediction_dusty_split{,_stats,_predictors}
│                                   #   .csv; contrasts weak vs strong within each side and the ends vs the
│                                   #   undetected middle within each class; per-box grids with the side x
│                                   #   class medians (colour = class; solid dust-rich / dashed undetected /
│                                   #   dotted no-dust) -> paper_ism_prediction_dusty_split_<box>.* and the
│                                   #   summary figure (per box the side fractions per class; the AUC per
│                                   #   predictor at each end) -> paper_ism_prediction_dusty_split_summary.*
│                                   #   Part 9 = the X-RAY EXPOSURE after quenching from the same histories:
│                                   #   E_x = int w_jet [f_gas < 0.2] dt from SFT to the anchor (the gated
│                                   #   channel; duty f_x, gas-poor time T_gaspoor, jet weight while gas-poor
│                                   #   r_x, ungated E_jet, the pre-SFT E_x, the gate's timing t_gate - t_SFT,
│                                   #   the quenching sequence = f_gas at SFT below / above 0.2) ->
│                                   #   ism_prediction_xray_exposure.csv, merged into Q / QM; the target is
│                                   #   the dust-to-gas ratio (ldg; the gate is a gas cut) next to M_dust/M*
│                                   #   (lfd, zero dust at P9_FLOOR): the exposure per class, the populations
│                                   #   (first X-ray episode never / before SFT / in [SFT,QT] / after QT, the
│                                   #   sequence x class -> ..._populations.csv), every quantity as a
│                                   #   predictor (Spearman with both targets, overall and per class, AUC at
│                                   #   the dust ends -> ..._predictors.csv), E_x / dt_SFT / T_gaspoor
│                                   #   terciles x class (-> ..._stats.csv), contrasts (never-exposed vs
│                                   #   exposed, the sequences per class), the figure (D/G vs E_x per class
│                                   #   with the s50nox / s50noagn medians as the zero-exposure reference, vs
│                                   #   time since SFT per onset population, vs T_gaspoor per r_x tercile,
│                                   #   the rank correlations) -> paper_ism_prediction_xray_exposure.*
│                                   #   Result: 98-100 % of the classified quenched galaxies are exposed and
│                                   #   the D/G saturates within ~0.5 Gyr (-1.4 -> -3.5); the 39 never-
│                                   #   exposed cis100 galaxies (1.3 %: undergrown BH, 90 % satellites) sit at
│                                   #   the observed log M_dust/M* = -3.0 with kappa_gas 0.85 = the s50nox
│                                   #   phenotype inside the fiducial box; among the exposed the first
│                                   #   episode's timing and the sequence (gas-poor before SFT keeps 0.5 dex
│                                   #   more D/G) set the dust, the class only modulates it (strong keeps
│                                   #   0.15-0.5 dex more at fixed exposure; the m25 figure-G direction).
│                                   #   Caveats: whole-galaxy (FOF) quantities, no sightlines; the
│                                   #   gas-particle cut is a gas mass floor 8x higher in m100 / m50 than
│                                   #   in m25; each variant's track is its OWN quenched population
├── paper_xray_classes_m25.ipynb  # PAPER FIGURES under the X-RAY CLASSIFICATION: one ordered notebook of pure
│                                   #   reads that rebuilds the figures of paper_figures_quenched_m25 (A) and
│                                   #   paper_ism_prediction_boxes (B) under one official division of the
│                                   #   quenched galaxies, XSCHEME (Part 0): "onset" (adopted 2026-08-30) = the
│                                   #   TIMING OF THE FIRST X-RAY EPISODE of B Part 6a (t_onset = first history
│                                   #   snapshot with x_coup = w_jet [f_gas < 0.2] >= 0.5, B Part 9 panels d-f):
│                                   #   early = by QT (B's lead + concurrent merged, user 2026-08-30; XREF, the
│                                   #   reference every other class is contrasted with), late = after QT, never
│                                   #   = no episode by the anchor; no event -> no_event. "onset_sft" splits the
│                                   #   same episode at SFT instead of QT: before (= B's lead) / after (=
│                                   #   concurrent + late; XREF) / never; "onset4" keeps lead /
│                                   #   concurrent apart; "rx" = the previous jet weight while gas-poor r_x =
│                                   #   E_x / T_gaspoor cut at R_X_CUT (0.97 = B Part 9's cis25 tercile edge;
│                                   #   0.5 the earlier tail) -> xlow / xhigh; the four-way onset label and the
│                                   #   rx label are always in the table (onset_class, rx_class). Every section reads
│                                   #   XCLASSES / XREF / XCOL / XNAME / XSHORT / XABBR, so the switch changes
│                                   #   every figure (N classes: one column / line per class, Mann-Whitney per
│                                   #   class vs XREF, stacked significance strips). Baseline = fast / slow
│                                   #   quenchers, log10(tau_q / t_QT) < FAST_CUT = -1.25. Part 0 ->
│                                   #   output/cis25/plots/paper_xray/xray_classes.csv (every galaxy of every
│                                   #   classified anchor of cis100 / cis50 / cis25; the m25 sample = the cis25
│                                   #   rows on (snap, gal_id): all 266 Q present) + counts / per-class medians
│                                   #   / crosstabs (x fast-slow, x old class, x the other scheme). Sections: 1
│                                   #   = A Parts 1b + 1c merged (paper_xray_feedback_tracks; columns fast/slow
│                                   #   | the classes); 2 = A Part 3 at log M* > 10.25 (A_V terciles, a class
│                                   #   column with < P3_SPLIT_MIN = 12 galaxies drawn unsplit) ->
│                                   #   paper_xray_ks_av_bins_R50_H2 + .csv; 3 = A Part 4d (both mass samples,
│                                   #   lines SF + the classes) -> paper_xray_radial_profiles_*; 4 = A Part 5
│                                   #   figure B (mgt10p25, one all-A_V column) -> paper_xray_{kinematics,
│                                   #   stellar,ism}_sequence_mgt10p25 + paper_xray_kappa_{sequence,intervals}
│                                   #   .csv; 5 = B Part 9 rows 1-3 from the exposure table (D/G vs E_x, vs
│                                   #   dt_SFT, vs T_gaspoor, the classes on every row, s50nox / s50noagn as
│                                   #   the zero-exposure reference; class contrasts inside terciles of the
│                                   #   two clocks) + THE EXPOSURE AS A DIAGNOSTIC INSIDE EACH CLASS (Spearman
│                                   #   of E_x, f_x, r_x, T_gaspoor, dt_SFT, t_onset - t_SFT with log D/G and
│                                   #   log f_d per box x class -> paper_xray_exposure_diagnostic.csv) ->
│                                   #   paper_xray_dg_vs_exposure{.png,.pdf,.csv,_classes.csv} + THE
│                                   #   DUSTY TAIL PER CLASS (unit-area distributions of M_dust/M* and
│                                   #   M_dust/M_gas per class, the class shares of the whole box
│                                   #   against the shares above the threshold -4 / -2.5, hypergeometric
│                                   #   P of the tail count -> paper_xray_dust_distributions +
│                                   #   _dust_tail_composition.csv, the cell placed after the ISM grid);
│                                   #   5c (after the tail cell) = THE COUPLING x ONSET CROSS (the weak /
│                                   #   strong division of B 6a x the X-ray classes: tail fraction + share
│                                   #   per cell, hypergeometric P, stratified Fisher / MW in both
│                                   #   directions, AUC of E_x / f_x / T_gaspoor / onset lead / w_pre / age
│                                   #   for tail membership, the tail split at age 2 Gyr — young under-dosed
│                                   #   vs old fully-dosed; figure: tail fraction per coupling x onset cell
│                                   #   + the dose-age plane -> paper_xray_coupling_onset{.png,.pdf,.csv});
│                                   #   5b = B Part 2's
│                                   #   figure-H grid per box + across the boxes, tracks per class + the small
│                                   #   classes (<= G5_POINTS_MAX) as points -> paper_xray_ism_grid_*; 6 = the
│                                   #   fiducial KS plane (all / one column per class, below / on / above B08
│                                   #   counts of the track ends next to the observed ones) ->
│                                   #   paper_xray_ks_fiducial_R50_H2 + .csv. Inputs: A's tables, ks_tracks_pt/
│                                   #   {ks_tracks,ks_galaxies}.fits, ks_tracks/ks_stage_kinematics.fits, the
│                                   #   boxes' catalogue cache + classes + exposure CSVs, obs_data/almac11; no
│                                   #   simbanator, no HIST; palette = B Part 9's onset colours (lead #fdae61,
│                                   #   concurrent #d7191c, late #2c7bb6, never black), onset_sft before
│                                   #   #e66101 / after #5e3c99, rx xlow #d95f0e / xhigh #31688e, fast
│                                   #   #7b3294 / slow #a6761d. Local results under "onset"
│                                   #   (2026-08-30, the cluster caches copied), log M* > 10.25: cis100 early
│                                   #   2694 / late 204 / never 39, median log D/G -3.41 / -2.60 / -1.75 (late
│                                   #   and never vs early p < 1e-14); cis50 164 / 12 / 0, -3.26 / -2.99 (p
│                                   #   0.7); cis25 144 / 12 / 4, -2.81 / -2.98 / -1.33 (late = early there).
│                                   #   Inside early (onset4) lead is 0.5 dex dustier than concurrent at fixed
│                                   #   dt_SFT in cis100 / cis50 but not in cis25; late's excess is the short
│                                   #   elapsed time (cis100 T1 -2.18 vs -3.41, T2-T3 equal); late = 95-100 %
│                                   #   fast quenchers, dt_SFT 0.44 Gyr. The diagnostic: inside cis100 late
│                                   #   rho(E_x, D/G) = -0.60 (E_x is a clock there), inside early E_x is
│                                   #   saturated (~1 Gyr, rho ~0) and only f_x / r_x carry -0.1 / -0.3; cis25
│                                   #   early rho(E_x) -0.35, rho(r_x) -0.47. The 4 m25 never galaxies are the
│                                   #   former xlow tail (core A_V 0.73 mag, M_dust/M* 2.5e-3 vs 2e-5). KS ends
│                                   #   below B08: early 39 / late 42 / never 25 % vs observed 50 %. ISM grid
│                                   #   cis100: the ALMA-C11 dust detections sit on the never track (+0.09
│                                   #   dex), +0.75 above late, +1.7 above early. Under "onset_sft" (local,
│                                   #   2026-08-30): cis100 before 1052 / after 1846 / never 39, log D/G -3.07
│                                   #   / -3.56 / -1.75 (before vs after p 4e-35, 0.4-0.6 dex in every dt_SFT
│                                   #   tercile); cis50 67 / 109, -2.98 / -3.47 (p 7e-4); cis25 47 / 109 / 4,
│                                   #   -2.78 / -2.84 / -1.33 (p = 1; core A_V 0.083 vs 0.089 mag) -- before =
│                                   #   100 % jet lead, w_pre 0.46 vs 0.18, older / lower-sSFR cores at SFT,
│                                   #   ~ the m25 strong pre-SFT coupling class; diagnostic inside cis25 after
│                                   #   rho(E_x, D/G) = -0.48, inside before only f_x / r_x (-0.43 / -0.54).
│                                   #   The dusty tail (distribution figure): the cis100 tail above
│                                   #   M_dust/M* -4 (20 % of the box) is tilted to late (18 % of the
│                                   #   tail vs 7 % of the box, P 4e-27) and never (7 vs 1 %, P 3e-28;
│                                   #   early falls 92 -> 75 %); the cis25 tail matches the box shares
│                                   #   (early 88 vs 90 %; onset_sft before 27 vs 29 %) except its 4
│                                   #   never galaxies (all above -4, 7 vs 2 %, P 0.01); cis50 under
│                                   #   onset_sft tilts to before (54 % of the tail vs 38 % of the box,
│                                   #   P 0.07; above D/G -2.5 58 vs 38 %, P 0.01).
│                                   #   Deployed 2026-08-30 late evening (fifth version: onset_sft switch
│                                   #   + the Section 5 dusty-tail distributions; default "onset"). The
│                                   #   user ran the fourth version on the cluster under both schemes
│                                   #   (a _new copy for "onset", since removed: one notebook again).
├── obs_data/almac11/               # observed tables the cluster notebooks read (ks_table.csv: KS
│                                   #   placement; almac11_gas_dust.csv: dust / CO detections + fiducial
│                                   #   CIGALE age, M*, M_dust, logDGR; age_sersic_sigma.csv: optical
│                                   #   R_e (COSMOS-Web / ACS), Sersic n, ALMA sizes; README:
│                                   #   conventions/provenance; committed, unlike output/)
└── obs_data/literature/            # published comparison samples (copied from the pilot_specphot
                                    #   tables, README = provenance): av_re_literature.csv = LEGA-C
                                    #   (de Graaff+21 MAGPHYS A_V, vdW+21 F814W R_e) + 3D-HST
                                    #   (Momcheva+16 FAST A_V, vdW+14 R_e) with log M*, SED ages, UVJ;
                                    #   spilker18_legac.csv = the 8 Spilker+18 CO(2-1) LEGA-C rows
```

---

## Configuration

Simulation paths live in `~/.simbanator/config.json` — set up once per machine, never hardcoded in scripts.

```python
import simbanator as sb

sb.add_simulation(
    "cis100",
    data_dir    = "/mnt/share/simbas/SIMBA_100",
    catalog_dir = "/mnt/share/simbas/SIMBA_100/Groups",
    file_format = "m100n1024_{snap:03d}.hdf5",
)

sb.list_simulations()        # show all registered simulations
sb.remove_simulation("cis100")
```

Example `~/.simbanator/config.json`:

```json
{
  "simulations": {
    "cis100": {
      "data_dir": "/mnt/share/simbas/SIMBA_100",
      "catalog_dir": "/mnt/share/simbas/SIMBA_100/Groups",
      "file_format": "m100n1024_{snap:03d}.hdf5"
    }
  }
}
```

---

## Workflows

### 1 — Simulation handle and output paths

```python
sim = sb.Simulation("cis100")
out = sb.OutputPaths(sim.name)

caesar_file = sim.get_caesar_file(snap=105)     # catalog HDF5
snap_file   = sim.get_snapshot_file(snap=105)   # particle snapshot
z           = sim.get_z_from_snap(105)           # redshift
cs          = sim.load_catalog(105)              # caesar.load(...)

out.progenitors           # output/cis100/progenitors/
out.filtered_snap(105)    # output/cis100/filtered_particles/snap_105/
out.sed                   # output/cis100/sed/
out.plots                 # output/cis100/plots/
out.subdir("custom_task") # output/cis100/custom_task/
```

---

### 2 — Build progenitor tracks

Loops through Caesar catalogs and writes a FITS table mapping each galaxy to its most-massive progenitor index at every snapshot.

```python
from simbanator.analysis.progenitors import caesar_read_progen
import os

snaplist = list(range(151, 5, -1))   # z=0 first, descending

caesar_read_progen(
    ids        = galaxy_ids,          # GroupIDs at the base snapshot
    outname    = "tracks.fits",
    snaplist   = snaplist,
    sb         = sim,
    output_dir = out.progenitors,
)
# Output: output/cis100/progenitors/tracks.fits
# Shape: (N_galaxies, N_snaps), values are catalog row indices; -1 = absent
```

---

### 3 — Merger detection

`process_galaxies_with_tracks` reads Caesar HDF5 catalogs directly — **no FITS conversion** of the catalogs is needed.

```python
from simbanator.analysis.mergers import process_galaxies_with_tracks, analyze_mergers
import os

track_path = os.path.join(out.progenitors, "tracks.fits")

galaxies = process_galaxies_with_tracks(
    track_fits_path      = track_path,
    box_size             = 100.0,        # Mpc/h — periodic box side length
    sb                   = sim,
    snaplist             = snaplist,     # must match the track column count
    search_radius_factor = 5.0,          # search sphere = factor × r_half
    mass_threshold       = 1e9,          # min neighbour stellar mass (M☉)
    rhalf_unit_factor    = 1e-3,         # kpc/h → Mpc/h unit conversion
)
# If len(snaplist) ≠ track column count, a warning is issued and the extra
# snapshots are skipped automatically — no crash.

major, minor = analyze_mergers(
    galaxies,
    array_size         = (len(snaplist), len(galaxies)),
    mass_threshold_maj = 0.25,   # mass ratio ≥ 0.25 → major merger
    mass_threshold_min = 0.10,   # 0.10 ≤ ratio < 0.25 → minor merger
)
# major, minor: integer arrays, shape (n_snaps, n_galaxies)
```

You can also pass explicit catalog paths instead of `sb` + `snaplist`:

```python
galaxies = process_galaxies_with_tracks(
    track_fits_path = track_path,
    box_size        = 100.0,
    caesar_paths    = [sim.get_caesar_file(s) for s in snaplist],
)
```

**Required Caesar HDF5 fields** (override at module level if your build differs):

| Module constant | Default path | Description |
|---|---|---|
| `_H5_POS` | `galaxy_data/pos` | Positions (N, 3), Mpc/h |
| `_H5_SMASS` | `galaxy_data/dicts/masses.stellar` | Stellar mass, M☉ |
| `_H5_RHALF` | `galaxy_data/dicts/radii.stellar_half_mass` | Half-mass radius, kpc/h |
| `_H5_H2` | `galaxy_data/dicts/masses.H2` | Molecular hydrogen, M☉ |
| `_H5_DUST` | `galaxy_data/dicts/masses.dust` | Dust mass, M☉ |

```python
import simbanator.analysis.mergers as m
m._H5_SMASS = 'galaxy_data/dicts/masses.stellar'   # override if needed
```

---

### 4 — Particle extraction

Copies a galaxy's (or halo's, or aperture's) particles into a self-contained HDF5 file for Powderday.

```python
from simbanator.analysis.particles import extract_particles

# Batch mode — one file per galaxy, snapshot opened once
extract_particles(
    cs         = sim.load_catalog(snap),
    simfile    = sim.get_snapshot_file(snap),
    snap       = snap,
    galaxy_ids = list_of_galaxy_ids,
    sim_name   = sim.name,
    prefix     = "m100n1024",    # prepended to filenames; omit for no prefix
    overwrite  = False,
)
# Output: output/<sim>/filtered_particles/snap_<NNN>/m100n1024_snap<NNN>_gal<GGGGGG>.h5

# Batch mode + radius — same files/names, but each holds ALL particles within a
# spherical region of `radius` (PROPER kpc) around the galaxy centre (CGM,
# satellites, ...) instead of the caesar member particles. Selection uses the
# periodic minimum image and Coordinates are unwrapped around the centre.
extract_particles(cs=cs, simfile=snap_file, snap=snap,
                  galaxy_ids=list_of_galaxy_ids, radius=100.0,
                  sim_name=sim.name, prefix="m100n1024")

# Single galaxy
extract_particles(cs=cs, simfile=snap_file, snap=snap,
                  galaxy_id=42, sim_name=sim.name)

# Spatial aperture (radius here in snapshot coordinate units, legacy behaviour)
extract_particles(cs=cs, simfile=snap_file, snap=snap,
                  center=[x, y, z], radius=50.0, sim_name=sim.name)
```

**Output filename pattern** (under `output/<sim>/filtered_particles/snap_<NNN>/`):

| Mode | With prefix | Without prefix |
|---|---|---|
| Galaxy | `<prefix>_snap<NNN>_gal<GGGGGG>.h5` | `snap<NNN>_gal<GGGGGG>.h5` |
| Halo | `<prefix>_snap<NNN>_halo<ID>.h5` | `snap<NNN>_halo<ID>.h5` |
| Aperture | `<prefix>_snap<NNN>_aperture.h5` | `snap<NNN>_aperture.h5` |

---

### 5 — SED modelling with Powderday

`MakeSED` manages the full Powderday loop: write parameter files, generate SLURM scripts, plot SEDs, and extract photometry.  
Requires `hyperion` and `caesar` (`pip install simbanator[sed]`).

```python
from simbanator.sed.makesed import MakeSED

makesed = MakeSED(
    sb             = sim,
    nnodes         = 1,
    model_run_name = "PSBG_dust_on",
    hydro_dir_base = out.filtered_particles,  # parent of snap_NNN/ dirs
    selection_file = "my_selection",
    run_tag        = "run_v1",                # subfolder under output/<sim>/sed/
)

# 1. Record the target galaxies
makesed.selection_gals(snaps=snaps, galaxyID=galaxy_ids)

# 2. Generate Powderday parameter files and job scripts
makesed.create_master(
    where       = "cluster",      # or "local"
    subset_type = "plist",        # or "region" (pass radius=...)
    partition   = "INTEL_PHI",
    prefix      = "m100n1024",    # must match extract_particles prefix
)

# 3. After Powderday finishes, plot a SED
makesed.plotsed(snap=105, gal=272)

# 4. Extract fluxes — single galaxy
makesed.extract_flux_single(snap=105, gal=272,
                             facility="HST", instrument="WFC3")

# 5. Batch flux extraction
flux_file, xmean_file = makesed.extract_flux_batch(
    snaps      = snap_array,
    gals       = gal_array,
    facility   = ["HST", "JWST", "Spitzer", "Herschel"],
    instrument = ["WFC3", "NIRCam", "IRAC",  "SPIRE"],
    wave_unit  = "micron",
    findx      = 0,           # inclination index
    aperture   = -1,          # Hyperion SED aperture index (-1 = largest/total);
                              #   loop 0..N-1 with distinct outname= for per-aperture catalogs
    uncertainties = True,     # Hyperion MC errors → <filter>_err columns (NaN if absent)
)
# Outputs: output/<sim>/sed/<run_tag>/sed_fluxes/all_galaxies_fluxes.fits
#          output/<sim>/sed/<run_tag>/sed_fluxes/all_xmean.fits
# With outname="fluxes_X.fits": companion files fluxes_X_xmean.fits /
# missing_sources_fluxes_X.txt, so per-aperture calls never overwrite each other.
```

Multi-aperture SEDs need the one-time powderday patch (reads `SED_APERTURE_*` from the
parameter master) documented in `powderday_flux_quenched_m25.ipynb`, which is the end-to-end
example: quenched (0.2/τ) log M*>10 galaxies with >20 gas particles in the high-res 25 Mpc box
(`cis25`) at z≈0.3–2, split by weak/strong AGN–ISM coupling over the quench window [SFT, QT],
run through dust_on/dust_off Powderday and extracted into per-aperture flux+error catalogs.

**Output directory tree under `output/<sim>/sed/<run_tag>/`:**

```
powderday_sed_out/
└── snap_<NNN>/
    ├── gal_<GGGGGG>/
    │   ├── snap<NNN>.galaxy<GGGGGG>.rtin
    │   └── snap<NNN>.galaxy<GGGGGG>.rtout.sed
    ├── master.snap<NNN>.job   (cluster) / run_local.sh (local)
    └── parameters_master.py
target_selection/
└── my_selection.h5
sed_fluxes/
    ├── all_galaxies_fluxes.fits
    └── all_xmean.fits
sed_plots/
└── snap_<NNN>/
    └── gal_<GGGGGG>.png
```

---

### 6 — Star-formation histories

```python
from simbanator.analysis.sfh_caesar import HDF5BuildHistory

sfh = HDF5BuildHistory(
    sb           = sim,
    progenitors  = out.progenitors,
    progfilename = "tracks.fits",
)
sfh.get_history_indx(galaxy_ids, start_snap=44, end_snap=151)
history = sfh.get_property_history(["sfr", "masses.stellar"])
```

---

### 7 — Radial profiles

```python
from simbanator.analysis.profiles import radial_profile

radii, profiles = radial_profile(
    snapfile        = sim.get_snapshot_file(snap),
    catfile         = sim.get_caesar_file(snap),
    galaxy_id       = 42,
    properties_dict = {
        "PartType0": ["Masses", "Metallicity"],
        "PartType4": ["Masses"],
    },
    radii = np.arange(0, 100, 2),   # kpc
    dens  = True,                   # surface density; False → mean
)
```

---

### 8 — Photometric filters

```python
import simbanator as sb

# Download the default set (SDSS ugriz + GALEX FUV/NUV)
sb.download_default_svo_filters(output_dir="filters/")

# Download a single filter
sb.download_svo_filter("SLOAN/SDSS.r", output_dir="filters/")

# Or download a custom list
sb.download_svo_filters(
    {"HST/ACS.F606W": "HST_ACS_F606W.dat"},
    output_dir="filters/",
)
```

---

### 9 — Geometry utilities

```python
from simbanator.utils.geometry import shrink_center, principal_axes, rotate_to_frame

center              = shrink_center(positions, masses=stellar_masses)
Ixx, e, evecs, axrat = principal_axes(positions - center, masses=stellar_masses)
pos_face_on         = rotate_to_frame(positions, inclination=0, evecs=evecs)
pos_edge_on         = rotate_to_frame(positions, inclination=0, evecs=evecs)[:, [0, 2, 1]]
```

---

## Dependencies

| Group | Packages |
|---|---|
| Core (always) | `numpy`, `scipy`, `astropy`, `h5py`, `matplotlib`, `unyt`, `Pillow`, `psutil` |
| `[sed]` | `hyperion`, `caesar` |
| `[full]` | `yt`, `caesar`, `py-sphviewer`, `fsps`, `svo_filters` |

Heavy optional imports (`yt`, `sphviewer`, `fsps`) are lazy — they load only when explicitly imported and do not break the core package if absent:

```python
from simbanator.visualization.rendering import RenderRGB   # needs yt + sphviewer
from simbanator.analysis.sfh_fsps import compute_sfh       # needs fsps
```

---

## Migration from `modules/`

The old `modules/` directory is kept for reference but is not part of the installed package.

```python
# Old
from modules.io_paths.simba import Simba
from modules.anal_func.filter_particles import filter_particles_by_obj

# New
from simbanator.io.simba import Simulation
from simbanator.analysis.particles import extract_particles
```

---

## License

MIT
