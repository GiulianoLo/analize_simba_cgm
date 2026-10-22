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
├── io/                   # I/O: simulation handles, paths, the visible config, tables, catalogues
│   ├── simba.py          # Simulation – path resolution, Caesar/snapshot loading
│   ├── paths.py          # OutputPaths – structured output directory manager
│   ├── config.py         # config/simulations.json read/write helpers (see config/README.md)
│   ├── catalogs.py       # caesar catalogues read with h5py (no caesar): read_galaxy_columns (dicts/ or flat,
│   │                     #   index= subset, NaN for gaps), dataset, parent_halo_columns, sim_attributes,
│   │                     #   catalog_path / file_prefix (from the config); nearest_snapshot, z_tag,
│   │                     #   time_redshift_maps, softening_pkpc; findsatellites (caesar search)
│   ├── products.py       # need (path or 'run <producer> first'), read_frame, read_selection; the step-4 registry
│   ├── qc.py             # hdf5_status: ok / missing / no-perm / empty / no-keys / no:<grp> / corrupt:<Err>
│   ├── sysinfo.py        # print_ram_usage
│   └── tables.py         # as_str (explicit byte decoding: no silent '== "AGN"' failures), read_table (decodes
│                         #   every byte column at read time), write_table (temp file + rename: atomic, gvfs-safe),
│                         #   float_column, to_frame (native byte order), align_to_rows, galaxy_keys, tagged_path;
│                         #   convert_hdf5_fits – Caesar HDF5 → FITS (legacy)
├── histories/            # everything along a galaxy's branch through time
│   ├── trees.py          # caesar_read_progen, read_progen – merger-tree track FITS
│   │                     #   main_progenitors / main_descendants (tree_data or progen_links sidecar),
│   │                     #   load_anchor_history, progenitor_index_matrix, align_by_ids
│   ├── mergers.py        # Progenitor, Galaxy, process_galaxies_with_tracks,
│   │                     #   analyze_mergers – companion detection & classification
│   │                     #   Units: positions Mpc/h, r_half kpc/h, masses M☉
│   ├── properties.py     # HDF5BuildHistory – property histories from Caesar catalogs
│   │                     #   trace_main_branches (one pass over the snapshots, each catalogue read once),
│   │                     #   safe_ratio, gas_oh12, dust_to_metal_devis (De Vis+19), time_derivative
│   ├── sfh_fsps.py       # compute_sfh, bin_sfh, load_sfh_file – FSPS SFHs
│   ├── sfh.py            # smooth_resample_sfh, recent_sfr – de-burst snapshot-cadence
│   │                     #   SFR tracks (Gaussian kernel + uniform resample);
│   │                     #   sfr_delayed_bq, fit_delayed_bq – CIGALE sfhdelayedbq
│   │                     #   form + bounded fit (shared by 7b′ and aperture truth);
│   │                     #   build_mfrac_lookup/mfrac_of – powderday-consistent FSPS
│   │                     #   surviving-mass fraction (linear-Z snap, cached .npz);
│   │                     #   archaeological_sfh – formed-mass SFH on a fixed 0→t_obs
│   │                     #   grid (linear in weights: disjoint subsets sum exactly);
│   │                     #   projected_region_sfh – SFH of r_in<R≤r_out along a
│   │                     #   sightline (Hyperion aperture geometry, nstar_min skip)
│   │                     #   formed_mass_sfh (unsmoothed), cumulative_ssfr, load_sfh_archive,
│   │                     #   sfh_archive_attrs, sfh_shape_stats
│   └── events.py         # find_quenching_times, load_quenching_events. SFT = crossing below 1/t,
│                         #   QT = the next crossing below 0.2/t, with the persistence rule.
│                         #   smooth_window DEFAULT 3 -> 1 (2026-09-02): a 3-snapshot boxcar on log sSFR is
│                         #   ~0.2 Gyr at SIMBA's cadence — the fast-quenching timescale — and it merged the
│                         #   fast quenchers into the slow mode (log10(tau_q/t_H) unimodal, BIC(1)-BIC(2)
│                         #   = -14 in m100n1024; at 1 it is bimodal, +693, trough -1.68). sSFR = 0 samples
│                         #   are clipped to eps and CARRY the quench: callers must not drop them.
│                         #   Pass smooth_window=3 to reproduce products built before that date.
│                         #   ssfr_track / quench_event / quench_records (keep_zero=True keeps SFR = 0
│                         #   snapshots; False reproduces products built with sSFR > 0), track_start,
│                         #   first_downward_crossing, values_at_rows, ratio_at_rows, regrid, log_track,
│                         #   aligned_stack, binned_event_profile
├── agn/                  # BH accretion and SIMBA's feedback modes (one module): JET_LOGMBH / JET_FEDD /
│                         #   XRAY_FGAS_MAX / W_FULL / MDOT_PER_MBH; eddington_ratio, mdot_from_fedd,
│                         #   bolometric_luminosity; jet_weight (require_accretion= / missing_bh= : the two
│                         #   conventions the notebooks used), xray_coupling (x_coup; gate= is the knob),
│                         #   xray_mode, agn_states (xray / choked / radiative / dormant + diagnostics),
│                         #   catalogue_agn_state; ignition_time, threshold_onset, relative_onset_time,
│                         #   onset_category, pre/post_event_mean; integrate_window, exposure_integrals
│                         #   (e_x, e_jet, t_gaspoor, f_x, r_x, ...), bh_window_metrics; bh_dataset_path,
│                         #   read_bh_history, build_bh_history
├── particles/            # radial and resolved analysis of particle files; ks_tracks_lib.py (repo root) now only
│                         #   re-exports the K-S code that moved into ks / kinematics / grains / profiles here
│   ├── cutouts.py        # extract_particles – per-galaxy/halo/aperture HDF5 subsets
│   │                     #   cutout_path, read_cutout, read_star_formation, reduced_path, load_reduced,
│   │                     #   reduced_has, parse_reduced_name, header_units, code_to_kpc/msun, write_extraction_plan
│   ├── profiles.py       # radial_profile – surface-density / mean radial profiles
│   │                     #   reduced-file Σ(R) of gas components / phases / stars, Σ and mass in apertures,
│   │                     #   temperature profile, member_radii, half_mass_radius, enclosed_radii,
│   │                     #   radius_from_curve_of_growth, annulus_surface_densities, the profile cache
│   ├── kinematics.py     # kappa_rot (Sales+12 K_rot/K), measure_zone_kinematics, KIN_ZONES
│   ├── ks.py             # apertures, HE_FACTOR, sfr_window, measure_ks, ks_columns, ks_region,
│   │                     #   select_tracks, measure_files, sf_law_depletion_time_gyr
│   ├── frames.py         # shrink_center, principal_axes, rotate_to_frame,
│   │                     #   sightline_unit_vectors, projected_radius (image-plane R)
│   │                     #   face_on_R / reduced_radius (pos @ evecs: the axes are the COLUMNS), select_gas,
│   │                     #   spin_unit
│   └── grains.py         # Z_to_OH12, Dust_to_Metal; gas_components (caesar H2 split, the 2026-08-27 recipe),
│                         #   growth_timescale, stellar_dust_budget, weighted/number_fraction,
│                         #   measure_zone_budget, nfw_concentration, nfw_vesc_kms
├── rt/                   # radiative transfer with powderday
│   ├── __init__.py       # MASTERS_DIR; MakeSED and friends load on first use (caesar + hyperion)
│   ├── outputs.py        # tree layout (rtout_path, tree_rtout_path), sed_is_complete (Peeled SEDs are written
│   │                     #   last), log_phase, rt_census, source_types, count_bh_particles, parse_master,
│   │                     #   staged_master_values, check_bh_field_names, read_rt_sed (hyperion inside; aperture=,
│   │                     #   sightline=), on_log_grid, stack_seds, sed_to_flux_density, dust_opacity, grid_centres
│   ├── makesed.py        # MakeSED – Powderday setup + flux extraction (needs hyperion)
│   │                     #   extract_flux_* take aperture= (Hyperion SED aperture index) and
│   │                     #   uncertainties= (MC errors → <filter>_err); list_sed_apertures QC;
│   │                     #   read_selection_centers – Stage-1 selection h5 → RT-grid centres
│   └── masters/          # powderday parameter masters + cosmology_setup_all_{cluster.cis,local.pc39}.sh;
│                         #   MakeSED resolves a bare paramf name here (simbanator.rt.MASTERS_DIR):
│                         #   parameters_master.py / parameters_master-nodust.py / parameters_master-agn.py / parameters_master-agn-nenkova.py
│                         #   parameters_master-nenkova-i90.py / -i60.py / -i30.py
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
├── photometry/           # photometry from the RT SEDs and SED fitting with CIGALE
│   ├── __init__.py       # filters + attenuation eager; flux_extraction and friends load on first use (hyperion)
│   ├── attenuation.py    # av_from_pair, attenuation_curve, av_in_window, bolometric_from_nulnu,
│   │                     #   nulnu_to_fnu_factor, fnu_to_nulnu, mbb_flux_mjy, anchor_av, attenuation_table
│   ├── fluxes.py         # flux_extraction, get_svo_filters – SED → photometry
│   │                     #   flux_unc= propagates an SED uncertainty → mJy_err / mag_err
│   │                     #   annular_flux_table – F(<r_out)−F(<r_in) between two
│   │                     #   cumulative-aperture flux tables (non-positive → NaN)
│   │                     #   attenuation_mag – A_λ = −2.5 log10(F_on/F_off), NaN-safe
│   │                     #   dust_luminosity – L_dust [W] of a raw powderday SED
│   │                     #   (∫νLν dlnν beyond 3 µm; optional dust_on−dust_off)
│   ├── cigale.py         # write_sfhfromfile_multi, nearest_metallicity_node (from the notebooks);
│   │                     #   CIGALE 2025.0 end-to-end: write_cigale_input (flux table →
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
│   └── filters.py        # download_svo_filters – fetch filter curves from SVO
├── obs/                  # comparative data: RELATIONS (K98, B08, RK19) + RELATIONS_EXTRA (VERTICO), relation,
│                         #   relation_y (shift_dex= for an IMF), IMF_TO_CHABRIER, tdep_ms_gyr (Tacconi+18);
│                         #   draw_relation, draw_ks_relations, overlay_ks_observations, draw_observed_points;
│                         #   read_cigale_best_model, read_observed_photometry; mahalanobis_cloud / _d2
├── selection/            # M25_RULE + quenched_sample (mask + individual cuts), dust_label, dusty_from_av,
│                         #   mass_bins, sf_controls. The candidate divisions live in projects/divisions.
├── stats/                # statistics used across projects (one module, plain arrays, seeds for bootstraps):
│                         #   mw_auc, auc_rank, stratified_auc, cell_rank / auc_within_cells, auc_per_column, fdr_bh,
│                         #   cramers_v(_corrected), fisher_2x2; spearman, partial_spearman, partial_corr, theil_sen_ci;
│                         #   wilson (clamped to contain p), beta_interval; bootstrap_median, group_medians,
│                         #   grouped_bootstrap_median, cluster_bootstrap, median_band, running_median (keys= for a
│                         #   galaxy bootstrap), running_quantile, binned_median, binned_stats, log_band_with_floor,
│                         #   median_iqr, jackknife; weighted_quantile, time_weighted_mean, terciles, ridge_residual;
│                         #   scaled_kde, kde_dip, n_modes, critical_bandwidth, silverman_test, kde_levels;
│                         #   kaplan_meier, km_quantiles, log_rank
├── plotting/             # history plots, paper style, particle rendering
│   ├── __init__.py       # plots + figures eager; rendering classes load on first use (sphviewer, yt)
│   ├── figures.py        # PAPER_RC + paper_style(), save_figure (png+pdf), figure_path (prefix router),
│   │                     #   apply_font_floor, panel_grid / equalise_rows (common row limits + first-column frame),
│   │                     #   panel_letter, panel_label, zone_frame, quiet_legend, scalebar, caption_lines,
│   │                     #   wrap_caption, to/from_plot_space, robust_limits, clipped_limits, draw_track,
│   │                     #   stage_medians, draw_median_track, significance_strip, count_histogram_behind,
│   │                     #   kde_contour_or_scatter, ks_axes
│   ├── plots.py          # HistoryPlots – generic multi-panel history figure
│   │                     # plot_merger_rate_by_phase – bar chart of mergers by phase
│   │                     # plot_main_galaxy_track – single-galaxy trajectory (unwrapped)
│   │                     # plot_neighborhood_track – track + nearby galaxies + merger events
│   │                     # plot_all_galaxy_tracks – all galaxy tracks overlaid (unwrapped)
│   │                     #   All track plots: positions in Mpc/h, radius param in Mpc/h
│   ├── animation.py      # create_animation – GIF from x/y frame sequence
│   └── rendering.py      # ParticleProjectionRender, RenderRGB, SingleRender – from a snapshot;
│                         #   make_camera, render_image, blend_gas_stars, project_mass_weighted (from the notebooks);
│                         #   ReducedRender – from a reduced cutout; frame="H2_L" (2026-09-04) orients faceon /
│                         #   edgeon on the member H2 angular momentum inside frame_rmax (set_frame,
│                         #   angular_momentum), "stellar" = the stored inertia frame; tests/test_rendering_frame.py
│                         #   (yt + caesar + py-sphviewer; the yt/caesar imports are optional and
│                         #    the snapshot classes raise a clear error when they are absent)
│                         # ReducedRender – the same maps, RGB composites and mass-weighted gas
│                         #   VELOCITY FIELD from a reduced particle cutout (build_reduced_
│                         #   particles_job.py): needs only py-sphviewer + h5py, so a grid of many
│                         #   galaxies x epochs is cheap. .sigma_map (calibrated to Msun/kpc^2),
│                         #   .weighted_map ('vlos', 'temp', ...), .velocity_field -> streamplot,
│                         #   .rgb (RenderRGB's recipe), .totals, .from_epoch / .epoch_path.
│                         #   pos @ evecs frame: faceon = cols (0,1), edgeon = (0,2); every map is
│                         #   image[y,x] for imshow(..., extent, origin='lower')
└── data/                 # bundled reference data
    └── snap_z_maps/      # bundled snapshot → redshift tables per simulation box
```

Old import paths still work: `simbanator.analysis.*`, `simbanator.sed.*`, `simbanator.utils.*`,
`simbanator.visualization.*` and `simbanator.data.convert` are one-line shims that hand back the module
at its new place (the same object, private names included). New code imports from the new paths.

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
│                                   #   `nh` [cm^-3] per gas particle (added 2026-09-02, the caesar
│                                   #   recipe of build_profiles_job._nH, so rho = nh m_p / 0.76):
│                                   #   the density behind the THERMAL SPUTTERING timescale
│                                   #   tau_sp = 0.17 Gyr (a/0.1um)(1e-27/rho)[(2e6/T)^2.5 + 1] that
│                                   #   agn_states_dusty_vs_pure Part 19 measures per particle - the
│                                   #   files already carried temp, this was the missing ingredient;
│                                   #   backfills like the others (one snapshot column, no geometry)
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
│                                   #   Part 2b (ks-21-plan-postsft, BUILD_PLAN_POSTSFT = True once, 2026-09-04):
│                                   #   the POST-SFT PLAN - every main-branch snapshot from t_SFT - 0.5 Gyr to the
│                                   #   anchor for the quenched galaxies (the same histories + progenitor index
│                                   #   Part 1 walks), written as a second plan for build_reduced_particles_job.py
│                                   #   (same job, same 100 kpc cutouts; the printed sbatch) + ks_postsft_map.csv
│                                   #   with the (anchor, gal_id) <-> (snap, gx) map and the times.
│                                   #   Part 3c (ks-33-budget, cached ks_stage_budget.fits): kl.measure_zone_budget
│                                   #   on EVERY particle of the 100 kpc cutout in spherical shells core 0-1 /
│                                   #   ring 1-3.16 / outskirt 3.16-10 / cgm_in 10-30 / cgm_out 30-100 kpc: dust,
│                                   #   H2, HI, gas masses, the member share of the dust and H2, the dust in gas
│                                   #   above 1e5 K, and the mass-weighted RADIAL VELOCITIES of gas / H2 / dust
│                                   #   about the member stars' systemic velocity (+ H(z) r, the files store
│                                   #   peculiar velocities; + = outflow), the dust / H2 on outflowing particles,
│                                   #   the dust-weighted spread of v_r. Pairs = the critical points + the post-SFT
│                                   #   snapshots whose cutouts exist (incremental cache, files without vel
│                                   #   re-measured once backfilled, like 3b). Also (2026-09-04 later) m_dust_in /
│                                   #   m_H2_in (inward-moving mass), m_dust_esc / m_H2_esc (mass ABOVE THE PARENT
│                                   #   HALO'S NFW ESCAPE SPEED: kl.nfw_vesc_kms with caesar m200c + proper r200c
│                                   #   and kl.nfw_concentration = Dutton & Maccio 2014, looked up straight from
│                                   #   the catalogue files with h5py into ks_halo_props.csv) and vesc_kms at the
│                                   #   shell's mid radius; a cached table without them is rebuilt. Consumer:
│                                   #   agn_states Part 25.
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
│                                   #   branch histories of mstar / sfr / mgas / mbh / bh_fedd (+ mdust /
│                                   #   zgas since 2026-08-31: masses.dust + metallicities.mass_weighted;
│                                   #   + mh2 = masses.H2 since 2026-08-31 for the noxray H2 legs; a
│                                   #   pre-rebuild cache fills the new keys NaN — rm ism_prediction_
│                                   #   histories_*.hdf5 + re-run Part 6 for the dust / H2 tracks) from the
│                                   #   in-catalogue trees (cis25: the progen_links sidecars) for every
│                                   #   anchor whose catalogue ladder END_SNAP..anchor is on disk (one pass,
│                                   #   each catalogue read once -> ism_prediction_histories_<box>.hdf5),
│                                   #   SFT / QT with find_quenching_times (QT_SMOOTH = 1 since 2026-09-02:
│                                   #   NO boxcar on log sSFR, and the SFR = 0 snapshots kept — the old
│                                   #   default of 3 erased the fast quenching mode, see quenching.py;
│                                   #   QT_SMOOTH=3 reproduces the older classes), w_pre over [SFT - 1 Gyr, SFT],
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
│                                   #   Part 9 = the X-RAY EXPOSURE after quenching from the same histories
│                                   #   (P9_BOXES now also lists cis50nox, gated on its histories file: the
│                                   #   would-be dose of the channel-off run once its catalogue ladder is
│                                   #   downloaded and Part 6a builds it):
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
├── noxray_analogues.ipynb      # the NO-X-RAY run vs the fiducial one and the nox ANALOGUES: pure
│                                   #   reads of the ism_prediction products (catalogue cache, Part 6a
│                                   #   classes, Part 9 exposures - nothing here opens a catalogue);
│                                   #   outputs -> output/box_resolution/noxray_analogues/. Part 1 =
│                                   #   s50nox vs s50 (same box, resolution and selection): per-quantity
│                                   #   medians + Mann-Whitney / AUC, the disc-verdict fractions
│                                   #   (DISC_KAPPA 0.5, old, extended = above the fiducial size relation
│                                   #   fit with a log(1+z) term, Fisher p) and 213 mass- and anchor-
│                                   #   matched pairs (MATCH_DM 0.2, Wilcoxon) ->
│                                   #   noxray_analogues_populations.{png,pdf,csv}. Result: the nox QG is
│                                   #   OLD (4.1 vs 2.0 Gyr), GAS-rotating (kappa_gas 0.77 vs 0.35; 87 vs
│                                   #   37 % gas discs), 30x dustier (log f_dust -3.1 vs -4.7), more
│                                   #   massive and lower-z, but NOT a stellar-spin disc (kappa_star 0.30
│                                   #   vs 0.33) and NOT extended at fixed M* and z (dlre ~ 0: the raw
│                                   #   R_e excess is mass + epoch) and 0.4 dex GAS-POORER; the matched
│                                   #   pairs keep age +0.7, kappa_gas +0.3, dust +1.4 dex. Part 2 = the
│                                   #   Mahalanobis match: FEATURES = age + log M_dust/M* ONLY (the
│                                   #   cheap observable pair; the structural set of the first version
│                                   #   is listed in the cell), an ANALOGUE = a quenched galaxy of
│                                   #   ANALOG_BOXES (cis50 / cis100 / cis25) within the nox cloud's own
│                                   #   ANALOG_Q (0.90) D^2 quantile -> noxray_analogues_match
│                                   #   {,_medians}.csv + figure. Result: 12.1 % of s50 (43), 11.3 % of
│                                   #   m100 (381), 10.1 % of m25 (18) are (age, f_dust) analogues;
│                                   #   reverse containment 34 vs 12 %. Part 2b = PART 1 ON THE
│                                   #   ANALOGUES (everything the match did not see is a PREDICTION):
│                                   #   contrasts vs the nox population itself + CDF grid ->
│                                   #   noxray_analogues_analogue_populations.{png,pdf,csv}, and the
│                                   #   (age, log f_dust) plane coloured by the DUST-TO-H2 ratio ->
│                                   #   noxray_analogues_dgr_plane.*. Result: the (age, f_dust)-selected
│                                   #   analogues are NOT the nox phenotype - younger (AUC 0.22-0.38),
│                                   #   0.3-0.4 dex gas-richer (AUC 0.75-0.86) and 0.5-0.6 dex LOWER in
│                                   #   D/G (AUC 0.11-0.14): dust-rich because gas-rich, not unscoured
│                                   #   survivors; their kappa_gas is ~0.5 vs the nox 0.77 in m50 /
│                                   #   m100 (m25's 18 do reach 0.83). Part 3 = the AGN history of the
│                                   #   analogues vs the rest (w_pre / class, onset, exposure, BH +
│                                   #   environment; Fisher / MW / AUC; chain check: satellite most-
│                                   #   massive-progenitor chains inherit the central's M_BH / f_Edd /
│                                   #   f_gas -> w_pre / E_x = upper limits, exposure contrasts repeated
│                                   #   on centrals / satellites) -> noxray_analogues_agnhistory
│                                   #   .{png,pdf,csv}. Result (m100, 381): the coupling class does NOT
│                                   #   make them (weak 45 vs 57 %: they lean STRONG, p 9e-5; w_pre AUC
│                                   #   0.56) - what separates them is the DOSE: never-exposed 14 vs
│                                   #   0 %, duty f_x 0.66 vs 0.89 (AUC 0.28; satellites 0.39 vs 0.88),
│                                   #   BH-not-grown 33 vs 1 %, late onset 18 vs 6 % -> UNDER-EXPOSED,
│                                   #   not weakly coupled (cis25 the same: f_x AUC 0.16). Part 4 =
│                                   #   SELECTING DUSTY QGs IN REAL LIFE (kappa_gas vs Sigma_e;
│                                   #   DUSTY_CUT -3.5): AUC of every structural quantity for dust-rich
│                                   #   membership, the F1-best two-cut rule kappa_gas >= a & log
│                                   #   Sigma_e <= s (completeness / purity vs the base rate) + the
│                                   #   one-cut gas-disc rule, the plane coloured by log f_dust with
│                                   #   s50nox as the reference panel -> noxray_analogues_structure
│                                   #   .{png,pdf,csv}. Result: in m100 / m50 the plane is weak
│                                   #   (kappa_gas AUC 0.53-0.55, purity ~20 vs 15 % base - the coarse
│                                   #   gas washes it out); in the resolved m25 box kappa_gas works
│                                   #   (AUC 0.69, rule purity 35 vs 12 % base; sSFR AUC 0.73): gas
│                                   #   rotation selects dusty QGs where the gas is resolved. Part 5 =
│                                   #   WHY ONLY m25 - resolution, statistics, or the BH? catalogue
│                                   #   tests -> noxray_analogues_resolution_tests.{png,pdf,csv}:
│                                   #   (a) statistics EXCLUDED - 2000 pseudo-m25 samples drawn from
│                                   #   m100 (per-anchor |dlog M*| <= 0.2 twins) give AUC 0.52
│                                   #   (0.47-0.58), P(>= m25's 0.69) = 0.001; (b) matched M_gas
│                                   #   [m100 floor, 3e9) - same reservoirs, 8x the particles in m25:
│                                   #   AUCs converge (m25 0.61, m100 0.60) so part of the box gap is
│                                   #   WHICH reservoirs each selection keeps; (c) the BH mix EXCLUDED
│                                   #   - the AUC inside M_BH strata stays (m25 weighted 0.64, m100
│                                   #   0.50); (e) the sign flip (dusty - rest median kappa_gas +0.18
│                                   #   in m25 vs -0.03 / -0.07 in m100 / m50) - pure estimator noise
│                                   #   cannot reverse a sign, pointing at a physically different
│                                   #   dusty population in the coarse boxes (the Part 2b gas-rich
│                                   #   passers-through). Part 5b = the DEGRADATION EXPERIMENT (the
│                                   #   remaining channel, the estimator): kappa_gas of every cis25
│                                   #   sample galaxy recomputed (Sales+10, member gas of the m25
│                                   #   reduced particle files output/cis25/reduced_particles/, all
│                                   #   178 covered; full-N value validated against the catalogue
│                                   #   kappa in the printout) then resampled to n100 = member gas
│                                   #   mass / 1.82e7 = the particle count m100 would give the SAME
│                                   #   reservoir (under-floor n100 < 21 galaxies flagged: not even
│                                   #   in the m100 selection), P5B_NREAL 200 realisations -> the AUC
│                                   #   at m100 sampling with and without the m100 selection, vs the
│                                   #   full-N and the observed m100 AUC ->
│                                   #   noxray_analogues_kappa_degraded.{png,pdf,csv}; needs the
│                                   #   reduced files (cluster; skipped cleanly elsewhere,
│                                   #   NOX_P5B_LIMIT caps a smoke test). Result (full 178, validation
│                                   #   corr 0.95, median |diff| 0.025; median n100 = 66 vs 259
│                                   #   member): degraded AUC 0.709 (0.691-0.728) vs full-N 0.717 -
│                                   #   sampling noise closes ~1 % of the m25 -> m100 gap: the
│                                   #   ESTIMATOR IS NOT THE BOTTLENECK, kappa_rot is stable at ~66
│                                   #   particles; the coarse boxes fail because their dusty QGs are
│                                   #   a physically different population (the sign flip + Part 2b),
│                                   #   i.e. resolution acts on the ISM physics / who quenches dusty,
│                                   #   not on the measurement. s50nox now carries its own histories
│                                   #   cache on the share (coupling classes + the 6c/6d channel-off
│                                   #   panels; the B Part 9 exposure merge still pending)
│                                   #   Part 6 = the RESILIENT SURVIVORS (P6_AGE 2 Gyr, P6_DOSE E_x >=
│                                   #   1 Gyr, P6_TAIL log f_dust >= -4: the X-ray notebook's 5c/5d old
│                                   #   fully-dosed dusty tail) taken as a POPULATION, not a mechanism
│                                   #   hunt: placed on f_dust-age, f_H2-age, M_dust/M_H2-sSFR and kappa_gas x
│                                   #   Sigma_e (Part 4's plane, its rule drawn), vs the destroyed
│                                   #   control (same age + dose, dust-poor), the young under-dosed
│                                   #   dusty tail and the nox cloud in every panel; per-axis AUC
│                                   #   contrasts, the Part 2 nox-analogue share and what the Part 4
│                                   #   rule catches per group -> noxray_resilient_planes.{png,pdf} +
│                                   #   noxray_resilient_contrasts.csv. Part 6b = WHAT THE DOSE DOES
│                                   #   NOT SEPARATE: the onset and the coupling of the resilient vs
│                                   #   the destroyed control (and the young tail) — coupling (w_pre,
│                                   #   weak/strong Fisher, ungated E_jet), timing (t_onset - t_SFT +
│                                   #   the lead/concurrent/late mix, t_gate - t_SFT, E_x before SFT,
│                                   #   f_gas at SFT, tau_q), the dose axes as the null reference;
│                                   #   centrals-only repeat + a per-box verdict line (which family
│                                   #   wins at fixed dose) -> noxray_resilient_onset_coupling
│                                   #   .{png,pdf,csv}. Result: the survivors are the EARLY-DOSED —
│                                   #   lead onsets 50 vs 37 % (m100, p 4e-4; m50 100 vs 28 %), the
│                                   #   gate opens before SFT, E_x-before-SFT higher — with a LOWER
│                                   #   duty f_x after SFT (AUC 0.40 m100, 0.19 m25) and a slower
│                                   #   quench (tau_q AUC 0.58): the same dose arrived earlier and
│                                   #   more diluted; the coupling class adds little (w_pre AUC 0.54
│                                   #   m100; only the 8 m50 survivors lean strong). Part 6c = the
│                                   #   COUPLING HISTORY ITSELF: the quenched sample as the plain 2x2
│                                   #   of the (age, f_dust) plane (old/young x dusty/dust-poor, NO
│                                   #   dose cut), each galaxy's x_coup = w_jet [f_gas<0.2], ungated
│                                   #   w_jet and gas-poor fraction rebuilt from the per-box
│                                   #   histories hdf5 and aligned to three clocks (AGN activation =
│                                   #   first f_Edd > 0, jet activation, SFT; left-censoring at the
│                                   #   track start counted), median +- bootstrap band per quadrant,
│                                   #   old-dusty-vs-old-dust-poor AUC at every grid time; the nox
│                                   #   box carried as its own would-be-coupling figure ->
│                                   #   noxray_coupling_tracks_<box>{.png,.pdf} + noxray_coupling_
│                                   #   tracks.csv. Result (m100): the BH side is the SAME machine
│                                   #   (w_jet tracks identical on the jet clock) — the dusty
│                                   #   quadrants differ in the GATE: their gas-poor fraction rises
│                                   #   later after activation (x_coup onset old dusty ~2.7 Gyr vs
│                                   #   young dust-poor ~1.5 Gyr after AGN act.), and on the SFT
│                                   #   clock the medians collapse together (AUC drifts to 0.42-0.44
│                                   #   only at t >= +1 Gyr: the coupling of the dusty side turns
│                                   #   off earlier); m25 shows it directly — the old-dusty x_coup
│                                   #   is INTERMITTENT after SFT (dips to 0 and back; AUC 0.33/0.27
│                                   #   at +0.5/+1 Gyr) while the old dust-poor lock at 1.
│                                   #   (2026-09-01) a fourth row: the DELIVERED coupling
│                                   #   x_coup x [f_Edd > FEDD_ACT6C 1e-4]. SIMBA's mode weight
│                                   #   RISES as f_Edd falls, so a dormant massive BH reads
│                                   #   w_jet = 1 = maximal eligibility while injecting nothing;
│                                   #   the delivery factor is Part 6i's accreting/dormant split
│                                   #   applied to the track, and the gap between the two rows is
│                                   #   the dormant-but-eligible fraction. Eligibility stays the
│                                   #   primary row (every dose - E_x / f_x / r_x of B Part 9 -
│                                   #   integrates it); delivered is a LOWER limit (f_Edd is
│                                   #   instantaneous at the ~100-250 Myr cadence). Result (m25):
│                                   #   the eligibility row of the old dusty is pinned at 1 after
│                                   #   SFT while the delivered row dips repeatedly - dormancy was
│                                   #   hiding the intermittency. The row labels now follow
│                                   #   XRAY_FGAS_MAX / FEDD_ACT6C instead of a hardcoded 0.2.
│                                   #   Part 6d = the ALIGNED SFH of the same 2x2: median log sSFR,
│                                   #   the distance from the MS (DMS = log sSFR - log sSFR_MS,
│                                   #   Speagle+14 eq. 28 at the track's own M*(t) and cosmic t,
│                                   #   -0.24 dex zero-point, same sSFR floor),
│                                   #   the SF-active fraction (sSFR > 0.2/t_H, the RM+19 rejuvenation
│                                   #   threshold) and the cumulative in-situ mass fraction on the
│                                   #   five clocks (6c's three + QT + anchor); per-galaxy burst
│                                   #   anatomy at native sampling
│                                   #   (n_rejuv = post-SFT upward crossings of the threshold, f_post
│                                   #   = in-situ share formed after SFT, n_peaks_pre at >= 0.3 dex
│                                   #   prominence) with old-dusty-vs-old-dust-poor AUC/Fisher and
│                                   #   Spearman vs the anchor log f_dust; (B) the m25 core (r < 1
│                                   #   kpc) vs outskirts (3.2-10 kpc) SFR25 / M_dust per critical
│                                   #   epoch from the KS 'pt' product (pure read) ->
│                                   #   noxray_sfh_tracks_<box>{.png,.pdf} +
│                                   #   noxray_sfh_{tracks,bursts}.csv +
│                                   #   noxray_sfh_m25_radial.{png,pdf,csv}. Result: rejuvenation
│                                   #   INCIDENCE does not separate the old quadrants (57 vs 56 %
│                                   #   m100) and the pre-SFT burst count is flat — what separates is
│                                   #   the POST-SFT star formation: old dusty stay SF-active longer
│                                   #   (sfact/lssfr AUC 0.56-0.59 at +0.5..+2 Gyr m100), form more
│                                   #   in-situ mass after SFT (f_post AUC 0.56 m100 p 7e-4, 0.64 m25)
│                                   #   and f_post tracks the anchor dust (Spearman +0.13 m100, +0.28
│                                   #   m25 among the old) — in nox the correlation FLIPS (-0.35, p
│                                   #   3e-14: channel off, the dust survives without late SF); and in
│                                   #   m25 both the residual SF and the surviving dust are CENTRAL
│                                   #   (core log M_dust AUC 0.88-0.89 at gas_min/end; outskirt dust
│                                   #   share ~0.4, i.e. NOT off-centre), with a transient outskirt-
│                                   #   dust excursion around SFT.
│                                   #   Part 6e = the ANCHOR-ALIGNED view (t - t_anchor, the run-up to
│                                   #   the observed snapshot): (A) median log sSFR + log f_gas per
│                                   #   quadrant, one column per box (the f_H2 / f_dust rows are auto-
│                                   #   added once the histories carry mh2/mdust), median t_SFT -
│                                   #   t_anchor ticked per quadrant; (B) the m25 core (r < 1 kpc) vs
│                                   #   outskirts (3.2-10 kpc) sSFR25 + f_H2 = M_H2/M* per critical
│                                   #   epoch (KS 'pt', anchor stage = 0) ->
│                                   #   noxray_anchor_tracks{.png,.pdf,.csv} +
│                                   #   noxray_anchor_m25{.png,.pdf,.csv}. Result: on the observer's
│                                   #   clock the old quadrants decline TOGETHER and split only in the
│                                   #   last ~1 Gyr (lssfr AUC 0.57/0.62 m100 at -1/-0.5 Gyr,
│                                   #   0.69-0.83 at the anchor; f_gas ~0.5 flat) — and in m25 the
│                                   #   split is CENTRAL: core f_H2 AUC 0.80/0.79 at ssfr_min/anchor
│                                   #   vs outskirt ~0.47 ns (the surviving molecular reservoir of the
│                                   #   old dusty sits inside 1 kpc). Part 6f = the BLIND w_jet-
│                                   #   STEADINESS test: candidate dust-poor = w_jet >= 0.9 for >= 80
│                                   #   % of the post-jet-on span, candidate dusty = the rest of the
│                                   #   jetting (no-jet / < 1 Gyr spans excluded; W6F_* knobs,
│                                   #   W6F_QUANT flips the classifier to the gated x_coup), placed on
│                                   #   the (age, lfd) plane with median tracks (m25: the KS r < 10
│                                   #   kpc dust leg anchored at the catalogue lfd + the in-situ
│                                   #   ageing proxy anchored at the catalogue age; other boxes once
│                                   #   mdust histories exist) -> noxray_wjet_plane{.png,.pdf} +
│                                   #   noxray_wjet_{classes,plane_tracks}.csv. Result: steady ->
│                                   #   dust-poor 85 % vs base 77 % (m100 Fisher p 0.003; m25 83 vs 75
│                                   #   %), and the head-to-head says the GATE carries the flicker
│                                   #   (steadiness AUC for dusty|old: 0.59 w_jet vs 0.62 x_coup m100,
│                                   #   0.68 vs 0.75 m25); in nox the split predicts NOTHING (99 %
│                                   #   dusty either way) — the channel, not the jet duty cycle per
│                                   #   se. Part 6g = the coupling TRANSLATED into physical units:
│                                   #   fraction in X-ray mode (x_coup >= 0.9 — the X-ray
│                                   #   probability), median log f_Edd (jets on < 0.2, full jets <
│                                   #   0.02 drawn: slow accretion reads directly), median log L_bol =
│                                   #   f_Edd L_Edd and the delivered strength log(x_coup L_bol), on
│                                   #   the SFT and anchor clocks ->
│                                   #   noxray_xray_mode_{sft,anchor}{.png,.pdf} +
│                                   #   noxray_xray_mode.csv. Result: every quadrant hits the X-ray
│                                   #   mode AT SFT (the probability jumps 0 -> 0.7-0.9), then the
│                                   #   dust-poor stay PINNED (0.95-1.0) while the old dusty keep
│                                   #   dropping out (0.5-0.75; fxon AUC 0.28-0.33 m25 after SFT);
│                                   #   accretion speed and power do NOT separate the old quadrants —
│                                   #   the delivered strength does (m100 anchor: dust-poor ~1e43.3
│                                   #   erg/s vs dusty ~2 dex lower over the final 2 Gyr: the
│                                   #   accreting ones sit out of the mode, the in-mode ones barely
│                                   #   accrete).
│                                   #   Part 6h = the AGN STATES behind the coupling + the stable-
│                                   #   coupling CALIBRATION: (A) five mutually exclusive states per
│                                   #   epoch (quasar f_Edd >= 0.2 / partial jet / full jet + gas-rich
│                                   #   = X-ray choked / X-ray mode / off = BH under the jet mass at
│                                   #   that epoch) stacked per quadrant on the anchor clock, anchor
│                                   #   composition printed (the mode is read from f_Edd, never from
│                                   #   L_bol — L_bol mixes f_Edd and M_BH); (B) the sweep f_on(window
│                                   #   x quantity x tau_on) — windows post-jet / post-SFT / last 2
│                                   #   Gyr, quantities w_jet / x_coup, tau_on 0.5 / 0.9 / ~1 — scored
│                                   #   by AUC(1 - f_on) for dusty|old per box (heat-map), the best
│                                   #   combo's Youden theta* printed as the recommended definition ->
│                                   #   noxray_xray_states{.png,.pdf,.csv} +
│                                   #   noxray_stability_sweep{.png,.pdf,.csv}. Result: NOT restarted
│                                   #   quasars — the quasar state at the anchor is 0-6 % everywhere;
│                                   #   the old-dusty out-of-mode share is partial-jet + gas-rich-
│                                   #   full-jet flicker during the run-up plus an 'off' block (23 %
│                                   #   m100, 42 % m25) that is a small/REASSIGNED BH at the anchor
│                                   #   epoch (91 % of those jetted earlier: the Part 3 satellite
│                                   #   chain-inheritance caveat). The calibration: the POST-SFT
│                                   #   window beats post-jet everywhere; best m100 = w_jet >= 0.5
│                                   #   post-SFT (AUC 0.69; theta* 0.90: stable -> dust-poor 85 %,
│                                   #   varying -> dusty 44 % vs base 21 %), best m25 = x_coup ~ 1
│                                   #   post-SFT (AUC 0.87; theta* 0.90: varying -> dusty 68 % vs base
│                                   #   24 %, stable -> dust-poor 94 %); cis50 only an inverted (N=9)
│                                   #   signal, nox nothing significant — the recommended stable-
│                                   #   coupling definition: ON for >= 90 % of the post-SFT window.
│                                   #   Part 6i = the anchor STATES against f_dust in STELLAR-AGE
│                                   #   BINS (pure catalogue: M_BH / f_Edd / f_gas at the anchor, no
│                                   #   histories): the 6h states stacked per 0.5-dex lfd bin (the
│                                   #   -6.5 zero-dust spike keeps its own bin), rows age 0-1 / 1-2 /
│                                   #   >= 2 Gyr, and the X-ray state SPLIT at FEDD_ACT6I (f_Edd
│                                   #   1e-4) into accreting (delivering: L_x ~ f_Edd M_BH) vs
│                                   #   dormant (an open valve, nothing flowing) ->
│                                   #   noxray_xray_states_fdust{.png,.pdf,.csv} + the old dusty-tail
│                                   #   vs dust-poor split printed. Result (0.6 gate): the dust-poor
│                                   #   occupancy is NOT dormancy — old dust-poor m25 98.7 % in mode
│                                   #   = 79 % accreting + 20 % dormant (m100 89 = 62 + 27); the old
│                                   #   dusty tail sits at 58-64 % (m25/m100) with the deficit in
│                                   #   'off' (small/reassigned BH) + partial jet, and its in-mode
│                                   #   members accrete at the same median log f_Edd ~ -3.2 —
│                                   #   eligibility differs between the dust sides, activity inside
│                                   #   the mode does not.
│                                   #   Part 6j = CYCLING vs STABLE coupling: the delivery-state
│                                   #   taxonomy per galaxy, on a window that OPENS PRE6J = 1 Gyr
│                                   #   BEFORE SFT (with t = 0 pinned at SFT a channel that ran
│                                   #   before it and shut off there is filed as 'never'; the
│                                   #   earlier history is still read, but only for pre_on = was
│                                   #   it ever ON before we looked, which is what now separates
│                                   #   never from dropout) and with runs shorter than DTOFF6J /
│                                   #   DTON6J = 0.3 Gyr DEBOUNCED away (OFF gaps closed, then ON
│                                   #   blips erased, interior runs only, durations integrated
│                                   #   with mid-point widths: the histories are sampled every
│                                   #   ~0.05 Gyr, so a 1-2 sample dip is accretion flicker, not
│                                   #   a duty cycle). The label uses f_on_post (duty since the
│                                   #   channel's OWN first ON), so widening PRE6J can only move
│                                   #   a galaxy out of 'never' - it can never re-read a locked-on
│                                   #   track as a dropout. ON = channel >= TAU6J 0.9; never /
│                                   #   dropout / cycling / stable over four channels (xact
│                                   #   delivered, xcoup eligibility, acc accretion alone, gate
│                                   #   alone), with (A) the dusty ladder per class (+ %pre),
│                                   #   (B) the cycling axis (n_off / off_tot) scored against the
│                                   #   duty axis (f_on_post), (C) the threshold sweep fitted on
│                                   #   CAL6J_BOX and applied unchanged elsewhere, (C2) the
│                                   #   STABILITY sweep - the same galaxies relabelled on every
│                                   #   (PRE6J_GRID x DT6J_GRID) point, the live point asserted
│                                   #   against the live labels - (D) the centrals-only repeat ->
│                                   #   noxray_cycling_split{.png,.pdf} +
│                                   #   noxray_cycling_{classes,ladder,axes,sweep,window}.csv.
│                                   #   RESULT (0.6 gate, corrected definitions): the two choices
│                                   #   matter very differently. The WINDOW ORIGIN barely moves
│                                   #   the X-ray channels (converged by 1 Gyr, %pre <= 9 %,
│                                   #   because the gas gate is shut while the galaxy still forms
│                                   #   stars) but is decisive on the accretion channel, where
│                                   #   'never' VANISHES entirely once the window opens before SFT
│                                   #   (75-100 % of every class is pre_on) - that class was pure
│                                   #   clock artifact. The DEBOUNCE is the consequential knob: at
│                                   #   dt = 0 'cycling' is 65-87 % of the sample at a 0.8-1.1
│                                   #   dusty lift (meaningless); at 0.3 Gyr it falls to 4-18 %
│                                   #   and lifts to 2.4x (m50) / 1.6x (m25) - but only 1.1x in
│                                   #   m100, so cycling is STILL not a dusty selector that
│                                   #   transfers (threshold sweep: best m100 rule 1.18x,
│                                   #   validating 2.75 / 1.43). What is monotone in every AGN-on
│                                   #   box AND stable across all nine (window, debounce) settings
│                                   #   is the ladder: m100 xact 54 / 28 / 23 / 14 % dusty for
│                                   #   never / dropout / cycling / stable (base 20 %), m100 xcoup
│                                   #   60 / 50 / 28 / 17 %, m25 xact 71 / 78 / 56 / 27 % (base
│                                   #   35 %); the starved (never + dropout) lift holds at 1.6-2.5
│                                   #   (m100 xact), 2.2-3.5 (m100 xcoup), 2.1-3.1 (m25) over the
│                                   #   whole grid while nox stays flat at 0.95-1.01. The dust
│                                   #   survives where the coupling NEVER arrived or stopped
│                                   #   arriving, not where it flickers. The satellite caveat is
│                                   #   much reduced: m100's 'never' is now 64 % central and still
│                                   #   39 % dusty among centrals alone (base 18 %).
│                                   #   Part 6k = the ladder's RARE CLASSES AT FIXED AGE AND MASS,
│                                   #   CENTRALS ONLY (reads 6j's C6J, no histories): starved =
│                                   #   never or dropout vs rest, per box x channel; the 5e
│                                   #   estimator (quantile cells of E6K_CONF age + log M*, MW U
│                                   #   summed over cells, labels permuted INSIDE each cell,
│                                   #   E6K_NPERM 2000) + the Mantel-Haenszel OR of being dusty
│                                   #   over the same cells (Haldane 0.5 when a cell separates
│                                   #   perfectly, marked '+') + a twin check (|dage| <= 0.3 Gyr,
│                                   #   |dlog M*| <= 0.15, <= 5 controls each, bootstrap band);
│                                   #   rows with < E6K_SMALL = 10 starved centrals print '!' and
│                                   #   are counts, not tests ->
│                                   #   noxray_cycling_fixedage{.png,.pdf} +
│                                   #   noxray_cycling_{fixedage,twins}.csv.
│                                   #   RESULT (0.6 gate, on the corrected 6j labels): starved
│                                   #   centrals are dustier at fixed age and mass on BOTH
│                                   #   channels. m100 xact (delivered) is now the strongest row:
│                                   #   416 starved centrals of 2157, 26 vs 16 % dusty, AUC 0.57
│                                   #   raw / 0.54 fixed (perm p 0.012), MH OR 1.74 (p 0.001,
│                                   #   9 cells), 406/416 twins matched, delta +0.08 [+0.06,
│                                   #   +0.10]. m100 xcoup agrees on a fifth as many objects with
│                                   #   a larger per-object effect but a marginal permutation:
│                                   #   112 starved, 26 vs 17 %, AUC 0.54 fixed (p 0.153), OR 1.67
│                                   #   (p 0.054), twins 112/112 at +0.11 [+0.07, +0.15]. m25 far
│                                   #   harder in both (16 / 8 starved, 62 / 100 % dusty vs 26 %,
│                                   #   AUC 0.67 / 0.94, p 0.040 / < 5e-4, OR 3.6 / 13.3+, twins
│                                   #   +0.37 / +0.72); m50 null; nox unscorable on xcoup (0
│                                   #   starved) and a 4-object '!' row on xact - the channel-off
│                                   #   run has nothing to be starved of. Of 7 scored rows the
│                                   #   m100 delivered OR (7 x 0.001) and the m25 eligibility row
│                                   #   clear Bonferroni; the m100 permutation AUC (7 x 0.012 =
│                                   #   0.08) does not on its own.
│                                   #   SUPERSEDES the pre-correction reading: on the old labels
│                                   #   (window pinned at SFT, no debounce) the test appeared to
│                                   #   split sharply by channel - delivered null (p 0.51),
│                                   #   eligibility significant (p 0.007, twins +0.16). That split
│                                   #   does NOT survive: it came from the undebounced counter
│                                   #   filing genuinely shut-down deliveries as 'cycling', which
│                                   #   emptied the starved class on xact (142 -> 416 objects once
│                                   #   excursions must last) and left it on xcoup. The surviving
│                                   #   statement is channel-agnostic and smaller: a central whose
│                                   #   X-ray coupling is not running at the anchor keeps more of
│                                   #   its dust than an age- and mass-matched central whose
│                                   #   coupling still is (+0.08 to +0.11 in dusty fraction in
│                                   #   m100, +0.37 to +0.72 in m25). Whether an eligible BH is
│                                   #   currently DELIVERING no longer distinguishes itself from
│                                   #   eligibility; both read the same under-exposure axis of
│                                   #   Part 3. AND THAT POOLED STATEMENT IS ITSELF DECOMPOSED IN
│                                   #   6l: 'not running now' is also 'ran for less time', and
│                                   #   split into its halves the contrast is carried ENTIRELY by
│                                   #   the never-exposed (m100 xact: never OR 3.14, dropout OR
│                                   #   1.21 at p 0.25).
│                                   #   Part 6l = THE DOSE BEHIND THE LABELS: is 'starved' just
│                                   #   'less exposed'? (reads 6j's C6J + the anchor M_BH / E_x
│                                   #   from QM; centrals only). 6j now carries the dose itself:
│                                   #   dose6j integrates each channel over the WHOLE history
│                                   #   (e_dose = int w dt, e_on = time above TAU6J, e_late = the
│                                   #   ON time inside the last LATE6J = 1 Gyr, t_last_on = how
│                                   #   long since it was last ON) and the ladder prints med E_on
│                                   #   per class. (A) the dose of starved vs running on those
│                                   #   axes + B's catalogue e_x, as medians and as an AUC for the
│                                   #   starved label; E6L_SETS scores 'starved' (never +
│                                   #   dropout), 'dropout' alone and 'never' alone, because the
│                                   #   pooled class mixes a no-dose population with a shut-down
│                                   #   one. (B) the 6k contrast run at fixed (age, log M*) and
│                                   #   again at fixed (age, log M*, e_on), same stratified AUC +
│                                   #   MH OR, same within-cell permutation. (C) the mirror tests
│                                   #   - is the dose a DIAL? the AUC of log M_dust/M* for the
│                                   #   low-dose half at fixed age + mass INSIDE one state at a
│                                   #   time, and for the long-since-shutdown half of the dropouts
│                                   #   (dust regrowth once the channel stops). (D) is 'never'
│                                   #   just 'no jet-mode BH'? the share below SIMBA's 10^7.5 jet
│                                   #   threshold on both sides, and the contrast repeated on the
│                                   #   BHs that cleared it with log M_BH held fixed ->
│                                   #   noxray_dose_confound{.png,.pdf} +
│                                   #   noxray_dose_{confound,bins,mirror,bhmass}.csv.
│                                   #   RESULT (0.6 gate, 2026-09-01): the confound is REAL -
│                                   #   m100's starved centrals spent 0.44 Gyr ON against 0.96 for
│                                   #   the running (AUC 0.24) and only 0.23 against 0.62 Gyr in
│                                   #   the last Gyr (AUC 0.11); m25 0.13 against 1.20. Splitting
│                                   #   the class settles 6k: m100 xact 'never' (122 centrals, med
│                                   #   e_on 0.15 Gyr) is 39 vs 16 % dusty, AUC 0.65 / OR 3.14 at
│                                   #   fixed age + mass and 0.63 / 2.60 at fixed dose (p < 5e-4
│                                   #   throughout), while 'dropout' (294 centrals, med e_on 0.58
│                                   #   Gyr) is 20 vs 16 %, AUC 0.50 (p 0.805), OR 1.21 (p 0.251)
│                                   #   - null at fixed dose as well. Nor is the dose a dial:
│                                   #   within the running class more delivery does NOT mean less
│                                   #   dust (AUC 0.53, p 0.127) and within the dropouts a longer
│                                   #   lag does not mean more dust (0.47, p 0.587, median lag
│                                   #   0.62 Gyr) - no regrowth signal. Read together that is a
│                                   #   THRESHOLD near the ~0.5 Gyr at which B Part 9's D/G
│                                   #   response saturates, with never / dropout on opposite sides
│                                   #   of it. The BH-mass repeat picks the channel: on xact in
│                                   #   m100 only 2 % of the never-exposed sit below the jet
│                                   #   threshold and the contrast survives among the 119 that
│                                   #   cleared it (38 vs 15 %, AUC 0.63, OR 2.79, p < 5e-4),
│                                   #   while on xcoup 9-19 % are below it and the 32 that remain
│                                   #   are 19 vs 17 % (OR 0.92); m25 is the extreme case (100 %
│                                   #   of its xcoup starved and 64 % of its xact never are sub-
│                                   #   threshold), so its headline 6k rows do not survive this
│                                   #   control at all. WHAT those sub-threshold BHs are is
│                                   #   settled in 6m: not small black holes but black holes the
│                                   #   CATALOGUE LOST (19/21, 17/18, 7/7, 8/8 of them), so (D)
│                                   #   removes a bookkeeping population for a reason it could not
│                                   #   see.
│                                   #   THE CORRECTED STATEMENT OF THE 6j-6l ARC: the dusty tail
│                                   #   is not about the AGN having STOPPED, it is about the AGN
│                                   #   never having DELIVERED. In m100 a central whose gated
│                                   #   X-ray coupling never reached ON for as long as DTON6J =
│                                   #   0.3 Gyr - despite a black hole above SIMBA's jet threshold
│                                   #   - is ~2.8x more likely to sit in the dusty tail at fixed
│                                   #   age, mass and BH mass; a central whose coupling ran and
│                                   #   then stopped is indistinguishable from one whose coupling
│                                   #   still runs, and among the exposed neither total dose nor
│                                   #   recency grades the dust further.
│                                   #   Part 6m = THE DUST AGAINST THE TIME SINCE THE LAST
│                                   #   INJECTION, AND AGAINST THE CAUSE OF THE INTERRUPTION
│                                   #   (reads 6j's C6J, centrals only). 6k/6l cut on one binary -
│                                   #   is the coupling running NOW - which pools cycling with
│                                   #   stable and says nothing about how long ago it stopped or
│                                   #   why. The two channels stated once: ELIGIBLE = xcoup =
│                                   #   w_jet x gate (a BH above 10^7.5 not accreting so hard the
│                                   #   jet weight falls away, in a galaxy gas-poor enough for
│                                   #   f_gas < 0.6 - SIMBA WOULD inject); DELIVERED = xact =
│                                   #   xcoup x [f_Edd > 1e-4] (it actually is). ON = channel >=
│                                   #   0.9 after debouncing runs under 0.3 Gyr. THE LAST
│                                   #   INJECTION = the END of the last believable ON run
│                                   #   (t_last_on, off the DEBOUNCED series so a single spike
│                                   #   cannot reset the clock): 0 while still ON, NaN if never
│                                   #   ON. E6M_LAG = (0.15, 0.50) Gyr cuts it into injecting /
│                                   #   dormant / shut, with 'never' apart.
│                                   #   THE WOBBLE NOW CARRIES A REASON (6j causes6j): every OFF
│                                   #   excursion is labelled by what held for most of its
│                                   #   duration - fedd_up = f_Edd rose above FEDD_W6J =
│                                   #   JET_FEDD/10^TAU6J = 0.025 so the JET WEIGHT ITSELF fell
│                                   #   below ON (the coupling lost because the BH accretes
│                                   #   HARDER: the new-gas episode); fedd_down = f_Edd fell below
│                                   #   1e-4 (the BH went quiet); gate_open = f_gas came back
│                                   #   above 0.6; bh_small / bh_lost. Each cause is scored
│                                   #   against the SMOOTH histories (no completed excursion) at
│                                   #   fixed (age, log M*, e_on).
│                                   #   E6M_DBH / DBH6J = THE BLACK HOLE THE CATALOGUE LOST: in ~2
│                                   #   % of the m100 histories (~7 % in m25) the tracked M_BH
│                                   #   collapses 2-3 dex in ONE snapshot at fixed log M* (8.0 ->
│                                   #   4.7 with log M* steady at 10.45) and then regrows - the
│                                   #   group finder dropping the BH, not a shrinking black hole.
│                                   #   Every channel goes off there, so those galaxies enter the
│                                   #   taxonomy as textbook shutdowns AND as excursions, and they
│                                   #   are 86 % DUSTY against 17 %. 6j carries lmbh_max beside
│                                   #   lmbh_anch; 6m flags them at 0.5 dex and runs every block
│                                   #   from (C) on without them.
│                                   #   RESULT (0.6 gate, 2026-09-01): the RECENCY AXIS IS FLAT -
│                                   #   m100 median log M_dust/M* = -4.81, -4.89, -4.81, -4.67,
│                                   #   -4.64, -4.73, -4.72 from < 0.06 to > 2 Gyr (box median
│                                   #   -4.79) at a median E_on that stays 0.7-1.0 Gyr; 'shut' 19
│                                   #   % dusty vs a 17 % base (AUC 0.52 p 0.575, OR 1.38 p
│                                   #   0.184), 'dormant' null, and the sweep says the same at
│                                   #   every boundary. The apparent signal WAS the lost BHs: m100
│                                   #   xcoup 'shut' looked like 19 centrals at 89 % dusty, lift
│                                   #   5.0, OR 26 - 89 % of it flagged, 2 survive; xact 'shut'
│                                   #   falls 26 -> 19 % (OR 2.16 p 0.001 -> 1.38 p 0.184); m25
│                                   #   loses its rare classes entirely. The only displaced class
│                                   #   is 'never' (120 BH-kept centrals, median -4.31 against
│                                   #   -4.79).
│                                   #   THE INTERRUPTION IS REAL, AND IT HAPPENS FOR BOTH REASONS
│                                   #   - m100 xact against 1731 smooth histories at fixed age +
│                                   #   mass + DOSE: fedd_up 89 gals, 22 vs 15 % dusty, median
│                                   #   -4.65 vs -4.80, AUC 0.57 (p 0.040), OR 1.94 (p 0.051);
│                                   #   fedd_down 303, 21 vs 15 %, AUC 0.55 (p 0.019), OR 1.66 (p
│                                   #   0.016); any excursion 390, 23 vs 15 %, AUC 0.56 (p 0.002),
│                                   #   OR 1.87 (p 0.001). Strongest cell of the cross: fedd_up x
│                                   #   dormant (44 % dusty, 9 centrals) - the BH woke up, broke
│                                   #   the coupling, and has not injected since. BUT THE GATE IS
│                                   #   NOT WHAT REOPENS: gate_open labels ONE excursion in all of
│                                   #   m100; on the factor channels the composite wobbles in 18 %
│                                   #   of centrals, acc in 15 %, xcoup in 4 %, THE GAS GATE IN 1
│                                   #   % (and those are dust-poor, 11 vs 17 %); the gas-gate-open
│                                   #   fraction of the final OFF stretch is 0.00 for every state
│                                   #   but 'never'. What breaks the coupling is the Eddington
│                                   #   ratio moving, not gas coming back.
│                                   #   WHERE THE ARC LANDS: two effects survive age, mass, dose,
│                                   #   BH mass and the lost-BH flag, both in m100's delivered
│                                   #   channel - the channel NEVER HAVING DELIVERED (never, 120
│                                   #   centrals: 38 vs 17 %, AUC 0.64, OR 2.26 p 0.001; OR 2.79
│                                   #   at fixed log M_BH in 6l) and the delivery HAVING BEEN
│                                   #   INTERRUPTED (OR 1.87 pooled, 1.94 on the f_Edd-up branch,
│                                   #   at fixed dose). What does not survive: how much was
│                                   #   delivered (6l), how long ago it stopped, and every rare-
│                                   #   class headline in m25.
│                                   #   FIGURE noxray_recency{.png,.pdf} = per box the
│                                   #   DISTRIBUTION of log M_dust/M* against the time since the
│                                   #   last injection (violins + points + medians, never-ON as
│                                   #   its own column), the same as cumulative distributions per
│                                   #   recency state, and again split by the cause of the
│                                   #   interruption; + noxray_recency_{states,bins,wobble,factors
│                                   #   ,offstretch,sweep}.csv
│                                   #   Part 7 = the m25 CLOSE-UP of
│                                   #   resilient vs destroyed (the only box with resolved kappa_gas /
│                                   #   ISM): (A) log f_dust / f_H2 / sSFR histories on the t - t_SFT
│                                   #   clock (histories hdf5; the dust and H2 legs once the mdust/mh2
│                                   #   rebuild ran — until then the middle panel falls back to f_gas), (B) radial f_dust / f_gas / f_H2 / age per annulus (m25
│                                   #   8a/7e tables, sightline medians, bootstrap bands), (C) the K-S
│                                   #   plane at R50_H2 (KS 'pt' tracks: sft -> qt -> end, end regions
│                                   #   vs B08) -> noxray_resilient_m25_{histories,profiles,ks}.
│                                   #   {png,pdf} + noxray_resilient_m25.csv; each block gates on its
│                                   #   product file (cluster)
│                                   #   (Part 7 B upgraded 2026-08-31 to the X-ray notebook's
│                                   #   Section-3 form: A_V + fdust/fgas/fH2 + age + kappa
│                                   #   gas/H2/star per annulus, SF controls grey line; the A_V /
│                                   #   kappa panels and the SF line gate on annulus_av_allincl /
│                                   #   annulus_kinematics / powderday_quenched_selection). Part 8 =
│                                   #   the Section-4-style CRITICAL-POINT SEQUENCES (m25): core
│                                   #   (ap3kpc) / outskirt (ann10kpc = ap10-ap3) at agn_ign ->
│                                   #   jet_on -> sft -> qt -> end, kappa_H2/kappa_star/sSFR/age/
│                                   #   fdust/fH2 per stage resilient vs destroyed, ALMA-C11
│                                   #   detection min-max bands (ks_tracks_pt/ks_tracks.fits +
│                                   #   ks_tracks/ks_stage_kinematics.fits) ->
│                                   #   noxray_resilient_m25_sequences{.png,.pdf,.csv}. Part 9 = the
│                                   #   Section-5a-style D/H2-vs-exposure planes (catalogue, always
│                                   #   runs): log M_dust/M_H2 vs E_x (fixed bins, E_x = 0 pile) /
│                                   #   dt_SFT / T_gaspoor per box, P6 groups scattered + binned
│                                   #   medians, all dotted, nox median dashed, rho printed for all
│                                   #   and inside old+dosed -> noxray_resilient_exposure{.png,.pdf,
│                                   #   .csv}. Part 10 = the Section-5b-style ISM GRID per box: lfd
│                                   #   (floor kept) and lfh2 vs age / Sigma_e / sSFR, running-
│                                   #   median tracks of resilient/destroyed/young_dusty + all
│                                   #   dotted + nox dashed, the ALMA-C11 QGs as points (dust / CO
│                                   #   ULs arrowed; H2 row = obs alpha_CO H2 vs sim 1.36 M_H2, like
│                                   #   for like) +
│                                   #   the (measured - track) scorecard over the detections ->
│                                   #   noxray_resilient_ismgrid_{cis50,cis100,cis25}{.png,.pdf} +
│                                   #   noxray_resilient_ismgrid.csv
├── agn_states_dusty_vs_pure.ipynb  # the AGN ACTIVATION HISTORY of the THREE DUST ROUTES of quenched galaxies
│                                   #   across the boxes: an ordered restart of noxray_analogues (which grew a
│                                   #   2x2 of age x dust and thirteen sub-parts on top of it). ONE flat dust
│                                   #   division; pure reads of the ism_prediction products (Part 1
│                                   #   catalogue cache, Part 6a classes + per-box history caches) ->
│                                   #   output/box_resolution/agn_states/. H2 AND DUST ONLY: every MEASURED ISM
│                                   #   axis is molecular (log 1.36 M_H2/M*, log M_dust/M*, log M_dust/M_H2);
│                                   #   M_gas survives in exactly two structural places - SIMBA's own X-ray gate
│                                   #   f_gas < XRAY_FGAS_MAX (a criterion of the RUN, not a measurement) and
│                                   #   the metal budget Z_gas M_gas + M_dust behind the dust-to-metal ratio,
│                                   #   where the metals genuinely live in the whole reservoir. 'lfgas' is still
│                                   #   carried in the DOSE frame so that budget can be formed; nothing plots it.
│                                   #   THE SAMPLE, DECLARED ONCE IN PART 0 AND APPLIED ONCE, so every part runs
│                                   #   on the SAME galaxies: the m25 quenched rule, then MASS_MIN (log M* >
│                                   #   10.25), Z_MAX (z <= 1) and AGE_MIN (age >= 2 Gyr), AGE_MAX optional /
│                                   #   unset; each cut's own toll and the per-box totals printed there, so the
│                                   #   sample is auditable in one place -> cis100 1373 / cis50 123 / cis25 80.
│                                   #   THIS IS WHAT IT IS FOR: before it existed the dose engine carried its own
│                                   #   z <= 0.5, which Parts 8-12 inherited through DOSE while Parts 13-15 read
│                                   #   every anchor, and the m25 close-up silently ran on 37 of 178 galaxies.
│                                   #   P7_ZMAX (Parts 8-13) and P13_ZMAX / P14_ZMAX (the m25 close-ups) survive
│                                   #   as EXTRA restrictions, all default None.
│                                   #   THE AGE CUT IS A DELIBERATE CHANGE OF PRINCIPLE: the notebook was built
│                                   #   with none, so that an old-vs-young difference between the dusty and the
│                                   #   dust-poor population would be MEASURED on a dust-selected pair rather than
│                                   #   defined. AGE_MIN trades that for a sample uniform across the notebook. It is
│                                   #   NOT calibrated on the observed sample and cannot be: the ALMA-C11 ages are
│                                   #   CIGALE sfhdelayedbq ONSET ages while this is caesar's MASS-WEIGHTED age
│                                   #   (see the age-axis note under Parts 14-15). And it is NOT neutral - the
│                                   #   young under-dosed galaxies it drops are one of the
│                                   #   three causes of the 'timing-explained' route (m25 under-dosed 60 -> 5) and
│                                   #   Part 8's dose-response weakens with them (cis100 rho over the exposed
│                                   #   -0.35 -> -0.12). AGE_MIN = None restores the original behaviour.
│                                   #   SELECTION (Part 1) = the census of that sample, then the single split at
│                                   #   DUSTY_CUT = log M_dust/M*: 'dusty QG' at or above, 'dust-poor QG' below
│                                   #   (the zero-dust P9_FLOOR spike lands in dust-poor). age is still REPORTED
│                                   #   as a contrast, now measured within the cut -> the cuts drawn on the lfd
│                                   #   and log M* distributions per box (agn_states_selection.{png,pdf}) +
│                                   #   per-quantity medians / AUC / Mann-Whitney -> agn_states_census.csv.
│                                   #   Part 2a = THE AGN HISTORY INTEGRATED IN TIME, AND THE THREE ROUTES.
│                                   #   (This was Part 7, drawn after the tracks; it moved to the front because
│                                   #   the routes are defined from it and Parts 2b-4 are now split on them.)
│                                   #   One pass over each box's main-branch chains -> one record per
│                                   #   galaxy-anchor: int Mdot_BH dt (the total AGN energy budget), the same
│                                   #   over the 'xray' snapshots, int w_jet Mdot_BH dt, the coupling int w_jet
│                                   #   dt with and without the accretion floor, the exposure E_x = int_xray dt,
│                                   #   t_quiet = t_anchor - t(last X-ray), t_since_acc, n_x / f_x / t_span, and
│                                   #   zgas at the anchor (the branch's last existing index - the catalogue
│                                   #   cache carries no metallicity). THE CAVEAT IS IN THE DEFINITIONS: w_jet =
│                                   #   clip(log10(0.2/f_Edd), 0, 1) is 1 for ANY f_Edd <= 0.02, so a dead BH
│                                   #   scores maximal coupling for free - the coupling integrals are gated at
│                                   #   P7_FEDD_ACT (1e-4, env-overridable), the accretion-weighted counterparts
│                                   #   are the un-foolable versions, and the printed line carries the share of
│                                   #   E_x delivered below the floor. Window = P7_WINDOW (None = the whole
│                                   #   branch; a float = that many Gyr; 'sft' = since t_SFT).
│                                   #   The integrals are built for every anchor of Part 0's sample; DOSE_ALL and
│                                   #   DOSE are now the same frame (P7_ZMAX default None), the distinction kept
│                                   #   only so that tightening Parts 8-13 alone still works. The full frame, with
│                                   #   z / cat / grp columns, -> agn_states_dose_vs_dust_galaxies.csv.
│                                   #   THE THREE ROUTES, from E_x, t_quiet and DUSTY_CUT (PF_EX 'fully dosed'
│                                   #   1 Gyr, PF_TQ 'still firing' 0.5 Gyr, CATS, route_cat / route_grp - ALL
│                                   #   DEFINED HERE, change them once): 'dusty, dosed & firing' (dusty, E_x >=
│                                   #   PF_EX AND t_quiet < PF_TQ - the channel had every chance and is firing
│                                   #   NOW: the AGN timing does not explain it), 'dusty, timing-explained'
│                                   #   (dusty, but never armed / under-dosed / long since quiet), 'dust-poor'.
│                                   #   Part 9 draws the FOUR-way version of the same rule (the timing-explained
│                                   #   route opened into its three causes); everything else uses the three.
│                                   #   STATES (Part 2b) = four EXCLUSIVE, EXHAUSTIVE states per snapshot of
│                                   #   the main branch, from f_gas = M_gas/M*, Mdot_BH = f_Edd M_BH /
│                                   #   4.5e7 yr (eta 0.1; derived, the catalogues carry masses.bh +
│                                   #   bh_fedd) and f_Edd, evaluated dormant -> xray -> choked ->
│                                   #   radiative: 'dormant' (Mdot_BH = 0, incl. no BH yet - the boxes
│                                   #   notebook's missing-BH-is-w_jet-0 convention), 'xray' (M_BH > 10^7.5,
│                                   #   full-velocity jet w_jet >= W_FULL i.e. f_Edd <~ 0.02, AND f_gas <
│                                   #   XRAY_FGAS_MAX), 'choked' (the jet criterion holds but a partial-
│                                   #   velocity jet or a gas-rich host kills the X-ray), 'radiative'
│                                   #   (accreting, no jets: f_Edd >= 0.2 or M_BH under the threshold).
│                                   #   The four stay four; the printed census splits each into its causes
│                                   #   (nobh / pjet / fjrich / quasar / lowmbh), PER ROUTE ->
│                                   #   agn_states_snapshot_census.csv. XRAY_FGAS_MAX is the one physics
│                                   #   knob: 0.2 = SIMBA (Dave+19) and what ism_prediction Part 6a uses,
│                                   #   env XRAY_FGAS_MAX=0.6 reproduces the noxray_analogues reading; only
│                                   #   the xray|choked split moves with it. agn_states() and the histories-cache
│                                   #   discovery (BOXES_OK / HAVE_MD / HAVE_MH2) live in Part 0, so Part 2a can
│                                   #   run before the state engine.
│                                   #   FIGURES = every chain on the SFT clock t - t_SFT (TG -3..+5 Gyr,
│                                   #   linear interpolation, no extrapolation, blanked under NMIN = 3 - lowered
│                                   #   from 5 when the tracks moved from dusty/pure to the three routes), one
│                                   #   column per box (m100 / m50 / m25; append 'cis50nox' to BOXES for its
│                                   #   would-be states), THE THREE ROUTES as three median tracks with bootstrap
│                                   #   16-84 bands and THE SOURCE COUNT AS A HISTOGRAM BEHIND EVERY PANEL (all
│                                   #   three on one scale, peak N printed in the corner). Two of the three
│                                   #   differ in AGN history BY CONSTRUCTION, so the contrast that carries
│                                   #   information is FIRING-DUSTY vs DUST-POOR: galaxies dosed alike, differing
│                                   #   only in whether the dust survived.
│                                   #   Part 3 = (A) the parameters log 1.36 M_H2/M* / log Mdot_BH / log f_Edd /
│                                   #   log M_BH / log sSFR / log M_dust/M* (the H2 and dust rows auto-dropped if
│                                   #   a cache lacks mh2 / mdust) -> agn_states_parameter_tracks.{png,pdf};
│                                   #   (B) the state occupancy state by state ->
│                                   #   agn_states_occupancy.{png,pdf}; (C) the same stacked per (box, route) ->
│                                   #   agn_states_composition.{png,pdf}; tracks + the per-epoch AUC of all three
│                                   #   contrasts in GPAIRS (firing vs dust-poor, timing vs dust-poor, firing vs
│                                   #   timing) -> agn_states_tracks.csv.
│                                   #   Part 4 = the SAME grid with the DISTANCE FROM THE 0.2/t_H QUENCHING
│                                   #   CURVE on the y-axis, Delta_q = log sSFR - log(0.2/t_H) evaluated at
│                                   #   each snapshot's own t_H (Delta_q = 0 IS the quenched definition, the
│                                   #   QT crossing): one row for all galaxies + one per AGN state,
│                                   #   restricted to the galaxies sitting in that state at that epoch (the
│                                   #   argmax of the four interpolated indicators) ->
│                                   #   agn_states_quenching_distance.{png,pdf,csv}. Env AGN_NBOOT trims the
│                                   #   bootstrap for a smoke test.
│                                   #   Part 8 = THE SAME DATA READ AS AN IMPLICATION (pure post-processing of
│                                   #   Part 2a's DOSE frame, no new reads). A correlation between the dust and
│                                   #   the dose is the wrong instrument: the relation is not a dose-response
│                                   #   but a ONE-WAY IMPLICATION. (A) the DUSTY FRACTION (Wilson 68 %,
│                                   #   hand-rolled - no statsmodels on the cluster) against E_x, the
│                                   #   never-exposed as their own point off the axis: the drop is a STEP in the
│                                   #   first bin and then a plateau; the SATURATION SCAN re-measures
│                                   #   rho(lfd, E_x) over the exposed above a rising E_x floor (P8_SCAN) - a
│                                   #   dose would hold, a threshold collapses. (B) FOUR CONDITIONALS with
│                                   #   Wilson intervals and raw counts: P(ever exposed | pure) = NECESSITY (can
│                                   #   a galaxy end up dust-poor without the channel?), P(pure | ever exposed) =
│                                   #   SUFFICIENCY, P(dusty | never exposed) = the rule, P(never exposed |
│                                   #   dusty) = THE CONVERSE, with the base rates as dotted lines so the
│                                   #   asymmetry reads as arithmetic (the never-exposed are a few % of the
│                                   #   population) and not as a weakening of the rule. (C) WHAT THE
│                                   #   NEVER-EXPOSED ARE: AUC of every catalogue / history property between
│                                   #   never-exposed and exposed (P8_CONTRAST), so it is visible whether the
│                                   #   label does work or just restates young / low-mass / small M_BH; the
│                                   #   printed census splits the never-exposed into those that DID launch a jet
│                                   #   (the gas or velocity gate shut the X-ray off) and those that never grew a
│                                   #   BH. THE CENSORING CAVEAT IS TESTED: the conditionals are re-printed over
│                                   #   the QUALIFIED sample (t_span >= P8_SPAN_FRAC 0.5 x the box's median
│                                   #   exposed span) with the count of never-exposed that survive it - a 'never'
│                                   #   over 1 Gyr of history is not a 'never' over 8 ->
│                                   #   agn_states_exposure_implication.{png,pdf,csv} + ..._saturation.csv +
│                                   #   ..._neverexposed.csv.
│                                   #   Part 9 = THE FIGURE: Parts 7/8/8b are the working, this is the argument
│                                   #   on one page for one box (PF_BOX, default the largest). TOP ROW: (a) the
│                                   #   channel is NECESSARY and then SATURATES - dusty fraction vs E_x with the
│                                   #   never-exposed off-axis and P(ever exposed | dust-poor) printed; (b) once
│                                   #   it stops the dust COMES BACK - dusty fraction vs t_quiet, the
│                                   #   still-firing galaxies as their own point; (c) EVERY dusty QG in exactly
│                                   #   one box by WHEN the channel fired: never armed / under-dosed (E_x <
│                                   #   PF_EX 1 Gyr) / dosed-then-quiet (t_quiet >= PF_TQ 0.5 Gyr, the dust
│                                   #   regrew) / DOSED AND STILL FIRING - the first three are the timing story,
│                                   #   the fourth is what it cannot reach (29 / 37 / 45 % of the dusty QGs in
│                                   #   cis100 / cis50 / cis25). BOTTOM ROW = THE RESIDUAL: those
│                                   #   dosed-and-still-firing galaxies, dusty vs dust-poor, as normalised
│                                   #   histograms on every axis the AGN history offers (age, E_x, int w_jet dt,
│                                   #   int_xray Mdot dt, M_BH, M*, sSFR, f_H2, Z_gas) - every
│                                   #   AGN-history one AUC ~ 0.5, sSFR the one mild exception (0.67 / 0.73 /
│                                   #   0.79, the dusty ones slightly LESS quenched) - and then on the
│                                   #   DUST-TO-METAL ratio and M_dust/M_H2. AT DUSTY_CUT = -4.5 (cis100 /
│                                   #   cis50 / cis25): AGN axes age 0.49 / 0.47 / 0.58, E_x 0.48 / 0.58 /
│                                   #   0.51, int w_jet dt 0.47 / 0.42 / 0.57, M_BH 0.48 / 0.56 / 0.53, and
│                                   #   f_H2 0.50 / 0.41 / 0.85; sSFR the one mild exception (0.65 / 0.57 /
│                                   #   0.87, the dusty ones slightly LESS quenched); then DTM 0.954 / 0.990 /
│                                   #   0.993 (+1.10 dex in cis100) and M_dust/M_H2 0.959 / 0.972 / 0.980
│                                   #   (+1.11 dex). METALLICITY GOES THE WRONG WAY (AUC 0.45 / 0.32 / 0.29:
│                                   #   the dusty ones are marginally metal-POORER), so the split is NOT
│                                   #   enrichment - it is the grain budget. The last two AUCs share a
│                                   #   numerator with the DUSTY_CUT split axis, so the argument rests on the
│                                   #   DENOMINATORS (H2, Z) being coin flips. Z_gas comes from zgas at the
│                                   #   anchor, which PART 2a RECORDS (a DOSE built before that column raises
│                                   #   with 're-run Part 2a'); PF_Z_HAS_DUST
│                                   #   carries the assumption that SIMBA depletes grain metals out of the gas
│                                   #   phase, so the metal budget is Z_gas M_gas + M_dust. ONE RESIDUAL ROW PER BOX (PF_ROWS = None ->
│                                   #   every box with >= PF_MIN_CLASS 3 on both sides), the individual galaxies
│                                   #   always rugged along the bottom so a 5- or 7-object histogram is not read
│                                   #   as a distribution; panel (c) likewise draws one stacked bar per box.
│                                   #   ADDING AN AXIS = one entry in PF_PROPS: any DOSE column works, anything
│                                   #   else is looked up in the Part 0 catalogue frame QM and merged on (box,
│                                   #   snap, gal_id) - which is how lssfr gets in, Part 2a never carried it -
│                                   #   and PF_DERIVED holds the recipes built from other columns (dg, dh2,
│                                   #   sfe); an unknown name raises with all three lists printed. THE LAST
│                                   #   ENTRY IS THE PUNCHLINE, drawn framed in red. A MATCHED comparison,
│                                   #   not a fit: same age, same dose, same BH, same H2. Two honesty notes in
│                                   #   the cell + the markdown: PF_EX / PF_TQ are cuts, so the decomposition and
│                                   #   the residual AUCs are printed for EVERY box (the split reproduces at
│                                   #   three resolutions); and the M_dust/M_H2 panel's AUC is half-circular
│                                   #   (the sample is split on M_dust/M*, which shares a numerator) - the claim
│                                   #   rests on the H2 panel beside it showing nothing ->
│                                   #   agn_states_the_argument.{png,pdf}
│                                   #   + ..._categories.csv + ..._residual_auc.csv.
│                                   #   Part 11 = THE DUST TRACKS OF THE THREE ROUTES on the SFT clock - the
│                                   #   comparison that is NOT dusty-vs-pure but unexplained-residual vs
│                                   #   timing-explained-dusty vs dust-poor, and NOT at the anchor but as a
│                                   #   RATE. Rows: log M_dust/M_H2, log dust-to-metal, log Z_gas, log f_dust,
│                                   #   log 1.36 M_H2/M*, log sSFR; columns = boxes; median + bootstrap 16-84
│                                   #   band, source
│                                   #   counts behind. Each track's slope over P11_FIT (0-3 Gyr after the SFT,
│                                   #   drawn as a shaded strip) is bootstrapped over the galaxies and labelled
│                                   #   RISES / FALLS / holds by whether the interval clears zero ->
│                                   #   agn_states_dust_tracks_routes.{png,pdf} + ..._slopes.csv. THE TEST: if
│                                   #   the residual's dust were MADE after quenching its M_dust/M_H2 would
│                                   #   rise while the dust-poor curve falls. RESULT (cis100, DUSTY_CUT -4.5):
│                                   #   ALL THREE FALL - M_dust/M_H2 slope -0.45 (residual) / -0.53
│                                   #   (timing-explained) / -0.64 (dust-poor) dex/Gyr, D/M -0.42 / -0.51 /
│                                   #   -0.62; the two dusty routes are much closer to each other than to the
│                                   #   dust-poor one. Z_gas, f_H2 and sSFR tracks are near-IDENTICAL, so
│                                   #   the routes differ in the dust alone, not in the fuel. No route makes
│                                   #   dust after the SFT: the dusty QGs are slower losers, not producers.
│                                   #   NB Part 9's panel (b) is a CROSS-SECTION (the long-quiet are more often
│                                   #   dusty), not regrowth - its title now says so.
│                                   #   Part 12 = THE PLAIN ONE: fixed,
│                                   #   LOG-SPACED t_sq bins (P12_NBIN 6, P12_LOG, P12_EDGES for explicit
│                                   #   edges), one point per route per bin - median t_sq on x, median property
│                                   #   on y, and the BOOTSTRAP ERROR ON THE MEDIAN as the bar. Nothing else on
│                                   #   the plot. P12_NMIN = 1 so a single-galaxy bin is KEPT, drawn as an open
│                                   #   symbol with no bar (a galaxy is not a median). Seven rows: log
│                                   #   M_dust/M_H2, log dust-to-metal, 12+log(O/H) with SOLAR drawn, log
│                                   #   M_H2/M*, log sSFR, log M*; columns = boxes. Two
│                                   #   conventions stated in the cell: the dust-to-H2 ratio
│                                   #   divides out the notebook's 1.36 helium factor so it is a ratio to the
│                                   #   BARE molecular mass, and 12+log(O/H) = 8.69 + log(Z/0.0134)
│                                   #   (Asplund+09) is SCALED FROM THE TOTAL METAL MASS FRACTION, i.e. solar
│                                   #   abundance ratios assumed - the catalogue carries Z, not oxygen. The
│                                   #   population's own 16-84 spread is in the CSV (<prop>_p16/_p84), not the
│                                   #   figure, which carries the error on the median instead ->
│                                   #   agn_states_tsq_binned_properties.{png,pdf,csv}.
│                                   #   RESULT: every property is FLAT in t_sq; the routes separate in the dust
│                                   #   ratios alone (cis100 at DUSTY_CUT -4.5: log M_dust/M_H2 -1.79
│                                   #   timing-explained / -2.36 firing / -3.48 dust-poor, DTM -0.45 / -1.06 /
│                                   #   -2.15) while 12+log(O/H) (8.83-8.96, all ABOVE solar), M_H2/M* (-1.43
│                                   #   for all three), sSFR and M* overlap between them.
│                                   #   Part 13 = THE m25 CLOSE-UP OF THE THREE ROUTES (noxray_analogues Part 7
│                                   #   re-cut on the three routes, under the names Parts 11-12 give them): the routes of
│                                   #   Parts 11-12 followed into the PARTICLE-LEVEL products of the only box where the
│                                   #   ISM and kappa_rot are resolved. THE ONE PART THAT IS NOT A PURE READ of the
│                                   #   ism_prediction products, and every block gates on its own file and prints what
│                                   #   is missing instead of failing, so the cell runs unchanged off the cluster.
│                                   #   (A) the log M_dust/M*, log 1.36 M_H2/M* and log sSFR HISTORIES on the t - t_SFT
│                                   #   clock (the same history cache, plus the molecular leg Part 11 does not carry):
│                                   #   median + bootstrap 16-84 per route, the individual chains behind them, the count
│                                   #   per epoch shaded -> agn_states_routes_m25_histories.{png,pdf}.
│                                   #   (B) the RADIAL ANATOMY per annulus (0-1 / 1-3.2 / 3.2-10 / 10-32 kpc) from
│                                   #   output/cis25/tables/{annulus_ism_truth,aperture_truth}.fits (+ annulus_av_allincl
│                                   #   -> the A_V panel, annulus_kinematics -> the kappa_H2 / kappa_star
│                                   #   panels, powderday_quenched_selection -> the grey SF-controls line, each optional):
│                                   #   A_V, M_dust/M*, M_H2/M*, stellar age and the rotation support, as
│                                   #   PER-GALAXY SIGHTLINE MEDIANS (a galaxy counts once per annulus) with a bootstrap
│                                   #   16-84 of the group median; the nstar_ap >= 20 sampling floor of the m25 notebook
│                                   #   kept -> agn_states_routes_m25_profiles.{png,pdf}.
│                                   #   (C) the K-S PLANE at r < R50(H2) from output/cis25/ks_tracks_pt/ks_tracks.fits +
│                                   #   ks_tracks_lib: the sft -> qt -> end track of each route, the m25 SF controls
│                                   #   behind, B08 +- scatter and K98, and the printed census of where each route's END
│                                   #   points sit w.r.t. B08 (below / on / above / censored) ->
│                                   #   agn_states_routes_m25_ks.{png,pdf}. Every point of the three blocks, with its
│                                   #   count, -> agn_states_routes_m25.csv. P13_NMIN = 3 is BELOW the notebook's NMIN on
│                                   #   purpose - in m25 the 'dusty, dosed & firing' residual is a handful of objects -
│                                   #   so every legend and every row carries its N and a three-galaxy median reads as
│                                   #   one. P13_ZMAX (None) is an OPTIONAL extra restriction: the cell reads Part 2a's
│                                   #   DOSE_ALL, i.e. every anchor of Part 0's sample that has a route, not Part 9's
│                                   #   frame F. Historically F was a z <= 0.5 subset and inheriting it cut m25 to 37 of
│                                   #   its 178 routed anchors (all 178 ARE in the K-S tracks); Part 0's Z_MAX / AGE_MIN
│                                   #   now fix the sample for every part, so the two agree by construction.
│                                   #   Part 14 = THE CRITICAL-POINT SEQUENCES OF THE THREE ROUTES (noxray_analogues
│                                   #   Part 8, itself the X-ray notebook's Section 4 lean, re-cut on Part 2a's
│                                   #   routes). The m25 galaxies followed through AGN ignition -> jet-mode onset
│                                   #   -> SFT -> QT -> track end, in the TWO SPHERICAL ZONES of the K-S
│                                   #   notebook's Part 3b: core (r < 3.2 kpc) and outskirt (3.2-10 kpc). Per
│                                   #   stage and zone: kappa_rot^H2, kappa_rot^star and the mass-weighted age
│                                   #   from output/cis25/ks_tracks/ks_stage_kinematics.fits (the rule-free
│                                   #   cache), and the ISM from the tracks' PROJECTED apertures in
│                                   #   ks_tracks_pt/ks_tracks.fits (core = ap3kpc, outskirt = ap10kpc - ap3kpc):
│                                   #   sSFR, M_dust/M*, M_H2/M*. Median + galaxy-bootstrap 16-84, a stage point
│                                   #   needs P14_NMIN (3) galaxies, each route offset slightly along x so three
│                                   #   coincident points stay separable. Zero dust / H2 is NaN, never a floor;
│                                   #   zero SFR sits at SSFR_FLOOR because the quenching is the point. The pink
│                                   #   bands are the ALMA-C11 detections' min-max (dust detections for age /
│                                   #   f_dust / sSFR, CO detections for f_H2, /1.36 so BARE molecular masses are
│                                   #   compared). THE AGE PANEL IS NOT LIKE FOR LIKE and is hatched + labelled to
│                                   #   say so: the sim age is caesar's MASS-WEIGHTED age, the ALMA-C11 age is
│                                   #   CIGALE's sfhdelayedbq ONSET age (formation time, weighted by nothing). For a
│                                   #   delayed-tau SFH the mass-weighted age sits ~2 tau BELOW the onset (0.6-1.8
│                                   #   Gyr at these ages for tau = 0.3-1), and the bq episode lowers it further, so
│                                   #   the observed band is an UPPER bound; the table carries no tau so it is not
│                                   #   corrected. P14_OBS_AGE_LIKE flips it if a mass-weighted observed age appears.
│                                   #   Every other panel (kappa, sSFR, f_dust, f_H2) IS like for like. NOT a pure
│                                   #   read - it gates on both K-S files and says which is
│                                   #   missing -> agn_states_routes_m25_sequences.{png,pdf,csv}. P14_ZMAX follows
│                                   #   P13_ZMAX (default None = every routed anchor of Part 0's sample).
│                                   #   THE QUESTION IT SETTLES: a distinct ROUTE separates already at SFT/QT; a
│                                   #   late STATE only at 'end'.
│                                   #   Part 15 = THE ISM GRID UNDER THE ROUTE SPLIT (noxray_analogues Part 10,
│                                   #   the X-ray notebook's Section 5b lean, re-cut on the routes) - THE ONE
│                                   #   PLACE THE NOTEBOOK MEETS THE OBSERVATIONS. Per box, a 2x3 grid: log
│                                   #   M_dust/M* (zero dust at P9_FLOOR) and log 1.36 M_H2/M* against stellar
│                                   #   age, log Sigma_e and log sSFR. On each panel: the three routes as
│                                   #   running medians in equal-count x bins with a bootstrap 16-84 band
│                                   #   (run_track15; a route under 2 x G15_NMIN falls back to 4 bins, and a route
│                                   #   with <= G15_POINTS_MAX 150 galaxies also draws its points), the 'all
│                                   #   quenched with a route' dotted track, the channel-off cis50nox all-quenched
│                                   #   DASHED track (G15_NOX; drawn in every box's panel as the no-X-ray
│                                   #   reference), and THE ALMA-C11 QUENCHED GALAXIES AS POINTS with dust and CO
│                                   #   upper limits arrowed. The H2 row compares the observed alpha_CO H2 (helium
│                                   #   included) to the simulated 1.36 M_H2 - like for like. Observed points are
│                                   #   built once from obs_data/almac11/{almac11_gas_dust,age_sersic_sigma}.csv
│                                   #   (the ctrl rows dropped, Sigma_e from re_kpc); missing files -> tracks only.
│                                   #   THE SCORECARD IS THE TEST: the median (measured - track) offset over the
│                                   #   DETECTIONS that fall inside each track's OWN x range, printed per box and
│                                   #   row with its count, so the route the observations sit on is the one
│                                   #   closest to zero. G15_LIKE says which x axes are the SAME quantity on both
│                                   #   sides - Sigma_e and sSFR are, the AGE column is NOT (sim mass-weighted vs
│                                   #   obs CIGALE onset), so the age panel is marked on the figure and EXCLUDED
│                                   #   from the printed scorecard; the CSV keeps every panel with an
│                                   #   x_like_for_like flag. Excluding it moves the offsets by < 0.1 dex, so the
│                                   #   headline result never rested on it -> agn_states_routes_ismgrid_<box>.
│                                   #   {png,pdf} + agn_states_routes_ismgrid.csv (tracks + offsets).
│                                   #   RESULT (DUSTY_CUT -4.5, 11 ALMA-C11 QGs): the observed dusty QGs sit
│                                   #   +0.41 to +0.82 dex above the TIMING-EXPLAINED track, +0.86 to +1.08 above
│                                   #   the FIRING-DUSTY one and +1.98 to +2.02 above the DUST-POOR one in every
│                                   #   box - so the two dusty routes bracket the observations and the dust-poor
│                                   #   track is excluded, while the H2 row overlaps for all three (+0.04 to
│                                   #   +0.24). The closest track of all is cis50nox (+0.14): with the X-ray
│                                   #   channel switched off entirely, SIMBA reproduces the observed dust.
│                                   #   Read it knowing the routes are defined AT THE ANCHOR: a track is the locus
│                                   #   of galaxies that ENDED in that route, not a path through the plane.
│                                   #   (noxray_analogues Part 9 - dust-to-H2 against the exposure and the clock -
│                                   #   was NOT ported: Parts 7-9 here already carry that comparison.)
│                                   #   Part 16 = THE DGR UNDER TWO DIVISIONS OF THE SAME SAMPLE, AND THE
│                                   #   QUENCHING-TIME DISTRIBUTION BEHIND THEM. The other division: the
│                                   #   quenching SPEED r_q = log10(tau_q / tau(z)), tau_q = t_QT - t_SFT, cut at
│                                   #   P16_TSQ_CUT = -1.25. With the default tau(z) = the age of the universe at
│                                   #   QT (which IS t_QT, since t_SFT / t_QT are cosmic times - no cosmology
│                                   #   needed, the ks_tracks_lib tau_q_over_tH convention) this is EXACTLY
│                                   #   paper_xray_classes_m25's FAST_CUT baseline, so the fast/slow classes of
│                                   #   the two notebooks are the same division; P16_TAU switches tau(z) to the
│                                   #   anchor's t_H_gyr or to the Hubble time 1/H(z_QT) (0.08-0.17 dex above the
│                                   #   age over z = 0.3-2: it moves the cut, not the ordering). P16_DGR picks
│                                   #   the axis: "dh2" = log M_dust/M_H2, the ratio to the BARE molecular mass
│                                   #   (the notebook's H2-only convention), or "dg" = log M_dust/M_gas over the
│                                   #   total gas; the figure stem carries the choice. Per box, three panels:
│                                   #   (a) the DGR histogram split by the three routes, (b) the SAME histogram
│                                   #   split fast / slow, (c) THE DISTRIBUTION OF r_q ITSELF - the sanity check
│                                   #   on the division, since a fast/slow dichotomy is only worth drawing if
│                                   #   tau_q/tau(z) really is two populations. (c) tests that two ways: the
│                                   #   KDE's interior minima (assumption-free, the trough reported with its
│                                   #   depth relative to the lower flanking peak) and dBIC(2 vs 1) of a 1-D
│                                   #   Gaussian mixture fitted by EM in the cell (no sklearn); 'bimodal' is
│                                   #   printed only when BOTH agree (dBIC > P16_DBIC = 6 AND a real trough).
│                                   #   Drawn with it: the PARENT population (P16_PARENT - every galaxy of the
│                                   #   box in the Part 6a class table, none of the notebook's cuts, so a mode
│                                   #   killed by the selection is visible), the cut, and the snapshot-cadence
│                                   #   floor median log10(dt_snap / t_QT) below which a quenching faster than
│                                   #   one snapshot cannot be resolved. (c) counts ONE POINT PER QUENCHING
│                                   #   EVENT: P16_UNIQ (0.02 Gyr) merges anchors of the same box whose
│                                   #   (t_SFT, t_QT) agree to that tolerance - the branch is re-interpolated at
│                                   #   every anchor, so the same galaxy's times repeat only to ~1e-3 Gyr; (a)
│                                   #   and (b) keep every anchor, they are ISM measurements. Shared bins
│                                   #   everywhere; P16_MATCH keeps only galaxies with BOTH a finite DGR and an
│                                   #   SFT/QT pair, so the two divisions are scored on identical galaxies; the
│                                   #   1-99 percentile range is the DGR axis and the tails are clipped into the
│                                   #   end bins and counted, never dropped. Printed: per group N / median /
│                                   #   16-84 / dusty and fast fractions, the AUC + dmed + Mann-Whitney + KS of
│                                   #   every route pair and of fast vs slow (the best route pair, fast-vs-slow
│                                   #   and the (c) verdict are annotated on the panels), the route x speed
│                                   #   cross-tab both ways, rho(r_q, DGR), and the (c) tests for sample and
│                                   #   parent -> agn_states_dgr_<dgr>_vs_quenching_speed.{png,pdf} +
│                                   #   agn_states_dgr_vs_quenching_speed{,_contrasts}.csv +
│                                   #   agn_states_quenching_time_bimodality.csv.
│                                   #   READ IT AS: (a) is not a discovery - DUSTY_CUT is inside the route
│                                   #   definition - it sets the scale of a real separation on this axis; (b)
│                                   #   is the test, since the quenching speed knows nothing about the dust.
│                                   #   RESULT (products of 2026-09-02 BEFORE the QT_SMOOTH fix, Z_MAX 2 /
│                                   #   AGE_MIN 3, 960 anchors -> 711 / 64 / 37 quenching events): the routes
│                                   #   separate the DGR at AUC 0.95-0.99, the fast/slow cut at 0.44 (cis100),
│                                   #   0.66 (cis50), 0.50 (cis25); every route straddles the cut (14-50 %
│                                   #   fast) and rho(r_q, DGR) = +0.07 / -0.27 / +0.21.
│                                   #   AND (c) IS WHAT FOUND THE MEASUREMENT BUG: r_q came out UNIMODAL in
│                                   #   every box and in the parent (dBIC -14), against the bimodality SIMBA's
│                                   #   quenching literature reports with these very cuts. Traced (variant test
│                                   #   over the same histories, tauq_variants in the session scratchpad) to
│                                   #   find_quenching_times' smooth_window = 3 default: a ~0.2 Gyr boxcar at
│                                   #   the ~75 Myr cadence, i.e. the fast-quenching timescale itself, which
│                                   #   stretched tau_q for the fast quenchers into the slow mode (secondary:
│                                   #   classify_anchor dropped the SFR = 0 snapshots, which carry the quench).
│                                   #   With smooth_window = 1 and the zeros kept the SAME histories give
│                                   #   dBIC +693 / +70 / +12, modes -2.11 / -1.09 and a trough at -1.68
│                                   #   (cis50 -1.64, cis25 -1.54) - the literature's -1.5 dip - and the fast
│                                   #   events are real (75-100 % of the next 0.5 Gyr below 0.2/t; their
│                                   #   tau_q ~ 40 Myr is sub-snapshot, so the fast mode's POSITION is
│                                   #   interpolation, its EXISTENCE is not). Fixed 2026-09-02 in
│                                   #   quenching.py (default 1) + Part 6a (QT_SMOOTH, zeros kept):
│                                   #   RE-RUN paper_ism_prediction_boxes Part 6a to rebuild
│                                   #   ism_prediction_agn_classes.csv, then every part of this notebook that
│                                   #   uses t_SFT / t_QT (2b-4, 11-13, 16) moves with it.
│                                   #   Part 17 = ONE SNAPSHOT, NO ROUTE, NO QUENCHING CLOCK - the raw form of
│                                   #   the whole question, written after the routes were challenged as an
│                                   #   artefact of our own binning. Take ONE snapshot (P17_SNAP, None = the
│                                   #   richest anchor of P17_BOX), every galaxy exactly once, bin by stellar
│                                   #   age (P17_NAGE quantile bins, columns) and inside each age bin by the
│                                   #   OBSERVED log M_dust/M* (P17_NFD quantiles THERE, so the split is never
│                                   #   the age-dust correlation), and draw the AGN history of each group on
│                                   #   the one clock that needs no quenching times: LOOKBACK FROM THE
│                                   #   SNAPSHOT. Rows: X-ray occupancy, jet criterion, the fraction with the
│                                   #   gate f_gas < XRAY_FGAS_MAX open, log M_gas/M* (the gate variable
│                                   #   itself - the second structural exception to the H2-only rule), log
│                                   #   Mdot_BH, log f_Edd, and E_x accumulated between that lookback and the
│                                   #   snapshot; bootstrap 16-84 bands, epochs under P17_NMIN blank.
│                                   #   P17_CUTS 'no_age' = the quenched rule + MASS_MIN only (AGE_MIN would
│                                   #   truncate the axis being binned); 'part0' = QM verbatim.
│                                   #   THE UNBINNED CONTROL is printed under it: Spearman rho between the
│                                   #   CONTINUOUS log M_dust/M* and E_x / f_x / t_quiet / X-ray-now / gate /
│                                   #   log M_BH / log M* / log f_gas / log f_H2, INSIDE each age bin - the
│                                   #   same question with no dust bins at all, which no bin edge can
│                                   #   manufacture - plus the per-cell table (N, medians, % in the X-ray
│                                   #   state AT the snapshot, % gate open, % never fired, E_x, t_quiet) ->
│                                   #   agn_states_snapshot_agnhist_<box>_snap<NNN>{.png,.pdf} +
│                                   #   _{tracks,cells,rho,galaxies}.csv.
│                                   #   IT ALSO EXPOSED A SAMPLE TRAP, now printed by Part 0 as well: the
│                                   #   histories caches were built by the boxes notebook Part 6a for ITS
│                                   #   quenched pool (PASSIVE_FACTOR 0.2, DUST_TO_H2_MIN 1e-4), so galaxies
│                                   #   admitted by looser constants here have NO main branch and drop out of
│                                   #   every history-based part - with DUST_TO_H2_MIN = 1e-15 in Part 0 that
│                                   #   is 31 % / 22 % / 67 % of the declared sample (cis100 / cis50 / cis25),
│                                   #   and the loss is along the DUST axis itself. Part 17 restricts to the
│                                   #   galaxies with a chain BEFORE binning and prints the toll; Part 0 prints
│                                   #   the coverage per box and warns below 90 %. Fix = match the boxes
│                                   #   notebook's constants, or rebuild its histories (OVERWRITE_HIST) with
│                                   #   these.
│                                   #   RESULT (cis100 snapshot 116, z = 0.70, 305 of 810 with chains, three
│                                   #   age x three dust bins): the dust-AGN link is ENTIRELY in the youngest
│                                   #   age bin - rho(lfd, E_x) = -0.70 at age 1.0-2.4 Gyr but -0.14 at
│                                   #   2.4-3.3 and -0.06 (f_x -0.00) at 3.3-5.1, where the X-ray occupancy
│                                   #   tracks of the three dust bins lie on top of each other. In that young
│                                   #   bin the dusty third is simply the one that has not yet crossed the gate
│                                   #   (rho(lfd, f_gas) = +0.60, its gate opens ~1.5 Gyr later), and it is
│                                   #   degenerate with mass (rho(lfd, log M*) = -0.54). AND EVERY GROUP IS
│                                   #   BEING DOSED NOW: 47-94 % of each (age, dust) cell is IN the X-ray state
│                                   #   at the snapshot, the dustiest thirds included (47 / 71 / 56 %). So at
│                                   #   fixed age above ~2.4 Gyr the AGN history does not distinguish dusty
│                                   #   from dust-poor: the Part 2a routes are mostly age + gas fraction, and
│                                   #   the residual 'dosed and still firing' group is not a special class.
│                                   #   Part 17 UPDATE (2026-09-02, after the routes were challenged): P17_SPLIT
│                                   #   chooses WHICH observed dust variable the colour bins are cut on - "lfd"
│                                   #   log M_dust/M* (what an observer reports, but largely a statement about
│                                   #   how much gas is left), "dh2" log M_dust/M_H2, "dtm" log dust-to-metal
│                                   #   (the grain-survival version; its Z_gas comes from the branch, the
│                                   #   catalogue carries none) - and the rho block scores ALL THREE whatever
│                                   #   the colours show. P17_BOXES is a LIST (default cis100 + cis50nox), one
│                                   #   figure per box, so the CHANNEL-OFF run is drawn beside the fiducial one:
│                                   #   in cis50nox the X-ray mode never fired, so a dust spread at fixed age
│                                   #   there cannot be AGN-made (its AGN-state rows are WOULD-BE states, said
│                                   #   on the figure). Part 17 no longer needs the box in BOXES_OK: any box
│                                   #   with catalogue rows and a histories cache runs ->
│                                   #   agn_states_snapshot_agnhist_<box>_snap<NNN>_<split>{.png,.pdf} +
│                                   #   agn_states_snapshot_agnhist_<split>_{tracks,cells,rho,galaxies}.csv.
│                                   #   RESULT with the survival variables (cis100 snap116, 305 galaxies,
│                                   #   histories rebuilt with QT_SMOOTH = 1): in the OLD age bin (3.3-5.1 Gyr)
│                                   #   dtm and dh2 do NOT follow the accumulated dose (rho(dtm, E_x) = -0.06,
│                                   #   f_x +0.01, both n.s.) but they DO follow the CLOCK SINCE THE LAST
│                                   #   EPISODE: rho(dtm, t_quiet) = +0.35 and rho(dtm, X-ray now) = -0.35,
│                                   #   with rho(dtm, log M_BH) = +0.38 - the dust comes back after the channel
│                                   #   stops, it is not a matter of how much it fired. In the middle bin the
│                                   #   same signs, weaker (+0.26 / -0.36); in the youngest bin every dust
│                                   #   variable is the GATE (rho(lfd, f_gas) = +0.60, E_x -0.70) and is
│                                   #   degenerate with mass (-0.54). cis50nox (51 galaxies, all 100 % would-be
│                                   #   X-ray on): lfd tracks f_gas at rho = +0.87 to +0.91 while dtm and dh2
│                                   #   correlate with NOTHING - without the channel, dustiness IS gas content.
│                                   #   Part 18 = MATCHED PAIRS, the controlled version of the same question.
│                                   #   Each DUSTY galaxy (top P18_QTOP of P18_SPLIT, default dtm) is paired
│                                   #   with the DUST-POOR one (bottom P18_QBOT) nearest in age, log M*,
│                                   #   log M_gas/M* and log Z_gas - the four things that trivially set a dust
│                                   #   mass - under a total distance cap (P18_MAXD) AND a per-axis caliper
│                                   #   (P18_CALIPER, what keeps Z honest since dust-to-metal has Z in its
│                                   #   denominator); P18_POOL 'auto' pairs inside one anchor where the anchors
│                                   #   are rich enough and across anchors (with z added to the axes) otherwise.
│                                   #   Three competing explanations are then scored on the pairs: (A)
│                                   #   DESTRUCTION - T_gaspoor, E_x, f_x, t_quiet, t since the gate shut,
│                                   #   f_gas now; (B) RESUPPLY - dlog M_gas / M_H2 / M* over 0.5-1 Gyr and the
│                                   #   gas peak minus now; (C) NOISE - N_gas, N_star and the GAS PARTICLE MASS
│                                   #   (a pure resolution variable with no dust in it), with P18_BOX = 'cis25'
│                                   #   as the 8x-resolution arm; plus (D) RECENCY, printed apart and flagged
│                                   #   SEMI-CIRCULAR (dlog M_dust over the window is another view of the dust
│                                   #   level itself) - it dates the difference rather than testing it. Output:
│                                   #   a forest plot of AUC per property coloured by block with the median
│                                   #   paired difference beside it, and a BALANCE panel (standardised mean
│                                   #   difference of every matching axis before / after pairing) ->
│                                   #   agn_states_matched_pairs_<box>_<split>{.png,.pdf} + _scorecard.csv +
│                                   #   _galaxies.csv (both members, tagged role).
│                                   #   RESULT (cis100, 155 pairs, split on dtm, gap +1.46 dex, match |SMD| <=
│                                   #   0.23): (A) the dusty member was MODESTLY LESS dosed - E_x -0.12 Gyr
│                                   #   (AUC 0.40, p 1.5e-4), f_x -0.02 (0.40) - while t_quiet shows nothing
│                                   #   once gas and metals are held fixed; (B) it grew MORE STELLAR MASS in the
│                                   #   last Gyr, +0.041 dex (AUC 0.65, p 9e-11), i.e. residual SF / accretion,
│                                   #   with the gas changes small and mixed; (C) THE NOISE HYPOTHESIS IS DEAD
│                                   #   IN THIS DIRECTION - the dusty members have MORE gas particles (AUC 0.60)
│                                   #   at the same particle mass, so they are better sampled, not worse; and
│                                   #   (D) the difference is RECENT: +1.5 dex of dust growth in 1 Gyr (AUC
│                                   #   0.93) against a matched twin that lost it. So a dusty QG here is one
│                                   #   that RE-GREW its dust recently, alongside a little late stellar growth,
│                                   #   not one that dodged the AGN. The small boxes cannot be matched at these
│                                   #   settings (cis25: 2 pairs; cis50nox: 19, and its dusty / dust-poor gap is
│                                   #   only 0.08 dex because every nox QG is dusty) - the cell says so and asks
│                                   #   for looser calipers rather than pretending.
│                                   #   Part 19 = THERMAL SPUTTERING, MEASURED PARTICLE BY PARTICLE (m25 only,
│                                   #   the box with the particle products AND the resolution). Parts 17-18 can
│                                   #   only ask WHEN the AGN fired; SIMBA destroys dust with tau_sp =
│                                   #   0.17 Gyr (a/0.1um)(1e-27/rho)[(2e6 K/T)^2.5 + 1] (Tsai & Mathews 95, the
│                                   #   form in Li, Narayanan & Dave 19), grain by grain. The cis25 reduced
│                                   #   particle files carry m_dust and temp per gas particle, and since
│                                   #   2026-09-02 `nh` as well (a BACKFILL field added to
│                                   #   build_reduced_particles_job.py: re-submitting a plan appends only that
│                                   #   dataset, no geometry redo), so tau_sp is evaluated PER PARTICLE and
│                                   #   collapsed onto tau_eff = M_dust / sum_i (m_dust,i / tau_sp,i) - the
│                                   #   e-folding time of the whole reservoir under the model's own destruction
│                                   #   term, not a proxy. Per galaxy: tau_eff, dust-weighted median tau_sp,
│                                   #   dust-weighted log T and log nh, the dust fractions with tau_sp <
│                                   #   P19_TFAST, above 1e5 and 1e6 K, and in star-forming gas (where it grows),
│                                   #   plus M_dust(particles)/M_dust(caesar) as a printed cross-check (1.0 with
│                                   #   P19_SEL='member', the like-for-like selection; 'aperture' or a radius in
│                                   #   kpc widen it to the CGM). Three panels: (1) the dust variable against
│                                   #   tau_eff coloured by age, (2) WHERE the dust sits on the (T, nh) plane the
│                                   #   law is written on, with the tau_sp contours and the points coloured by
│                                   #   dust-to-metal, (3) E_x against tau_eff - does the dose predict the
│                                   #   environment at all. The tables give the same as Spearman rho: dust vs
│                                   #   sputtering inside each age bin, then AGN vs sputtering. Self-contained:
│                                   #   the dust-to-metal and the AGN summaries are rebuilt from the cis25
│                                   #   histories in the cell, so it does not need Part 17 to have run ->
│                                   #   agn_states_sputtering_cis25{.png,.pdf} + _galaxies.csv + _rho.csv.
│                                   #   THE POINT: if the dust follows tau_eff while tau_eff does NOT follow E_x,
│                                   #   the AGN decides when the gas turns hot and thin but the ISM phase
│                                   #   structure decides what survives. _galaxies.csv is what turns Part 18's
│                                   #   block A from proxies into the measured destruction rate (add tau_eff /
│                                   #   f_dust_fast to P18_BLOCKS and merge the table into D18).
│                                   #   IF NO FILE CARRIES `nh` the cell prints the backfill command instead of
│                                   #   running: sbatch --array=0-15 submit_reduced_particles.sh
│                                   #   output/cis25/caesar_sfh/prof_kstracks/dust_profile_plan_kstracks.hdf5
│                                   #   (and the prof_boxcomp plan for snapshots 105 / 134).
│                                   #   Part 20 = THE DUST BUDGET ALONG THE TRACK (m25): Part 19's measurement
│                                   #   walked over the KS-track stages (sf_peak / agn_ign / jet_on / sft / qt /
│                                   #   post_quench / gas_min / anchor, from ks_track_epochs.fits) - because Part
│                                   #   19 sees only the SURVIVORS' environment, and if sputtering ever mattered
│                                   #   it mattered earlier - PLUS the other half of SIMBA's budget, the GROWTH
│                                   #   term tau_acc = TAU_ACC0 (a/0.1um)(n_ref/n)(Z_sun/Z)(T_ref/T)^0.5, so the
│                                   #   question at each epoch is which clock is shorter (f_dust_growing = the
│                                   #   dust fraction with tau_acc < tau_sp). tau_acc needs the per-particle
│                                   #   metallicity: `zgas`, added to build_reduced_particles_job.py beside `nh`
│                                   #   (same additive backfill); without it the cell measures the sputtering
│                                   #   half and prints the sbatch line. The phase split carries over -
│                                   #   tau_sp_nsf (non-star-forming dust) is the row whose clock is physical,
│                                   #   and a Hubble-time line is drawn on it. Tracks split by the dust the
│                                   #   galaxy ENDS with (P20_SPLIT terciles at the anchor) ->
│                                   #   agn_states_dustbudget_track_cis25{.png,.pdf} + .csv.
│                                   #   Part 21 = THE CHAIN: AGN HISTORY -> GAS STRUCTURE -> DUST, the
│                                   #   hypothesis that survives Parts 17-19 (the AGN does not destroy grains, it
│                                   #   decides whether the galaxy can hold dense rotating molecular gas, and
│                                   #   dust grows where that gas is). PURE READS of the m25 KS-track products:
│                                   #   ks_track_measurements.fits gives Sigma_H2 / Sigma_dust in the CORE
│                                   #   (ap3kpc) and in the DISC ANNULUS (ap10 minus ap3, areas included) plus
│                                   #   R50(H2) and R50(star); ks_stage_kinematics.fits gives kappa_rot of H2 /
│                                   #   gas / stars and the H2-star alignment in the same zones;
│                                   #   ks_track_epochs.fits the stage map and timings - joined to this
│                                   #   notebook's AGN histories (E_x, E_x since QT, f_x, t_quiet, X-ray now).
│                                   #   Four blocks: (1) AGN vs structure, (2) structure vs dust, (3) MEDIATION -
│                                   #   rho(AGN, dust) against the partial with the best single mediator AND with
│                                   #   the whole structural block held fixed (p21_partial_multi, rank-
│                                   #   residualised), (4) CAUSE BEFORE EFFECT - E_x accumulated UP TO a stage
│                                   #   against the structure AT it, and that structure against the final dust.
│                                   #   P21_CIRCULAR bars the dust-made structural variables (Sigma_dust, D/H2
│                                   #   per zone) from mediating: they are the outcome seen through a smaller
│                                   #   aperture -> agn_states_chain_cis25{.png,.pdf} + _galaxies.csv + _rho.csv.
│                                   #   RESULT (266 quenched anchors, 134 galaxies): (1) the AGN DOES set the
│                                   #   structure - X-ray ON NOW goes with a LARGER, more diffuse H2 body
│                                   #   (R50(H2) +0.43, R50_H2/R50_star +0.40, kappa_H2 core -0.27, Sigma_dust
│                                   #   core -0.29) while t_quiet goes the other way (R50(H2) -0.34,
│                                   #   R50_H2/star -0.37, kappa_H2 core +0.25, Sigma_dust core +0.25);
│                                   #   (2) the dust follows a COMPACT, ROTATING, DENSE H2 body (kappa_H2 core
│                                   #   +0.47/+0.33, Sigma_H2 disc +0.37/+0.26, R50_H2/star -0.37, H2
│                                   #   concentration -0.23; dtm / dh2 respectively); (3) THE MEDIATION SPLITS IN
│                                   #   TWO: the INSTANTANEOUS state is mediated by the structure (X-ray now
│                                   #   -0.43 -> -0.20 with dtm, -0.40 -> -0.15 with dh2; t_quiet +0.37 -> +0.20)
│                                   #   while the ACCUMULATED DOSE survives and even strengthens (E_x -0.27 ->
│                                   #   -0.50 dtm, -0.31 -> -0.49 dh2) - a memory term the current gas
│                                   #   configuration does not carry, and NOT a metallicity artefact
│                                   #   (rho(E_x, Z_gas) = -0.05 n.s., and dh2 has no Z in it);
│                                   #   (4) cause before effect: E_x up to a stage predicts R50(H2) AT that stage
│                                   #   (+0.37 / +0.40 / +0.39 at qt / post_quench / gas_min) and Sigma_H2(disc)
│                                   #   at every stage predicts the dust the galaxy ends with (+0.31 to +0.38).
│                                   #   Part 22 = THE PICTURE: Part 21 drawn instead of scored - same inputs, no
│                                   #   new measurement, three figures and a per-galaxy csv per box. The dust
│                                   #   axis is P22_DUST, default log M_dust/M_H2.
│                                   #   ONE TRACK PER GALAXY: an anchor row is not a galaxy (a galaxy quenched
│                                   #   at several anchors contributes overlapping chains, the same early
│                                   #   history up to 5x), so the chains are de-duplicated on the main branch
│                                   #   itself - a fingerprint of M*(t) at three fixed times - and only the
│                                   #   LATEST anchor of each is kept: 262 rows -> 175 galaxies.
│                                   #   agn_states_dustclock_boxes: ONE COLUMN PER BOX (P22_BOXES = cis100 /
│                                   #   cis50 / cis25, so the three resolutions compare side by side), two rows:
│                                   #   (a) THE FAN, the main panel - the dust on each galaxy's own t - t_QT
│                                   #   clock, split four ways by AGN dose x gas geometry at each box's own
│                                   #   medians: one curve before quenching, four after; (b) THE DOSE, the dust
│                                   #   against f_x_post, one line per THIRD of the geometry with P22_DBINS
│                                   #   equal-count bins INSIDE each third - three falling lines that never
│                                   #   cross. THE GEOMETRY AXIS DIFFERS BY COLUMN and the titles say so: m25 is
│                                   #   split on the KS-track H2 EXTENT RELATIVE TO THE STARS, -log R50(H2)/R50(*)
│                                   #   (P22_GEOM = compactness, the default since 2026-09-04 - see WHY THE
│                                   #   GEOMETRY AXIS IS NO LONGER kappa_rot below); the
│                                   #   coarse boxes have no KS-track products and are split on the catalogue's
│                                   #   WHOLE-GALAXY kappa_rot of the gas (P22_GEOM_CAT = kappa_gas, caesar
│                                   #   rotation.gas_kappa_rot) over Part 0's QM sample, t_SFT / t_QT from the
│                                   #   boxes notebook's classes, the same de-duplication and dose as m25.
│                                   #   P22_LIKE = True cuts the m25 column the same way (strictly like for
│                                   #   like, at the price of the KS geometry Figs 2-3 / Part 23 rest on). One
│                                   #   agn_states_dustclock_<box>.csv per box. First real run (2026-09-03):
│                                   #   cis100 1625 / cis50 197 / cis25 175 galaxies; the fan is the same in all
│                                   #   three - one curve before QT, the low-dose rotating group alone stays
│                                   #   dusty after it (cis100 -4.1 vs -4.7 at +2 Gyr; cis50 -4.0 vs -4.8) - and
│                                   #   (b) falls with the dose in every box, the top geometry third on top.
│                                   #   agn_states_dustsplit_cis25 = WHEN the halves part company: (a) Sigma_H2
│                                   #   core and (b) kappa_rot(H2) core at each critical point (P22_STAGES =
│                                   #   agn_ign / jet_on / sft / qt / anchor; sf_peak dropped) as BINNED POINTS, not
│                                   #   curves - the stages are epochs of a sample, each galaxy reaching them at
│                                   #   its own time; (a) the two halves are INDISTINGUISHABLE the whole way
│                                   #   (the dose does not decide how much molecular gas is left), (b) identical
│                                   #   until QT then the low-dose half RE-SETTLES while the high-dose half
│                                   #   stays flat at 0.16; (c) THE DUST-GAS PLANE WALKED IN BINS OF t - t_QT
│                                   #   (P22_TQT): behind everything THE WHOLE SCATTER - every critical point
│                                   #   and the anchor of every galaxy, red / blue by half (P22_SCAT_STAGES,
│                                   #   None = every stage row; 730 + 818 points) - then both halves on the SAME
│                                   #   bins, each galaxy once per bin, a grey constant-dust-to-gas guide, and
│                                   #   at the bottom right of each HIGH-DOSE point a small box with the time
│                                   #   since quenching (bin centre) and N galaxies low / high dose (boxes that
│                                   #   would collide are stacked underneath each other): they fall together
│                                   #   until quenching then split VERTICALLY - same Sigma_H2, log Sigma_dust
│                                   #   4.9 vs 3.2 by +3 Gyr. P22_PLANE = "ks" for the KS version.
│                                   #   agn_states_dustgas_ssfr_cis25 = THE REJUVENATION TEST: the SAME WALKED
│                                   #   PLANE AS (c), but stepped through the CRITICAL POINTS jet_on -> sft ->
│                                   #   qt -> anchor (P22_SSTAGE, arrows for the direction of time) instead of
│                                   #   bins of time, ONE PANEL PER BIN OF THE GALAXY'S sSFR AT THE ANCHOR
│                                   #   (P22_SEDGES, terciles of the GALAXIES): the panel is a property of the
│                                   #   GALAXY, not of the epoch, so a whole track stays inside one panel and
│                                   #   each panel follows a FIXED SUBSAMPLE through its own history (a genuine
│                                   #   evolutionary track, not a different set of galaxies at every stage). A
│                                   #   galaxy that formed NO STAR in the last 100 Myr at the anchor has no log
│                                   #   sSFR: it is not missing data but the deadest galaxy there is, so it goes
│                                   #   in the BOTTOM panel (printed + in panel a's title) and the edges are
│                                   #   quantiles of the galaxies that DO have one, so they stay finite.
│                                   #   Each point = the median galaxy at that epoch with its bootstrap error,
│                                   #   drawn above P22_NMIN3 (Fig 3's own floor: a panel x half x epoch cell is
│                                   #   thinner than a Fig 2 epoch); the
│                                   #   grey dashed constant-dust-to-gas line is THE SAME LINE IN EVERY PANEL,
│                                   #   so the panels compare and the vertical distance from it IS
│                                   #   Sigma_dust/Sigma_H2. The cell prints that ratio + N at every drawn
│                                   #   point. It now shows WHERE ALONG THE SEQUENCE the two halves part
│                                   #   company, not only that they end apart. (The previous one-star-per-half
│                                   #   version gave a 1.07 dex gap in the lowest-sSFR panel and none in the
│                                   #   other two; re-run for the per-critical-point numbers.)
│                                   #   TWO THINGS IT SETTLES, both printed: (i) "time since the last X-ray
│                                   #   episode" is NOT an axis here - 83 % are IN the X-ray state at the anchor
│                                   #   and only ~13 % of post-quenching snapshots are out of it, ~72 % of those
│                                   #   in the top third of kappa_H2 (getting out needs re-accretion over the
│                                   #   f_gas gate, which mostly the rotating galaxies manage); the axis that
│                                   #   works is f_x_post, HOW MUCH of the post-SFT era was spent in it (rho with
│                                   #   the dust -0.51, rho with z = 0.00 against -0.39 for E_x [Gyr]);
│                                   #   (ii) the X-ray gate is not doing the work - 99 % of the sample is
│                                   #   already below XRAY_FGAS_MAX (median f_gas 0.03), the dose is flat in it
│                                   #   (rho -0.04), and the picture survives on the DUST-TO-GAS ratio
│                                   #   (P22_DUST = "dg", rho -0.48) which has the gas divided out.
│                                   #   With the kappa axis dose and geometry were near-independent (rho -0.29;
│                                   #   the extent axis: +0.09) and each survived the other: rho(dose, dust |
│                                   #   kappa) -0.44, rho(kappa, dust | dose) +0.37.
│                                   #   Knobs: P22_BOXES, P22_GEOM_CAT / P22_LIKE (Fig 1's coarse-box geometry
│                                   #   axis, and whether m25 is cut on it too), P22_GEOM (compactness = -log
│                                   #   R50(H2)/R50(*) default | sigma_R50 = log Sigma_H2 inside R50(H2), He
│                                   #   included, the K-S abscissa | kappa_H2 / kappa_gas, whose group names then
│                                   #   carry the cut itself instead of 'rotating' | sigma_H2 of the core, which
│                                   #   only tracks the particle count, rho +0.89), P22_GCUT ("median" | a
│                                   #   number: 1/3 = isotropic kappa, 0.5 = rotation-supported), P22_NH2MIN /
│                                   #   P22_NH2CHK (the resolution floor of the split and the printed check),
│                                   #   P22_DOSE, P22_DUST, P22_CORE,
│                                   #   P22_STAGES, P22_TQT, P22_SCAT_STAGES, P22_SSTAGE, P22_SEDGES, P22_NMIN3,
│                                   #   P22_DBINS, P22_PLANE, P22_NMIN.
│                                   #   WHY THE GEOMETRY AXIS IS NO LONGER kappa_rot (2026-09-04): the 'rotating'
│                                   #   split was kappa_rot(H2) of the core above its MEDIAN, and that median is
│                                   #   0.21 - BELOW the isotropic value 1/3 (one of three velocity components is
│                                   #   azimuthal; a disc sits above ~0.5): 31 % above 1/3, 21 % above 0.5, so
│                                   #   nobody rotated and the label was the upper half of a non-rotating,
│                                   #   radially moving distribution. Its dust signal lived in the resolution
│                                   #   floor: median 128 member H2 particles, 43 % below 100, and below 100
│                                   #   kappa scatters 0.10-0.67 with the compact few-particle knots scoring
│                                   #   highest; at >= 100 particles rho(kappa, dust) = +0.14 and the low/high-
│                                   #   dose contrast flips sign (+0.64 / -0.52 dex). The size ratio keeps its
│                                   #   sign there (+0.82 / +0.17 dex with the global cut), its strength at fixed
│                                   #   M_H2 (rho -0.39 for the extent), is independent of the dose (rho +0.09)
│                                   #   and is what the data measure (R_CO/R_e: the ALMA-C11 QGs span -0.4..+0.1
│                                   #   in the log, the simulated bulk -0.43..+0.04). Sigma_H2 inside R50 is the
│                                   #   strongest single dust predictor (rho +0.50, +0.49 at fixed M_H2) but IS
│                                   #   the K-S abscissa, so it is an option, not the default. In M_H2 x extent
│                                   #   the quenched galaxies are dusty in TWO places: the ultra-compact H2-poor
│                                   #   remnants (extent < -0.6, M_H2 < 2e8: dh2 ~ -1.4) and the H2-rich extended
│                                   #   discs (M_H2 > 1e9, R50 ~ 2.7 kpc, kappa ~ 0.45: ~ -1.7); the bulk between
│                                   #   is dust-poor at any kappa. Smoke-tested locally 2026-09-04 (the Part 22 +
│                                   #   23 cell sources on the cluster tables): the two dose halves end +1.08 dex
│                                   #   apart in dh2; low-dose compact minus extended +1.36 dex, high dose -0.02.
│                                   #   Part 23 = THE SAME DIVISION IN THE RESOLVED PRODUCTS (m25). NOT a pure
│                                   #   read: it needs the m25 notebook's tables and the ks_tracks_quenched_m25
│                                   #   run (each block gates on its file and prints what is missing). The four
│                                   #   dose x geometry groups are re-derived for EVERY quenched anchor row
│                                   #   (P23_ALLANCH; 262 rows) with Part 22's cuts, not re-medianed.
│                                   #   agn_states_dosegeom_profiles_cis25 = the RADIAL ANATOMY per annulus
│                                   #   (0-1 / 1-3.2 / 3.2-10 / 10-32 kpc, per-galaxy sightline medians, the m25
│                                   #   SF galaxies in grey): A_V, M_dust/M*, M_H2/M*, RESOLVED sSFR, M_dust/M_H2,
│                                   #   stellar age, kappa_rot(H2) and kappa_rot(*). The sSFR panel = P23_SFRCOL
│                                   #   (sfr100, the KS convention; sfr25 the other) over the SAME region's M*, so
│                                   #   the 'is it just still forming stars?' objection is answered radius by
│                                   #   radius; a dead annulus sits ON the P23_SSFRMIN floor (dotted line, 'no SF')
│                                   #   instead of vanishing off the log axis - Part 0's SSFR_FLOOR = 1e-15 would
│                                   #   stretch the panel over five decades to show one dead annulus.
│                                   #   RESULT: A_V, M_dust/M* and M_dust/M_H2
│                                   #   separate the low-dose ROTATING group from the other three by an order of
│                                   #   magnitude IN THE INNER KPC and converge outwards (a CORE phenomenon);
│                                   #   M_H2/M* separates them at NO radius; kappa_rot(*) is flat and identical
│                                   #   for all four - the stellar body is not what differs, only the gas.
│                                   #   The kappa panels carry the isotropic 1/3 (dashed) and 0.5 (dotted) as
│                                   #   reference lines - every quenched group sits BELOW 1/3 in kappa_rot(H2);
│                                   #   with P22_GEOM = kappa_H2 the core panel is partly circular (the split
│                                   #   variable) and the cell says so, with the extent axis no annulus panel is.
│                                   #   agn_states_dosegeom_ks_cis25 = the K-S PLANE WITH THE ALMA-C11 RESOLUTION
│                                   #   APPLIED (2026-09-04): the INTRINSIC tracks are the solid ones (P23_KS_MAIN
│                                   #   = intrinsic, R = R50(H2), the physics); where ALMA-C11 WOULD MEASURE each
│                                   #   group is the faded dashed overlay (P23_KS_OVERLAY), a slope-1 arrow on a
│                                   #   stage point when > 50 % of its galaxies come out unresolved. The rule is
│                                   #   TWO-TIER (corrected the same day - the first version used R = max(R50,
│                                   #   3.8 kpc), i.e. every galaxy as if its uv fit had failed, which collapsed
│                                   #   all four groups to log Sigma_H2 0.67-0.80): a uv-plane Gaussian fit
│                                   #   measures BELOW the beam, the smallest size the ALMA-C11 fits recovered is
│                                   #   1.6 kpc (hc9) against b_maj/2 = 3.3-4.5 kpc (beams of 1.3-1.6 arcsec at
│                                   #   z = 0.35-0.42), so a galaxy with R50(H2) >= P23_UV_MIN_KPC (1.6) keeps
│                                   #   its own size and one below it gets the P23_BEAM_KPC limit (3.8, the
│                                   #   median b_maj/2 of the three unresolved sources); P23_UV_MIN_KPC = None
│                                   #   gives the one-tier rule back. Under it the groups keep their order and
│                                   #   THE COMPACT GROUPS ARE THE ONES THAT COME OUT UNRESOLVED (at end, log
│                                   #   Sigma_H2 intrinsic -> observed, unresolved fraction: low/compact 1.84 ->
│                                   #   0.77, 66 %; low/extended 1.35 -> 1.16, 20 %; high/compact 1.57 -> 1.09,
│                                   #   43 %; high/extended 1.26 -> 1.23, 9 %) - a prediction for the sample.
│                                   #   The anchors run to z = 2 where 1 arcsec is 8 kpc, so a fixed ANGULAR
│                                   #   beam would penalise the high-z anchors more, not less; the uv-fit tier
│                                   #   is the correction that matters. The GALAXIES LARGER THAN THE BEAM
│                                   #   (P23_TAIL: R50(H2) > 3.8 kpc at end - 9 of 266 anchor rows, R50 4.0-9.4
│                                   #   kpc, six of them extended with kappa 0.63-0.80 = the rotating ones, the
│                                   #   counterparts of the observed 8-9 kpc rotating CO discs hc3 / hc5 / hc7)
│                                   #   are drawn galaxy by galaxy and listed. The H2 convention cannot reorder
│                                   #   the observed sources: alpha_CO / r31 shift every observed point sideways
│                                   #   together (4.36 -> 0.8: -0.74 dex; r31 0.5 -> 1: -0.30), a size upper
│                                   #   limit moves along slope 1, parallel to B08. ONE PANEL (P23_KS_XLIM 0-4,
│                                   #   P23_KS_YLIM -3.2-2, legend upper left): the binned per-group stage
│                                   #   medians through the SAME CRITICAL POINTS AS PART 22's Fig 3 (jet_on ->
│                                   #   sft -> qt -> end, arrows for the direction of time, no individual
│                                   #   tracks; `end` = the anchor as the K-S products define it, the last
│                                   #   snapshot still holding H2 - put sf_peak / agn_ign back in P23_KS_STG
│                                   #   for the burst itself)
│                                   #   against ONE relation, the molecular Bigiel+08 - the total-gas K98 is
│                                   #   deliberately not drawn - with the ALMA-C11 quiescent galaxies
│                                   #   (obs_data/almac11/ks_table.csv, filled = size measured, open + arrow =
│                                   #   unresolved / Sigma lower limits); the simulated tracks END where the
│                                   #   observed points live. The OFFSET from that relation per stage = the
│                                   #   post-starburst test, PRINTED + in the CSV (the old panel b is gone):
│                                   #   all four groups sit ~1 dex ABOVE B08 through the
│                                   #   pre-quenching epochs jet_on / sft (it held at agn_ign and sf_peak too,
│                                   #   when those were drawn - the phase is burst-like for everyone) and the
│                                   #   lower-kappa ones (kappa split, 2026-09-03 run) are only marginally higher (+1.02/+1.05 vs
│                                   #   +0.86/+1.01 at sft); what separates them is SPEED - tau_q 0.21-0.22 Gyr
│                                   #   against 0.28-0.31 - and by `end` the low-dose groups are back ON the
│                                   #   relation while the high-dose ones have fallen BELOW it. The printed
│                                   #   burst amplitude (peak sSFR in the Gyr before SFT over the 3 Gyr before
│                                   #   that) is +0.12..+0.33 dex with no ordering: post-starburst-like in the
│                                   #   speed of the shutdown, not in the size of the burst.
│                                   #   Extent split (local smoke test 2026-09-04): offsets at end +0.17 / +0.07 /
│                                   #   -0.09 / -0.21 (low compact / low extended / high compact / high extended),
│                                   #   tau_q 0.23-0.27 Gyr with no ordering.
│                                   #   Knobs: P23_ALLANCH, P23_ANNULI, P23_KS_AP, P23_KS_STG, P23_KS_SFR,
│                                   #   P23_BEAM_KPC / P23_UV_MIN_KPC (the two-tier resolution rule), P23_KS_MAIN /
│                                   #   P23_KS_OVERLAY (which track is solid, whether the other is drawn),
│                                   #   P23_TAIL (the galaxies larger than the beam),
│                                   #   P23_KS_XLIM / P23_KS_YLIM, P23_REL, P23_SFCLOUD, P23_NMIN, P23_M25,
│                                   #   P23_OBS.
│                                   #   Part 24 = THE PICTURES (m25): one representative galaxy PER CLASS at each
│                                   #   critical point (jet_on -> sft -> qt -> anchor), rendered from the REDUCED
│                                   #   PARTICLE CUTOUTS via simbanator.visualization.rendering.ReducedRender - the
│                                   #   same particles Parts 13/23 measure, no snapshot and no caesar load. The
│                                   #   THE FOUR ARE SEEDED THEN MATCHED, not chosen class by class: (1) the SEED is
│                                   #   P24_SEEDCLS (hi_lo = high dose + the lower side of P22_GEOM, the class that has to
│                                   #   carry the low-dust end) restricted to log M_dust/M_H2 < P24_SEEDMAX (-3),
│                                   #   and among those the BEST-RESOLVED galaxy - the only place resolution
│                                   #   decides; (2) the other three are whichever member of their class is CLOSEST
│                                   #   TO THE SEED in stellar age and log M*, with P24_RESW = 0 so matching alone
│                                   #   decides. The seed's anchor is a PREFERENCE, NOT A CAGE: one anchor holds a
│                                   #   handful of drawable galaxies per class and insisting on it gave quadruples
│                                   #   off by 2.9 Gyr and 1.0 dex, so a class it cannot match to within P24_COSTMAX
│                                   #   is taken from whichever anchor matches best and the cell names it (every
│                                   #   panel carries its own z anyway). The SEED is searched too: if the other
│                                   #   three cannot be matched to the best-resolved low-dust galaxy, the next
│                                   #   P24_SEEDTRY in resolution order are tried and the least-bad kept.
│                                   #   The cost is in
│                                   #   units of P24_TOLAGE / P24_TOLM (1 = off by one tolerance) and is printed per
│                                   #   galaxy, with a warning above 1. P24_DGR picks the ratio meant everywhere:
│                                   #   "dh2" = M_dust/M_H2 (what P22_DUST means) | "dg" = M_dust/M_gas. Pool =
│                                   #   every anchor row (Part 23's A23) when P24_ALLANCH and Part 23 has run -
│                                   #   ANC22 keeps only each galaxy's LATEST anchor, so a z > P24_ZMIN cut on it
│                                   #   throws away most of the sample. P24_PICK names one by hand, seed included.
│                                   #   AGE + log M* come from Part 0's CAT (the FULL catalogue cache), NOT from QM:
│                                   #   QM is CAT after the quenched rule and the mass / age / history cuts, which
│                                   #   are not the KS products' cuts, so a drawable anchor row can be missing from
│                                   #   QM and carry no age - that is what labelled a class 'nan'. A galaxy that
│                                   #   still has none is held out of the pool and counted in the printout (log M*
│                                   #   falls back to the KS table's log_mstar_anchor); a class left with nothing is
│                                   #   drawn unmatched + best resolved with a message. Nothing prints as nan: a
│                                   #   missing number reads n/a.
│                                   #   ON THE FIGURE: the P24_SIDE margin (right, width P24_MARGIN) carries, for
│                                   #   every row, the class name large + bold in its colour and that galaxy's log
│                                   #   M*, age, log M_dust/M_H2 at the anchor and its match cost; each PANEL its
│                                   #   own z and log M_dust/M_H2 from the member particles at that epoch, in a
│                                   #   small white box. Panels are 60 kpc a side (P24_EXT = 30) with the gutter
│                                   #   nearly closed (P24_WSPACE) - they are images, not plots. Map panels take the
│                                   #   colormap's dark end as background so white streamlines and labels read.
│                                   #   THE WHOLE 100 kpc CUTOUT is loaded, not the catalogue members (P24_MEMBERS
│                                   #   = False): sphviewer takes hsml from the 32 nearest LOADED neighbours, so
│                                   #   members-only inflates hsml and the galaxy comes out blobby and too small.
│                                   #   agn_states_gallery_rgb_cis25 = the composite, ONE COLOURED CHANNEL PER
│                                   #   COMPONENT (P24_CHAN: stars white, gas blue, dust orange, Screen-blended,
│                                   #   named in the first panel) stretched by log_stretch = P24_DYN decades below
│                                   #   the P24_PMAX percentile of the POSITIVE pixels. That stretch is the fix for
│                                   #   the uniform bright blob: RenderRGB's percentile clip (P24_STRETCH =
│                                   #   "percentile") saturates whenever the source fills less than 100-vmax % of
│                                   #   the frame, which is every galaxy in a wide frame.
│                                   #   agn_states_gallery_{dust,H2,temp}_cis25 = the physical maps, ONE SHARED
│                                   #   colour scale over all 16 panels, with the mass-weighted GAS VELOCITY FIELD
│                                   #   as streamlines (face-on: a rotating disc closes them into circles, a
│                                   #   dispersion-supported one does not - kappa_rot drawn, not scored). P24_MAPDEF
│                                   #   gives each field its KIND: "sigma" = a surface density (.sigma_map,
│                                   #   calibrated to Msun/kpc^2) | "weighted" = the mass-weighted LOS mean
│                                   #   (.weighted_map), which is what temp needs - a temperature is not a column
│                                   #   and must not be summed along the sightline; zgas and vlos (symmetric scale)
│                                   #   ride the same machinery, just add them to P24_MAPS.
│                                   #   TWO DIALS AT THE TOP OF THE CELL, flagged with an arrow: P24_SIZE = HOW BIG
│                                   #   EACH IMAGE IS, in kpc A SIDE (60 = a 60 x 60 kpc box; the cutouts end at
│                                   #   rmax_kpc = 100 from the centre, and the cell warns once past that), and
│                                   #   P24_FS = HOW BIG THE TEXT IS, one multiplier over every label in the part.
│                                   #   P24_SIZE = None derives the frame instead: P24_EXTFAC x the median R50(*) of
│                                   #   the drawn epochs, and only THEN does P24_EXTLIM apply - a GUARD ON THAT
│                                   #   AUTOMATIC VALUE, not a size control: raising its upper bound does nothing
│                                   #   unless the automatic value was already clipped. The cell prints which
│                                   #   decided.
│                                   #   Knobs: P24_SIZE, P24_FS, P24_STAGES, P24_MAPS, P24_MAPDEF, P24_STREAM,
│                                   #   P24_NGRID (the velocity field's RESOLUTION - it does not set how many lines
│                                   #   are drawn), P24_STREAM_DENSITY (2026-09-04: matplotlib seeds the streamlines
│                                   #   on its own 30 x 30 base grid scaled by this - 0.5 = half as many lines; the
│                                   #   user's NGRID 40 -> 5 changed nothing because of this), P24_STREAM_SOURCE
│                                   #   (2026-09-04: WHICH gas feeds the field, in the panel's frame - "slab"
│                                   #   [default] = every gas particle of the loaded region within +-
│                                   #   P24_STREAM_DEPTH kpc [3] of the mid-plane along the line of sight, enough
│                                   #   particles for smooth flows without the CGM in front / behind; "members" =
│                                   #   the CAESAR members only [too few, the user's verdict]; "region" = everything
│                                   #   loaded = what the field used before, CGM + neighbours), P24_STREAM_BULK
│                                   #   ("members" = the galaxy's own systemic velocity from its member gas / stars
│                                   #   subtracted, not the selection's mean), P24_STREAM_WEIGHT ("m_gas" | "m_H2"
│                                   #   | "m_dust"); the cell prints the particles used vs in the frame, the csv
│                                   #   carries stream_gas / stream_n_used / stream_n_in_frame; P24_STREAM_LW,
│                                   #   P24_STREAM_ARROW, P24_TAU,
│                                   #   P24_SPOS, P24_MEMBERS, P24_SIDE, P24_MARGIN, P24_WSPACE, P24_XSIZE, P24_CMAP,
│                                   #   P24_PCT, P24_MAPDYN (a log map
│                                   #   shows at most that many decades below its top - with the whole region loaded
│                                   #   the empty outskirts otherwise drag the low percentile down), P24_REDUCED,
│                                   #   P24_PREFIX
│                                   #   -> agn_states_gallery_cis25.csv: per panel the ids, epoch, z, frame size and
│                                   #   BOTH frame_* (what is drawn: the region inside the frame) and member_* (the
│                                   #   catalogue galaxy, which reproduces ks_track_epochs exactly).
│                                   #   THE PROJECTION FRAME (2026-09-04, user: faceon / edgeon did nothing): the
│                                   #   stored evecs are the STELLAR INERTIA frame (principal_axes of the member
│                                   #   stars), degenerate for a spheroid and blind to the gas disc. P24_FRAME =
│                                   #   H2_L (default) builds the frame from the angular momentum of the member H2
│                                   #   inside P24_FRAME_RMAX 10 kpc (P24_FRAME_MEMBERS: members only, whatever
│                                   #   P24_MEMBERS renders), faceon down that spin, edgeon along the disc; gas_L /
│                                   #   dust_L / star_L / stellar are the options; files without vel or with < 20
│                                   #   weighted particles fall back to stellar and the cell prints per panel the
│                                   #   frame used, n and the tilt between the H2 spin and the stellar minor axis.
│                                   #   P24_PICK (2026-09-04) = lo_hi (100, 45), lo_lo (90, 43), hi_hi (95, 60),
│                                   #   hi_lo (90, 31): age 1.29-1.55 Gyr, log M* 10.34-10.53, log M_dust/M* -3.40 /
│                                   #   -4.05 / -5.77 / -4.24, log M_H2 8.85 / 8.83 / 8.36 / 8.95 - chosen on the
│                                   #   M_dust/M* contrast (dh2 alone flags the H2-poor remnants).
│                                   #   STILL TILTED (2026-09-04, later): the frame WAS applied, but that pick's H2
│                                   #   has kappa_rot 0.1-0.2 at every epoch (below the isotropic 1/3) - its spin
│                                   #   is noise, there is no disc to align. Now: ReducedRender.angular_momentum
│                                   #   returns the spin COHERENCE |sum l|/sum |l| (1 disc, ~1/sqrt n random),
│                                   #   set_evecs imposes a frame, P24_FRAME_STAGE = sft gives each ROW one frame
│                                   #   (the spin at SFT), the report prints coherence / own coherence / tilt per
│                                   #   panel; P24_PICK = the ROTATING quadruple 116/61, 116/119, 116/42, 110/71
│                                   #   (kappa 0.77 / 0.64 / 0.69 / 0.49, age 3.5-4.0, log M* 10.17-10.64, dust
│                                   #   -3.38 / -4.74 / -4.26 / -5.31, 200-850 gas particles, three at anchor 116)
│                                   #   - the only matched set of four with a disc; rotation above kappa 0.5 is
│                                   #   12/36 lo_hi, 4/34 lo_lo, 7/70 hi_hi, 2/39 hi_lo. The non-rotating tight
│                                   #   set stays in P24_PICK_DUSTCONTRAST.
│                                   #   Part 25 = WHERE THE DUST WENT AND WHETHER THE H2 CAME BACK (2026-09-04): the
│                                   #   budget along the branch since SFT, core -> CGM, per dose x geometry group
│                                   #   on t - t_SFT. (A) the histories cache, every snapshot: whole-galaxy M_dust,
│                                   #   M_H2, their ratio, the X-RAY OCCUPANCY (fraction in the X-ray state at each
│                                   #   moment since SFT), the t_quiet distribution at the anchor (never-exposed
│                                   #   counted apart) and the per-galaxy REGROWTH (anchor minus post-SFT minimum;
│                                   #   P25_REGROW 0.3 dex = came back) -> agn_states_budget_branch_cis25. Smoke
│                                   #   test: low/compact H2 back 43 % (median +0.28 dex), dust back 59 % (+0.52),
│                                   #   H2 minimum 1.2 Gyr after SFT; low/extended 25 / 25 % (median 0); the
│                                   #   high-dose groups sit at X-ray occupancy ~1 from +0.2 Gyr, low/compact
│                                   #   0.4-0.7 with 40 % out of the state at the anchor (t_quiet up to 2 Gyr).
│                                   #   (B) the critical points, MEMBER particles (ks_track_measurements): M_dust
│                                   #   and M_H2 in the core 0-1 / ring 1-3.2 / outskirt 3.2-10 kpc at sft -> qt ->
│                                   #   post_quench -> gas_min -> anchor placed at each group's median t - t_SFT,
│                                   #   plus the core's share of the < 10 kpc mass -> agn_states_budget_zones_cis25.
│                                   #   Smoke test: the low/compact core keeps 10^7.9-8.2 of H2 and 10^6.6 of dust
│                                   #   from QT to the anchor (core share ~1), the other three fall to 10^6.6-7.9
│                                   #   and 10^2.4-4.8. THE 3-D CHECK (m25 8j0 kinematics, every particle, 1 kpc
│                                   #   sphere): the two EXTENDED groups hold a median of ZERO H2 particles inside
│                                   #   1 kpc (5 gas) against 21 (23) for low/compact - a real central cavity the
│                                   #   projected annuli of Part 23 cannot see; their H2 sits in the 1-3 kpc ring.
│                                   #   (C) the whole 100 kpc cutout from ks_tracks/ks_stage_budget.fits (the K-S
│                                   #   notebook's Part 3c, every particle, shells core / ring / outskirt / cgm_in
│                                   #   10-30 / cgm_out 30-100 kpc, radial velocities about the stellar systemic
│                                   #   velocity): ISM against CGM dust, the CGM share, v_r of the dust in the
│                                   #   outskirt and the inner CGM (+ = outflow), the outflowing and the hot share
│                                   #   of the CGM dust, the core H2 with its particle count -> agn_states_budget_
│                                   #   cgm_cis25 at the critical points, and _track on the continuous clock (bins
│                                   #   P25_TB) once Part 2b's post-SFT cutouts exist (ks_postsft_map.csv). Preview
│                                   #   on the Part 24 low/compact example: CGM dust outflowing at 120-200 km/s at
│                                   #   SFT (f_out 0.75); by the anchor ISM dust -1.7 dex, CGM dust -0.5 dex, the
│                                   #   CGM holds ~8x the ISM dust and every shell is INFLOWING (-10..-70 km/s);
│                                   #   its 1 kpc sphere is gas-free - but it is NOT representative: the first
│                                   #   cluster run (1335 cutouts, critical points) has the CGM share of the dust
│                                   #   rising 0.4 -> 0.8-0.9 in every group (ISM dust -1.2..-2.3 dex, CGM dust
│                                   #   -0.3..-0.7: the surviving dust is the CGM dust), v_r(10-30 kpc) +12 at
│                                   #   ignition -> +150 at jet_on / SFT (f_out 0.9) -> +100 at QT -> ~0 (high dose)
│                                   #   or +98 (low/compact) at the anchor, only 24 % of the 10-30 kpc rows below
│                                   #   zero; the one population-level inflow is the OUTSKIRT 3.2-10 kpc of the
│                                   #   high-dose groups after QT (-40..-70 km/s). Knobs: P25_TG, P25_STG, P25_TB, P25_REGROW,
│                                   #   P25_NMIN, P25_KSDIR. Tables -> agn_states_budget_cis25.csv. 2026-09-04 later:
│                                   #   'outflowing' SHARPENED to ABOVE THE LOCAL ESCAPE SPEED (v_r > v_esc of the
│                                   #   PARENT halo's NFW potential at the particle's radius, caesar m200c / r200c,
│                                   #   Dutton & Maccio 2014 c200; K-S Part 3c looks the halo up per (snap, gx) in
│                                   #   ks_halo_props.csv and rebuilds a cache without m_dust_esc / m_dust_in):
│                                   #   Fig A panel (f) is now the CGM share of the dust on the continuous clock
│                                   #   (the regrowth scatter is the fallback and its numbers stay printed); Fig C
│                                   #   panel (e) = the ESCAPING share solid with the loose v_r > 0 share dashed;
│                                   #   panel (d) = v_r of the CGM 10-30 kpc (solid) AND of the outskirt 3.2-10 kpc
│                                   #   (dashed open); NEW Fig D agn_states_budget_flows_cis25 = per zone (ring /
│                                   #   outskirt / CGM) and per group the share of the zone's dust (top) and H2
│                                   #   (bottom) moving OUTWARD (v_r > 0, solid) and ESCAPING (dashed) on the
│                                   #   continuous clock - above the 0.5 line more of the zone's material heads
│                                   #   for the CGM than for the core; inward = the complement; anchor values
│                                   #   printed. First real outward shares at the anchor (outskirt dust): low/
│                                   #   compact 0.49, low/extended 0.40, high/compact 0.19, high/extended 0.33 -
│                                   #   the high-dose compact outskirt dust is mostly falling inward.
│                                   #   Part 26 = THE DOSE x EXTENT SPLIT OF THE 100 Mpc BOX WITH THE CATALOGUE RADII,
│                                   #   AGAINST THE ALMA-C11 SAMPLE (2026-09-04, redone the same day as POPULATION
│                                   #   TRACKS): Part 22's Fig 1 frame (FIG1_22, the same dose) gets caesar's
│                                   #   radii.gas_half_mass / radii.stellar_half_mass and masses.dust / H2 / stellar
│                                   #   read straight from the catalogue files (h5py; cached
│                                   #   agn_states_catalogue_radii_<box>.csv, P26_CATDIR overrides the config.json
│                                   #   catalog_dir; ALSO ages.mass_weighted -> age_mw, so the m25 frame - Part
│                                   #   22's KS-track anchors, which carry no age column - gets its age from the
│                                   #   catalogue: P26_BOXES ["cis100", "cis25"] runs, an old cache without the
│                                   #   column is rebuilt; file stems fall back to m100n1024 / m50n512 / m25n512
│                                   #   without a config), the box split at its own medians of f_x_post and of
│                                   #   log R50(gas)/R50(*) - the catalogue extent tracks the m25 particle H2 extent
│                                   #   with Spearman +0.95, 92 % on the same side of the split. FIGURE (2x2): each
│                                   #   class as its population track = the running median along x (a sliding window
│                                   #   over the P26_KFRAC 0.30 nearest galaxies, >= P26_NMIN 10) with the 16-84
│                                   #   dispersion band, in the observed regime (z <= P26_ZMAX 0.5, log M* >=
│                                   #   P26_LOGM 10.8; P26_TRACK_SEL "zmax" = every mass); planes age -> M_dust/M*,
│                                   #   age -> M_H2/M* (the observed alpha_CO H2 / 1.36), age -> extent (R_CO/R_e
│                                   #   for the data, the split cut dotted), M_H2/M* -> M_dust/M*; the observed box
│                                   #   P26_OBOX dashed; the regime galaxies inside it on all four axes drawn ONE BY
│                                   #   ONE, numbered, listed in panel (b) -> agn_states_boxes_vs_alma_inbox.csv;
│                                   #   the ten CO-detected ALMA-C11 QGs with ERROR BARS (CIGALE age and M_dust,
│                                   #   CO-flux M_H2, uv-fit size interval (+) R_e error for the extent; arrows =
│                                   #   upper limits, P26_ULIM per axis: dust undetected, CO unresolved). Prints per
│                                   #   class / selection the medians and in-box counts, per source the counterparts
│                                   #   within P26_TOL -> agn_states_boxes_vs_alma.{png,pdf,csv}. P26_BOXES
│                                   #   ["cis100"] (add "cis50": one figure per box), P26_POINTS = the regime
│                                   #   galaxies as faint points behind the bands. HISTORY TRACKS (added the same
│                                   #   day): P26_HIST ("lfd", "lfh2") draws, dashed on the two age panels, each
│                                   #   class's median main-branch chain of the quantity vs the mass-weighted age
│                                   #   ALONG the branch (HIST_FILE, the chains Part 22 dosed; grid P26_HIST_AGE,
│                                   #   nodes with >= P26_HIST_NMIN chains, P26_HIST_BAND = the 16-84 range across
│                                   #   the chains) with a circle at the class's median age at SFT: solid = where
│                                   #   the quenched galaxies sit at their anchors, dashed = where they came from
│                                   #   (-> the 'history track' rows of the csv). RESULT (m100, 1625 quenched, 309
│                                   #   in the regime): the mass gap closes and the H2 content is reproduced (log
│                                   #   M_H2/M* -1.55..-1.72 per class vs -1.45 observed), the extent by rank
│                                   #   (-0.15..-0.30 all-gas vs -0.07 CO/R_e: not the same scale), but a DUST GAP
│                                   #   opens - class medians at log M_dust/M* -4.7..-6.3 vs -3.0 observed, bands
│                                   #   wide because the dust distribution is bimodal (dust-poor floor + dusty
│                                   #   tail) - and only 8 of 309 (2.6 %) fall in the observed box: 5 low/extended
│                                   #   (11 % of that class), 2 low/compact, 1 high/extended, 0 high/compact (snap
│                                   #   125 gal 88/365/424/300/280, snap 134 gal 154/1284/287; log M* 10.84-11.45,
│                                   #   age 4.2-6.5 Gyr). The ALMA sample is the dusty upper edge of SIMBA's massive
│                                   #   quenched population, populated by the LOW-dose galaxies, the extended ones
│                                   #   first. Per source: hc4 / hc12 have 3 massive counterparts each, hc11 one
│                                   #   (low/compact, d 0.6), jw22 / hc2 / hc7 / hc10 none within tolerance.
│                                   #   HISTORY RESULT: every class starts at log M_dust/M* -2.5 (age 1 Gyr) and
│                                   #   reaches SFT at age 2.0-2.4 Gyr; the LOW-dose median chains pass THROUGH the
│                                   #   observed box (-3.1 at age 3, -4.0 / -4.3 at 4) while the high-dose ones are
│                                   #   below it by age 3 (-3.8 / -4.1): ALMA-like galaxies are the low-dose classes
│                                   #   seen ~1-2 Gyr after SFT, and the 8 in-box galaxies are their late stragglers.
│                                   #   Part 27 = THE COMPANIONS (2026-09-04): for every box of P27_BOXES (["cis100"];
│                                   #   one figure each, first box = plain product names, others _<box>; each needs
│                                   #   its Part 26 frame = P26_BOXES + histories; cutout stem from the config's
│                                   #   file_format or P27_PREFIXES; anchors output/<box>/reduced_particles, branch
│                                   #   cutouts output/<box>/filtered_particles, plan output/<box>/caesar_sfh/
│                                   #   prof_almamatch/) each CO-detected ALMA-C11 source's
│                                   #   CLOSEST galaxy of the Part 26 regime sample (P27_POOL "regime" | "zmax") in
│                                   #   the observable space P27_AXES (age 0.5 Gyr, lfd 0.3, lfh2 0.3, extent 0.3,
│                                   #   log M* 0.15 = the tolerances; the distance in tolerance units, summed in
│                                   #   quadrature; an observed upper limit penalises only galaxies above it),
│                                   #   P27_NBEST runner-ups, RENDERED (ReducedRender, Part 24's stars/gas/dust
│                                   #   composite = P27_STYLE "rgb"; the DEFAULT "contours" = each side's STARS with
│                                   #   the DUST and H2 as CONTOURS: the F814W stamp + the imaging pipeline's continuum
│                                   #   (orange) and CO(3-2) moment-0 (blue) maps at P27_OBS_LEVELS (3, 5, 8, 12, 20)
│                                   #   x rms (first level also negative dashed, beam drawn), the companion's log
│                                   #   Sigma_* + its own projected Sigma_dust / Sigma_H2 (x P27_HE 1.36 He) AS THEY
│                                   #   ARE, contoured at P27_SIM_FRAC (5, 10, 20, 40, 80) % of each map's peak
│                                   #   (P27_SIM_LEVELS "peak", the default; "obs" = the same Msun/pc^2 as the data
│                                   #   through M_H2/S_CO, M_dust/S_nu and the beam area, dict(H2=, dust=) by hand,
│                                   #   P27_SIM_FALLBACK where the dust is undetected), beam-SMOOTHED (P27_BEAM) only
│                                   #   so the contours trace the distribution, not the particles; a THIRD ROW = the
│                                   #   companion FACE-ON AS ITS GAS (P27_FACEON_ROW, the composite with P27_FACE_CHAN
│                                   #   gas + dust, no stars); prints per source the data's 3 sigma in Msun/pc^2
│                                   #   against the companion's peak for reference) FROM THE OBSERVED LINE
│                                   #   OF SIGHT: North up,
│                                   #   East left, tilted to the source's inclination and turned so the projected
│                                   #   major axis sits at its position angle, next to the source's ACS F814W stamp
│                                   #   at the same P27_SIZE (30 kpc a side). Geometry from obs_data/almac11/
│                                   #   almac11_orientation.csv (+ almac11_stamps_f814w.npz with the F814W stamps AND
│                                   #   the pipeline's continuum / CO moment-0 cutouts around the same centre, their
│                                   #   rms, beams and M_H2/S_CO, M_dust/S_nu; built LOCALLY by
│                                   #   make_orientation_table.py there): COSMOS-Web axis ratio + angle (the SE++
│                                   #   angle IS the sky PA E of N - it agrees with the CO uv-fit PAs and the stamp
│                                   #   moments), ACS gim2d q + stamp-moment PA where COSMOS-Web has no fit (hc5,
│                                   #   hc9, hc10, hc12), CO uv PA, 3D-Barolo i for hc5. P27_INC "kin_first"
│                                   #   (Barolo, else cos^2 i = (q^2 - q0^2)/(1 - q0^2), P27_Q0 0.2) | "optical" |
│                                   #   a number | "match_q" (tilt the simulated galaxy until its APPARENT q equals
│                                   #   the observed one; edge-on when it cannot); P27_PA "opt" | "co" | "kin";
│                                   #   P27_FRAME "stellar" (the member stars' principal axes: the tilt leaves the
│                                   #   MINOR axis, what a Sersic q measures) | "star_L" / "H2_L" / "gas_L";
│                                   #   P27_AZIMUTH = the in-plane tilt direction. Every companion's apparent q is
│                                   #   measured from the projected stellar mass (moments inside P27_MOM_R x R50)
│                                   #   and scanned face-on -> edge-on (P27_INC_STEP). PARTICLES: the anchor's
│                                   #   reduced cutout (output/cis100/reduced_particles, NOT built yet: the cell
│                                   #   writes the plan output/cis100/caesar_sfh/prof_almamatch/
│                                   #   dust_profile_plan_almamatch.hdf5 with every listed companion lacking one and
│                                   #   prints the sbatch line), until then the RT CUTOUT OF THE SAME MAIN BRANCH
│                                   #   nearest in time within P27_RT_DT 1.5 Gyr (output/cis100/filtered_particles,
│                                   #   the ALMA-C11 analogue campaign's members-only cutouts at snaps 128-132,
│                                   #   linked through the catalogues' tree_data progen/descend, converted on the
│                                   #   fly to the reduced schema in OUT/rt_reduced/; P27_SOURCE "auto" | "reduced"
│                                   #   | "rt"); a companion with no data is drawn through the closest runner-up
│                                   #   that has some (a STAND-IN, dashed edge, P27_STANDIN). Products
│                                   #   agn_states_companions.{png,pdf} (rows: F814W stamp / companion at the
│                                   #   observed geometry / optional face-on, P27_FACEON_ROW; edge colour = class)
│                                   #   + agn_states_companions.csv (geometry used + origin, matches + runner-ups
│                                   #   with both sides' observables, what was drawn from where and its dt, the
│                                   #   apparent q and PA, the face-on and edge-on q). FIRST RUN (local, branch
│                                   #   cutouts): closest matches hc2 -> 134/506 (high/compact, d 0.5, dust-poor:
│                                   #   hc2's dust is a limit), hc4 and hc12 -> 125/88 (low/compact), hc3 -> 125/365,
│                                   #   hc7 -> 125/424, hc10 -> 134/287 (low/extended), hc5 -> 125/280, hc9 ->
│                                   #   125/953, hc11 -> 134/22, jw22 -> 134/1429 (d 4.8: below the regime mass
│                                   #   cut); 6 of 10 drawn from branch cutouts (3 stand-ins), 4 wait for the plan;
│                                   #   the requested PAs are reproduced to 1-3 deg, but EVERY companion is rounder
│                                   #   than its source from any direction (edge-on q 0.69-0.79 vs observed
│                                   #   0.13-0.72): the simulated quenched bodies are spheroids, the ALMA-C11
│                                   #   galaxies discs. CONTOURS: the data's CO sits on the stellar bodies at 3-20
│                                   #   sigma; every companion's beam-convolved H2 peak (30-80 Msun/pc^2) clears the
│                                   #   source's 3 sigma level (10-35) = would be CO-detected, its dust peak
│                                   #   (0.01-1.2) mostly stays below the continuum 3 sigma (0.35-2) = the Part 26
│                                   #   dust gap as a picture.
│                                   #   Part 28 = THE DUST-AGE PLANE OF EVERY RUN AGAINST THE DATA, AND THE EXPOSURE
│                                   #   BEHIND EACH BOX'S HIGH / LOW DOSE SPLIT (2026-09-08; cells p28-md / p28-fdust-
│                                   #   age-dose, needs Parts 0 + 22, Part 26 optional). ON PART 0'S ANCHOR ROWS, not
│                                   #   Part 22's de-duplicated galaxies (a galaxy quenched at several anchors counts
│                                   #   once per anchor on both sides of every share: cis100 1683 rows vs 1625
│                                   #   galaxies). LEFT: log M_dust/M* vs the mass-weighted age for m100 / m50 / m25
│                                   #   AND the channel-off m50 run (P28_NOX), each as TWO population tracks (Part
│                                   #   26's running median over P28_KFRAC 0.30 of the class, >= P28_NMIN 10, 16-84
│                                   #   band): dusty (>= DUSTY_CUT, solid) and dust-poor (dashed), P28_SEL "sample"
│                                   #   (Part 0's QM) | "regime" (Part 26's z <= 0.5, log M* >= 10.8); the ALMA-C11
│                                   #   QGs (age_sersic_sigma.csv, kind main, N = 11, dust ULs arrowed) and the m100
│                                   #   galaxies inside Part 26's observed box numbered as there (P28_INBOX; INBOX26
│                                   #   when Part 26 ran, else Part 26's rule re-applied to the anchor rows); one hue
│                                   #   per run (BOXC28), P28_LAYOUT "single" (one plane) | "rows" (one plane per box
│                                   #   beside its dose panel). RIGHT, one panel per P28_BOXES box: the distribution
│                                   #   of P28_DOSE (= P22_DOSE, the X-ray duty fraction since SFT; Part 22's
│                                   #   _chains22 / _dose22 re-run over EVERY anchor row of the box, the m25 chains
│                                   #   Part 22's own _CH22), dusty vs dust-poor histograms (P28_NBINS 20, P28_LOGY),
│                                   #   the background shaded on either side of the cut (P28_DCUT "part22" = Part
│                                   #   22's cut of the box, so the halves ARE Parts 22-27's | "median" | a number),
│                                   #   and in the bottom corners the four SHARES dusty / dust-poor x low / high dose
│                                   #   as percentages of P28_DENOM "sample" (every anchor row of Part 0's sample in
│                                   #   the box: the four sum to the chain coverage) | "dosed" (sum 100 %). ->
│                                   #   agn_states_fdust_age_dose.{png,pdf,csv} (the shares) +
│                                   #   agn_states_fdust_age_tracks.csv. RESULT (the cluster products, Part 0 at
│                                   #   Z_MAX 8 / AGE_MIN 3, chain coverage 100 / 100 / 99 %): cuts 0.82 / 0.81 /
│                                   #   0.85; shares dusty-low / dusty-high / poor-low / poor-high = cis100 11.9 /
│                                   #   2.9 / 39.2 / 46.0 %, cis50 9.9 / 1.0 / 42.4 / 46.8 %, cis25 13.3 / 17.3 /
│                                   #   10.7 / 57.3 % - in the two big boxes the dusty QGs are 4-10x more often LOW
│                                   #   dose, m25 (75 rows) does not show it. The nox run (178 quenched) has NO
│                                   #   dust-poor class at -4.5; its dusty track (lfd -3.2 at 3-7 Gyr) runs through
│                                   #   the ALMA-C11 points and the m100 in-box galaxies, the AGN-on dusty tracks sit
│                                   #   at -3.6..-4.1 and the dust-poor ones at -6..-6.3.
│                                   #   Part 29 = THE CUTS ON TRIAL (2026-09-08, replaces the first 'dust vs compactness'
│                                   #   version the same day; cells p29-md / p29-compactness, needs Part 28): are the
│                                   #   MEDIAN splits of Parts 22-27 - on the X-ray duty fraction since SFT and on a
│                                   #   compactness - natural or arbitrary? Part 28's anchor rows, the dusty fraction
│                                   #   only. Axes: the dose (P29_DOSE); P29_AXES ["r50", "conc"] = Parts 22-26's
│                                   #   -log R50(gas)/R50(*) from Part 0's cache and the gas's OWN CONCENTRATION
│                                   #   log R80/R20 (the user's r20/r80; caesar radii.gas_r20 / gas_r80 x a, read
│                                   #   once per box through the config's catalog_dir or P29_CATDIR and cached in
│                                   #   agn_states_gas_concentration_<box>.csv with the stellar r20/r50/r80 beside
│                                   #   them; "conc_rel" = gas minus stellar concentration). FIG A
│                                   #   agn_states_cuts_axes: one row per axis x one column per box - the dusty
│                                   #   fraction in P29_NBINS 10 equal-count bins (clamped Wilson), the axis's own
│                                   #   histogram behind, the cut in use dashed (Part 22's for the dose, the median
│                                   #   for a compactness), and the CONTRAST SCAN (dusty fraction below minus above
│                                   #   every candidate cut between the P29_SCAN 0.10-0.90 quantiles, P29_NSCAN 61)
│                                   #   thin on the same scale with its maximum dotted red; printed per axis the AUC
│                                   #   dusty-vs-dust-poor (cut-free), the contrast at the cut vs the best, and the
│                                   #   RESOLUTION GUARD (Spearman with log n_gas / log M_gas, the AUC inside n_gas
│                                   #   >= P29_NGAS_CHK 200). FIG B agn_states_cuts_plane: dose x compactness (one
│                                   #   row per axis) in P29_GRID 4 x 4 EQUAL-COUNT cells drawn in rank space (data
│                                   #   values at the edges, medians at 0.5; a box under 4 x 4 x P29_NMIN rows falls
│                                   #   back to the bare 2 x 2 quadrants), coloured by the dusty fraction, the four
│                                   #   median quadrants = the classes with their own fraction written in, cells
│                                   #   under P29_NMIN 8 hatched. CSVs agn_states_cuts_{axes,scan,plane}.csv.
│                                   #   RESULT (the cluster products, Part 0 at Z_MAX 8 / AGE_MIN 3): (1) THE DOSE
│                                   #   CUT IS ON THE FLAT PART. The dusty fraction falls from 45 % at f_x_post ~0.3
│                                   #   to 12 % at 1 (cis100; 33 -> 7 % cis50, 67 -> 24 % cis25) as a CONTINUUM,
│                                   #   flat only above ~0.85; the largest contrast sits at 0.42 / 0.49 / 0.80 and
│                                   #   the median cut 0.82 / 0.81 / 0.85 keeps 52 / 63 / 63 % of it - 'low dose'
│                                   #   mixes the genuinely low-dosed (45 % dusty) with the 0.5-0.8 galaxies (10-15
│                                   #   %). AUC(dose) 0.26 / 0.22 / 0.33. (2) THE R50 COMPACTNESS CARRIES NO DUST
│                                   #   INFORMATION in the full sample: AUC 0.48 (cis100, p 0.4) / 0.29 (cis50) /
│                                   #   0.47 (cis25); its dusty fraction is U-shaped (34 -> 7 -> 24 % in cis100) so
│                                   #   the median cut keeps 4 % of the best contrast, and the plane's low-dose
│                                   #   quadrants are 23 vs 24 % (compact vs extended) - the extent split of Parts
│                                   #   22-26 is a split at fixed dust; in cis25 it tracks the gas particle count
│                                   #   (Spearman -0.65 with log n_gas). (3) THE GAS CONCENTRATION DOES: AUC 0.63 /
│                                   #   0.63 / 0.83, the dusty fraction rising 18 -> 44 % (cis100), 29 -> 48 %
│                                   #   (cis50), 0 -> 100 % (cis25) towards concentrated profiles, the best cut at
│                                   #   0.29 / 0.29 / 0.41 (above the median 0.21 / 0.21 / 0.22); uncorrelated with
│                                   #   the r50 axis (Spearman +0.01 / +0.03 / -0.01) and NOT a resolution artefact
│                                   #   (inside n_gas >= 200 the AUC is 0.71 / 0.82 in cis100 / cis25, with log
│                                   #   n_gas only +0.26..+0.33). In the dose x concentration plane the quadrants
│                                   #   are 28 / 19 / 10 / 2 % (cis100: low-conc / low-flat / high-conc / high-flat)
│                                   #   with a gradient INSIDE every quadrant (10-49 % across the low-dose
│                                   #   concentrated cells): one gradient, not four populations.
│                                   #   Part 30 = THE DOSE x CONCENTRATION AND THE E_x x CONCENTRATION PLANES (2026-09-08;
│                                   #   cells p30-md / p30-planes, needs Part 29): one column per box, top the X-ray
│                                   #   duty fraction since SFT vs log R80/R20 of the gas (the dashed vertical = the
│                                   #   cut in use, P30_XCUT "part28" | "best" | a number), bottom E_x [Gyr] vs the
│                                   #   same (dashed = PF_EX 1 Gyr, P30_EXCUT); every anchor row a point (dusty on
│                                   #   top), THE COMPACT / EXTENDED DIVISION = the horizontal line at P30_CCUT on the
│                                   #   concentration (0.30 = a uniform disc's R80/R20 = 2 by default, the same in
│                                   #   every box; "best" = Part 29's largest-contrast cut per box, drawn thin red
│                                   #   when it differs, P30_SHOW_BEST; "median"), the reference profiles ticked on
│                                   #   the right (P30_REFS: uniform sphere 0.20, uniform disc 0.30, exponential disc
│                                   #   0.56), each quadrant's N and dusty fraction in a box ->
│                                   #   agn_states_dose_concentration_planes.{png,pdf,csv}. RESULT at the 0.30
│                                   #   division (cis100): duty-fraction plane low/concentrated N 88, 52 % dusty;
│                                   #   high/concentrated 59, 36 %; low/flat 772, 20 %; high/flat 764, 4 % - the
│                                   #   concentrated side is 9 % of the sample and holds the dustiest classes; E_x
│                                   #   plane (PF_EX 1 Gyr) low/concentrated 23, 74 %; high/concentrated 124, 40 %;
│                                   #   low/flat 170, 25 %; high/flat 1366, 10 %. cis50: 12 / 67 %, 4 / 50 %, 94 /
│                                   #   13 %, 93 / 0 %; cis25: 6 / 100 %, 11 / 55 %, 12 / 33 %, 45 / 16 %. The
│                                   #   quenched galaxies' gas is typically a uniform blob (median 0.21 = a uniform
│                                   #   sphere); the dusty ones are the minority with a core.
│                                   #   Part 31 = THE REMOVAL DEPTH (2026-09-08; cells p31-md / p31-depth, needs Parts 28-29):
│                                   #   the outcome as a continuous quantity, depth = log M_dust(anchor) - log M_dust(epoch),
│                                   #   P31_EPOCH "sft" (the branch snapshot nearest t_SFT) | "peak" | "qt", the same for the
│                                   #   H2 and the total gas (from the histories), against the CHANNEL (the duty fraction) and
│                                   #   the WELL: the parent halo's M200c and the NFW escape speed at the galaxy's position
│                                   #   (caesar galaxy_data/parent_halo_index -> halo_data virial_quantities.m200c [Msun] /
│                                   #   r200c [kpccm x a] / masses.total / circular_velocity, the galaxy-halo distance from
│                                   #   minpotpos with the minimum image; a central at its R50(*) (P31_VESC_R), a satellite at
│                                   #   its distance to the host centre; Dutton & Maccio 2014 c(M, z); cached
│                                   #   agn_states_halo_props_<box>.csv, P31_CATDIR overrides the config; P31_CENTRALS = centrals
│                                   #   only). FIGURE agn_states_removal_depth, one column per box: (1) dust depth vs duty, one
│                                   #   running median per log M* quartile; (2)-(3) dust depth vs P31_WELL ["log_m200c", "vesc"]
│                                   #   per duty quartile; (4) H2 depth vs the first well axis per duty quartile. PRINTED + CSV
│                                   #   agn_states_removal_depth.csv: Spearman of each depth with duty / log M* / log M_BH /
│                                   #   log M200c / v_esc / log M_halo,tot / V_c / central / conc / log M_gas(SFT) / log
│                                   #   M_dust(SFT) / age / t_quiet, raw, at fixed duty, at fixed log M*, at fixed both (3x3
│                                   #   cells); _galaxies.csv = every row with its depths, halo and v_esc. Written because
│                                   #   Parts 29-30 found the mass a third axis no dose definition absorbs (the X-ray ENERGY
│                                   #   delivered, Part 2a's m_acc_x, has AUC 0.60 the wrong way and 0.48 at fixed M*; the
│                                   #   energy per M_gas / M_H2 / M_dust at SFT 0.39-0.46; only the TIME in the state works,
│                                   #   and at a given time the massive galaxies lose 1.4 dex less dust: -3.3 dex at 10^10.3 vs
│                                   #   -1.9 at 10^11.2). RESULT (local run on the cluster products; the sandbox
│                                   #   figure is what the cluster will draw): median dust depth -3.09 / -3.12 /
│                                   #   -2.33 dex (cis100 / cis50 / cis25), dusty rows -1.07 vs dust-poor -3.23
│                                   #   (cis100); centrals 80 / 74 / 84 %, median log M200c 12.4, v_esc ~650 km/s.
│                                   #   Spearman with the DUST depth, cis100, raw / fixed duty / fixed M* / fixed
│                                   #   both: duty -0.28 / - / -0.22 / -0.04; log M* +0.32 / +0.26 / - / +0.17; log
│                                   #   M_BH +0.35 / +0.26 / +0.23 / +0.19; log M200c +0.33 / +0.26 / +0.21 / +0.20;
│                                   #   v_esc +0.35 / +0.27 / +0.22 / +0.21; conc +0.24 / +0.23 / +0.19 / +0.19;
│                                   #   central -0.04; log M_dust(SFT) -0.01 / -0.13 / -0.33 / -0.33 (a bigger
│                                   #   reservoir at fixed M* loses MORE in dex). cis50 the same signs, weaker
│                                   #   (M200c +0.06 at fixed M*, M_BH +0.24); cis25 M200c +0.36 / v_esc +0.34 at
│                                   #   fixed M*, conc +0.43. So the well keeps dust at fixed dose AND fixed
│                                   #   stellar mass, modestly (+0.2), in every box. THE H2 DEPTH IS NOT
│                                   #   INFORMATIVE: it is set by log M_gas(SFT) at -0.73 (-0.63 at fixed both) -
│                                   #   the anchor H2 sits near the resolution floor for everyone, so the log
│                                   #   ratio is minus the reservoir; the dust depth is not floored that way
│                                   #   (log M_dust(SFT) raw -0.01).
│                                   #   Part 32 = THE DIVISION ON TRIAL (2026-09-08, v2 the same day with kappa_gas; cells p32-md /
│                                   #   p32-score, needs Part 31's frames or Part 29's): every candidate dusty / dust-poor divider scored
│                                   #   on the same anchor rows - single properties (Mann-Whitney AUC toward dusty; the list now carries
│                                   #   the catalogue whole-galaxy kappa_rot of the gas, "kappa_gas" = caesar's rotation.gas_kappa_rot,
│                                   #   the axis Part 22 splits the coarse boxes on, which the first version lacked, and kappa_star),
│                                   #   LINEAR SCORES of 2-4 properties (P32_SETS; a ridge logistic regression by Newton steps, hand-
│                                   #   rolled, no sklearn; out-of-fold AUC of a repeated stratified k-fold P32_KFOLD 5 x P32_NREP 5,
│                                   #   P32_RIDGE 1.0 = sklearn's C = 1; the sets now include kappa_gas, duty+kappa_gas (Part 22 pair),
│                                   #   kappa_gas+M* (no dose), duty+kappa_gas+M*, duty+kappa_gas+conc+M*, kappa_gas+M*+age) and TWO-CUT
│                                   #   AND RULES from a blind quantile scan on a pair of axes (largest Youden J = recall - FPR; P32_SCAN
│                                   #   / P32_NSCAN / P32_NMIN). The drawn score (P32_SCORE "duty+kappa_gas+M*") is thresholded by the
│                                   #   data, not by hand: P32_THRESH "youden" (the cut of largest J; the deployed default, the user's
│                                   #   setting) | "prior" (out-of-fold p >= the box's dusty fraction) | a probability. FIGURE
│                                   #   agn_states_division_score, one column per box: (1) the P32_PLANE plane duty x log M* (points,
│                                   #   dusty on top), the score's iso-probability line at the threshold (the other terms at their
│                                   #   medians), the best two-cut rule of the plane (dashed rectangle + N / dusty share / recall), the
│                                   #   current dose cut (CUT28, dotted) and the current rule's members ringed (duty < CUT28 & conc >
│                                   #   P32_CCUT = Part 30's 0.30); (2) the dusty fraction along each candidate score of P32_CURVES (out-
│                                   #   of-fold, equal-count percentile bins, Wilson bands; steeper = better, AUC in the legend, the
│                                   #   prior dotted, the drawn score's threshold dashed); (3) log M_dust/M* vs the drawn score's log-
│                                   #   odds, running median + 16-84 (in-sample: a display of the gradient the classes cut). CSVs
│                                   #   agn_states_division_score.csv (the AUC ladder), _rules.csv (the current rule, Part 22's red group
│                                   #   = duty < CUT28 & kappa_gas > its median, the dose alone, the score at its threshold, the score at
│                                   #   the current rule's size, the best two-cut rules on (duty, M*) / (duty, conc) / (conc, M*) /
│                                   #   (kappa_gas, M*) / (duty, kappa_gas): n, dusty share inside, share of all dusty recovered, J),
│                                   #   _coef.csv (standardised + raw coefficients).
│                                   #   RESULT v1 (local run, DUSTY_CUT -4.5, threshold "prior", no kappa_gas): NO DIVIDER IS CLEAN - the
│                                   #   best linear score reaches AUC 0.83 / 0.89 / 0.84 (cis100 / cis50 / cis25) and any rule recovering
│                                   #   most of the dusty galaxies selects a group one third dusty; the outcome at fixed causes is a
│                                   #   gradient. Within that the missing axis looked like the stellar mass: cis100 duty 0.74, conc 0.63,
│                                   #   M* 0.75, duty+conc (the current pair) 0.78, duty+M* 0.80, duty+conc+M* 0.83, conc+M*+age without
│                                   #   any dose 0.78; the two-cut rule found blind, duty < 0.79 & log M* > 10.52 (n 536, 33 % dusty, 72
│                                   #   % of the dusty, J 0.47), and cis50 found the same rule independently (duty < 0.78 & log M* >
│                                   #   10.61).
│                                   #   RESULT v2 (local run, the deployed knobs: DUSTY_CUT -4.0 and "youden" as set on the cluster; the
│                                   #   -4.5 / "prior" values in brackets): THE AXIS PART 32 v1 WAS MISSING IS THE ONE PART 22 ALREADY
│                                   #   HAD. kappa_gas is the strongest single divider in the coarse boxes, AUC 0.82 / 0.85 (0.80 / 0.83)
│                                   #   against log M* 0.75 / 0.82 and conc 0.61 / 0.66; it is uncorrelated with M* (rho 0.00) and holds
│                                   #   inside every M* quartile (0.79-0.87) and every gas-particle-count bin (0.81-0.85 at n_gas
│                                   #   50-300), so it is not the few-particle artefact of the m25 core kappa. kappa_gas+M* WITHOUT ANY
│                                   #   DOSE scores 0.87 / 0.93 (0.88 / 0.94), above every dose-bearing pair (duty+conc 0.78 / 0.81,
│                                   #   duty+M* 0.80 / 0.86, duty+kappa_gas 0.83 / 0.88), and the duty adds nothing on top
│                                   #   (duty+kappa_gas+M* 0.87 / 0.92): the dose is already written into the gas kinematics (Part 21's
│                                   #   chain read as a scoreboard); the concentration R80/R20 of Parts 29-31 was a weak proxy of it. The
│                                   #   blind two-cut on (kappa_gas, M*) lands ABOVE the isotropic 1/3: kappa_gas > 0.40 & log M* > 10.40
│                                   #   in cis100 (n 457, 21 % dusty, 76 % recovered, J 0.53; at -4.5: > 0.37 & > 10.52, J 0.49), > 0.32
│                                   #   & > 10.48 in cis50 (J 0.67). RULES cis100 at -4.0: current (duty < 0.82 & conc > 0.30) n 88, 28 %
│                                   #   dusty, 20 % of the dusty, J 0.16; Part 22's red group (duty < 0.82 & kappa_gas > 0.30) n 511, 18
│                                   #   %, 73 %, J 0.46; the score at youden (0.10) n 300, 31 %, 75 %, J 0.62 (at -4.5 / prior: n 435, 44
│                                   #   %, 76 %, J 0.59); cis50 the score n 35, 37 %, 87 %, J 0.75. cis25 (Part 0's 75-row sample, not
│                                   #   the KS-track frame of Parts 22-23): kappa_gas 0.92 and conc 0.88 are the axes, the score reaches
│                                   #   0.97 at -4.0 (0.81 at -4.5, n 12 / 23 dusty - read with care). So Part 22's division is NOT
│                                   #   replaced: it gains the stellar mass as a third axis (Part 33 draws it), and no divider is clean.
│                                   #   Part 33 = PARTS 22 AND 23 UNDER THE DIVISIONS (2026-09-08; cells p33-md / p33-compare, needs
│                                   #   Parts 22 and 32 in memory, Part 23 for the m25 figure): the SAME frames (Part 22's FIG1_22 /
│                                   #   ANC22, Part 23's A23, RF and the K-S arrays - nothing re-measured) re-cut under three divisions
│                                   #   and drawn side by side: (a) as drawn = Part 22's dose x geometry at the medians (kappa_gas in the
│                                   #   coarse boxes, the H2 extent in m25), its own masks and colours; (b) the Part 32 cut = P32_SCORE
│                                   #   at its threshold (S32): predicted dusty / predicted dust-poor on the rows Part 32's frame covers
│                                   #   (every row in m100 / m50, whose Part 22 frames ARE Part 32's rows; in m25 only Part 0's sample,
│                                   #   39 of 175 galaxies / 49 of 262 anchor rows, said in the title; P33_CUT32 "rule" takes Part 32's
│                                   #   blind two-cut rule on P32_PLANE instead, dose + M* only, every row); (c) the best cut = Part 22's
│                                   #   division with a THIRD axis, the stellar mass (P33_THIRD, split at its median or P33_MSPLIT): dose
│                                   #   x geometry x M*, eight cells, the massive half solid with its band, the light half dotted in the
│                                   #   same colours. FIGURE 1 agn_states_division_fans: rows (a, b, c) x one column per box, the Part 22
│                                   #   fan (P22_DUST on t - t_QT), every legend entry with N and the DUSTY SHARE (log M_dust/M* >=
│                                   #   DUSTY_CUT at the anchor: QM's lfd in the coarse boxes, the KS epochs' catalogue masses in m25),
│                                   #   the level of each track before / after QT (P33_WIN) and the fan opening printed. FIGURE 2
│                                   #   agn_states_division_part23_cis25: rows (a, b, c) x (the radial M_dust/M*, the radial M_dust/M_H2
│                                   #   - P33_PROF, per-galaxy sightline medians, bootstrap 16-84 - and the K-S track through P33_KS_STG
│                                   #   on the intrinsic R50(H2) apertures with each group's end R50, unresolved share under Part 23's
│                                   #   two-tier rule and offset from Bigiel+08 under the panel). CSV agn_states_division_compare.csv
│                                   #   (per box / division / group: N, dusty share, fan levels; m25 also the inner-kpc ratios and the
│                                   #   K-S end point). Knobs P33_BOXES, P33_CUT32, P33_THIRD / P33_MSPLIT, P33_WIN, P33_PROF,
│                                   #   P33_KS_STG, P33_NMIN.
│                                   #   RESULT (local run, DUSTY_CUT -4.0 / youden; the -4.5 values in brackets): cis100 the fan opening
│                                   #   after QT (max - min of the group medians in log M_dust/M_H2, 0.6-2 Gyr) is 0.58 dex as drawn,
│                                   #   1.01 (0.85) under the Part 32 cut and 1.30 under dose x kappa_gas x M*: the red cell of (c), low
│                                   #   dose + rotating + massive (n 290), is 24 % (46 %) dusty at -3.4 while all seven other cells are
│                                   #   <= 5 % (<= 14 %) at -4.1 to -4.7; as drawn the red curve was 17 % (33 %) dusty at -3.95. cis50
│                                   #   1.10 -> 1.68 -> 1.78 dex, the red cell 19 % (30 %) dusty, the rest <= 8 %. Before QT the fan
│                                   #   stays closed under every division (one galaxy before quenching, several after). m25: the Part 32
│                                   #   classes (39 galaxies) open 1.03 dex; dose x extent x M* 1.87 dex, but the extended low-dose cells
│                                   #   lose their track after QT (their H2 runs out: fewer than P22_NMIN tracks) - the m25 fan is the
│                                   #   dose halves plus the compact H2 (65-82 % dusty cells). PART 23 (262 anchor rows): the inner-kpc
│                                   #   dust contrast belongs to the EXTENT axis - as drawn, low dose + compact H2 sits at log M_dust/M*
│                                   #   -3.33 in the inner kpc against -4.96 to -5.52 for the other three (M_dust/M_H2 +0.25 vs -1.77 to
│                                   #   -2.37); with the third axis the massive compact low-dose cell keeps -3.93 / -0.66 and the light
│                                   #   one -3.08 / +0.49, so the mass adds nothing to the core signal there; the Part 32 classes on the
│                                   #   49 covered rows separate by 0.65 dex only (-5.03 vs -5.68). In the K-S plane the compact groups
│                                   #   end unresolved (58-79 %) and the extended ones resolved (7-21 %) under any division that keeps
│                                   #   the extent, and the dose ordering of the Bigiel+08 offsets at 'end' (low dose on the relation,
│                                   #   +0.12 to +0.37; high dose below, -0.05 to -0.30) survives every second and third axis; the Part
│                                   #   32 classes, which do not carry the extent, end at the same R50 (2.1 / 2.4 kpc) and the same
│                                   #   offset (-0.04 / -0.05).
│                                   #   Part 34 = THE GEOMETRY AXIS AGAINST THE PARTICLE COUNT (2026-09-09; cells p34-md /
│                                   #   p34-resolution, pure reads of S32 + QM): kappa_rot(gas) is measured from 50-250 gas
│                                   #   particles, so it is scored against the ISOTROPIC SAMPLING NULL - E[kappa] = 1/3 and
│                                   #   s_iso(N) = sqrt(Var(f)/N_eff) = sqrt((4/45)/0.6N) = 0.385/sqrt(N) (P34_VARF, P34_WEFF)
│                                   #   = +-0.044 at the m100 median N = 76. FOUR TESTS, one per figure row: (a) the null
│                                   #   (kappa vs N with the 1/3 +- s_iso band; the sd the null allows, the share above
│                                   #   1/3 + 2 s_iso, the class gap in s_iso units, and whether the false positives are the
│                                   #   few-particle ones), (b) the AUC of raw kappa INSIDE bins of N (untouchable by any
│                                   #   re-weighting: a function of kappa is AUC-identical in a bin), (c) the shrinkage family
│                                   #   1/3 + (kappa-1/3) N/(N+N0) - the EB posterior mean for a noise inflated by
│                                   #   f = N0 sd_int^2 / C_iso^2 - drawn against f, with the deconvolved CEILING of perfect
│                                   #   de-noising and a SHAM driven by N shuffled at fixed log M*, (d) the ladder of every
│                                   #   version against log N_gas / kappa_star / log M* + the out-of-fold scores. RESULT
│                                   #   (DUSTY_CUT -4): the noise is real and small - s_iso 0.044 / 0.045 / 0.025 (m100 / m50 /
│                                   #   m25) against a dusty - dust-poor gap of 0.42 / 0.40 / 0.65 = 9.5 / 8.7 / 26 sigma; the
│                                   #   null would give sd 0.043 against the observed 0.226 and 2.3 % above 1/3 + 2 s_iso
│                                   #   against 33 %; the dust-poor galaxies above kappa 0.5 have median N_gas 68 vs 76 for
│                                   #   their class and the AUC inside N quintiles is flat (0.81, 0.88, 0.77, 0.88, 0.85) - NOT
│                                   #   the m25 few-particle kappa_H2 artefact. Deconvolved, the sampling noise is 3.6 / 4.0 /
│                                   #   2.0 % of the within-class variance, so perfect de-noising is worth +0.004 / +0.005 /
│                                   #   +0.002 AUC; the physical shrinkage (w(N) = 0.95-0.98) gives 0.822 -> 0.823, the same as
│                                   #   the sham. The variants that gain more (significance 0.831, N/(N+800) 0.835 / 0.883) sit
│                                   #   ABOVE the ceiling: at those N0 one assumes a noise 250-500x the isotropic value and the
│                                   #   variable is (kappa-1/3) x N - log N_gas alone scores 0.76 because the gas carries the
│                                   #   dust (near-circular with the label). A de-noised kappa does not replace Part 32's mass
│                                   #   axis either (kappa + log M* 0.874 / 0.925 out-of-fold against 0.820 / 0.836 for any
│                                   #   single de-noised kappa; rho(kappa, log N_gas) = -0.06). WHAT IS WORTH TAKING is the
│                                   #   threshold, not the weighting: cutting at 1/3 + 2 s_iso instead of the median 0.30 lifts
│                                   #   Part 22's red group from 17.8 % to 23.7 % dusty in m100 (25.0 % m50, 87.5 % m25) at
│                                   #   ~70 % recall, and a FIXED cut at 0.42 matches it to within 24 of 1683 galaxies -> set
│                                   #   P22_GCUT = 0.42 if the cleaner red group is wanted. Products
│                                   #   agn_states_kappa_resolution.{png,pdf} + agn_states_kappa_resolution.csv.
│                                   #   Part 35 = WHY kappa_gas AND log M* (2026-09-09; cells p35-md / p35-why, needs Part 0 +
│                                   #   Part 2a for the anchor metallicity, Part 28's frames for the duty axis if present):
│                                   #   the label DECOMPOSED instead of correlated - log M_dust/M* = log[M_dust/(Z M_gas)] +
│                                   #   log Z + log M_gas/M*, an identity that holds galaxy by galaxy (residual = machine
│                                   #   zero), so the dusty - dust-poor gap splits into a DEPLETION, a METALS and a GAS term.
│                                   #   FIG A = the waterfall of the three terms per box + each term against each candidate
│                                   #   cause (kappa_gas, log M*, the duty fraction), one running median per box; FIG B = the
│                                   #   kappa x M* plane coloured by the median depletion in equal-count cells + the ladder of
│                                   #   raw vs rank-partial rho toward the depletion (kappa, M*, Z, Sigma_gas, dose, N_gas).
│                                   #   RESULT (DUSTY_CUT -4): the gap is +2.4 / +2.6 / +2.1 dex and ALL of it is the depletion
│                                   #   - the metallicities match to 0.1 dex (AUC 0.47) and the DUST-POOR are the more gas-rich
│                                   #   (-0.2 dex, AUC 0.33). A dust-poor QG keeps ~1 metal in 1000 in grains, a dusty one 1 in
│                                   #   5. Both axes act on that term and nothing else: partial rho(kappa, depletion | M*, Z,
│                                   #   dose) = +0.47 / +0.37 / +0.31, partial rho(log M*, depletion | kappa, Z, dose) = +0.55 /
│                                   #   +0.55 / +0.32, rho(kappa, log M*) = +0.02 (orthogonal, nearly additive: the quadrants of
│                                   #   the median cut are 1 / 5 / 3 / 21 % dusty in m100). The dose keeps a small partial of its
│                                   #   own (-0.17 / -0.19 / -0.02) and is NOT mediated by the kinematics (rho(dose, kappa | M*)
│                                   #   ~ 0 in m100) - Part 32's "the dose is already written into the gas kinematics" was too
│                                   #   strong. In m25, the box that resolves the gas, the strongest term is Sigma_gas = 0.5
│                                   #   M_gas / pi R50(gas)^2 (+0.39 against +0.28 for M*): the grain-growth density of SIMBA's
│                                   #   dust model, for which the mass axis is the coarse boxes' stand-in. Part 34's resolution
│                                   #   control is repeated for BOTH axes inside quartiles of N_gas. Products
│                                   #   agn_states_why_factors.{png,pdf}, agn_states_why_plane.{png,pdf},
│                                   #   agn_states_why_{gaps,ladder,quadrants}.csv.
│                                   #   Part 36 = WHEN THE DIVISION IS MADE (2026-09-09; cells p36-md / p36-when, needs Part 0;
│                                   #   the kinematics it BUILDS itself): every chain on TWO clocks - t - t_SFT and t - t_x1,
│                                   #   the FIRST snapshot at which the X-ray channel fires along the main branch - followed
│                                   #   through both, split by the label it ends with. THE SIDECAR: the boxes notebook's
│                                   #   histories carry no kinematics, so this part walks the SAME main branch (caesar
│                                   #   tree_data/progen_galaxy_star, most massive progenitor, the histories' own snapshot
│                                   #   ladder) and reads kappa_rot(gas), kappa_rot(*), radii.{gas,stellar}_half_mass,
│                                   #   masses.HI and ngas from each catalogue -> agn_states_kin_histories_{box}.hdf5 (one file
│                                   #   open per snapshot, ~100 per box, cached; P36_BUILD False = read only what is there;
│                                   #   P36_CATDIR overrides the config's catalog_dir). It is matched to the histories on the
│                                   #   snapshot numbers and the galaxy ids, so a partial build leaves NaNs, never a misaligned
│                                   #   branch. FIGS: agn_states_when_{sft,x1} (rows kappa_rot(gas) / the metal share in grains /
│                                   #   the gas / the sSFR, columns box, one median track per outcome class),
│                                   #   agn_states_when_power (the AUC toward the FINAL label at every epoch: when each axis
│                                   #   becomes a divider) and agn_states_when_plane (the kappa x log M* plane at four epochs,
│                                   #   the class medians joined by arrows). RESULT: t_x1 - t_SFT has median +0.07 Gyr with 56 %
│                                   #   inside a quarter Gyr - the gas-poor gate opens AS the galaxy truncates, so the two
│                                   #   clocks are one; before it the two classes are the same galaxies (metal share -0.4 for
│                                   #   both, AUC 0.50 at -1 Gyr, the same gas and sSFR); after it the dust-poor lose ~2.5 dex
│                                   #   of depletion in ~1.5 Gyr while the dusty stay flat; kappa_gas does the same and not
│                                   #   before it (both classes are discs at -1 Gyr, AUC 0.41; the dusty re-settle to 0.8-0.9,
│                                   #   the dust-poor fall to ~0.2 = +35 s_iso on Part 34's yardstick) - m25 so far, the coarse
│                                   #   boxes need their sidecar built. log M* is the exception: it already separates them 3 Gyr
│                                   #   BEFORE the truncation - the pre-existing half of the division, the geometry the made
│                                   #   half. Caveats: the depletion track is the label itself at the anchor (what it dates is
│                                   #   the START of the divergence), the late epochs are the subset whose anchor is that late,
│                                   #   rows are anchors not distinct galaxies. Products above + agn_states_when_{tracks,power,
│                                   #   clocks}.csv.
├── dusty_qgs_kappa_selection.ipynb  # THE ORDERED RESTATEMENT of agn_states_dusty_vs_pure's answer (2026-09-09,
│                                   #   47 cells, Parts 0-18 with a Part 4b, deployed): ONE sample, ONE selection,
│                                   #   and every part below cut on it -> output/box_resolution/kappa_selection/
│                                   #   (prefix kappa_, so nothing of the old notebook is overwritten).
│                                   #   THE SELECTION (Part 2), BOTH AXES (2026-09-09, the user's "we keep
│                                   #   kappa+mass as default"): selected <=> kappa_rot(gas) > 1/3 + ROT_SIG
│                                   #   s_iso(N_gas) AND log M* > ROT_MASS (= 10.5, Part 0; None recovers the purely
│                                   #   kinematic rule everywhere). One helper, rot_rule() in Part 0, is called by
│                                   #   Part 2 for the sample and by Parts 6-8 for the K-S products' own larger
│                                   #   anchor set, so the division cannot drift between them; the second class is
│                                   #   "the rest" (a galaxy is in it for either reason). NOTE ROT_MASS 10.5 is the
│                                   #   SELECTION cut, MASS_MIN 10.25 the SAMPLE cut - different things. s_iso(N) =
│                                   #   sqrt((4/45)/(0.6 N)) = 0.385/sqrt(N) - the particle count enters as the
│                                   #   UNCERTAINTY of the measurement, never as a factor on kappa itself (that
│                                   #   would put the gas mass, i.e. most of the dust label, inside the morphology
│                                   #   indicator - Part 34 of the old notebook). The dust cut survives only as a
│                                   #   LABEL to score against. Part 2 draws the kappa-N plane with the cut curve,
│                                   #   the significance z = (kappa-1/3)/s_iso against the unit normal the isotropic
│                                   #   null predicts, the dusty fraction in equal-count z bins (Wilson 68 %) and
│                                   #   the cut scanned over 0-5 sigma. RESULT (DUSTY_CUT -4, MASS_MIN 10.25,
│                                   #   AGE_MIN 3): sd(kappa) is 5-9x the shot noise, 33 / 32 / 39 % of the sample
│                                   #   is above 2 s_iso against the 2.3 % an isotropic population would give. With
│                                   #   the mass axis ON the selected class is 373 / 44 / 19 galaxies carrying dusty
│                                   #   | selected 23.9 / 27.3 / 31.6 % at 71 / 80 / 50 % completeness; WITHOUT it,
│                                   #   557 / 64 / 29 at 18.3 / 18.8 / 34.5 % and 82 / 80 / 83 % - the cell prints
│                                   #   that trade box by box next to the rule written without the mass term, so the
│                                   #   price is never hidden. The median split keeps the completeness and loses the
│                                   #   purity; a FIXED cut at the same median level differs by 24 of 1683 galaxies,
│                                   #   which is exactly what the N dependence is worth.
│                                   #   THE SECOND AXIS, drawn as its own figure (kappa_selection_mass, added
│                                   #   2026-09-09 on the user's question "is the division less prominent because
│                                   #   the mass channel is missing?"): the z x log M* plane with the dusty share of
│                                   #   each quadrant written in it and a hand-rolled STRATIFIED 5-fold logistic
│                                   #   out-of-fold AUC on rank-transformed columns in the title. ANSWER: yes in the
│                                   #   coarse boxes, no in m25. Out of fold z / z+M* = 0.829 / 0.850 (m100), 0.852 /
│                                   #   0.902 (m50), 0.892 / 0.886 (m25); quadrant dusty shares (lo/lo -> hi/hi)
│                                   #   1 / 3 / 7 / 28 % (m100), 0 / 5 / 10 / 26 % (m50), 8 / 0 / 31 / 38 % (m25).
│                                   #   So M* triples the purity of the selected group in the coarse boxes at ~2/3
│                                   #   of the completeness and does NOTHING in the box that resolves the gas (AUC
│                                   #   0.53) - exactly Part 35's reading, that Sigma_gas is the physical axis and
│                                   #   M* the coarse boxes' stand-in. The notebook TAKES IT ANYWAY (ROT_MASS 10.5,
│                                   #   2026-09-09) and states the m25 cost every time it prints the rule; the
│                                   #   figure now draws BOTH cuts, shades the selected quadrant and keeps the box
│                                   #   median as a dotted line for scale.
│                                   #   Part 3 = ONE pass over the main-branch chains (the tracks on BOTH clocks,
│                                   #   t - t_SFT and t - t_x1, plus the AGN clocks, the dose and the post-truncation
│                                   #   dust budget per galaxy) and THE FIGURE: log M_dust/M* on the truncation clock
│                                   #   per class, one row per box, with the ANCHOR distribution of the same quantity
│                                   #   beside it on the same axis -> kappa_clock_lfd.{png,pdf} + kappa_clock.csv +
│                                   #   kappa_branch_galaxies.csv. The classes are IDENTICAL for the 3 Gyr before the
│                                   #   truncation and part after it (anchor gap +1.49 / +1.11 / +1.29 dex, AUC 0.81 /
│                                   #   0.79 / 0.69 with the mass axis on). It also builds the four SURFACE-DENSITY
│                                   #   tracks Part 4b needs (lsig_star, lsig_gas, lsig_h2, lr_star), Sigma_X =
│                                   #   0.5 M_X / (pi R50_X^2) with the stars on R50(*) and the gas AND H2 both on
│                                   #   R50(gas) - caesar carries no H2 radius, so Sigma_H2 is an ISM density, not a
│                                   #   resolved measurement (the resolved statement is Parts 5B / 7A). It also carries the KINEMATIC SIDECAR: the histories hold masses
│                                   #   only, so kappa_rot(gas) and the half-mass radii along the branch come from
│                                   #   agn_states_kin_histories_<box>.hdf5 - the SAME cached product Part 36 of
│                                   #   agn_states_dusty_vs_pure builds, read if present and walked out of the caesar
│                                   #   catalogues (progen_galaxy_star, most massive progenitor) if not, matched to
│                                   #   the histories on snapshot numbers and galaxy ids (P3_KIN_BUILD, P3_CATDIR).
│                                   #   Part 4 = the channel, the truncation and the counterfactual, TWO rows: (1)
│                                   #   t_x1 - t_SFT (median +0.07 Gyr, 56 % within a snapshot) drawn WITH the two
│                                   #   other clocks the firing needs - t_gate (f_gas < XRAY_FGAS_MAX) and t_big
│                                   #   (M_BH > 10^7.5). The channel fires +0.00 Gyr after its own gas gate opens
│                                   #   (90 % within 0.25 Gyr) and +1.24 Gyr after its mass criterion was met, so the
│                                   #   coincidence CANNOT be read as the cause of the truncation and the cell says
│                                   #   so; (2) on the t_x1 clock, each galaxy referred to its own value at the
│                                   #   firing: log sSFR, log f_gas and log M_dust/M* in dex, the COMPACTNESS of the
│                                   #   cold gas -log R50(gas)/R50(*) with them and kappa_rot(gas) itself on its own
│                                   #   right-hand axis (dashed) - the sSFR and the gas are already 1.2 dex down, the
│                                   #   DUST is flat and then collapses 2 dex, and the axis the notebook selects on is
│                                   #   itself moving at the firing. THE CONTROL is cis50nox, the same box with the
│                                   #   X-ray heating compiled out (its t_x1 = the WOULD-BE firing): its galaxies
│                                   #   quench on the same clocks and their dust does not move: the rotating class's
│                                   #   net is -0.55 dex there against -2.7 / -3.3 / -1.9 in the fiducial boxes, and
│                                   #   169 of 178 of its quenched chains are "rotating" (95 %). The post-truncation
│                                   #   budget is PRINTED, not drawn (the user dropped the row): net AUC 0.75 / 0.73 /
│                                   #   0.77 and a real rebound (>= 0.5 dex recovered from the post-SFT minimum) in
│                                   #   51 / 65 / 58 % of the rotating class against 19 / 48 / 34 %, so the separation
│                                   #   is slower destruction AND re-growth; the share that first hit the zero-dust
│                                   #   floor (and so has its rebound measured from P9_FLOOR) is printed with it ->
│                                   #   kappa_cause.{png,pdf,csv}.
│                                   #   Part 4b (NEW 2026-09-09, the user's "rotating discs seem to have more compact
│                                   #   ISM - did they evolve differently?") = the gas rotation and the ISM / stellar
│                                   #   SURFACE DENSITIES of the two classes, five quantities (kappa_rot(gas),
│                                   #   log Sigma_H2, log Sigma_gas, log Sigma_*, -log R50(gas)/R50(*)) on both sides
│                                   #   of the same clock: FIG 1 the clocks on t - t_SFT (rows = boxes, columns =
│                                   #   quantities; read the LEFT half - apart before 0 = inherited, parting at 0 =
│                                   #   made), FIG 2 the DISTRIBUTIONS AT THE TRUNCATION with an arrow from each
│                                   #   class's median there to its median at the anchor (the shift written on it),
│                                   #   FIG 3 the class-median GAP against the clock with a joint-bootstrap band
│                                   #   (kappa on its own right axis) -> kappa_sigma_{clocks,atsft,when}{.png,.pdf}
│                                   #   + kappa_sigma.csv. The cell prints each class's median at -2 / 0 / +2 Gyr and
│                                   #   at the anchor with the AUC between them, three CONTROLS under the table
│                                   #   (log M*, R50(*), R50(gas) - the numbers every Sigma is built from) and a
│                                   #   per-box verdict splitting the five into "already apart 2 Gyr before" and
│                                   #   "apart only at or after". m25 (the only box with a sidecar so far): NOTHING is
│                                   #   inherited - kappa 0.54 -> 0.92, Sigma_H2 0.42 -> 0.65 (+0.28 dex at the
│                                   #   anchor), Sigma_gas 0.44 -> 0.63, Sigma_* 0.46 -> 0.75, while the SIZE RATIO
│                                   #   never separates at all (AUC 0.43-0.48 everywhere). CAVEAT printed with it:
│                                   #   log M* is the rule's own second axis, so Sigma_* is half selection by
│                                   #   construction and only its pre-truncation column is a statement.
│                                   #   Part 5 = the m25 close-up on the two classes
│                                   #   (Parts 13B/13C + 14 of the old notebook re-cut): the critical points in the
│                                   #   two spherical zones, the radial anatomy per annulus and the K-S plane ->
│                                   #   kappa_m25_{sequences,profiles,ks}{.png,.pdf} + kappa_m25_closeup.csv.
│                                   #   kappa_rot(H2) is the SAME at ignition (0.70 both) and separates only at 'end'
│                                   #   (0.51 vs 0.22), the dust offset is present at EVERY annulus while the stellar
│                                   #   age and kappa_rot(*) are identical.
│                                   #   Parts 6, 7, 8 = the old notebook's Parts 22, 23, 24 wholesale, with ONE
│                                   #   change: the division is Part 2's selection, both axes (P6_GEOM "kappa_z_cat"
│                                   #   cut at P6_GCUT = ROT_SIG in EVERY box, not at each box's median, ANDed with
│                                   #   ROT_MASS; "kappa_z_H2" is the same construction on the resolved core H2 and is
│                                   #   what Fig 2 (b) walks through the stages, since a catalogue quantity exists
│                                   #   only at the anchor). THE AGN DOSE IS GONE AS A DIVISION (2026-09-09, the
│                                   #   user's "fully remove the dose division"): the four-way dose x class grid,
│                                   #   P6_SPLIT, P6_DOSE and the panel that scored the dose as an alternative
│                                   #   divider are all removed; E_x / f_x / f_x_post are still computed and written
│                                   #   to the CSVs, and nothing is ever split by them. Fig 1 (b) is now THE RULE
│                                   #   DRAWN: the dust against the kinematic significance in equal-count bins (as
│                                   #   many as each box can fill at P6_NMIN a bin), one line per side of the mass
│                                   #   cut, on a linear axis clipped to P6_ZLIM +-25, with the significance cut
│                                   #   vertical - the dust rises along the significance WITHIN each mass half and the
│                                   #   two halves are offset, which is the two axes of the rule read separately. The
│                                   #   full division moves dh2 by +1.58 / +0.84 / +0.98 dex (rho(significance, dh2)
│                                   #   +0.39 / +0.26 / +0.49, rho(log M*, dh2) +0.46 / +0.59 / +0.27). In the m25
│                                   #   K-S frame the selected class is small (21 of 174), so some Fig 3 panels fall
│                                   #   below P6_NMIN3 and simply do not draw. The knobs keep their positions one for
│                                   #   one under the P6_/P7_/P8_ prefixes, so a knob tuned in agn_states_dusty_vs_pure
│                                   #   carries across by renaming -> kappa_dustclock_boxes, kappa_dustsplit_cis25,
│                                   #   kappa_dustgas_ssfr_cis25 (+ the per-box CSVs); Part 7 ->
│                                   #   kappa_resolved_{profiles,ks}_cis25 + kappa_resolved_cis25.csv (the catalogue
│                                   #   class predicts the RESOLVED kappa_rot(H2) at every radius, 0.4-0.6 against
│                                   #   0.1-0.2, which is not circular: different tracer, different notebook);
│                                   #   Part 8 -> kappa_gallery_*_cis25 (needs the reduced cutouts + simbanator).
│                                   #   PARTS 9-11 (NEW 2026-09-09, the user's "let's consider ISM and CGM too, just
│                                   #   for the 25mpc run"): the 25 Mpc box ONLY - the box that resolves the gas and
│                                   #   the only one with particle cutouts along the branch.
│                                   #   Part 9 = THE BRANCH INDEX + three clocks. The index is the most-massive-
│                                   #   progenitor walk (anchor_snap, gal_id) -> gx(snap) from caesar
│                                   #   tree_data/progen_galaxy_star where the catalogue has it and from
│                                   #   output/cis25/progen_links/ (find_progen_m25_job.py) otherwise, cached as
│                                   #   kappa_m25_branch_ids.csv; it is what lets Part 10 find a PROGENITOR's particle
│                                   #   file, and it makes the de-duplication EXACT (an anchor that LIES ON a later
│                                   #   anchor's branch is the same galaxy - not Part 6's M*(t) fingerprint, which
│                                   #   cannot de-duplicate a branch starting after its reference times). THE CACHE IS
│                                   #   INCREMENTAL AND SELF-CHECKING (2026-09-09, the user's "why on part 12 do the
│                                   #   rotators go from 34 to 10"): a chain the cache does not carry is walked and
│                                   #   APPENDED, and what it covers is printed every run. It had to become that,
│                                   #   because the walk used to STOP at the first snapshot where every chain then
│                                   #   alive died - the anchors below that gap never joined it - and a cache written
│                                   #   that way (or written for a narrower sample) then truncated Parts 9-13 to its
│                                   #   own subset with no warning: 87 of the 162 m25 anchor rows, 5 of the 10
│                                   #   anchors, 49 galaxies. Complete, the same sample is 162 rows -> 79 galaxies
│                                   #   (P9_DEDUP False keeps all 162 chains); the reduced particle files were
│                                   #   already there for them (1793 of the 3046 sit on nodes the truncated index
│                                   #   never listed). THE DE-DUPLICATION ITSELF is not a loss of statistics - a
│                                   #   galaxy is a quenched anchor twice in the median, seven times at most, and
│                                   #   would otherwise enter one median that often - but it does decide WHICH anchor
│                                   #   the class is read at, and kappa_rot(gas) is measured on the gas that is left,
│                                   #   so the latest anchor is systematically the most isotropic reading of one and
│                                   #   the same object: P9_CLASS = "latest" (the anchor kept, 13 rotating of 79) |
│                                   #   "ever" (rotating at ANY anchor it is in the sample at, 27) | "sft" (at the
│                                   #   anchor nearest its own truncation, 19) - all three counts are printed
│                                   #   whichever is in force. THE FIGURE, 2x2 on t - t_SFT: (a) the X-RAY MODE AS AN OCCUPANCY -
│                                   #   the share of each class with the channel firing, Wilson 68 %; (b) lfd; (c)
│                                   #   log M_dust/M_H2; (d) the DUST-TO-METAL ratio M_dust/(M_dust + Z M_gas). The
│                                   #   point of (b)-(d) is the DENOMINATOR: the metal share falls only under grain
│                                   #   DESTRUCTION and is flat under dilution / outflow / pristine accretion.
│                                   #   Neither ratio is floored (Part 3's lfd is) and the cost is printed (0.4 % of
│                                   #   the branch steps). RESULT: the channel is on in 15 % of BOTH classes at
│                                   #   -1 Gyr and ~90 % of both within a snapshot of SFT (AUC 0.46-0.54 throughout -
│                                   #   the mode is switched BY the truncation, it is not distributed differently
│                                   #   between the classes); all three dust ratios fall together after 0 (the metal
│                                   #   share -0.4 -> -1.5/-2.1 dex by +2 Gyr) -> kappa_m25_clocks{.png,.pdf,.csv}.
│                                   #   Part 10 = WHERE THE DUST IS AND HOW IT MOVES, from the reduced particle files
│                                   #   (ISM *and* CGM inside 100 kpc, with the caesar membership flag, m_dust and
│                                   #   the peculiar velocity per particle). THE PARTITION, one particle in exactly
│                                   #   one zone: core = the inner half of the member gas (r < R50_gas), outskirt =
│                                   #   the outer half, CGM = gas in the aperture NOT on the member list. The ISM
│                                   #   boundary is measured BOTH ways and cached, so P10_SPLIT flips between the
│                                   #   half-mass radius (default) and a fixed 3 kpc (Parts 6-7's ap3kpc) with no
│                                   #   re-read - the default is scaled because these galaxies are compact (R50 of the
│                                   #   member gas ~1.7 kpc at the truncation, so a fixed 3 kpc leaves the outskirt
│                                   #   EMPTY in a fifth of the nodes; the cell prints both). v_r = (v_pec - v_sys).
│                                   #   r_hat + H(z) r, positive outward, v_sys = the mass-weighted velocity of the
│                                   #   MEMBER STARS (the gas is what may be flowing). EVERYTHING IS MEASURED TWICE
│                                   #   over the same particles - weighted by m_dust and by m_gas - which IS the
│                                   #   question "is the dense, dusty gas an exception?". Figures: 1 the BUDGET
│                                   #   (M_dust,CGM / M_dust,tot on the clock, the same share of the gas as control,
│                                   #   the three zones' dust masses) -> kappa_m25_dust_budget; 2/3 the FLOW (v_r and
│                                   #   the mass fraction with v_r > P10_VFAST = 100 km/s, per zone, per class, dust-
│                                   #   then gas-weighted) -> kappa_m25_flow_{dust,gas}; 4 the FLUX through a fixed
│                                   #   shell (Mdot_out for dust and gas + the DUST-TO-GAS RATIO OF WHAT LEAVES
│                                   #   against the ISM it comes from) -> kappa_m25_flux_shell. The clock is BINNED
│                                   #   (P10_TB, 0.5 Gyr): a node exists only where a branch snapshot has a cutout
│                                   #   (median ~28 per galaxy in m25, ~1200 distinct files), each galaxy contributes
│                                   #   its own median inside a bin, the class median is over galaxies. The census is
│                                   #   CACHED per (snap, gx) in kappa_cis25_particle_zones.fits (~7 GiB read once;
│                                   #   only MISSING ROWS are filled, so adding a column needs the file deleted) ->
│                                   #   + kappa_m25_particle_{clock,nodes}.csv.
│                                   #   Part 11 = WERE THE ANCHOR-ROTATORS ALREADY DENSER IN H2 AT THE TRUNCATION?
│                                   #   Sigma_H2 = He x 0.5 M_H2 / (pi R50(H2)^2) - the K-S abscissa, from the m25
│                                   #   K-S track products - at every critical point, with M_H2 and R50(H2) reported
│                                   #   SEPARATELY (a density is a ratio) and the stellar mass CONTROLLED FOR: every
│                                   #   AUC is printed raw and pooled within P11_MTERC = 3 terciles of log M* at the
│                                   #   anchor, because Part 2's rule already contains log M* > 10.5. Where Part 10's
│                                   #   census exists the same density is measured independently from the member H2
│                                   #   PARTICLES at the nearest branch node (within 0.25 Gyr) and drawn dashed.
│                                   #   ANSWER: NO - at sft 2.66 vs 2.86 dex, AUC 0.53 raw / 0.51 at fixed M*,
│                                   #   p = 0.84 (28 galaxies, 9 / 19: the intersection of Part 0's sample with the
│                                   #   K-S products' own anchor set), and nothing separates at any critical point ->
│                                   #   kappa_m25_sigma_h2_at_sft{.png,.pdf,.csv}. Consistent with Part 4b: the two
│                                   #   classes are the same galaxies until the truncation happens to them.
│                                   #   PART 12 (NEW 2026-09-09, the user's "we need to look at the radial density
│                                   #   profiles. I suspect the rotating gas sources have a hole in the center"): THE
│                                   #   RADIAL PROFILES AT THE CRITICAL POINTS, THE CENTRAL HOLE, AND WHAT IT DOES TO
│                                   #   THE DUST. Parts 5/7 read radii through the m25 notebook's four coarse annuli
│                                   #   (0-1, 1-3.2, 3.2-10, 10-32 kpc) for the K-S anchor set; this goes back to the
│                                   #   particles - 14 log bins from 0.25 to 32 kpc, every retained branch. THE FOUR
│                                   #   CRITICAL POINTS, all off the SAME chain Part 9 built (which now also stores the
│                                   #   per-step AGN state code `st` and the BHAR `mdot` - re-run Part 9 once): agn_ign
│                                   #   = the first step whose BHAR reaches P12_AGNFRAC = 0.5 of its own pre-SFT peak
│                                   #   (the m25 Part 2d windows rule, rebuilt from the histories), xray_on = the first
│                                   #   step with the X-ray channel firing (Part 0's agn_states), sft, anchor. Median t
│                                   #   - t_SFT: -1.1 / -0.2 / 0 / +2.6 Gyr. THE GEOMETRY: FACE-ON in the plane of the
│                                   #   MEMBER GAS's own angular momentum, R cylindrical, only the slab |z| < P12_ZMAX
│                                   #   = 10 kpc (P12_FRAME switches to the stellar frame or off); the misalignment to
│                                   #   the stellar body is printed - it is large, which is this notebook's premise,
│                                   #   not a bug. THE HOLE: hole = log10(Sigma_peak / Sigma(<P12_RIN = 1 kpc)), the
│                                   #   peak outside the aperture against the mean inside it; a ring scores large and
│                                   #   positive, a declining profile negative (the aperture is a cumulative mean, the
│                                   #   peak an annulus, so the two regimes do not overlap and the 0.3 dex threshold
│                                   #   sits clear of both). TWO GUARDS make it a measurement and not sparseness:
│                                   #   Sigma_in is floored at ONE PARTICLE, and a galaxy is classified only where
│                                   #   n_exp (the count the peak would put inside the aperture) reaches P12_NEXP = 5.
│                                   #   On the 35 m25 files tested the metric spans -1.07 to +1.65 dex (median +0.22,
│                                   #   17/35 above the cut) and the strong holes hold ~10 particles inside 1 kpc where
│                                   #   the peak would put 200-750, with 300-2900 member gas particles in the galaxy -
│                                   #   a 20-50x deficit, not Poisson. FIGURES: 1 = 3 rows (Sigma_gas, Sigma_H2,
│                                   #   Sigma_dust) x 4 stages, class medians + bootstrap band + the individual
│                                   #   galaxies, an EMPTY bin drawn at the per-quantity floor (a hole is exactly where
│                                   #   Sigma -> 0, so empty bins cannot be dropped) and the softening shaded ->
│                                   #   kappa_m25_stage_profiles; 2 = the hole itself (the two groups' profiles, the
│                                   #   metric's ECDF per class, the hole fraction per class at each stage with Wilson
│                                   #   bars) -> kappa_m25_hole; 3 = THE POINT: the DUST CLOCKS split by the hole at
│                                   #   P12_HOLE_AT = "sft" - lfd, the dust-to-metal ratio, f_H2, plus the hole x
│                                   #   kinematic class cross, so it can be read off whether the dust follows the hole
│                                   #   or the rotation -> kappa_m25_hole_clock + kappa_m25_hole{,_galaxies}.csv.
│                                   #   CACHE: kappa_cis25_profiles.npz stores only MASSES AND COUNTS per bin, on BOTH
│                                   #   grids (physical kpc and R/R50*, P12_XAXIS flips the figure), for the member gas
│                                   #   and for every particle in the slab, so every threshold and every surface
│                                   #   density is recomputed at draw time; the bins and the geometry are stored with
│                                   #   it and checked on load. By default only the ~4 stage nodes per galaxy are read
│                                   #   (a few hundred MB); P12_ALLNODES = True adds the rest incrementally.
│                                   #   Part 12b (NEW 2026-09-09, the user's "rotation seems to be the best division
│                                   #   for observing the different mass winds and cgm effects... any idea why?") =
│                                   #   THE OPENING ANGLE OF THE OUTFLOW, the mechanism test: the classes are
│                                   #   identical before the truncation (Part 4b) and get the same dose (Part 6), so
│                                   #   what differs is the COUPLING, and the geometric form of that claim is about
│                                   #   ANGLES. mu = |r_hat . n_hat| about the MEMBER GAS's angular-momentum axis;
│                                   #   solid angle is uniform in mu, so a profile over its own mean IS dM/dOmega in
│                                   #   units of the isotropic expectation. The wind = v_r >= P12B_VFAST (Part 10's
│                                   #   v_r verbatim) in P12B_SHELL, gas- AND dust-weighted. THE CONTROL IS THE
│                                   #   RESERVOIR: a disc starts planar whatever its wind does, so the statement is
│                                   #   the normalisation-free f_esc(mu) = M(fast, mu) / M(all gas, mu) - the share of
│                                   #   the gas AT THAT ANGLE that is leaving - and its polar/plane ratio R_esc (1 =
│                                   #   no preferred direction, > 1 = through the poles), which is what Fig 3 splits
│                                   #   on (P12B_SPLITON, also "fpol" / "dfpol"). BESIDE IT A FRAME-FREE MEASUREMENT,
│                                   #   because n_hat is noise exactly where there is no disc: the direction tensor
│                                   #   T_ij = sum w r_i r_j of the wind (eigenvalues 1/3,1/3,1/3 isotropic ->
│                                   #   (1,0,0) collimated -> (1/2,1/2,0) planar), the ANGLE between its major axis
│                                   #   and n_hat (60 deg for a random axis, drawn as the null) and the mass-weighted
│                                   #   HALF-OPENING ANGLE (60 deg isotropic). Figures: 1 = 3 rows x 4 stages, the
│                                   #   reservoir / the wind / f_esc -> kappa_m25_outflow_angles; 2 = the collimation
│                                   #   2x2 (f_pol wind vs reservoir per stage, the alignment ECDF against the random
│                                   #   axis, the opening angle per stage, the dust against the gas f_esc) ->
│                                   #   kappa_m25_outflow_collimation; 3 = the dust clocks split by the geometry with
│                                   #   the geometry x class cross -> kappa_m25_outflow_clock +
│                                   #   kappa_m25_outflow{,_galaxies}.csv. Cache kappa_cis25_outflow_angles.npz holds
│                                   #   the whole (mu x r x v_r) cube per node, gas / dust / counts / momentum, so
│                                   #   every threshold but the wind definition is recomputed at draw time.
│                                   #   CLUSTER RESULT (2026-09-09, 257 stage nodes, 11 s): the rotators ACQUIRE the
│                                   #   polar escape channel across the truncation - R_esc 0.99 at the AGN ignition
│                                   #   (the rest 1.16), then 1.67 / 1.60 / 1.82 at xray_on / SFT / anchor against
│                                   #   1.02 / 1.11 / 1.03, AUC 0.63 / 0.70 / 0.75 (p 0.15 / 0.03 / 0.007).
│                                   #   Part 12c (NEW 2026-09-09, the user's "is the deviation from isotropy an
│                                   #   artifact of the low statistics? and doesn't SIMBA fix the collimation
│                                   #   anyway?") = THE NULL AND THE PRESCRIPTION, a pure read of 12b's cache.
│                                   #   (1) THE NULL, three ways, none of which re-reads a file (the cube carries
│                                   #   COUNTS as well as masses): the per-galaxy binomial z of the polar wind count
│                                   #   against Binomial(N_wind, p_reservoir) - the disc's geometry already inside p,
│                                   #   an UPPER bound since wind particles are launched in events; the count-weighted
│                                   #   R_esc beside the mass-weighted one; and the controls rho(R_esc, N_wind) vs
│                                   #   rho(R_esc, L-coherence). Plus a hand-rolled BENJAMINI-HOCHBERG over the 3 x 4
│                                   #   contrast grid, and the STRUCTURED prediction tested once (absent at ignition,
│                                   #   present at SFT + anchor). ANSWER: not an artefact - N_wind is ~7000 in the
│                                   #   median, the rotators' median z is +22 at xray_on (77 % of them individually
│                                   #   past z = 3), counts and masses agree to 0.01, rho(R_esc, N_wind) = -0.16..0.04
│                                   #   (p > 0.16) while rho(R_esc, L-coherence) = +0.28..+0.46 (p 0.016 -> 0.0004);
│                                   #   BUT no single cell of the 12 survives BH at q = 0.05 (best p 0.0067 vs the
│                                   #   0.0042 threshold), so the class contrast at ONE stage is suggestive only. What
│                                   #   carries it is the time course: rotators 0.99 -> 1.68 (AUC 0.76, p 0.013), the
│                                   #   rest 1.16 -> 1.06 (p 0.11).
│                                   #   (2) THE PRESCRIPTION. SIMBA's kinetic AGN feedback is ALWAYS bipolar, ejected
│                                   #   +-L of the ~256 gas elements nearest the BH, and that axis reorients with them
│                                   #   (Dave+19) - so "is the wind collimated" is not a question, and 12b never asks
│                                   #   it. What is NOT prescribed is whether the launch axis points OUT OF the ISM
│                                   #   disc and whether it STAYS there, which is what R_esc measures. Part 12b's
│                                   #   census therefore also stores L_in over the P12B_NLAUNCH = 256 NEAREST gas
│                                   #   elements (the code's own rule, not a fixed radius) with the RADIUS they reach
│                                   #   - in these quenched galaxies the centre is empty (4 test files: NO member gas
│                                   #   inside 1 kpc at all, the 256 reaching 2.7-3.2 kpc), so the axis the jet
│                                   #   follows is set kiloparsecs out, which is Part 12's hole re-entering. 12c then
│                                   #   scores the angle L_in vs the ISM disc and the WANDER of L_in between
│                                   #   consecutive stages (same-snapshot pairs dropped) -> kappa_m25_outflow_null
│                                   #   {.png,.pdf,.csv}, a 2x2: the per-galaxy z ECDF, R_esc vs L-coherence, R_esc on
│                                   #   the stage clock, and the launch axis's wander + tilt against the random-axis
│                                   #   null. WHAT THE CACHED CENSUS ALREADY SAYS (2026-09-10, the user's "does
│                                   #   R_esc ~ 2 mean the jets are perpendicular to the disc?"): NO - and the earlier
│                                   #   "coincidence between the cone and the disc" phrasing is WITHDRAWN. The wind's
│                                   #   own mass-weighted axis sits 60 / 75 / 64 / 71 deg from the disc normal over the
│                                   #   four stages for the rotators, where a RANDOM axis gives 60 (at xray_on the KS
│                                   #   against that null gives p = 0.011, on the IN-PLANE side); only 8 % of them lie
│                                   #   within 30 deg of the pole against 13 % for a random axis; lam1 = 0.49-0.50 is
│                                   #   the PLANAR signature (1/2, 1/2, 0), not the collimated one (1, 0, 0), with a
│                                   #   43-47 deg half-opening against 54.7 isotropic; and the polar half-sky still
│                                   #   takes only 0.44 of the fast mass (0.41 dust-weighted). R_esc > 1 comes from the
│                                   #   DENOMINATOR: the rotators' RESERVOIR polar share is 0.26-0.31 while their
│                                   #   wind's is 0.44 - the wind is MORE ISOTROPIC THAN THE MEDIUM it sits in, and
│                                   #   against a planar reservoir that reads as a polar channel. The rest's R_esc ~ 1
│                                   #   is forced, not physical: their reservoir share is 0.49-0.52, so they have
│                                   #   neither pole nor plane and the ratio is 1 by construction. THE CLAIM, for the
│                                   #   paper: the wind's geometry is DECOUPLED from the disc's - in a rotator that
│                                   #   decoupling opens an escape channel because the disc is planar, in the rest
│                                   #   there is nothing to be decoupled from. NOT the rotation-support tautology
│                                   #   (v_r = 0 in a supported plane by construction): split by shell the rotator
│                                   #   excess PEAKS IN THE CGM - 0.5-3 kpc ~ 1.0, 3-10 1.3-2.1, 10-30 2.25-2.36,
│                                   #   30-100 1.44-1.69 at xray_on/sft - so it is not manufactured at disc radii.
│                                   #   Caveat: the tensor runs over ALL fast gas in 3-100 kpc, the ACCUMULATED
│                                   #   outflow rather than the instantaneous jet; L_in (the 256 nearest, Part 12b's
│                                   #   new columns) is the closest proxy to the live axis and 12c block (e) scores it.
│                                   #   PART 12d (NEW 2026-09-10, cells p12d-md/p12d-accretion, the user's hypothesis:
│                                   #   "if a galaxy's AGN is accreting fast it might go fast from jet-mode to x-ray,
│                                   #   starting the quenching and preserving the disc; if not, the jet mode stays
│                                   #   longer, precesses more, and the disc gets disturbed before the x-ray fires"):
│                                   #   THE ACCRETION HISTORY BEHIND THE SPLIT. What 12b/12c leave open is a mechanism;
│                                   #   what they fix is a TIMING - the classes are identical at the ignition, the JET
│                                   #   LEG is the SAME length in both (t_x1 - t_ign = 0.58 vs 0.51 Gyr, p 0.63, so the
│                                   #   precession time is equal and cannot be what separates them), and the whole
│                                   #   divergence is AFTER the gate opens (t_SFT - t_x1 = +1.47 Gyr for the rotators
│                                   #   against -0.04 for the rest, p 0.024: the rest truncate AS the channel opens).
│                                   #   So the upstream variable neither part measures is the accretion rate itself.
│                                   #   ONE PASS over ism_prediction_histories_{box}.hdf5 (the file Parts 3 and 9 already
│                                   #   read), restricted to the Part 9 chains, with Part 0's agn_states() VERBATIM so
│                                   #   the gates are the ones every other Part dates its stages with -> per galaxy:
│                                   #   log f_Edd at the ignition / on the jet leg / after the gate / at its pre-SFT
│                                   #   peak, the time above the jet threshold and in the jet state, log M_BH at the
│                                   #   ignition and its growth gate -> anchor, and THE DUTY CYCLE (the fraction of the
│                                   #   post-gate window actually firing, the same over t_x1 -> t_SFT, the episode
│                                   #   count, the longest firing-free gap, the quiet time to the anchor). THE TEST IS
│                                   #   NOT THE CLASS CONTRAST: that grid is scored with the 12c Benjamini-Hochberg and
│                                   #   reported, but 13 rotators cannot carry it - the hypothesis is put as a
│                                   #   CORRELATION over every galaxy at once, Spearman of each quantity against
│                                   #   delay = t_SFT - t_x1, pooled and per class, which needs no class balance.
│                                   #   FALSIFIED IF f_Edd at the ignition is equal AND uncorrelated with the delay;
│                                   #   CONFIRMED IF the rotators accrete faster and the delay falls as f_Edd rises.
│                                   #   -> kappa_m25_accretion{.png,.pdf} (2x3: f_Edd on the t - t_x1 clock with the
│                                   #   jet and full-velocity thresholds drawn, the duty-cycle ECDF, the delay against
│                                   #   the duty cycle, the pre-SFT time budget of the four AGN states, M_BH vs f_Edd at
│                                   #   the ignition against the jet mass gate, and the contrast as an AUC forest with
│                                   #   the BH verdict) + kappa_m25_accretion.csv + kappa_m25_accretion_galaxies.csv.
│                                   #   Knobs P12D_FDR/P12D_TB/P12D_FEMIN/P12D_QTY. A PURE READ - it needs Parts 9 and
│                                   #   12 in the kernel (KEEP9, CLS9, ST12) and opens no particle file.
│                                   #   PART 13 (NEW 2026-09-09, the user's hypothesis: "the galaxies with lower dust
│                                   #   content are the ones that ejected more gas into the cgm (with higher
│                                   #   velocities) BEFORE starting the jet or x-ray phase. After that the agn feedback
│                                   #   increased the cgm temperature destroying the dust and stopping cold inflows
│                                   #   back into the ism... In the other case, jet started with compacted ISM, cold
│                                   #   inflows choked the x-ray channel and a compact ism was left, where dust can
│                                   #   reform"): THE EJECTION AND THE HEATING, the CGM ON THE CLOCK OF THE X-RAY
│                                   #   ONSET. The chain is stated as six separately falsifiable links - H1 more (and
│                                   #   faster) gas+dust displaced to the CGM BEFORE t_x1 in the future dust-poor; H2
│                                   #   the CGM heats after it; H3 the heat DESTROYS the grains (the CGM D/G falls, it
│                                   #   does not merely dilute); H4 the cold inflow shuts off; H5 in the other case
│                                   #   cold gas keeps arriving and f_gas stays ABOVE SIMBA's own X-ray gate, so the
│                                   #   channel is CHOKED not firing; H6 a compact ISM is left where dust can re-form -
│                                   #   and each is scored in a window BEFORE (P13_PRE = -1.5..-0.25 Gyr) and one AFTER
│                                   #   (P13_POST = +0.5..+2.5) the onset, separately: if the difference H1 needs
│                                   #   before the onset is only there after it, the causal order fails and the
│                                   #   scorecard says so. THE CLOCK is t - t_x1, the first step with the X-ray channel
│                                   #   on (Part 12's xray_on off Part 9's state code), which sits ~-0.2 Gyr from the
│                                   #   truncation in the median. THE SPLIT IS THE OUTCOME - the dust label at the
│                                   #   anchor (P13_SPLIT 'pop'; 'kcls' and Part 12's 'hole' are the alternatives) - so
│                                   #   this part is an ANATOMY, never a prediction, and the mass is the confound in
│                                   #   every panel (a heavier halo has a hotter CGM whatever the AGN did): every AUC
│                                   #   is printed raw AND pooled within P13_MTERC = 3 terciles of log M*. WHAT IS
│                                   #   MEASURED, from the same reduced files Part 10 reads but in its own cache
│                                   #   kappa_cis25_cgm_phases.fits (a SECOND full pass - the two censuses keep
│                                   #   separate caches so neither forces a 7 GiB rebuild of the other): the CGM in
│                                   #   PHASES (cold T < 1e5, warm, hot T > 1e6: gas mass, dust mass, count) in four
│                                   #   radial shells, with the mass-weighted AND the DUST-WEIGHTED mean log T - the
│                                   #   second is the point of H3 (in one m25 galaxy the CGM sits at <logT>_M = 5.9
│                                   #   while its surviving dust sits at 4.5: the grains are in the cold phase and the
│                                   #   heat removes that phase); the FLUX through r = 10 kpc split into in/out AND
│                                   #   cold/total for gas and dust (H4 is then a measured number, not an inference);
│                                   #   the ISM COMPACTNESS (gas and dust inside 1 kpc, R50 of the member gas) so H6
│                                   #   and Part 12's hole are two views of one claim. FIGURES: 1 the ejection
│                                   #   (M_dust,CGM/M*, the SHARE of the dust outside, v_r 90th pct of the outer ISM,
│                                   #   the cumulative dust across the shell - only galaxies sampled from P13_CUMSTART
│                                   #   = -2 Gyr enter that one) -> kappa_m25_cgm_ejection; 2 the heating (<logT>_M,
│                                   #   the cold mass fraction, <logT>_Mdust, the CGM D/G with the ISM's own dashed
│                                   #   behind it) -> kappa_m25_cgm_heating; 3 the return (the COLD inflow at the
│                                   #   shell, f_gas against the gate, the share of each group FIRING vs CHOKED, the
│                                   #   ISM inside 1 kpc) -> kappa_m25_cgm_return; 4 the test (the per-galaxy
│                                   #   before/after scalars against the dust the galaxy ACTUALLY ended with, Spearman
│                                   #   rho) -> kappa_m25_cgm_test + kappa_m25_cgm{,_nodes}.csv + the printed SCORECARD
│                                   #   (both medians, AUC, [AUC at fixed M*], MW p, and whether the direction is the
│                                   #   one the hypothesis predicts). CAVEAT THAT DECIDES THE READING: f_gas crossing
│                                   #   the gate at t = 0 is BY CONSTRUCTION (the gate is part of the definition of
│                                   #   t_x1) - the falsifiable statement is whether a group climbs back OVER it
│                                   #   afterwards.
│                                   #   H7 (NEW 2026-09-09, the user's follow-up: "are the dustier sources the ones
│                                   #   that expelled ISM into the CGM with velocity smaller than the escape one?"):
│                                   #   what separates the two may not be HOW MUCH left but whether it left BOUND -
│                                   #   below the local escape velocity it is a FOUNTAIN that comes back with its
│                                   #   dust, above it it is gone with the reservoir the dust would re-form from. The
│                                   #   potential CANNOT come from the cutouts (gas + stars only, no dark matter): it
│                                   #   is the CAESAR halo's - M200c, R200c of galaxy_data/parent_halo_index, NFW,
│                                   #   concentration from Dutton & Maccio (2014) - plus the baryons the cutout does
│                                   #   resolve as -Phi_b(r) = G[M(<r)/r + sum_{r_i>r} m_i/r_i] (the second term is
│                                   #   NOT optional: G M(<r)/r alone goes to zero at the centre where the potential
│                                   #   is deepest, and the profile it gives is not monotone and cannot be inverted).
│                                   #   P13_VESC_TO 'r200' (default, the speed needed to LEAVE THE HALO) | 'inf'.
│                                   #   MEASURED per node: v_r/v_esc quantiles and the bound mass fraction for the
│                                   #   CGM and the outer ISM, gas- and dust-weighted; the outward shell flux SPLIT at
│                                   #   v_esc (mdot_out_{g,d}{b,u}) and its cumulative integrals (the dust that left
│                                   #   BOUND vs UNBOUND); M200c / R200c / conc / central per node, cached in
│                                   #   kappa_cis25_halo.csv (one catalogue open per snapshot). AND THE APOCENTRE,
│                                   #   because in these halos v_esc(10 kpc) ~ 900-1100 km/s and almost nothing a
│                                   #   quenching galaxy ejects is formally unbound - the binary saturates, so the
│                                   #   informative number is HOW FAR OUT it gets before turning around: solve
│                                   #   -Phi(r_apo) = -Phi(r) - v^2/2 for a radial orbit in the same potential
│                                   #   (rapo50/rapo90/xapo50 per zone and weight), censored at 5 R200c. FIGURE 1b
│                                   #   the escape (v_r/v_esc of the outer ISM with the line at 1, the apocentre of
│                                   #   the ejected dust, the dust RETURNING through the shell against the dust
│                                   #   leaving it, the cumulative bound vs unbound dust) -> kappa_m25_cgm_escape;
│                                   #   figure 4 gains the H7 panel; six H7 rows join the scorecard. A census written
│                                   #   before these columns is DETECTED and re-measured from scratch (one extra
│                                   #   pass). FIXED at the same time: the scorecard's direction column compared a
│                                   #   group name against the string "dusty" and therefore read NO on every row
│                                   #   whenever P13_SPLIT was not 'pop'; the claims are now mapped onto whichever
│                                   #   group of the split in force ends up dust-rich (GK13[0] in all three).
│                                   #   CAVEATS: for a SATELLITE parent_halo_index is the group halo, so v_esc(r) at
│                                   #   a galaxy-centric radius is the wrong halo's about the wrong origin - the
│                                   #   `central` flag is stored per node; the concentration is a mean relation, not
│                                   #   the halo's own profile; the apocentre is a RADIAL-ORBIT UPPER BOUND (angular
│                                   #   momentum brings it back sooner).
│                                   #   Part 14 (NEW 2026-09-10, REWRITTEN the same day - the first version, the elapsed
│                                   #   share (t_SFT - t_x1)/t_SFT against the dust at QT, was rejected by the user: "no I
│                                   #   don't like it") = THE JET LEG AS THE UPSTREAM CAUSE, the user's own hypothesis put
│                                   #   on trial: a galaxy that spends a LONG time in the jet mode BEFORE the truncation
│                                   #   keeps its rotating disc and SUPPRESSES the X-ray channel by compactifying its ISM
│                                   #   (the dust survives, the high H2 density lets it grow back), while one that
│                                   #   truncates AT the gate opening with no long jet leg has its ISM stirred, loses the
│                                   #   dust, ends pressure-supported, goes on accreting - and the gate stays open so the
│                                   #   dust cannot re-form. THE CHAIN: jet leg -> compact rotating ISM -> the channel
│                                   #   suppressed -> the dust survives. Unlike every other part it splits on the CAUSE
│                                   #   and looks at the OUTCOME. Why the jet leg: Part 12d already had it - time in the
│                                   #   jet state before the SFT 1.26 Gyr (rotating) vs 0.44 (rest) and the strongest
│                                   #   predictor of the survival delay (rho +0.65), with the X-ray duty after the gate
│                                   #   0.87 vs 0.97 (its top-ranked contrast). THE CAUSE, before the SFT only, on the
│                                   #   branch sampling, Part 0's agn_states verbatim: t_jet_pre (the jet state - the
│                                   #   default axis), t_chk_pre (CHOKED = a jet with the gate shut, the jet leg without
│                                   #   the channel), and both as fractions of the sampled pre-SFT branch (P14_CAUSE).
│                                   #   THE MIDDLE LINK: duty_post, gap_max, n_epi_post, the share of the post-SFT branch
│                                   #   under the gate, f_gas at SFT / anchor, and the post-SFT gas budget (dlfgas_post,
│                                   #   regas = the "keeps accreting" half). THE OUTCOME at the anchor from the CATALOGUE
│                                   #   rows (Part 4b's _anc4b logic, so the two cannot drift): kappa_gas, compactness
│                                   #   -log R50(gas)/R50(*), Sigma_H2, Sigma_gas, R50(gas), f_dust, f_H2 - and the
│                                   #   compound the question names (COMPACT ROTATOR + high f_dust + low f_H2) as four
│                                   #   booleans drawn separately so a broken link shows on the ladder instead of hiding
│                                   #   in the conjunction (P14_CQTY picks what "compact" means; cuts = the box median).
│                                   #   FIG 1 kappa_jetleg_outcome, one row per box: (a) the LADDER P(outcome | cause) in
│                                   #   equal-count bins with Wilson 68 %, the four rungs + the conjunction; (b)
│                                   #   compactness x dust coloured by the jet leg; (c) dust x H2 the same; (d) the middle
│                                   #   link, duty_post against the cause. FIG 2 kappa_jetleg_clocks: THE ORDER OF EVENTS
│                                   #   - the LONG and SHORT jet-leg groups on the t - t_SFT clock for comp / Sigma_H2 /
│                                   #   kappa / f_gas (with SIMBA's gate) / f_dust / f_H2, each group's median jet start
│                                   #   and first firing marked on the axis. That figure is where the hypothesis can die:
│                                   #   the compactification must appear LEFT OF ZERO, during the jet leg; if the groups
│                                   #   part only after the truncation the jet leg is a label on the outcome, not its
│                                   #   cause. CONFOUND: the jet leg needs M_BH > 10^7.5 and grows with M* and epoch, so
│                                   #   the groups are split INSIDE quartiles of log M* (P14_MCTRL) and every score is
│                                   #   repeated as a partial rank correlation at fixed log M* and log M_BH, with
│                                   #   log_mstar / log_mbh / t_sft carried as CONTROL rows of the link table. THE
│                                   #   DECISIVE CONTROL is cis50nox (jets ON, X-ray OFF): the jet leg -> disc link must
│                                   #   SURVIVE there and the jet leg -> dust link must COLLAPSE; that double expectation
│                                   #   is its own printed block. A pure read of ism_prediction_histories_<box>.hdf5 on
│                                   #   Part 3's rows + Part 3's tracks + QM -> kappa_jetleg_{outcome,clocks}{.png,.pdf} +
│                                   #   kappa_jetleg.csv + kappa_jetleg_galaxies.csv (one row per branch).
│                                   #   FIG 1 is OFF by default since 2026-09-10 (P14_FIG1): the user found it
│                                   #   uninformative and keeps FIG 2, the clock - which is what Part 15 rebuilds.
│                                   #   Part 15 (NEW 2026-09-10, the user's reading of Part 14's FIG 2 turned into a
│                                   #   division) = THE STATE AT THE JET ACTIVATION. WHAT FIG 2 SHOWED: in every box
│                                   #   the SHORT jet leg is activated ON A SPIKE - gas compactness, Sigma_gas,
│                                   #   Sigma_H2 and the dust peak together while kappa drops; in cis50nox the
│                                   #   compactness barely moves but the H2 density still peaks and the rotation still
│                                   #   falls; AFTER the SFT the two groups run PARALLEL in everything except the dust
│                                   #   (in cis50nox the slopes are shallower, f_dust decays smoothly and far more
│                                   #   slowly, kappa climbs back fast). THE READING: the X-ray channel is the factor -
│                                   #   it steepens the cold ISM's disappearance AND stirs what is left, and since
│                                   #   SIMBA puts half the X-ray energy into ISM gas as heat and half as a RADIAL KICK
│                                   #   (Dave+19), the stirring is momentum delivered to whatever dense gas is there
│                                   #   when the channel opens. A galaxy long in the jet mode has already lost part of
│                                   #   its central H2: the channel finds less to push and its dust survives. SO THE
│                                   #   DIVISION IS A STATE, NOT A DURATION: Sigma_H2 AT the jet activation, above /
│                                   #   below the box median (P15_EPOCH reads the same axis at t_x1, where the
│                                   #   mechanism says the coupling happens, or at the SFT; P15_CAUSE swaps it for
│                                   #   Sigma_gas / f_H2 / sSFR / compactness / kappa - block (a) scores ALL of them
│                                   #   against the same outcome so the axis is chosen on the data). The proposed names
│                                   #   - POST-STARBURST vs NORMAL QUIESCENT - are a claim, so they are measured:
│                                   #   f_burst = the stellar mass formed in the last P15_BURST Gyr before the SFT,
│                                   #   integrated from the branch's own SFR so a merger cannot fake it; b_ssfr = the
│                                   #   sSFR peak over that branch's own baseline; and the anchor age (a post-starburst
│                                   #   branch must be bursty AND end young). FIG 1 kappa_jetstate_clocks = Part 14's
│                                   #   clock with the new split and the JET ACTIVATION as the origin (each group's
│                                   #   median SFT dashed, first firing dotted, in its own colour). FIG 2
│                                   #   kappa_jetstate_dustage = the plane the user asked for, log M_dust/M* against
│                                   #   the mass-weighted age AT THE ANCHOR - THE TRACKS AND THEIR CRITICAL POINTS
│                                   #   ONLY, the anchor cloud dropped 2026-09-10 at his request: triangle = the sSFR
│                                   #   peak (the maximum over P15_PKW Gyr before the SFT), circle = the jet
│                                   #   activation, square = the SFT, star = the median anchor reached along the
│                                   #   dashes; the age along the branch is integrated BACK from the catalogue's anchor
│                                   #   age - exact once nothing forms, a lower limit in the star-forming leg (drawn
│                                   #   dotted) because merged-in stars are not credited. THE DECAY (NEW 2026-09-10):
│                                   #   from the SFT to the anchor every galaxy's dust, sSFR and H2 is fitted with a
│                                   #   DOUBLE EXPONENTIAL y = A1 exp(-t/tau1) + A2 exp(-t/tau2), tau1 <= tau2, in
│                                   #   log10 (_decay15), and compared by BIC on the same points with the single
│                                   #   exponential it nests - which IS a straight line in log10 y. A point within
│                                   #   P15_DFLOOR dex of a floor is CENSORED, not fitted (P9_FLOOR for the dust,
│                                   #   SSFR_FLOOR for the sSFR), because a flat run of floored values would BE the
│                                   #   flattening the fit is asked to find; t_floor keeps when the branch reached it.
│                                   #   THE CLAIM UNDER TEST is the user's: the two branches leave the SFT on THE SAME
│                                   #   initial slope and the pre-depleted one then BENDS to something flatter (a long
│                                   #   tau2, a t_break inside the sampled window), the sSFR being shallower for it
│                                   #   throughout - in AUC terms s_early near 0.5 and s_late not. Beside every fit,
│                                   #   the MODEL-FREE slopes over the first and the last P15_DWIN Gyr, which is the
│                                   #   same statement without a model. FIG 3 kappa_jetstate_decay = the points (the
│                                   #   branches themselves) with the median fitted curve per branch and the median
│                                   #   t_break dashed; FIG 4 kappa_jetstate_slopes = the initial slope against the
│                                   #   late one GALAXY BY GALAXY, the diagonal being no bend at all (above it the
│                                   #   decay flattened, below it it steepened). FIG 5 kappa_jetstate_after = THE OPEN
│                                   #   QUESTION (does the jet continue in the stirred branch and turn PREVENTIVE - no
│                                   #   accretion, no dust - while the other branch's CGM cools and keeps its ISM
│                                   #   dense?): (a) the AGN state budget of the post-SFT branch per group, (b) the
│                                   #   budget plane, the gas that came back against the dust that went. FIG 6
│                                   #   kappa_jetstate_auc (NEW 2026-09-10) = THE ANCHOR TEST: is a ROTATING galaxy
│                                   #   dusty too, and is one with a STRONG OUTFLOW RIGHT NOW not? - the two dust
│                                   #   classes as histograms of each axis, the dusty share bin by bin drawn over them
│                                   #   with Wilson intervals, and the AUC that summarises both, raw and with the axis
│                                   #   RANKED INSIDE log M* cells. THE OUTFLOW IS MEASURED THROUGH A SHELL AT EACH
│                                   #   GALAXY'S OWN R50(stars), NOT at Part 13's fixed 10 kpc: |r - R50*| < P15_ORW
│                                   #   R50* in the ANCHOR reduced particle file (one file per branch, cached to
│                                   #   kappa_jetstate_outflow_<box>.csv), the member stars giving both the radius and
│                                   #   the rest frame, the Hubble flow inside the cutout added, Mdot_out = sum_{v_r >
│                                   #   0} m v_r / width for the gas / the cold gas / the dust, with the
│                                   #   90th-percentile outflow speed beside them; P15_OQ picks the axis (default
│                                   #   Mdot_out/M*, so that 'strong' is not just 'massive'). This is the one block of
│                                   #   Part 15 that reads particles, and it exists only where reduced particle files
│                                   #   do (m25, and whatever m100 snapshots are on disk). FIG 7 kappa_jetstate_cgm
│                                   #   (only with Parts 9-13 in the kernel) = Part 13's m25 CGM census re-split on
│                                   #   this division: <log T>_M, the cold share, the cold inflow at 10 kpc, the
│                                   #   displaced dust. CONTROLS: the axis is ranked inside cells of log M*
│                                   #   (P15_MCTRL), every score is repeated partial at fixed log M*, log M_BH AND at
│                                   #   fixed EPOCH - Part 14's jet leg carries a +1.5 to +2.8 Gyr difference in t_SFT
│                                   #   by itself, so any state measured at t_jet inherits that risk - and with
│                                   #   P15_ZCTRL the groups are re-formed inside epoch cells and the key contrasts
│                                   #   printed twice. THE STIRRING TEST (d kappa over P15_STIR Gyr across the X-ray
│                                   #   onset against the Sigma_H2 the onset FOUND, whose null is cis50nox where the
│                                   #   state fires in the bookkeeping but no energy is ever deposited) is kept as code
│                                   #   but is OFF since 2026-09-10 (P15_STIRTEST): the user dropped it; switching it
│                                   #   on restores block (e) and a third panel of FIG 5. Where Part 3's kinematic
│                                   #   sidecar was never built the surface density has no radius and the axis falls
│                                   #   back to the mass fraction of the same phase, loudly (P15_FALLBACK /
│                                   #   P15_MINCOV). A pure read of the histories on Part 3's rows + Part 3's sidecar,
│                                   #   plus one anchor particle file per branch for the outflow ->
│                                   #   kappa_jetstate_{clocks,dustage,decay,slopes,after,auc,cgm}{.png,.pdf} +
│                                   #   kappa_jetstate.csv + kappa_jetstate_galaxies.csv (one row per branch) +
│                                   #   kappa_jetstate_outflow_<box>.csv.
│                                   #   Part 16 (NEW 2026-09-10, the user's own hypotheses written up as five
│                                   #   falsifiable claims) = THE ORDER OF EVENTS, AND THE CGM CYCLE THE
│                                   #   TRUNCATION STARTS. Part 15 ended on a division that does not divide
│                                   #   (Sigma_H2 at the jet activation is worth AUC 0.48 in m25, while
│                                   #   kappa_gas at the anchor is worth 0.68-0.74 everywhere), so this part
│                                   #   asks what makes the ROTATION, and what the halo does afterwards. C1 THE
│                                   #   ORDER: a branch that reaches the jet mode early reaches the SFT early
│                                   #   and fires the X-ray channel early - one chain, not three draws. C2 THE
│                                   #   ROTATION IS MADE BY THE JET: that early jet leaves a HIGHER kappa at the
│                                   #   anchor because what it removes is the LOW-j gas (the central kpc
│                                   #   empties, R50(gas) grows, the rotating part survives); cis50nox is the
│                                   #   counterfactual. C3 THE CGM TEMPERATURE HAS THREE PHASES with the
│                                   #   truncation as the hinge - it RISES before it (the galaxy is still
│                                   #   forming stars: SN shock heating), FALLS just after it (the ISM is
│                                   #   displaced outward, cold gas is ADDED to the CGM and the mass-weighted
│                                   #   mean is diluted, so the cold phase must rise with it), and RISES again
│                                   #   once the X-ray channel is on, with the cold inflow and the CGM dust
│                                   #   falling as it does. C4 THE TWO CLASSES ARE THE SAME FOR ~1 Gyr after the
│                                   #   SFT and only then part company (the rotators' CGM cools, keeps its dust,
│                                   #   goes on accreting) - measured as a DIVERGENCE TIME with a bootstrap over
│                                   #   galaxies, not eyeballed. C5 IS THE DUST JUST THE STAR FORMATION: every
│                                   #   galaxy on the f_dust-sSFR plane with the region it could have FILLED IN
│                                   #   ONE Gyr at its own sSFR shaded behind it. 'EARLY' IS A RANK, NEVER A
│                                   #   DATE: t_jet is a cosmic time and each branch is watched to a different
│                                   #   anchor, so _crank16 ranks it inside cells of the anchor time x log M*
│                                   #   (P16_CTRL, P16_CELLQ) - early FOR ITS KIND - and the raw and ranked
│                                   #   correlations are printed side by side. THE CGM HALF READS PART 13's OWN
│                                   #   NODE TABLE (kappa_m25_cgm_nodes.csv) and RE-CLOCKS it on this part's
│                                   #   t_SFT and RE-SPLITS it on this part's classes (P16_SPLIT), so P13_CLOCK
│                                   #   / P13_SPLIT cannot change what (c) and (d) mean; without that table (c)
│                                   #   and (d) say so and are skipped. THE DUST BUDGET of (e) is f_dust^new =
│                                   #   y_d sSFR dt with y_d = 1.5e-3 to 6e-3, SIMBA's condensation channel
│                                   #   (delta_SNII = 0.15, delta_AGB = 0.2 on a metal yield 0.01-0.03 per unit
│                                   #   mass formed); the WHOLE range is shaded, the star-forming locus is drawn
│                                   #   through the same plane as an internal control, and per branch the same
│                                   #   excess is repeated with the mass its OWN SFH formed in that Gyr (no
│                                   #   constant-sSFR assumption). A galaxy with no star formation at all can
│                                   #   make nothing, so it is drawn in its own strip beside the plane rather
│                                   #   than dropped off the edge. FIG 1 kappa_timing_order = every branch's
│                                   #   three events on one timeline sorted by t_jet, and what the rank of the
│                                   #   first one leaves: the jet leg, kappa at the anchor, the dusty share, the
│                                   #   X-ray taken afterwards. FIG 2 kappa_timing_jetclock = kappa_gas,
│                                   #   R50(gas), Sigma_gas, f_dust and the X-ray occupancy on t - t_jet, early
│                                   #   jet against late. FIG 3 kappa_timing_lowj = the same leg in the m25
│                                   #   PARTICLES (the gas and the dust inside 1 kpc, R50(gas), the ISM mass) -
│                                   #   C2's mechanism, measured. FIG 4 kappa_cgm_cycle = the temperature, its
│                                   #   cold phase, the inner-halo temperature, the cold inflow, the CGM dust
│                                   #   and its dust-to-gas ratio on t - t_SFT, with C3's three windows
│                                   #   (P16_WIN) shaded and the X-ray occupancy behind. FIG 5
│                                   #   kappa_cgm_divergence = class MINUS class bin by bin with t_div, the
│                                   #   first epoch at or after the SFT where the difference holds for
│                                   #   P16_DIVSIG bins running - WHATEVER ITS SIGN, so the value AT t_div is
│                                   #   annotated beside it. FIG 6 kappa_dust_budget = the plane with the Gyr of
│                                   #   dust production shaded, and the per-branch excess over what its own SFH
│                                   #   could have made. The (c) scorecard is one straight-line slope per galaxy
│                                   #   through <log T>_M(CGM) in each window against what C3 says makes it (the
│                                   #   sSFR of the window, the ISM mass lost, the cold CGM mass gained, the
│                                   #   X-ray share, the cold inflow and the CGM dust as they fall); the d
│                                   #   columns are LATE MINUS EARLY inside the window, so a galaxy that LOSES
│                                   #   ISM has d < 0 and the predicted correlation with a COOLING CGM is
│                                   #   POSITIVE. A pure read of the histories on Part 3's rows + Part 3's
│                                   #   sidecar + Part 13's node table + Part 0's catalogue cache ->
│                                   #   kappa_timing_{order,jetclock,lowj}{.png,.pdf} +
│                                   #   kappa_cgm_{cycle,divergence}{.png,.pdf} + kappa_dust_budget{.png,.pdf} +
│                                   #   kappa_timing.csv (every printed number and drawn point) +
│                                   #   kappa_timing_galaxies.csv (one row per branch) + kappa_cgm_cycle.csv
│                                   #   (one row per m25 branch: the three temperature slopes and what moves
│                                   #   them).
│                                   #   Part 17 (NEW 2026-09-10) = THE REGROWTH TIME, PARTICLE BY PARTICLE. Part
│                                   #   16 measured WHEN the CGM dust appears and goes; this part measures
│                                   #   WHETHER IT CAN. SIMBA makes and destroys dust grain by grain, at rates
│                                   #   set by the density, the temperature and the metallicity of the gas the
│                                   #   grain sits in, and all three are in the reduced files (they carry nh and
│                                   #   zgas since 2026-09-02 for exactly this). THE GROWTH LAW (Hirashita 2000
│                                   #   / Asano+13, the form Li, Narayanan & Dave 2019 implement in SIMBA):
│                                   #   tau_grow = 25.4 Myr (a/0.1um) (n_H/100)^-1 (T/20K)^-1/2 (Z/Zsun)^-1,
│                                   #   with the single-grain rate LIMITED BY THE METALS STILL IN THE GAS -
│                                   #   dm_d/dt = m_d f_Z / tau_grow, f_Z = Z m_gas / (m_dust + Z m_gas), the
│                                   #   same bounded dust-to-metal form Part 9 draws. THE SPUTTERING LAW (Tsai &
│                                   #   Mathews 1995, SIMBA's own): tau_sp = 0.17 Gyr (a/0.1um) (1e-27/rho)
│                                   #   [(T_0/T)^2.5 + 1], rho = n_H m_p / X_H; tau_sp is the timescale of the
│                                   #   RADIUS, so the MASS e-folding is tau_sp/3 and that is what the budget
│                                   #   uses. TWO THINGS THAT MUST NOT BE READ AS PHYSICS: for STAR-FORMING gas
│                                   #   the temperature is the effective equation of state and its tau_sp means
│                                   #   nothing (the SF and non-SF ISM are never merged, exactly as the old
│                                   #   notebook's sputtering part did); and above P17_TGROW the sticking
│                                   #   coefficient collapses, so every growth column is the COLD gas of its
│                                   #   zone and the hot-CGM growth panel is empty on purpose. Both laws are
│                                   #   LINEAR in the grain radius, so P17_AGR slides the whole distribution and
│                                   #   changes no comparison - the part is written around ratios and shares.
│                                   #   (a) FIG 1 kappa_dust_timescales = the two clocks of a grain: tau_grow
│                                   #   and tau_sp of the dust that is ACTUALLY THERE (dust-mass-weighted), zone
│                                   #   by zone (the non-SF ISM, the SF ISM, the cold CGM, the hot CGM) on t -
│                                   #   t_SFT, with 1 Gyr and a Hubble time drawn across them - the user's
│                                   #   question, 'the regrowth timescale in single particles in the ISM and the
│                                   #   CGM', is this figure. (b) FIG 2 kappa_dust_budget_cgm = MADE OUT THERE,
│                                   #   OR BROUGHT? the in-situ growth summed over the particles of a zone, the
│                                   #   sputtering against it, and the MEASURED change of that zone's dust mass
│                                   #   between consecutive nodes of the same branch; plus 'the time to make it'
│                                   #   = log M_dust / Mdot_grow, how long the zone would need to build the dust
│                                   #   it holds at the rate it is growing at now - above a Hubble time (1.14 in
│                                   #   the log) it cannot have made it and the dust was brought there, and Part
│                                   #   13's flux through the shell says which way it went. (c) FIG 3
│                                   #   kappa_dust_outcome = the outcome: every timescale and every environment
│                                   #   it is made of scored against the dust at the anchor (AUC raw and with
│                                   #   the axis ranked inside terciles of log M*, the outcome plane drawn for
│                                   #   the strongest axes). It reads the SAME files Parts 10 and 13 read, KEYED
│                                   #   BY PART 13'S NODE TABLE so the branch walk is not repeated (so Part 13
│                                   #   must have been run for the m25 box, and Part 16 must have left
│                                   #   kappa_timing_galaxies.csv or D16 in the kernel), and keeps its OWN cache
│                                   #   kappa_<box>_dust_timescales.fits, one row per (snapshot, galaxy),
│                                   #   appended to. Where nh or zgas is missing from a file every timescale is
│                                   #   NaN and the cell prints the one-line backfill
│                                   #   (build_reduced_particles_job appends only the missing dataset - no
│                                   #   geometry redo). -> kappa_dust_timescales{.png,.pdf} +
│                                   #   kappa_dust_budget_cgm{.png,.pdf} + kappa_dust_outcome{.png,.pdf} +
│                                   #   kappa_dust_timescales.csv + kappa_<box>_dust_timescales_galaxies.csv +
│                                   #   the FITS census.
│                                   #   Part 18 (NEW 2026-09-10) = THE DECAY AFTER THE TRUNCATION, TIMED - the
│                                   #   revisit of Part 15's block (c). Its double exponential is NOT identified
│                                   #   on this data (4 free parameters on a ~10-point, ~1.2 Gyr window): 43 % of
│                                   #   its fits sit on a bound (tau1 at 0.05, tau2 at 50-70 Gyr) and 75 % give
│                                   #   f_slow < 1 %, so its tau1/tau2 are the bounds - do not quote them; its
│                                   #   median-track figure also flattens partly by SURVIVORSHIP (the fastest
│                                   #   branches reach the dust floor and leave the sample). Part 18 refits the
│                                   #   same claim as a BROKEN EXPONENTIAL in log10 y (a straight line per leg, so
│                                   #   the fit is linear at fixed break and the break is profiled over the
│                                   #   observed times), over EVERY SNAPSHOT from the truncation to the anchor
│                                   #   (P18_WIN "all", the user's rule 2026-09-10: nothing dropped, nothing
│                                   #   censored - P9_FLOOR is SIMBA saying zero dust and is fitted as the value it
│                                   #   is, so a branch that reaches zero has a FLAT second leg and no tau2 at all).
│                                   #   tau = log10(e)/slope, tau0 = the single exponential the break is scored
│                                   #   against by BIC, plus the model-free t_1dex; the error bars take sigma from
│                                   #   the OFF-FLOOR points (a run of identical floored values has no scatter).
│                                   #   P18_ALT fits the narrow treatment (window to the minimum, floors censored)
│                                   #   alongside: tau1 is unchanged (0.138 vs 0.150), tau2 doubles (0.45 vs 0.26),
│                                   #   16 % vs 1 % end with no tau2 - that IS the floor tail. SHAPE CLASSES first,
│                                   #   timescales after: in m100 26 % steep->shallow (both legs falling), 13 %
│                                   #   rebound (it fell and then STOPPED - the floor, or real regrowth), 12 %
│                                   #   shallow->steep (a plateau then a collapse), 25 % single, 21 % too short.
│                                   #   SPLIT BY THE DUST LABEL IT DIVIDES: steep->shallow is 30 % of the dust-poor
│                                   #   and 2 % of the dusty; the branches that fall and then STOP are 33 % of the
│                                   #   dusty vs 10 %, and for the dusty ones that arrest is REAL REGROWTH (median
│                                   #   2.4 dex back from the minimum, half of them off the zero-dust floor) while
│                                   #   the dust-poor ones simply stop AT the floor.
│                                   #   THE DISTRIBUTION QUESTION (is tau bimodal?) is
│                                   #   answered with SILVERMAN's critical-bandwidth test (smoothed bootstrap),
│                                   #   with the GMM drawn only as description (a mixture beats one Gaussian on
│                                   #   skewness alone; the Ashman D and whether the mixture is itself two-humped
│                                   #   are printed with it) and TWO confounders controlled explicitly: the
│                                   #   snapshot spacing (tau0 = 0.26 Gyr against dt = 0.07-0.18 Gyr; 12 % of the
│                                   #   e-foldings are shorter than one step) and THE ANCHOR (ten anchors z = 3-0
│                                   #   move the median log tau0 by 0.3 dex on their own). ANSWER: log tau is
│                                   #   right-skewed, not Gaussian, everywhere, and Silverman REJECTS ONE MODE
│                                   #   NOWHERE (p = 0.14 / 0.43 / 0.20 / 0.21 pooled, and the model-free t_1dex
│                                   #   never rejects either) - ONE mode with a long slow tail, the dusty QGs
│                                   #   inside that tail, not a second population. THE DUST LINK: they are slower
│                                   #   in the dust (AUC 0.81, 0.84 at fixed anchor in m100) but that test is
│                                   #   CIRCULAR and is marked (*); the non-circular versions - the same galaxies'
│                                   #   sSFR and H2 decay times - SPLIT BY RESOLUTION: in the coarse boxes the
│                                   #   dusty ones lose those FASTER (AUC 0.45 / 0.41 in m100, 0.36 / 0.35 in m50),
│                                   #   so what is slow about them is the dust alone, while in m25 - the box that
│                                   #   resolves the gas - they are slower in EVERYTHING (0.66 / 0.59).
│                                   #   Part 15 is optional; where it
│                                   #   has been run its parameters and its branch split are carried along for the
│                                   #   comparison. -> kappa_decay_{gallery,classes,audit,taudist,plane,dustlink}
│                                   #   {.png,.pdf} + kappa_decay.csv + kappa_decay_galaxies.csv.
├── starburst_vs_aging.ipynb        # THE SIMPLE NOTEBOOK beside dusty_qgs_kappa_selection (2026-09-10, rebuilt
│                                   #   through v9 on 2026-09-11; 19 cells, Parts 0-6 + 1b + 1c, deployed +
│                                   #   run end-to-end in 45 s).  ONE pass over the chains, ONE selection, and
│                                   #   everything else follows it -> output/box_resolution/burst_aging/
│                                   #   (prefix ba_; nothing of the kappa notebook is touched).
│                                   #   THE SELECTION IS A REGION OF A MAP, NOT A THRESHOLD ON A NUMBER (Part 1b,
│                                   #   cells p1b-md/p1b-peel - THE PEEL). One quantity (P1D_ON = lfd) is drawn
│                                   #   against the clock (P1B_CLOCK = jet | sft | x1 | qt), one column per epoch,
│                                   #   each scaled to its own peak; the TWO BRIGHT REGIONS that appear after the
│                                   #   clock point are peeled off in the order the eye reads them. STAGE 1 the
│                                   #   bright cells ABOVE THE VALLEY -> group 1 (a track joins at P1D_NHIT epochs
│                                   #   inside); STAGE 2 the same map with those tracks REMOVED, the most
│                                   #   populated bright blob of the residual -> group 2; STAGE 3 whatever never
│                                   #   entered either -> group 3, the intermediates. CALIBRATED ON THE BIG BOXES:
│                                   #   P1D_CAL = ["cis100","cis50"] finds the regions ONCE on the two pooled and
│                                   #   peels THE SAME CELLS off every box, so m25's 82 galaxies are never asked
│                                   #   to find structure they cannot resolve - only whether they fall in it. The
│                                   #   colour scale is FIXED at stage 1 and each column is DIMMED by the fraction
│                                   #   of its galaxies still in play, so an emptied region looks empty instead of
│                                   #   renormalising itself back to full brightness; the cyan line is the median
│                                   #   of what is still in play, 16-84 dotted. HOW A REGION IS FOUND: a cell is
│                                   #   bright within P1D_BRIGHT of the peak OF ITS OWN SIDE of the valley (a
│                                   #   minor upper ridge is not erased by a major lower one), the valley row is
│                                   #   left out so the two sides cannot connect through it, connectivity is
│                                   #   worked out on a mask dilated by P1D_BRIDGE epochs along the clock (a ridge
│                                   #   graded at +1.0 and +1.5 is ONE ridge, not two a track can never be inside
│                                   #   twice) and the jumped columns are filled, so the region is a band and not
│                                   #   a comb; of the candidates the MOST POPULATED wins. A pile-up on a known
│                                   #   censoring floor (P1B_FLOORS: zero dust at FLOOR, zero SFR at SSFR_FLOOR)
│                                   #   is NEVER a mode. P1D_REG replaces all of it with two rectangles by hand.
│                                   #   HOW RESTRICTIVE: P1D_BRIGHT (0.45 the whole ridge, 0.60 the default, 0.75
│                                   #   its core), P1D_NHIT (2 lets a galaxy that merely CROSSES join, 3 the
│                                   #   default, 4 asks it to live there), P1D_FHIT (a share of the epochs it
│                                   #   exists for) and P1D_PURE (a track qualifying for BOTH regions belongs to
│                                   #   NEITHER - a crosser, sent to the intermediates; the overlap is measured
│                                   #   BEFORE it). P1D_SCAN prints the whole (bright x nhit) grid on the
│                                   #   calibration pool - group sizes, crossers, overlap - and marks the row in
│                                   #   force, so restrictiveness is chosen off a table. It is also the guard
│                                   #   against a threshold that looks harmless: on the jet clock 0.45 swells the
│                                   #   upper region into the lower one (1003 crossers of 3541, overlap 99 %)
│                                   #   where 0.60 gives 47 and 6 %. WHAT THE SELECTION IS, carried onto the other
│                                   #   quantities (FIG 2, ba_peel_lfd_where; AUC between the two extreme groups,
│                                   #   m100 / m50 / m25): kappa_rot(gas) 0.98 / 0.98 / 1.00, sSFR 0.98 / 0.97 /
│                                   #   1.00, Sigma_H2 0.90 / 0.88 / 0.97, Mdot_BH 0.79 / 0.80 / 1.00, and
│                                   #   INVERTED in -log R50,gas/R50,* (0.08 / 0.05 / 0.09) and in the
│                                   #   mass-weighted stellar age (0.10 / 0.16 / 0.12). THE DUST SPLIT IS A
│                                   #   KINEMATIC ONE: the galaxies of the upper dust region are the ones whose
│                                   #   gas is still rotating, in EXTENDED rather than contracted gas, younger,
│                                   #   still forming stars and still feeding their black hole - and kappa_gas
│                                   #   separates them ALREADY AT THE TRUNCATION, before the dust has separated at
│                                   #   all. cis50nox has no lower group at all (the X-ray channel off = nothing
│                                   #   destroys the dust), which is the control's job. Part 1 gained "age" and
│                                   #   "lmdot" as tracked quantities for this figure (both straight out of the
│                                   #   histories file / agn_states). PART 1c (cells p1c-md/p1c-qtime) - DOES THE
│                                   #   SELECTION DIVIDE THE QUENCHING TIME? The peel knows nothing about tau_q,
│                                   #   and log tau_q/t_QT is the one axis here that is genuinely bimodal. Answer:
│                                   #   PARTLY. AUC(high vs low) 0.73 / 0.62 / 0.71 raw and 0.65 / 0.56 / 0.64
│                                   #   inside (epoch x mass) cells; the fast share is 23 % of the upper group
│                                   #   against 50 % of the lower one in m100 (29/46 in m50, 64/95 in m25) - the
│                                   #   dusty survivors are preferentially SLOW quenchers, but the two axes are
│                                   #   not the same axis. The reverse composition is printed too (of m100's fast
│                                   #   quenchers: 10 % upper, 37 % lower, 53 % intermediate). ->
│                                   #   ba_peel_ltauq{.png,.pdf,.csv} PART 2 ADOPTS IT (P2_ADOPT = "peel"): group
│                                   #   1 -> the 'burst' key, group 2 -> 'aging', the intermediates in neither,
│                                   #   the way a tercile middle is - so Parts 3-6 all describe the same
│                                   #   selection. The rest of Part 2 is the SPEC-LANGUAGE BROWSER you switch to
│                                   #   by putting an axis in P2_ADOPT instead: "<q> @ <epoch>", "<q> @ <A> > <B>"
│                                   #   (the change), ": max|min|mean|rng|slope|d", or any column of FR; <epoch>
│                                   #   is any event of P1_EV with an offset in Gyr (sft, qt+0.5, jet-0.3). P2_CUT
│                                   #   then takes "dip" | "median" | "tercile" | "qNN" | a number | a (low, high)
│                                   #   region | a dict per box, and the cell prints p05 / terciles / median / p95
│                                   #   / the dip per box to read a number off. P2_LABELS renames the groups
│                                   #   everywhere downstream. Every input is a catalogue-level product plus ONE
│                                   #   cache, so no particle file is opened outside Part 6. READS
│                                   #   ism_prediction_{catalogue.fits, agn_classes.csv, histories_<box>.hdf5}
│                                   #   (paper_ism_prediction_boxes P1/P6a), agn_states_kin_histories_<box>.hdf5
│                                   #   (dusty_qgs_kappa_selection P3, READ ONLY) and kappa_m25_cgm_nodes.csv (its
│                                   #   P13) for Part 6 only. Part 0 = the sample (the m25 quenched rule + log M*
│                                   #   > 10.25 + age >= 1.5; DUSTY_CUT = -4.0) -> 8240 -> 5151 rows, cis100 4266
│                                   #   / cis50 481 / cis25 162 / cis50nox 228; it also holds the shared KDE block
│                                   #   (_nmode / _hcrit / _silverman / _dip) and the controlled-score helpers
│                                   #   (_cellrank / _auc_ctrl, CTRL_COLS = t_anchor x log M*). Part 1 = ONE pass
│                                   #   -> F (event times, every tracked quantity read AT each of the five point
│                                   #   epochs jet/sft/x1/qt/anc AND over the window P1_WIN: _pk, _mn, _w0, _w1,
│                                   #   _dw), T (the tracks on four clocks) and R1 (the RAW series of EVERY
│                                   #   tracked quantity, which is what lets Part 2's spec language answer a new
│                                   #   question without re-reading a file). De-duplication by a log M*(t)
│                                   #   fingerprint, latest anchor kept: 5135 branches -> 3752 galaxies. THE
│                                   #   WINDOW IS A KNOB (P1_WIN / P1_WOFF, two event keys and an offset in Gyr
│                                   #   for either end); C, C0, C - C0 and t_feed all follow it.
│                                   #   PART 6b (cells p6b-md/p6b-grains) - THE GRAIN CLOCKS, the user's ask: a
│                                   #   PURE RE-READ of dusty_qgs_kappa_selection Part 17's per-PARTICLE
│                                   #   timescales (dust-mass weighted per zone per node), re-split on this
│                                   #   selection and re-clocked on this notebook's t_SFT ->
│                                   #   ba_grain_clocks{.png,.pdf,.csv}. tau_grow (Hirashita 2000 / Asano+13 in
│                                   #   Li, Narayanan & Dave 2019's SIMBA form) and tau_sp (Tsai & Mathews 1995,
│                                   #   SIMBA's own) in four zones: the non-star-forming ISM, the star-forming
│                                   #   ISM, the cold CGM, the hot CGM. TWO PANELS ARE PHYSICALLY EMPTY AND ARE
│                                   #   LABELLED, NOT DRAWN: SIMBA's star-forming gas sits on an effective EoS and
│                                   #   has no temperature, so its tau_sp is meaningless; above 1e5 K the sticking
│                                   #   coefficient collapses, so tau_grow in the hot CGM is formal. Both
│                                   #   conventions are Part 17's own. tau_sp is the RADIUS clock (the MASS
│                                   #   e-folds in tau_sp/3) and both laws are linear in the assumed grain size,
│                                   #   so the CONTRAST is what carries meaning. In +0.5..+2.5 Gyr (m25, 11 upper
│                                   #   vs 19 lower): non-SF ISM tau_grow -0.34 vs -0.61 dex (AUC 0.76), cold-CGM
│                                   #   tau_grow 1.47 vs 1.98 (AUC 0.25 - the upper group's cold CGM can still
│                                   #   grow dust inside a Hubble time, the lower one's cannot), cold-CGM tau_sp
│                                   #   3.46 vs 3.91, hot-CGM tau_sp 0.29 vs 0.35. ISM AND CGM ARE CAESAR'S MEMBER
│                                   #   LIST, NOT A RADIUS (Part 13's definition, inherited by Parts 6 and 6b):
│                                   #   ISM = the gas particles caesar assigns to the galaxy, CGM = everything
│                                   #   else within the cut-out's 100 kpc aperture (P13_RMAX); phases cold < 1e5
│                                   #   K, warm, hot > 1e6 K. WHY POINTS GO MISSING IN PARTS 6 / 6b: a clock bin
│                                   #   is 0.5 Gyr wide and is drawn only where at least P6_NMIN = 3 GALAXIES of
│                                   #   that group have a finite value in it (a per-galaxy median is taken inside
│                                   #   the bin first). The m25 census covers 30 of this notebook's galaxies (11
│                                   #   upper + 19 lower; the intermediates are in neither key) and each galaxy
│                                   #   only has nodes at the snapshots with a reduced particle cut-out, so the
│                                   #   ends of the clock fall below three and the line breaks: 12 of 14 bins are
│                                   #   drawn for the lower group, 11 of 14 for the upper one.
│                                   #   PART 6c (cells p6c-md/p6c-where) - WHERE THE CGM DUST SITS, AND WHETHER IT
│                                   #   CAN COME BACK, the user's hypothesis: the dust-poor galaxies have no
│                                   #   maintenance heating, their dust is simply out there and cannot return,
│                                   #   while the dusty ones heat their CGM AND cycle cold gas back in. Four
│                                   #   measurements of "how far out", all Part 13's: R50 of the CGM dust built
│                                   #   from the four shells (0-10, 10-30, 30-60, 60-100 kpc), the share inside 30
│                                   #   kpc, the mass-weighted median APOCENTRE of the outgoing dust, and its
│                                   #   BOUND share - on the clock, plus the radial profiles (dust, gas,
│                                   #   temperature, dust-to-gas shell by shell) in +0.5..+2.5 Gyr. ->
│                                   #   ba_cgm_where{.png,.pdf,.csv}. THE ANSWER, in +0.5..+2.5 Gyr (m25, 10 vs 17
│                                   #   galaxies): the ejection half of the hypothesis is RIGHT and the heating
│                                   #   half is INVERTED. r_apo 48 vs 77 kpc (AUC 0.12, p 0.001), bound share 0.98
│                                   #   vs 0.85 (AUC 0.87, p 0.002), R50 30 vs 34 kpc (AUC 0.28), dust inside 30
│                                   #   kpc 0.50 vs 0.43 - the dust-poor galaxies' dust went further out and a
│                                   #   sixth of it is unbound, so it cannot come back. BUT the X-ray channel is
│                                   #   on in 100 % of BOTH groups' nodes there (the duty cycles are now panels of
│                                   #   Part 6a, from the census's own xray_on / choked columns) and it is the
│                                   #   DUSTY group whose 30-60 kpc shell is HOTTER (log T 5.6 vs 4.9): they are
│                                   #   not the unheated ones, they are the ones whose dust stayed close enough to
│                                   #   be re-accreted. The cold inflow through 10 kpc is the same in both (AUC
│                                   #   0.51). PART 5 (v11): the X-ray-off box is out of it by default (P5_BOXES -
│                                   #   every branch there is censored, so its Kaplan-Meier curve is a flat line
│                                   #   and its medians are NaN), and the head-start panel is now a CONTOUR plot
│                                   #   (the 50 and 90 % regions of each group, over the galaxies whose death was
│                                   #   SEEN; the censored ones stay as open triangles because a lower limit
│                                   #   cannot enter a density) on a LOGARITHMIC lifetime axis. PART 6b (v11): the
│                                   #   timescales are drawn in GYR on a log axis instead of as log10 - 1 Gyr and
│                                   #   a Hubble time are lines to read against, not numbers to convert.
│                                   #   PART 7 (cells p7-md/p7-m25) - THE 25 Mpc CLOSE-UP, PORTED from
│                                   #   dusty_qgs_kappa_selection Part 5 and re-split on THIS notebook's selection
│                                   #   (its KCLS_OF lookup replaced by the peel's (anchor snap, gal_id) -> group;
│                                   #   the knobs keep a P7_ prefix because this notebook's Part 5 is the dust
│                                   #   lifetime). The only part that reaches outside the ism_prediction products:
│                                   #   it reads the m25 notebook's particle tables and the K-S track products and
│                                   #   each block gates on its own files. (A) the critical points -
│                                   #   kappa_rot(H2), kappa_rot(*), sSFR, stellar age, f_dust, f_H2 at ignition
│                                   #   -> jet -> SFT -> QT -> end in the core (r < 3.2 kpc) and outskirt (3.2-10
│                                   #   kpc) zones, with the ALMA-C11 detections as bands; (B) the radial anatomy
│                                   #   per annulus (0-1, 1-3.2, 3.2-10, 10-32 kpc) with the m25 star-forming
│                                   #   galaxies as the grey control; (C) the R50(H2) tracks SFT -> QT -> end on
│                                   #   the K-S plane against Bigiel+08, with the end regions counted. ->
│                                   #   ba_m25_{sequences,profiles,ks}{.png,.pdf} + ba_m25_closeup.csv. THE ANSWER
│                                   #   (A) IS A TIMING ONE AND IT IS NOT CIRCULAR - the groups come from the dust
│                                   #   map, everything drawn is measured from the PARTICLES by another notebook:
│                                   #   kappa_rot(H2) is IDENTICAL at ignition (0.72 vs 0.70 in the core) and
│                                   #   splits AT THE JET (0.76 vs 0.28 core, 0.85 vs 0.15 outskirt), while
│                                   #   kappa_rot(*) is identical at every stage - the division is made in the
│                                   #   GAS, at the jet, and never touches the stellar structure. The dust itself
│                                   #   only separates at QT/end, which is Part 1b's map seen in the resolved
│                                   #   products. m25: 34 galaxies carry a group (14 upper, 20 lower), 19 of them
│                                   #   are in the K-S tracks.
│                                   #   PART 8 (cells p8-md/p8-gallery) - THE PICTURES, ported from
│                                   #   dusty_qgs_kappa_selection Part 8: one representative galaxy per group
│                                   #   rendered at jet_on -> SFT -> QT -> anchor from the REDUCED PARTICLE
│                                   #   CUTOUTS through simbanator.visualization.rendering.ReducedRender (FIG 1
│                                   #   the RGB composite, stars + gas Screen-blended and attenuated by the dust
│                                   #   column; FIG 2+ one physical map per P8_MAPS field on a shared log scale
│                                   #   with the mass-weighted GAS VELOCITY FIELD as streamlines - face-on a
│                                   #   rotating disc closes them into circles). -> ba_gallery_rgb_cis25 +
│                                   #   ba_gallery_<field>_cis25 + ba_gallery_cis25.csv. THE SHIM IS THE PORT:
│                                   #   that cell was written against its own Part 6 tables, so the top of this
│                                   #   one rebuilds EPO6 (one row per galaxy x critical point: the snapshot
│                                   #   nearest THIS notebook's t_jet / t_SFT / t_QT / t_anchor and the galaxy's
│                                   #   index there, from kappa_m25_branch_ids.csv), ANC6 (the side of the peel +
│                                   #   the kinematic significance the representative is picked on) and GRP6
│                                   #   ("sel" = the upper side of the dust map, "rest" = the lower one) -
│                                   #   everything below the shim is unchanged. 34 m25 galaxies carry a full chain
│                                   #   (14 upper, 20 lower), 136 (galaxy, epoch) rows. It needs the cutouts AND a
│                                   #   working simbanator import and gates cleanly on both, so the notebook still
│                                   #   runs end to end off the cluster.
│                                   #   WHO GETS DRAWN (v14, the user: "we can make it simpler and just set some
│                                   #   ranges of redshift, mass, age and ssfr for the seeds point, then a number
│                                   #   of seeds to plot for each group"): the seed-and-match machinery is GONE.
│                                   #   P8_ZRANGE / P8_MRANGE / P8_AGERANGE / P8_SSFRANGE are windows (either end
│                                   #   None = no bound), P8_NGASMIN is the minimum number of GAS particles at the
│                                   #   anchor - SIMBA carries dust ON the gas particles, so that IS the
│                                   #   dust-carrier count - and P8_NPER is how many galaxies per group the figure
│                                   #   gets. P8_ORDER picks which survivors: "res" (the best resolved, the
│                                   #   default), "median" (closest to the group median in the kinematic axis and
│                                   #   the dust ratio) or "random" (seeded). P8_PICK now takes a LIST per group.
│                                   #   The cell prints the toll of every window and the full line of every galaxy
│                                   #   drawn. PART 1 (v14): "ldh2" = log M_dust/M_H2 and "ldg" = log M_dust/M_gas
│                                   #   joined P1_TRK (the same helium convention as lfh2) because the dust-to-H2
│                                   #   ratio could not be drawn at all before - and PART 3 now RAISES on a P3_TRK
│                                   #   name Part 1 does not track, listing the legal ones, instead of dropping
│                                   #   the row silently, which is exactly how that attempt failed with no error
│                                   #   to read. In m100 the two groups end +1.9 dex apart in ldh2 at +2 Gyr.
│                                   #   PART 9 (cells p9-md/p9-forecast) - THE OBSERVER'S FORECAST, and the one
│                                   #   part that is NOT about the selection: what a SURVEY would find in the
│                                   #   quenched population, redshift by redshift. The sample is the CATALOGUE at
│                                   #   every anchor of the ladder (ten redshifts, z = 2.02 -> 0.31, every galaxy
│                                   #   in the box), not the branch sample. P(disky | QG) with a Wilson interval,
│                                   #   P(f_dust > 1e-4 | disky QG) with the NON-disky QGs as a dotted contrast,
│                                   #   their product P(disky AND dusty | QG), and panel (a) every disky quenched
│                                   #   galaxy's f_dust one by one with the median and 16-84 where N allows. ->
│                                   #   ba_disky_forecast{.png,.pdf,.csv}. "Disky" = kappa_rot(*) > P9_KSTAR (0.5,
│                                   #   the usual rotation-dominated line; the catalogue measures it over hundreds
│                                   #   of star particles so the isotropic noise is ~0.02 and a fixed cut is
│                                   #   honest) and a THRESHOLD LADDER is printed first because the answer depends
│                                   #   strongly on it: at k > 0.40 9-10 % of QGs are disky and 15 % of those are
│                                   #   dusty, at 0.50 it is 1 % and 29 %, at 0.55 0.4 % and 43 %. P9_RULE chooses
│                                   #   the quenched definition - "ssfr" (mass + sSFR, the default) or "notebook"
│                                   #   (Part 0's, which also demands 50 gas + 500 star particles and a dust/H2
│                                   #   floor and therefore removes the gas-poorest, biasing P(dusty) UP; 28429 vs
│                                   #   7689 quenched rows, and both are printed). THE ANSWER (m100, where the
│                                   #   statistics are; m50 is drawn first as the user asked but a disky QG is a 1
│                                   #   % object, so it has 0-10 per redshift): P(disky | QG) 2.2 % at z = 2
│                                   #   falling to 0.9 % at z = 0.3; P(dusty | disky QG) 43 % at z = 2 falling to
│                                   #   25 % at z = 0.3, against 4-16 % for the NON-disky QGs at the same
│                                   #   redshifts - a disky quenched galaxy is 3-6x more likely to be dusty; the
│                                   #   joint P(disky AND dusty | QG) 0.9 % at z = 2 and 0.24 % at z = 0.3.
│                                   #   THE SHIM'S SNAPSHOTS COME FROM THE GALAXY'S OWN CHAIN (fixed 2026-09-11,
│                                   #   the user: "the rgb images are all identical even if i free all selection
│                                   #   boundaries"): the first version mapped each critical point to the nearest
│                                   #   snapshot of the CATALOGUE LADDER, which is ten snapshots per box - so the
│                                   #   four points collapsed onto two or three of them for 27 of 34 galaxies and
│                                   #   the row came out as the same picture two or three times. Part 1 now keeps
│                                   #   the SNAPSHOT NUMBER beside the time in every branch series (R["snap"]),
│                                   #   and the shim picks the nearest snapshot of THAT chain - every snapshot the
│                                   #   galaxy exists at, which is also what the cutouts are named after. 26 of 34
│                                   #   galaxies now get four distinct snapshots; the 8 that do not have two
│                                   #   critical points inside one snapshot gap (a genuinely fast quencher) and
│                                   #   the cell prints that count so it cannot be mistaken for the bug again.
│                                   #   IS kappa_* THE RIGHT REQUIREMENT? (the user, 2026-09-11). It is the right
│                                   #   AXIS and it is NOT a morphology: kappa_rot(*) measures ROTATIONAL SUPPORT,
│                                   #   an observer classifies from LIGHT, and a bulge-dominated disc - an S0 - is
│                                   #   a disc in an image while sitting at kappa_* ~ 0.35-0.50, below any
│                                   #   "rotation-dominated" line. The cell now leads with P(dusty) as a
│                                   #   CONTINUOUS function of kappa_* (panel a), and it rises monotonically and
│                                   #   identically in the two boxes: 4 % at 0.3, 7 % at 0.38, 12 % at 0.42, 18 %
│                                   #   at 0.48, 26 % at 0.52, 43 % above 0.55. So the axis is right and the
│                                   #   THRESHOLD is a visible choice: P9_KALT draws the alternatives (0.40, 0.45)
│                                   #   as thin lines beside the adopted 0.50 in every probability panel. WHAT IS
│                                   #   NOT A SUBSTITUTE: the stellar size - the mass-size residual is
│                                   #   uncorrelated with kappa_* (rho +0.05) and runs the OTHER way (P(dusty) 13
│                                   #   % for the compact half against 3 % for the extended one), so "extended =
│                                   #   disky" would select the opposite population; the cell prints that as a
│                                   #   warning, not as an option. The figure is now 2x3 - (a) the kappa_*
│                                   #   continuum, (b) the kappa_* distribution of the QGs per redshift, (c) every
│                                   #   disky QG's f_dust one point each, (d)(e)(f) the three probabilities with
│                                   #   Wilson bands as filled regions, percentage axes, one figure-level legend
│                                   #   and no top/right spines.
│                                   #   PART 3b (cells p3b-md/p3b-plane) - THE TRACK IN THE f_dust-AGE PLANE, the
│                                   #   user's ask: the clock thrown away and two tracked quantities put against
│                                   #   each other, walked along it. The LINE is the median of each group at every
│                                   #   clock step (age(t), lfd(t)) - a parametric curve, annotated with the clock
│                                   #   time every P3B_DT Gyr; the BAND the 16-84 in f_dust there; the MARKERS the
│                                   #   median of both quantities read AT each event (jet, truncation, quenching
│                                   #   time, anchor), which is a DIFFERENT average from a point of the line and
│                                   #   is said so; the CLOUD every galaxy at its anchor. P3B_X / P3B_Y take any
│                                   #   two tracked quantities. -> ba_plane_<y>_<x>{.png,.pdf,.csv}. THE PICTURE:
│                                   #   the two groups start TOGETHER at the jet (age 0.7-0.9 Gyr, lfd -2.4 to
│                                   #   -2.5 in every box) and separate afterwards - the upper group is still
│                                   #   above the dusty line at age 3 Gyr while the lower one is at -4.5 by age
│                                   #   1.6; at the anchor they are 3.96 vs 3.24 Gyr old and -5.76 vs -6.02 in
│                                   #   m100. A B/T INSTEAD OF kappa_*? (the user, 2026-09-12): nothing in the
│                                   #   catalogue decomposes the light, so what Part 9 can offer is the
│                                   #   two-component READING of kappa_* itself - a kappa_disc disc plus an
│                                   #   isotropic bulge gives B/T = (kappa_disc - kappa_*) / (kappa_disc - 1/3),
│                                   #   drawn as a SECOND AXIS on top of panel (a) (P9_BT, P9_KDISC = 0.85). It is
│                                   #   a relabelling, not a measurement: B/T is monotone in kappa_* here, so it
│                                   #   cannot rank galaxies differently - kappa_* 0.50 is B/T 0.68, 0.45 is 0.77,
│                                   #   0.40 is 0.87. A real B/T needs a bulge-disc decomposition of the mock
│                                   #   images. PART 8 (v18): P8_SKIP takes the NEXT P8_NPER of the ordering
│                                   #   instead of the first ones, and P8_LIST prints the ranked pool per group -
│                                   #   key, z, age, log M*, log sSFR, the dust ratio and the gas-particle count,
│                                   #   with the drawn ones marked - so a galaxy can be named in P8_PICK instead
│                                   #   of hunted for with the windows.
│                                   #   NO TWO PANELS OF A ROW ON THE SAME SNAPSHOT (v19, the user: "even if a
│                                   #   quenching event is short and so star and end are the same snapshot, then
│                                   #   we can simply show the snapshot after that degenerate one"): the four
│                                   #   critical points are resolved to the nearest snapshot of the galaxy's chain
│                                   #   and then made STRICTLY INCREASING - a colliding point is pushed FORWARD to
│                                   #   the next free snapshot (the state just after the degenerate one, which is
│                                   #   what the column means anyway), the anchor holding the last place, and if
│                                   #   that runs past the end of the chain they are pushed back from the anchor
│                                   #   instead. All 34 m25 galaxies now get four DISTINCT snapshots (8 of them
│                                   #   needed a step) where before 27 of 34 repeated a panel; the count of
│                                   #   stepped galaxies is printed.
│                                   #   THE RENDER CACHE IS KEYED ON THE GALAXY (v20, the user: "I plotted 4
│                                   #   sources per class and those 4 sources are exactly the same"): the cell
│                                   #   this was ported from drew exactly ONE galaxy per class, so REN8[(class,
│                                   #   stage)] named a panel uniquely - with P8_NPER > 1 every galaxy of a class
│                                   #   wrote into the same slot and every row of that class drew the LAST one
│                                   #   rendered, while the printed pool and the per-galaxy lines showed the
│                                   #   right, different galaxies. The key is now (class, galaxy key, stage) in
│                                   #   all three places that touch it (write, RGB read, the shared colour scale
│                                   #   of the maps). The frame table also names the galaxy of each row now (it
│                                   #   printed a blank label), and the titles say "N galaxies per group" instead
│                                   #   of "one galaxy per class".
│                                   #   NO TWO ROWS ON THE SAME HISTORY, AND THE GROUPS HAVE THEIR NAMES (v21, the
│                                   #   user: "the seed must be discarded if its history is identical to another
│                                   #   seed... let's change the groups names to dusty QG and dust-poor QG... put
│                                   #   the flow lines on the gas only picture... the kappa must be added as
│                                   #   information on the side, while on the side we should turn dust-to-gas into
│                                   #   just fdust"). FOUR CHANGES. (1) A candidate whose four-epoch chain shares
│                                   #   ANY (snapshot, index) with a galaxy already taken is SKIPPED and the next
│                                   #   one of the ordering is tried, with a printed line saying so - v19 stopped
│                                   #   two panels of one ROW from being the same snapshot, this stops two ROWS
│                                   #   from being the same galaxy seen through different branch ids (the same
│                                   #   physical object can carry more than one anchor row, and the ordering put
│                                   #   them next to each other). (2) P1D_NAMES = ("dusty QG", "dust-poor QG",
│                                   #   "intermediate") - the three groups are NAMED in Part 1b and every figure
│                                   #   from there down uses those names (P1D_COL zips onto them, Part 2's
│                                   #   P2_LABELS/P2_SHORT are just the first two), so no legend says "high" or
│                                   #   "low" any more. (3) The STREAMLINES ARE ON THE VIRIDIS H2 MAP ONLY
│                                   #   (P8_STREAM_ON = ["H2"], P8_MAPS = gas, dust, H2, temp; the FIELD is still
│                                   #   the full gas flow - P8_STREAM_SOURCE "slab", P8_STREAM_WEIGHT "m_gas" -
│                                   #   P8_STREAM_ON only chooses which map it is drawn over, and the molecular
│                                   #   one is where a flow is read against the ISM that is still there): over all
│                                   #   four fields the lines covered the very structure the other maps exist to
│                                   #   show; P8_STREAM_DENSITY 0.55 and P8_STREAM_LW 0.7 keep them a flow rather
│                                   #   than a thicket. (4) THE SIDE NUMBERS are now log M*, age, log
│                                   #   M_dust/M* (f_dust - the quantity the selection is actually made on,
│                                   #   replacing the dust-to-gas ratio) and BOTH kappas, kappa_rot(gas) and
│                                   #   kappa_rot(*) - so Part 7's result (the split is in the GAS kinematics and
│                                   #   never in the stellar ones) can be read off the pictures themselves. The
│                                   #   shim carries kappa_gas_cat / kappa_star_cat into EPO6 for it, and the per-
│                                   #   galaxy line printed by the picker gained lfd and kappa_gas.
│                                   #   THE GALLERY'S FURNITURE (v22, 2026-09-12, the user: "move the description
│                                   #   at the right so that it is aligned with each row... the colorbar is too
│                                   #   small, to far down and its label too close to it... I want to decide if to
│                                   #   add a contour indicating the end of the ISM region and starting of cgm").
│                                   #   THE SIDE BLOCK IS ONE BLOCK, CENTRED ON ITS ROW: the name and the five
│                                   #   numbers are measured (a line is fontsize x linespacing / 72 inches tall,
│                                   #   the figure height turns that into the fraction fig.text works in) and the
│                                   #   whole block's MIDDLE is put on the row's middle, so every row reads across
│                                   #   instead of drifting. THE COLOUR BAR HAS ITS OWN BAND at the foot of the
│                                   #   figure, measured in INCHES so it does not thin out as rows are added
│                                   #   (P8_CBAR_LEN 0.46 of the width, P8_CBAR_H 0.26 in thick, P8_CBAR_GAP 0.30
│                                   #   in under the panels, P8_CBAR_PAD 9 pt between the bar and its label) and
│                                   #   the FIGURE GROWS by that band, so the panels never pay for it; the bar
│                                   #   rides under the bottom row rather than at the bottom of the page. THE
│                                   #   ISM/CGM BOUNDARY is now drawable (P8_ISMEDGE, worth having once P8_SIZE is
│                                   #   wider than the galaxy): the ISM is CAESAR's MEMBER LIST and not a radius
│                                   #   (Part 6's definition), so its edge is a SHAPE - the contour of the member
│                                   #   gas in projection at the level enclosing P8_ISMEDGE_ENC (0.95) of the
│                                   #   member gas mass in the frame, everything outside it CGM. A NUMBER instead
│                                   #   of "members" draws a plain circle of that radius in kpc, None nothing,
│                                   #   P8_ISMEDGE_ON restricts it to some figures, and the cell SAYS SO when the
│                                   #   contour runs off the frame (the box is smaller than the ISM and the line
│                                   #   means nothing). THE GROUPS ARE PLURAL ("dusty QGs" / "dust-poor QGs") and
│                                   #   Parts 4 and 5 no longer say "starburst" / "aging" anywhere a reader can
│                                   #   see - those two cells had the Part 0 names hard-coded in their headlines
│                                   #   and tables (the internal DICT KEYS are still 'burst' / 'aging', and the
│                                   #   printed lines that name the keys say so on purpose); every printed table's
│                                   #   group column now sizes itself to the names.
│                                   #   A SIZE CIRCLE IN EVERY PANEL (v23, 2026-09-12, the user: "maybe we should
│                                   #   add a pysical size bar to each subplot with the corresponding half mass
│                                   #   radius of the baryon component", then "instead of a bar, let's just put a
│                                   #   circle instead, like you would do with an alma beam"). Two markers now:
│                                   #   the WHITE BAR, first column only, is the fixed kpc ruler; the AMBER
│                                   #   CIRCLE, in every panel, has the radius of the galaxy's OWN half-mass
│                                   #   radius AT THAT EPOCH and sits in the bottom-right corner the way an ALMA
│                                   #   beam is put in a map, with the number written against it. P8_R50MARK =
│                                   #   "baryon" (the member stars + gas, the dust riding on the gas) | "star" |
│                                   #   "gas" | "H2" | "dust" | None; P8_R50MARK_POS = "centre" puts the same
│                                   #   circle CONCENTRIC on the galaxy instead - where the aperture actually is -
│                                   #   and a circle too wide for the corner (diameter > 80 % of the frame) falls
│                                   #   back to that by itself, dashed and unfilled. P8_R50MARK_MEMBERS keeps it
│                                   #   to the CAESAR members (the galaxy, not the CGM sharing the frame) and
│                                   #   P8_R50MARK_2D measures it PROJECTED in the frame drawn, which is what an
│                                   #   image measures, rather than in 3D like the catalogue. It is computed from
│                                   #   the particles of that panel (sort by radius, cumulate the mass, take the
│                                   #   half), printed as a range over the grid, and written to
│                                   #   ba_gallery_<box>.csv as r50_kpc / r50_of - so a row shows the ISM
│                                   #   contracting or spreading out in kpc beside the picture of it, and the
│                                   #   amber circle against the cyan ISM/CGM contour is the galaxy against its
│                                   #   own boundary.
│                                   #   PRODUCTS
│                                   #   ba_galaxies.csv (P1) · ba_peel_<q>_<clock>{.png,.pdf,.csv} +
│                                   #   ba_peel_<q>_where{.png,.pdf} + ba_peel_members.csv (P1b) ·
│                                   #   ba_peel_ltauq{.png,.pdf,.csv} (P1c) · ba_split_rule.csv + ba_division.csv
│                                   #   + ba_split (P2) · ba_clocks.csv + ba_clocks (P3) · ba_observe.csv +
│                                   #   ba_observe (P4) · ba_lifetime{,_galaxies}.csv + ba_lifetime (P5) ·
│                                   #   ba_cgm{,_anchor}.csv + ba_cgm (P6); each figure .png AND .pdf. RESULT Part
│                                   #   3 = the clock grid on this selection; Part 4 = the observer's question
│                                   #   (one galaxy, one observation, drawn uniformly in cosmic time from its own
│                                   #   quenched life); Part 5 = t_dark, the LAST down-crossing of f_dust = 1e-4
│                                   #   after which the branch never returns, Kaplan-Meier with a log-rank test
│                                   #   and 100 % censored in cis50nox; Part 6 = the m25 ISM-CGM census re-clocked
│                                   #   on this notebook's own t_SFT. CAVEATS A VALLEY IS NOT TWO POPULATIONS and
│                                   #   neither is a floor - grade the column, refuse the shoulder, refuse the
│                                   #   censoring floor. Sigma_H2 is a DENSITY, not a resolved size (caesar has no
│                                   #   H2 radius). An extremum over a window is a biased estimator and a CHANGE
│                                   #   over a window whose length varies is partly that length. The groups differ
│                                   #   in cosmic time and in mass whatever they are divided on, so every AUC is
│                                   #   quoted raw and inside (epoch x mass) cells. The de-duplication is a
│                                   #   fingerprint, not the exact progenitor index. Part 4's two probabilities
│                                   #   answer different questions (one draw vs a snapshot survey). Part 5
│                                   #   censors, it does not drop. Part 6's census was measured under Part 13's
│                                   #   knobs and is re-clocked here, not re-measured. HISTORY: v3 the
│                                   #   compactification C and the bimodality battery; v4 the configurable window
│                                   #   + the P2_TEST switch + the t_feed cell; v5 the Part 2 rewrite (the spec
│                                   #   language, the cut you choose); v6 the 2D distribution on the clock; v7 the
│                                   #   correspondence map; v8 the peel; v9 (this one) made the peel THE selection
│                                   #   and dropped the 2D-map, correspondence-map, t_feed and checks cells with
│                                   #   it.
│                                   #   PARTS 10-11 (2026-09-13) - THE PAPER FIGURES, two cells that
│                                   #   only READ what Parts 1/1b/2 already built. PART 10 (cells p10-md/p10-paper1)
│                                   #   -> ba_paper_dust_modes{.png,.pdf,.csv}: one figure, three rows, all of them
│                                   #   log M_dust/M*. ROW 1 = THE SELECTION, one panel per resolution - Part 1b's
│                                   #   stage-1 map on the jet clock (each column scaled to its own peak) with the two
│                                   #   peeled REGIONS filled and outlined on it and each group's median track walked
│                                   #   through them (714 dusty / 1039 dust-poor in m100, 91/117 in m50, 15/18 in m25).
│                                   #   ROW 2 = the same dust as a plain distribution at the FIVE EVENTS of a galaxy's
│                                   #   own history: t_jet, t_x-ray, t_q,start (= t_SFT), t_q,end (= t_QT) and t_obs
│                                   #   (the anchor). ROW 3 = at FIVE REDSHIFTS, read at the ladder's own snapshot
│                                   #   nearest each target (z = 3.00, 1.50, 0.99, 0.70, 0.31 = snaps 62/90/105/116/134)
│                                   #   and labelled with its TRUE z - Part 1 now carries the histories' own z in every
│                                   #   series, so the row interpolates nothing and uses no cosmology of its own. WHAT
│                                   #   COUNTS AS TWO MODES IS PART 1b's OWN RULE (P1B_PROM / P1B_DEEP) applied to a 1-D
│                                   #   sample, so the figure and the selection cannot disagree about the word; the
│                                   #   censoring floor is counted, drawn as the grey strip and kept OUT of the KDE (a
│                                   #   delta function at the storage floor is not a population). THE ANSWER IS NOT THE
│                                   #   EXPECTED ONE: no single distribution is bimodal along a galaxy's own history -
│                                   #   ONE mode at t_jet, t_x-ray, t_q,start, t_q,end and at the anchor alike (m100
│                                   #   dusty share 100 -> 78 -> 99 -> 46 -> 9 %: the whole population slides down
│                                   #   together; only m25's anchor comes close, deepest dip 0.78 on 82 galaxies). Two
│                                   #   modes appear ONLY at fixed cosmic time and only in the middle of it: z = 0.99
│                                   #   (valley -4.9, depth 0.51) and z = 0.70 (valley -4.4, depth 0.32) in m100 and
│                                   #   m50, and NOT at z = 3.00, 1.50 or 0.31. AND THAT SECOND MODE IS THE MIXTURE:
│                                   #   PF1_ZSPLIT draws the already-quenched subset of the reference box as a thin
│                                   #   dark-red curve inside the same panel and it is SINGLE-MODED AT EVERY REDSHIFT
│                                   #   (deepest dip 0.87-1.00), so the two modes of the full sample are quenched
│                                   #   (dust-poor) against not-yet-quenched (dusty) galaxies and not two modes of the
│                                   #   quenched population. The two ridges of row 1 therefore live in the TIME
│                                   #   dimension of the map - a fast-falling and a slow-falling branch - and not in any
│                                   #   single-epoch distribution, which is caveat 1 of this notebook exactly. Knobs
│                                   #   PF1_BOXES (add NOX_BOX for a fourth panel), PF1_EPOCHS / PF1_ZS, PF1_ZQ (row 3
│                                   #   over the quenched ones only), PF1_ZSPLIT, PF1_BW, PF1_FTOL, PF1_MED, PF1_CUT,
│                                   #   env BA_PSILV=1 for Silverman's p per panel. PART 11 (cells p11-md/p11-paper2)
│                                   #   -> ba_paper_tracks{.png,.pdf,.csv}: the two groups on the truncation clock in
│                                   #   SIX quantities - log M*, the mass-weighted stellar age, kappa_rot(gas),
│                                   #   -log R50,gas/R50,*, D/G and D/M - for the reference box (PF2_BOX, m100) with the
│                                   #   other resolutions as thin medians behind it and THE NUMBER OF GALAXIES BEHIND
│                                   #   EVERY TRACK drawn in the back of its own panel, on ONE scale for the whole
│                                   #   figure, so an epoch carried by a handful cannot be read as if it were carried by
│                                   #   a thousand; each group's median jet activation (v) and first X-ray firing (|)
│                                   #   are marked on every panel. m100 gaps (dusty - dust-poor) at 0 / +1 / +2 Gyr:
│                                   #   log M* +0.36/+0.42/+0.41, age +0.85/+0.80/+0.72 Gyr, kappa_gas +0.49/+0.35/+0.18,
│                                   #   compactness -0.10/+0.02/+0.02, D/G +0.05/+1.86/+1.74, D/M +0.06/+1.83/+1.61 -
│                                   #   the two groups are IDENTICAL in dust at the truncation and 1.8 dex apart one Gyr
│                                   #   later, and D/M carries it as strongly as D/G, so what separates them is the
│                                   #   DEPLETION and not the enrichment. The jet fires 2.16 Gyr before the truncation
│                                   #   in the dusty group and 0.67 Gyr before it in the dust-poor one; the first X-ray
│                                   #   firing is at +0.05 / +0.10 Gyr in both. FOR THEM PART 1 GAINED TWO THINGS: the
│                                   #   dust-to-metal track ldtm (metal budget Z_gas M_gas + M_dust, the
│                                   #   agn_states_dusty_vs_pure convention; P1_ZDUST = True if a catalogue's zgas
│                                   #   already counts the grains) and the ladder's own redshift in every raw series -
│                                   #   both read the histories' zgas / z columns, so a cache written before them leaves
│                                   #   ldtm and row 3 empty and the cell prints the coverage.
│                                   #   PARTS 12-15 + the two knob changes (2026-09-14, deployed) - FOUR MORE
│                                   #   CHAPTER FIGURES, all pure reads of what Parts 1/1b/2/5/6b/6c already built,
│                                   #   and the answers they gave. THE KNOBS: Part 3b now takes P3B_PLANES = [("age",
│                                   #   "lfd"), ("kappa_gas", "lfd")] and draws one figure per plane (ba_plane_lfd_age
│                                   #   + ba_plane_lfd_kappa_gas: the dynamical state against the dust, walked along
│                                   #   the clock with the events on it - the chapter's question as a picture); Part
│                                   #   10 gained PF1_NOX (row 1 only: the X-ray-off box as a fourth map panel with
│                                   #   THE SAME CELLS laid over it - it prints 102 dusty / 0 dust-poor, the control
│                                   #   in one panel) and PF1_ZSPLIT (row 3: the already-quenched subset of the
│                                   #   reference box as a thin dark-red curve in the same panel; it is single-moded
│                                   #   at every redshift, deepest dip 0.87-1.00, so the z ~ 1-0.7 double mode of the
│                                   #   full sample is the quenched / not-yet-quenched MIXTURE). PART 12 (cells
│                                   #   p12-md/p12-paper3) -> ba_paper_onset{.png,.pdf,.csv}: kappa_gas, Sigma_H2 and
│                                   #   -log R50,gas/R50,* read AT t_jet and AT t_SFT as violins per group, one row
│                                   #   per resolution, AUC raw | ranked inside (epoch x mass) cells, plus the lead
│                                   #   time t_SFT - t_jet. THE ANSWER (m100, 714 vs 1039): kappa_gas already
│                                   #   separates at the jet, 0.69 vs 0.35 (AUC 0.75 | 0.63), and more at the
│                                   #   truncation, 0.77 vs 0.26 (0.88 | 0.76); Sigma_H2 does NOT (0.44 | 0.38 at
│                                   #   the jet, 0.39 | 0.35 at SFT - the dusty ones are if anything LESS dense);
│                                   #   the compactness is INVERTED (0.22 | 0.32: the dusty gas is extended, the
│                                   #   dust-poor gas contracted); and the lead time is the strongest single number
│                                   #   of the onset, 2.16 vs 0.67 Gyr (0.94 | 0.80). m50 identical; m25 the same
│                                   #   direction on 15 vs 18. PART 13 (p13-md/p13-paper4) -> ba_paper_origin
│                                   #   {.png,.pdf,.csv} + ba_paper_origin_galaxies.csv: the LEDGER of the surviving
│                                   #   dust from the raw series - the class over the post-truncation leg against
│                                   #   f_dust = 1e-4 (kept / regrown / faded), the REBOUND of log M_dust from its
│                                   #   post-truncation minimum, and the EX-SITU GROWTH log M*(anchor) - log M*(t_QT)
│                                   #   as the merger axis (a passive galaxy forms nothing in situ; PF4_JUMP = 0.04
│                                   #   dex = a 1:10 event), with the gas book beside it. THE ANSWER (m100): the
│                                   #   dusty group is 8 % kept, 4 % regrown, 88 % FADED by its anchor (dust-poor
│                                   #   0 / 2 / 98; m25 21 / 7 / 71) - the peel's dusty galaxies are dusty on the
│                                   #   jet clock and most have faded by the time they are observed, which is Part
│                                   #   5's lifetime read from the other end; mergers after t_QT are 6 % vs 5 % (0
│                                   #   in m25), the rebound does NOT track the arrivals (rho +0.12 / +0.00), and
│                                   #   of the 4 % regrowers 38 % merged - so mergers can feed at most ~40 % of a 4 %
│                                   #   class and the surviving dust is dust that has NOT YET GONE, not dust that was
│                                   #   brought in; the gas rose after t_QT in 15 % (m25 dusty 33 %). PART 14
│                                   #   (p14-md/p14-paper5) -> ba_paper_timescales{.png,.pdf,.csv}: THE LADDER -
│                                   #   Part 6b's tau_grow (non-SF ISM, SF ISM, cold CGM) and tau_sp/3 (the MASS
│                                   #   clock; non-SF ISM, cold CGM, hot CGM), Part 6c's CGM gas turnover M_gas(<100
│                                   #   kpc)/Mdot_in, Part 5's t_dark (KM median + quartiles) per box and the
│                                   #   follow-up window, per group on one log axis, read in +0.5..+2.5 Gyr. THE
│                                   #   NUMBERS (m25, 10 vs 17): tau_grow 0.46 vs 0.24 Gyr in the non-SF ISM, 0.013
│                                   #   vs 0.007 in the SF ISM, 29 vs 96 in the cold CGM; tau_sp/3 = 38 vs 36 Gyr in
│                                   #   the non-SF ISM, 950 vs 2700 in the cold CGM, 0.64 vs 0.75 in the HOT CGM;
│                                   #   turnover 15 vs 6.4 Gyr; t_dark 1.1 vs 0.28 Gyr (m100), 1.0 vs 0.28 (m50),
│                                   #   2.6 vs 0.77 (m25) against a follow-up of 1.7-3 Gyr. WHAT IT SAYS: the only
│                                   #   destruction clock shorter than the observed lifetime is sputtering in the
│                                   #   HOT CGM (~0.7 Gyr) - in the cool ISM it is 40 Gyr and in the cold CGM
│                                   #   effectively never - so grains die only where they are hot, i.e. after
│                                   #   ejection; and the ISM growth clock (0.2-0.5 Gyr) is shorter than t_dark in
│                                   #   BOTH groups, so a galaxy that keeps its cool ISM keeps making dust. PART 15
│                                   #   (p15-md/p15-paper6) -> ba_paper_timeline{.png,.pdf,.csv}: every event of
│                                   #   each pathway on the truncation clock, median + 16-84 (KM quartiles for the
│                                   #   dust), one lane per group, one row per box. THE SEQUENCE (m100): the jet at
│                                   #   -2.16 vs -0.67 Gyr, the first X-ray firing at +0.05 vs +0.10, the end of the
│                                   #   transition at +0.79 vs +0.32, the dust gone for good at +0.95 (11 % censored)
│                                   #   vs +0.26 (1 %), observed at +1.66 vs +1.85. Notes cell: caveats 12 (the
│                                   #   merger axis is a mass difference, the class is the observable) and 13 (the
│                                   #   rungs are contrasts, not values). ALL SIX PAPER FIGURES WERE RUN LOCALLY ON
│                                   #   THE REAL FOUR-BOX DATA before deploying (the ism_prediction histories, kin
│                                   #   sidecars, catalogue and the two kappa_selection node CSVs copied into a
│                                   #   scratch root; the whole notebook runs there in ~1 min).
│                                   #   THE PAPER PASS (2026-09-14, deployed) - the user's rules for chapter
│                                   #   figures, now in Part 0 as PAPER_RC and applied to Parts 3b / 10 / 12 / 14 +
│                                   #   the new Part 16: NO TITLES anywhere (panels are labelled inside; a row gets
│                                   #   a boxed rotated label), the runs' REAL NAMES on every figure (BOX_LABEL is
│                                   #   now m100n1024 / m50n512 / m25n512 / m50n512 s50nox, the "(sim100)" suffixes
│                                   #   are gone), no diagnostic text in the panels (every statistic stays in the
│                                   #   printout + CSV), one histogram per panel, large labels. Part 2 defines PFC
│                                   #   (the peel's red / blue) where the division is made, so Part 3b can use it.
│                                   #   PART 10: rows 2-3 are the m100n1024 HISTOGRAM and the f_dust = 1e-4 division
│                                   #   line alone (PF1_KDE / PF1_ANNOT / PF1_ZSPLIT off; PF1_NOX off - the control
│                                   #   panel is one knob), boxed row labels "the selection" / "critical points" /
│                                   #   "redshifts" (PF1_ROWLAB). PART 12: the AUC boxes and the lead-time panel
│                                   #   gone (PF3_AUC off, PF3_LEAD None), plain two-line labels (PF3_LAB), the
│                                   #   event ticks say "jet activation" / "quenching onset". PART 3b: P3B_BOXES
│                                   #   (the nox column out), names inside the panels, bold clock numbers, the
│                                   #   16-84 band only where the track is single-valued in x (dotted lines where
│                                   #   it doubles back, which the kappa_gas plane does). PART 14 rebuilt: m25n512
│                                   #   ONLY, x axis "timescale [Gyr]", t_dark -> "dust lifetime t_dust", the
│                                   #   window section -> EVOLUTIONARY TIMESCALES (time since quenching t_obs -
│                                   #   t_q,start, quenching duration t_q,end - t_q,start, jet lead time t_q,start -
│                                   #   t_jet), "CGM turnover" with its formula underneath, and PF5_WHEN = "all":
│                                   #   THE GRAIN CLOCKS AND THE TURNOVER ARE NOW READ OVER EVERY NODE OF A GALAXY'S
│                                   #   HISTORY (~26 per galaxy, 6.9 Gyr of clock), not only in +0.5..+2.5 Gyr after
│                                   #   the truncation; the printout gives both. Over the whole history (10 vs 17):
│                                   #   tau_grow 0.28 vs 0.21 Gyr in the non-SF ISM, 0.010 vs 0.012 in the SF ISM,
│                                   #   15 vs 32 in the cold CGM; tau_sp/3 = 34 vs 33 Gyr in the non-SF ISM, 730 vs
│                                   #   870 in the cold CGM, 0.14 vs 0.41 in the HOT CGM; CGM turnover 20 vs 11 Gyr;
│                                   #   t_dust 2.6 vs 0.77 Gyr; time since quenching 2.2 vs 3.0; quenching duration
│                                   #   0.20 vs 0.05; jet lead time 2.8 vs 0.37 Gyr. The order of the rungs is the
│                                   #   same as in the post-truncation window; the hot-CGM sputtering clock is even
│                                   #   shorter over the whole history (0.14 vs 0.64 Gyr for the dusty group).
│                                   #   PART 16 (cells p16-md/p16-paper7) -> ba_paper_guidance{.png,.pdf,.csv}: THE
│                                   #   OBSERVER'S GUIDANCE, the user's ask ("the probability of observing a disky
│                                   #   QG and find it to be dusty, for different degrees of dustiness and of
│                                   #   rotation, one subplot per redshift"). One panel per anchor of the ladder
│                                   #   (10, z = 0.30-2.02, m100n1024, the WHOLE quenched population under Part 9's
│                                   #   sSFR rule: 5908 QGs at z = 0.3 down to 324 at z = 2.0), P(f_dust >= x |
│                                   #   disky QG) as a continuous function of the dust threshold x (-5.5..-2.5),
│                                   #   one curve per degree of disky with a 68 % Wilson band, the no-morphology
│                                   #   curve dashed, and the share of the quenched population each cut keeps
│                                   #   written in the panel (so the joint probability is on the page too;
│                                   #   PF7_MODE = "joint" draws it directly). "Disky" is Part 9's B/T reading of
│                                   #   kappa_* (kappa_disc = 0.85) - AND IT SATURATES: the median quenched galaxy
│                                   #   has kappa_* = 0.30, i.e. B/T = 1, only 1.5 % have B/T <= 0.7 and B/T <= 0.5
│                                   #   is a few dozen galaxies, so the degrees of disky are B/T <= 0.8 / 0.9 / 0.95
│                                   #   = kappa_* >= 0.44 / 0.39 / 0.36 = the 5 / 13 / 20 % most rotation-supported
│                                   #   QGs, labelled by kappa_* first (notes caveat 14). THE GUIDANCE (m100, at
│                                   #   f_dust >= 1e-4): every QG 5 % dusty at z = 0.3, 8 % at z = 1, 17 % at z = 2;
│                                   #   the 20 % most rotation-supported 9 -> 14 -> 23 %; the 13 % 11 -> 17 -> 28
│                                   #   %; the 5 % most rotation-supported 15 -> 20 -> 35 % - a factor 2-3 over the
│                                   #   whole population at every redshift, rising with z; at f_dust >= 10^-3 the
│                                   #   numbers are 0-6 % for every cut, and the joint probability (share x P)
│                                   #   never exceeds ~5 %.
│                                   #   PARTS 17-18 (2026-09-14, deployed) - the radial profiles and the
│                                   #   displacement / return figure, both m25n512, both pure reads. PART 17
│                                   #   (cells p17-md/p17-paper8) -> ba_paper_profiles{.png,.pdf,.csv}: the paper
│                                   #   version of Part 7 (B) - the projected annuli of the imaging products' truth
│                                   #   tables (annulus_ism_truth / aperture_truth / annulus_av_allincl, four
│                                   #   sightlines per galaxy at its anchor, one value per galaxy = the sightline
│                                   #   median, bootstrap 16-84 of the group median) out to the 32-100 kpc annulus,
│                                   #   i.e. the projected CGM: Sigma_dust, Sigma_H2, M_dust/M* in the annulus, D/G,
│                                   #   A_V and the stellar age, the two groups against the box's star-forming
│                                   #   galaxies (powderday_quenched_selection pop = SF, grey dashed). Only the
│                                   #   peel's galaxies inside the powderday sample are drawn (11 dusty / 8
│                                   #   dust-poor / 266 SF). THE PROFILES: Sigma_dust inside 1 kpc 2e6 (SF) / 2e4
│                                   #   (dusty) / 5e3 (dust-poor) Msun/kpc2 - the quenched galaxies have lost two to
│                                   #   three orders of magnitude of central dust, the dusty ones a factor 4 more
│                                   #   than the dust-poor at every radius inside 10 kpc, and the two meet at 32-100
│                                   #   kpc (2.6e2 vs 2.3e2); M_dust/M* RISES outward in both groups (dusty 6e-6 ->
│                                   #   1.1e-3 from the centre to 32-100 kpc): the dust that is left is in the
│                                   #   outskirts; D/G 4e-4 vs 1e-4 in the centre and equal beyond 10 kpc; A_V 0.05
│                                   #   vs 0.04 in the centre against 1.9 for the star-forming galaxies. PART 18
│                                   #   (p18-md/p18-paper9) -> ba_paper_displacement{.png,.pdf,.csv}: the census
│                                   #   (Part 13 + 17 nodes, 0.12 Gyr apart, re-split + re-clocked; Part 6b's Z6B
│                                   #   when it exists) followed BY PLACE - the ISM and four CGM shells to 100 kpc,
│                                   #   every galaxy normalised to its own total at the truncation, stacked on the
│                                   #   clock (the white above the stack = destroyed or beyond 100 kpc) - and BY
│                                   #   FLUX through r = 10 kpc: outward, inward, inward-cold, the cumulative return
│                                   #   fraction int(in)/int(out) since the truncation, the cold return RATE against
│                                   #   the in-situ growth rate of the ISM dust (Part 17's dm/dt = m_d f_Z/tau_grow,
│                                   #   SF + non-SF ISM), and the bound share of the CGM dust. THE ANSWER (+0.5..+2.5
│                                   #   Gyr, 10 vs 17): the dust within 100 kpc is 15 % (dusty) vs 5 % (dust-poor)
│                                   #   of what the galaxy had at the truncation - 85-95 % of it is destroyed or
│                                   #   gone past 100 kpc within two Gyr; of what remains 41 % vs 20 % is in the
│                                   #   ISM, 4-5 % in the inner CGM, nothing at 60-100 kpc. IT DOES COME BACK: the
│                                   #   all-phase return fraction is 0.90 (dusty) vs 0.37 (dust-poor, AUC 0.82) -
│                                   #   in the dusty group almost everything that crosses 10 kpc outward crosses
│                                   #   back - and 98.5 % vs 85 % of the CGM dust is bound. BUT IT DOES NOT MATTER
│                                   #   FOR THE ISM DUST: the COLD return is 5 % (dusty) / 11 % (dust-poor) of the
│                                   #   ejected, and the cold return rate is 4e-4 (dusty) / 2e-3 (dust-poor) of the
│                                   #   rate at which the ISM grows its own dust; the cumulative cold return would
│                                   #   account for at most half of the dusty group's present ISM dust (0.48, an
│                                   #   upper bound that lets every returning grain survive) and 4x the dust-poor
│                                   #   group's - i.e. it could refill THEIR nearly empty ISM but does not. So the
│                                   #   recycling is real, mostly hot, and the surviving dust of the dusty group is
│                                   #   grown in place, not brought back (notes caveat 15: the budget is within 100
│                                   #   kpc, the return fraction counts every crossing).
│                                   #   THE SECOND PAPER PASS + PART 19 (2026-09-14, deployed). PART 16: the
│                                   #   dust palette (dark to pale brown by degree of disky, every QG grey dashed),
│                                   #   B/T only in the labels ("B/T <= 0.8: 4 %"), N_QG = 5908 in the panel, the
│                                   #   x axis "dusty if log M_dust/M* >=", six anchors z = 0.30 / 0.70 / 0.99 /
│                                   #   1.31 / 1.79 / 2.02 (PF7_ZS). PART 17: the panels are Sigma_dust, Sigma_gas,
│                                   #   M_dust/M*, A_V, Sigma_* and the stellar age, and every panel carries the
│                                   #   WHOLE-GALAXY value (r < 32 kpc, PF8_TOT) as a diamond at the right: total
│                                   #   Sigma_dust 3.1e4 (SF) / 3.0e3 (dusty) / 1.1e3 (dust-poor) Msun/kpc2, f_dust
│                                   #   3.0e-3 / 1.1e-4 / 6.1e-5, A_V 0.96 / 0.072 / 0.020, age 1.7 / 4.9 / 4.0 Gyr.
│                                   #   PART 18 rebuilt as the user asked: no bound share, no flux panel (both stay
│                                   #   in the CSV) - (a, b) the INSIDE / OUTSIDE shares of the dust within 100 kpc
│                                   #   (ISM + four shells, stacked to 1, one panel per group), (c) how much is LEFT
│                                   #   (the total within 100 kpc over the total at the truncation, LOG axis -
│                                   #   readable now: the dusty group falls to 0.15 by +1 Gyr and rebounds to 0.65
│                                   #   at +2, the dust-poor to 0.02), (d) the return fraction int(in)/int(out)
│                                   #   (all phases 0.90 vs 0.37; cold 0.05 vs 0.11), (e) the cold return rate over
│                                   #   the in-situ growth rate (1e-3). PART 19 (cells p19-md/p19-paper10) ->
│                                   #   ba_paper_ks{.png,.pdf,.csv}: THE KENNICUTT-SCHMIDT PLANE, the paper version
│                                   #   of Part 7 (C) - the two groups' median tracks t_q,start -> t_q,end -> end
│                                   #   inside R50(H2), individual tracks faint, the m25 star-forming galaxies as
│                                   #   the cloud, against the VERTICO resolved relation (Jimenez-Donaire+23, 37
│                                   #   Virgo spirals at 720 pc: t_dep = 1.86 Gyr, N = 0.97 +- 0.07, 0.42 dex)
│                                   #   written as a depletion time holding at a pivot of 10 Msun/pc2 with the
│                                   #   slope bending it, INSTEAD OF BIGIEL+08 (still in KS_RELATIONS, one entry of
│                                   #   PF10_RELS away). THE CONVENTIONS ARE ALIGNED: the tracks' Sigma_H2 carries
│                                   #   helium (x1.36, VERTICO's alpha_CO = 4.35 includes it) and the reference
│                                   #   Sigma_SFR is moved to SIMBA's Chabrier IMF by KS_IMF_TO_CHABRIER[KS_REF_IMF]
│                                   #   (Kroupa -> Chabrier x0.94 = -0.03 dex by default; Salpeter would be -0.20).
│                                   #   The IMF the z0MGS FUV + WISE4 calibration is on could not be verified from
│                                   #   here (the A&A full text is 403, the arXiv PDF over the fetch limit, the
│                                   #   abstract and the IRSA column page do not say) - it is a knob and the shift
│                                   #   applied is printed under the figure (notes caveat 16). The constants of
│                                   #   ks_tracks_lib (HE_FACTOR, B08, K98) are copied in verbatim so the cell does
│                                   #   not import the lib and runs anywhere the tracks file is.
│                                   #   PART 18, THIRD SHAPE (2026-09-14, deployed): a 2 x 2 - (a, b) the inside /
│                                   #   outside shares of the dust within 100 kpc (ISM + four shells, stacked to
│                                   #   1) per group; (c) the return fraction int(in)/int(out) through 10 kpc on
│                                   #   the left axis and, on a SECOND axis, the share of the returning dust that
│                                   #   comes back IN COLD GAS (T < 1e5 K; the census splits its fluxes at 1e5 K
│                                   #   only, cold / hot, so there is no warm flux - what is not cold is hot);
│                                   #   (d) ALL the dust returning per unit time over the dust the ISM grows per
│                                   #   unit time. THE NUMBERS (+0.5..+2.5 Gyr, 10 vs 17): returned / ejected 0.90
│                                   #   vs 0.37; the returning dust is 2 % cold (dusty) / 12 % cold (dust-poor),
│                                   #   98 / 88 % hot; all-phase return rate over in-situ growth 0.017 / 0.026 -
│                                   #   the return never dominates what the ISM makes itself, by a factor ~50.
│                                   #   The "how much is left" panel, the fluxes and the bound share stay in the
│                                   #   CSV (notes caveat 15 rewritten).
│                                   #   POLISH (2026-09-14 evening, deployed): PART 16 - the P(dusty | disky) curves
│                                   #   are now one searchsorted per cut with a vectorised Wilson band (0.2 s to
│                                   #   build; the 300-dpi PNG + PDF save is the slow half, ~1 s here, and the cell
│                                   #   prints both numbers - lower PAPER_DPI in Part 0 if the cluster's save is
│                                   #   slow). PART 18 - in panel (c) COLOUR SAYS THE QUANTITY (returned / ejected
│                                   #   dark grey on the left axis; the cold fraction of the return teal on a
│                                   #   teal-coloured right axis) and LINE STYLE SAYS THE GROUP (dusty solid,
│                                   #   dust-poor dashed; PF9_LS), the same style rule in panel (d); the axis labels
│                                   #   are short ("dust share within 100 kpc", "returned / ejected (through 10
│                                   #   kpc)", "cold fraction of the return [%]", "log (returning / grown in the
│                                   #   ISM)"). PART 19 - the individual tracks are off (PF10_TRACKS) and the
│                                   #   OBSERVED POINTS are drawn as KDE CONTOURS enclosing 50 / 90 % of the density
│                                   #   (PF10_CONT / PF10_CLEV): the star-forming cloud filled grey, each group's
│                                   #   points at the end of its track (PF10_CSTAGE) as coloured contours over the
│                                   #   faint points, with the median tracks t_q,start -> t_q,end -> end on top.
│                                   #   POLISH II (2026-09-14 evening, deployed): NO GRIDS on any paper figure
│                                   #   (PAPER_RC axes.grid False + the ten grid() calls removed); PART 16 y axis to
│                                   #   80 % (PF7_YMAX), the redshift in a rounded box, N_QG and the B/T shares
│                                   #   larger; PART 18 legends INSIDE - the places between the two stacks, the
│                                   #   lines (groups, the two quantities) in panel (1,1) with head-room to +1.2,
│                                   #   no bottom margin; PART 12 labels 16.5 pt, ticks 14.5 pt, row names 18 pt,
│                                   #   the reference-line captions 13 pt, panels 5.6 x 4.0 in.
│                                   #   POLISH III (2026-09-14 evening, deployed): PART 12 labels 19 pt / ticks
│                                   #   16.5 / row names 21, the reference captions 15 pt on an opaque box, panels
│                                   #   6.2 x 4.5 in; PART 16 texts 15-17 pt, legend 14.5 pt, the bands are called
│                                   #   "68 % credible interval", no tick on the shared panel edges; PART 18 the
│                                   #   places legend at 20 pt (handles 1.8 x 1.3), nudged left, wspace 0.50;
│                                   #   PART 19 the legend inside (lower right, white-backed) with a two-line
│                                   #   VERTICO entry; PART 17 the annulus-edge lines removed (the light separator
│                                   #   before the "total" column stays).
│                                   #   PART 12 + THE ANCHOR (2026-09-14, deployed): a third column t_obs
│                                   #   ("observed") beside the two onset events, set apart by a dashed line
│                                   #   (PF3_SEP). It closes the argument: at the anchor the kappa_gas difference
│                                   #   is GONE (0.29 vs 0.28 in m100, AUC 0.52 | 0.50), Sigma_H2 has collapsed in
│                                   #   both groups (7.7 vs 7.6) and the compactness gap is 0.05 dex (AUC 0.71 |
│                                   #   0.54) - the dynamical difference is a property of the onset, not of the
│                                   #   observed state, which is why it has to be read from the history.
│                                   #   PART 12b (cells p12b-md/p12b-paper3b, 2026-09-14, deployed) ->
│                                   #   ba_paper_onset_m25{.png,.pdf,.csv}: Part 12's high-resolution companion,
│                                   #   m25n512 ONLY, one panel per quantity - kappa_gas, the compactness, Sigma_H2,
│                                   #   the sSFR, the stellar mass and the dust-to-metal ratio - at the jet, the
│                                   #   truncation and the anchor (set apart by the dashed line); a value on the
│                                   #   sSFR storage floor is kept out of the violin and counted (PF3B_FLOORS). It
│                                   #   reuses Part 12's helpers and event list. THE NUMBERS (14 vs 20): kappa_gas
│                                   #   0.63 vs 0.34 at the jet (AUC 0.67), 0.59 vs 0.25 at the truncation (0.82),
│                                   #   0.25 vs 0.14 at the anchor; the dusty gas is EXTENDED and LESS DENSE at the
│                                   #   onset (Sigma_H2 8.1 vs 9.0 at the jet, 7.9 vs 8.5 at the truncation, AUC
│                                   #   0.21 / 0.23) and denser only at the anchor (7.2 vs 6.9); the sSFR is LOWER
│                                   #   in the dusty group at the jet and the truncation (-9.1 vs -8.9, -9.8 vs
│                                   #   -9.4: the dust-poor group is the starburst) and higher at the anchor (-11.1
│                                   #   vs -11.7); the stellar masses are the same (10.5 vs 10.5 at the jet); the
│                                   #   dust-to-metal ratio is the same at the onset (-0.4 vs -0.2, -0.5 vs -0.3)
│                                   #   and separates only at the anchor (-1.8 vs -3.1, AUC 0.75) - the depletion is
│                                   #   made after quenching, not before.
│                                   #   PART 19, THE OBSERVABLE SPLIT (2026-09-14, deployed): PF10_SPLIT = "fdust"
│                                   #   labels EVERY quenched galaxy of the box in the tracks by the catalogue's
│                                   #   f_dust at its anchor against 1e-4 (89 dusty / 177 dust-poor in the track
│                                   #   sample, 85 / 156 with an end point) instead of Part 1b's peel (11 / 8) -
│                                   #   the contours have statistics now; "peel" puts the notebook's groups back.
│                                   #   THE NUMBERS: at the truncation both sit 1.1-1.2 dex above VERTICO (dusty
│                                   #   log Sigma_H2 3.4, dust-poor 3.7); at the end the dusty ones are +0.47 dex
│                                   #   above it (Sigma_H2 1.8, Sigma_SFR -1.1) and the dust-poor +0.16 (1.3,
│                                   #   -1.8): the dusty end points straddle the star-forming contour's lower half,
│                                   #   the dust-poor ones the relation itself.
│                                   #   PART 19 AT z < 0.5 (2026-09-14, deployed): PF10_ZMAX = 0.5 keeps only the
│                                   #   tracks anchored at z = 0.31 and 0.49 - the quenched galaxies AND the
│                                   #   star-forming cloud - the epoch a z = 0 Virgo reference can be held against
│                                   #   (None = every anchor). 22 dusty / 37 dust-poor in the track sample, 20 / 29
│                                   #   with an end point. THE NUMBERS: at the truncation both 1.1 dex above VERTICO
│                                   #   (log Sigma_H2 3.2); at the end the dusty ones +0.51 dex above it (Sigma_H2
│                                   #   1.7, Sigma_SFR -1.1 - inside the star-forming contour) and the dust-poor
│                                   #   ON it (-0.01 dex; 1.0, -2.3): at low redshift the dusty quenched galaxies
│                                   #   of m25 are, on the K-S plane, still star-forming discs; the dust-poor ones
│                                   #   have arrived on the Virgo relation.
│                                   #   TERSE LABELS (2026-09-14, deployed): Part 12b axis labels are the quantity
│                                   #   alone (PF3B_LAB) and the run name is off (PF3B_NAME); Part 19's legend
│                                   #   (PF10_TERSE) names things and nothing else - dusty QGs / dust-poor QGs,
│                                   #   the circles "Observed", "star-forming galaxies", "50 % / 90 % contour",
│                                   #   "VERTICO (Jimenez-Donaire+23)". CHECKED: the z < 0.5 end points are the
│                                   #   anchors (100 %); on the catalogue SFR every one is below sSFR = 0.2/t_H
│                                   #   (dusty median -0.39 dex, dust-poor -0.64), on the 100-Myr SFR 71 % / 81 %
│                                   #   are; their K-S position is LOCAL - 0.18 Msun/yr inside R50(H2) = 1.1 kpc
│                                   #   (dust-poor 2.7 kpc) - a compact residual core, not a star-forming disc.
│                                   #   LARGE LABELS (2026-09-14, deployed): PAPER_RC is now 18 pt base / 21 pt
│                                   #   axis labels / 18 pt ticks / 17 pt legends, every explicit fontsize in the
│                                   #   paper cells (57 of them) was scaled x1.3, Parts 12 and 12b draw at 23 / 20
│                                   #   pt with 7.8-in panels (three event columns), Part 16 inherits PAPER_RC;
│                                   #   margins and the legend placements were re-fitted (Part 10 bottom 0.145,
│                                   #   Part 16 bottom 0.15, Part 18 15.6 x 11.2 in with the places legend held at
│                                   #   20 pt and shorter y labels, the ladder's 1 Gyr / Hubble tags above the
│                                   #   frame).
│                                   #   THE ONSET AS TRACKS + THE SELECTION ALONE (2026-09-15, deployed; 55 cells). PART 12
│                                   #   (p12-md/p12-paper3) -> ba_paper_onset{.png,.pdf,.csv} + ba_paper_onset_tracks.csv
│                                   #   is now the FULL TRACKS: the median track of each group on the jet clock (PF3_CLOCK)
│                                   #   with its bootstrap 16-84 band (med_band), one row per resolution, three gas
│                                   #   quantities (kappa_gas, Sigma_H2, -log R50,gas/R50,*), and the CRITICAL POINTS ON
│                                   #   THEM - each event of PF3_EV at the group's median offset from the clock and at the
│                                   #   median of the quantity READ AT the event with its 16-84 bar (Part 3b's convention: a
│                                   #   marker is not a point of the line, and both are printed). The observation has no
│                                   #   clock position (the dusty anchors sit 4.3 / 4.2 / 5.7 Gyr after the jet, past the
│                                   #   clock's end), so PF3_STRIP = ["anc"] draws it in a NARROW STRIP beside each panel,
│                                   #   the two groups side by side, the way the violins read it. PF3_STYLE = "violins"
│                                   #   gives the earlier figure back unchanged (PF3_LEAD / PF3_AUC work in both styles);
│                                   #   PF3_MARK, PF3_TLIM (-2 to +4 Gyr; a marker past it is skipped and printed),
│                                   #   PF3_XERR, PF3_STRIPW, PF3_HLPOS (per-quantity side / above-below of the reference
│                                   #   labels). THE OFFSETS (m100): t_q,start at +2.28 Gyr (dusty) vs +0.69 (dust-poor);
│                                   #   the AUCs are unchanged from the violin version. NEW PART 10b (p10b-md/
│                                   #   p10b-selection, between Parts 10 and 11) -> ba_paper_selection{.png,.pdf,.csv}: THE
│                                   #   SELECTION ALONE - Part 10's first row as one 1 x 3 figure, the jet-aligned
│                                   #   log M_dust/M* map of each X-ray-on box (Part 1b's stage-1 per-epoch KDE, each column
│                                   #   scaled to its own peak, colour bar PFS_CBAR) with the dusty and dust-poor REGIONS
│                                   #   filled and outlined on it and nothing else (PFS_MED puts the median tracks back,
│                                   #   PFS_NOX adds the X-ray-off panel, PFS_COUNTS / PFS_ALPHA / PFS_TLIM / PFS_YLIM /
│                                   #   PFS_CUT); the CSV carries the counts (630/1073, 83/121, 14/20) and each region's
│                                   #   extent (dusty t = +1.25..+3.75 Gyr, log f_dust -3.65..-2.38; dust-poor +1.00..+4.00,
│                                   #   -6.42..-4.15). Both cells tested locally on copies of the cluster products (Parts
│                                   #   0-2 + 10b + 12, every knob path); backup starburst_vs_aging.pre-p10b-p12tracks.bak.
│                                   #   PART 12b FOLLOWS (2026-09-15, deployed): the user - "it does not work on part 12b" -
│                                   #   the m25 companion still drew violins because it only borrowed Part 12's violin
│                                   #   helper. Part 12's track panel is now a helper, _pf3_track_panel(ax, axs, box, q, F,
│                                   #   gm, t0, floor, nmin, nboot, hline, ylabel, xlabel, striplab) (+ _pf3_auc_text for
│                                   #   PF3_AUC; _pf3_point takes a floor and an nmin), and Part 12b (p12b-md/p12b-paper3b)
│                                   #   calls it: PF3B_STYLE = None follows PF3_STYLE (or "tracks" / "violins" on its own),
│                                   #   six m25 panels in a 2 x 3 grid, each the median track + bootstrap band of both groups
│                                   #   on the jet clock with the jet and quenching-onset markers on it and the observation in
│                                   #   the strip, the sSFR storage floor censored out of track, reads and violins alike
│                                   #   (PF3B_FLOORS, counts printed), PF3B_NMIN = 3, PF3B_NBOOT; -> ba_paper_onset_m25
│                                   #   {.png,.pdf,.csv} + ba_paper_onset_m25_tracks.csv. m25 offsets: t_q,start +2.89 Gyr
│                                   #   (dusty) vs +0.37 (dust-poor). Part 12's own figure is unchanged by the refactor
│                                   #   (verified in the harness); backup starburst_vs_aging.pre-p12b-tracks.bak.
│                                   #   PART 10c, THE PEEL STAGE BY STAGE (2026-09-15, deployed; 57 cells): the user - "make
│                                   #   the dusty region selection plot as a row showing the progressive peeling just for
│                                   #   the 100 mpc run". New cells p10c-md/p10c-peel (after Part 10b) -> ba_paper_peel
│                                   #   {.png,.pdf,.csv}: Part 1b's FIG 1 row for ONE box in the paper style - stage 1 the
│                                   #   whole m100n1024 map with the dusty region (630 of 3169), stage 2 the same map with
│                                   #   those tracks removed and the dust-poor region (1073 of the 2539 left), stage 3 the
│                                   #   residual (1466 left, 1460 intermediates); every column keeps stage 1's scaling and is
│                                   #   dimmed by the share still in play (PFP_DIM), so the upper ridge visibly empties at
│                                   #   stage 2. Knobs PFP_BOX / PFP_STAGES / PFP_DIM / PFP_MED (the median of what is in
│                                   #   play) / PFP_TEXT (the two-line note per panel) / PFP_CBAR / PFP_ALPHA / PFP_TLIM /
│                                   #   PFP_YLIM. Backup starburst_vs_aging.pre-p10c-peel.bak. ANSWERED THE SAME DAY: the
│                                   #   dusty / dust-poor split of the map is NOT a threshold - the regions are found per
│                                   #   epoch column (KDE at 0.5 x Scott, a valley when two peaks stand 0.15 above the floor
│                                   #   and the minimum falls below 0.75 of the lower one, bright = within 0.6 of the peak of
│                                   #   its own side); the dusty region spans t = +1.25..+3.75 Gyr, log f_dust -3.65..-2.38,
│                                   #   the dust-poor one +1.00..+4.00, -6.42..-4.15; the -4 line (DUSTY_CUT) is the
│                                   #   observable label of Parts 4-6 and the Part 10 histograms, never the selection.
│                                   #   PARTS 12 / 12b ON THE TRUNCATION CLOCK (2026-09-15, deployed): the user - "i am trying
│                                   #   to run 12b to have tracks aligned at sft and to show tjet, tqt and not the tsft
│                                   #   markers (apart from the vertical line) and no horizontal threshold. Also to extend
│                                   #   tracks back in time more to see jet ignition of dusty". 12b had no clock / event /
│                                   #   range knobs of its own (it inherited Part 12's) and the tracks were Part 1's TG grids
│                                   #   (-3..+4). Now: _pf3_track_panel(..., clock=, events=, strip=, tlim=, tstep=) RE-GRIDS
│                                   #   the tracks from the raw snapshot series (_pf3_stack, to_grid on any grid), so a range
│                                   #   beyond TG is free; PF3_EVLAB labels every event (jet/sft/qt/x1/anc), PF3_TSTEP the
│                                   #   step. PART 12b knobs: PF3B_CLOCK = "sft", PF3B_EV = [jet, qt, anc] (the truncation is
│                                   #   the dashed line, not a marker), PF3B_STRIP, PF3B_TLIM = (-6, 4), PF3B_TSTEP,
│                                   #   PF3B_HLINE = {} (no reference lines); m25 offsets from the truncation: t_jet -2.89
│                                   #   (dusty) / -0.37 Gyr (dust-poor), t_q,end +0.24 / +0.05. PART 12 adopted the user's
│                                   #   own cluster edits as defaults: PF3_CLOCK = "sft", PF3_EV = jet / qt / anc (their
│                                   #   "tq" corrected - there is no such event key, the cell would have failed at the first
│                                   #   read), PF3_TLIM = (-4, 4) so the dusty jet (-2.3 Gyr in m100/m50) is on the figure,
│                                   #   PF3_NAMES = False (the run names they had commented out; the rows stay in BOXES
│                                   #   order); PF3_CLOCK = "jet" + ("sft", ...) in PF3_EV gives the jet-aligned version back.
│                                   #   Verified Part 12's tracks are bit-identical on the overlapping grid points before and
│                                   #   after the re-gridding. Backup starburst_vs_aging.pre-p12-sftclock.bak.
│                                   #   2026-09-18 Part 8 (cells p8-md/p8-gallery): the SFH PEAK and TODAY as the two outer
│                                   #   columns — P8_STAGES = [sfr_peak, jet_on, sft, qt, anchor, z0]. "sfr_peak" is read off
│                                   #   Part 1's series (P8_SFPEAK: sfr_max = the SFR maximum over the whole chain, default;
│                                   #   ssfr_pre = Part 1's t_pk; ssfr_max) and takes its place in each row's OWN time order
│                                   #   before the no-two-panels-on-one-snapshot pushes. "z0" (P8_Z0SNAP 151) is the z = 0
│                                   #   DESCENDANT: the chains end at their anchor, so every z = 0 galaxy's main-progenitor
│                                   #   line is walked down the progen sidecars (output/cis25/progen_links, else the
│                                   #   catalogue tree_data) to the anchor snapshot; the lowest-index z = 0 galaxy whose line
│                                   #   passes through the anchor galaxy is the descendant (108 of the 160 branch-map keys,
│                                   #   77-85 % at the z 0.25-0.7 anchors; the rest merged in as the minor partner -> the
│                                   #   panel says so). z0 is SOFT (P8_SOFT): no cutout = an empty panel naming (snap, gx),
│                                   #   never a dropped galaxy, and the cell writes the plan for every missing descendant of
│                                   #   the whole selection -> ba_gallery_z0_plan_cis25.hdf5; ONE sbatch --array=0-0
│                                   #   submit_reduced_particles.sh <plan> builds them (no z=0 cutout exists yet: the
│                                   #   reduced files stop at snapshot 134). The CSV carries drawn / why per panel. Verified
│                                   #   in a mock harness (fake renderer + sidecars + cutouts, the cell end to end); the
│                                   #   descendant walk verified on the real sidecars. Backup
│                                   #   starburst_vs_aging.pre-p8sfpeak-z0.bak.
│                                   #   SAME DAY, after the z=0 run: a panel printed log M_d/M_H2 ~ 32 — not a wrong
│                                   #   galaxy (the descendants' M* runs continuously anchor -> z=0, and the progen links at
│                                   #   151 lose 17 % of lines like every snapshot near it) but the label dividing 75 Msun
│                                   #   of member dust by a 1e-30 floor: three of the four drawn descendants are GAS-FREE
│                                   #   member lists at z=0. The ratio is now printed only with both masses > 0, else
│                                   #   'no H2'. Backup starburst_vs_aging.pre-p8dgrfloor.bak.
│                                   #   2026-09-18 (v36) THE FONTS AND THE QUENCHING NAMES: Part 10c (the m100 peel) and
│                                   #   Part 11 (the tracks on the SFT clock) at DOUBLE text — one dial each (PFP_FS / PF2_FS
│                                   #   = 2, every size and line width follows it), the panels grown with it (PFP_PANEL 8.6 x
│                                   #   9.0, PF2_PANEL 7.2 x 5.6), Part 11 inside its own rc_context (it had none: 11 / 9.5 pt
│                                   #   base sizes), PF2_YLAB = {age: 'stellar age'} (the mass-weighted label is taller than a
│                                   #   panel at 2x), PF2_TITLE False (paper figures carry no title). The quenching times
│                                   #   named for what they are on the tracks and the CGM figures: t_SFT -> t_{q,start},
│                                   #   t_QT -> t_{q,end}, anchor -> t_obs (Parts 6 / 6b / 6c / 18 x-axes, Part 11's _CLAB11
│                                   #   + the truncation mark, Part 12's _CLAB12 whose legend already said t_{q,start};
│                                   #   Part 10c's _CTXP too). Rendered locally (Parts 0-2 + 10c + 11 on the cluster's
│                                   #   products copied to the scratchpad, BA_NBOOT=30) before the deploy. Backup
│                                   #   starburst_vs_aging.pre-p10c-p11-fonts-tq.bak.
│                                   #   SAME DAY (v37): THE BAND ON EVERY FRACTION IS THE BETA POSTERIOR, not Wilson —
│                                   #   Part 0 frac_band(k, n, z, prior) = k/n with the Beta(k+a, n-k+a) quantiles at
│                                   #   Phi(+-z) (BETA_PRIOR 'flat' a=1 | 'jeffreys' a=1/2; z=1 -> 68 %), vectorised over k;
│                                   #   `wilson` is now an alias of it so Parts 1c/4/6/9 pick it up unchanged, Part 16 calls
│                                   #   it directly (its own _wilson_vec removed; legend names the prior). Exact at k=0 / k=n
│                                   #   (Wilson gave [0, 0.077] for 0/12, Beta-flat [0.013, 0.132]); identical above n~50.
│                                   #   NOT cosmic variance: the galaxies are treated as independent draws of ONE box; the
│                                   #   catalogue carries no positions, so an octant jackknife would need them added to
│                                   #   paper_ism_prediction_boxes Part 1 first. Backup starburst_vs_aging.pre-beta-band.bak.
│                                   #   SAME DAY (v38) Part 16b (cells p16b-md/p16b-contrast, after Part 16): DISKY AGAINST
│                                   #   SPHEROIDAL — the user's reading of Part 16 ('the diskier, the less dusty') came from the
│                                   #   in-panel SHARES (4/12/20 %), not the curves; the observer's question is the contrast at
│                                   #   equal numbers. ONE cut PF7B_BT = 0.9 splits every QG of the anchor into disky /
│                                   #   spheroidal, each with its own P(dusty at x) (frac_band), and a log ratio strip under
│                                   #   every panel (band = 16-84 of the ratio of PF7B_NDRAW Beta draws, drawn only where both
│                                   #   classes hold PF7B_KMIN dusty). Answer (m100, f_dust >= 1e-4, per 100 of each): 10.5 vs
│                                   #   3.7 at z 0.3 (ratio 2.8 [2.5, 3.2]) ... 28 vs 14 at z 2 (2.0 [1.6, 2.6]); the ratio is
│                                   #   flat in the threshold down to ~-3.5 -> ba_paper_guidance_contrast{.png,.pdf,.csv}. Part
│                                   #   16 kept (nested cuts, for the appendix); the user's cluster edit PF7_BT = [0.95, 0.85,
│                                   #   0.75] adopted. Backup starburst_vs_aging.pre-p16b.bak.
│                                   #   SAME DAY (v39) Part 16c (p16c-md/p16c-populations, diagnostic): WHERE DOES 'DISKY'
│                                   #   BEGIN — kappa_* histograms (B/T on the top axis) of the SF and the quenched galaxies per
│                                   #   anchor, the Parts 16/16b cuts dotted, the DENSITY CROSSING solid + printed (each
│                                   #   population normalised to one). m100: QGs one narrow peak at kappa_* 0.30 (84th 0.37),
│                                   #   SFGs bimodal (a spheroid component at 0.3 + the disc mode at 0.55-0.6, valley 0.40-
│                                   #   0.45); the crossing sits in that valley, kappa_* 0.38 (z 0.3) -> 0.44 (z 2), B/T 0.91
│                                   #   -> 0.79, with 55-62 % of SFGs and 10-14 % of QGs above it. So the natural threshold is
│                                   #   kappa_* ~ 0.40 = B/T ~ 0.85-0.9 under kappa_disc 0.85 — Part 16b's cut; QGs almost
│                                   #   never reach the SF disc mode (kappa_* > 0.5: 2-6 %). kappa_disc calibrated on the SF
│                                   #   disc mode (~0.58) would put the same crossing at B/T ~ 0.75. -> ba_bt_populations
│                                   #   {.png,.csv}. Backup starburst_vs_aging.pre-p16c.bak.
│                                   #   SAME DAY (v40) Part 16b v2: EXCLUSIVE B/T CLASSES instead of one cut — PF7B_EDGES
│                                   #   [0.8, 0.9] -> B/T <= 0.8 / 0.8-0.9 / 0.9-1 / = 1 (PF7B_SAT: the kappa_* <= 1/3 galaxies
│                                   #   where the relabelling saturates, ~70 % of the QGs, as their own class and the REFERENCE
│                                   #   of the ratio strip). m100 at f_dust >= 1e-4, dusty per 100: z 0.3 = 15.2 / 8.0 / 6.7 /
│                                   #   2.9 (x5.2 / 2.7 / 2.3 vs the spheroids), z 0.7 = 15.9 / 9.7 / 5.1 / 4.7, z 1 = 20 / 14 /
│                                   #   8 / 6.4; monotonic at z <= 1, at z >= 1.3 only the B/T <= 0.8 class stands apart (x2-4)
│                                   #   and 0.8-1 merges with the spheroids (tens of galaxies). Part 16c's top axis capped at
│                                   #   B/T = 1 with the clipped region shaded (the user asked how B/T > 1: the two-component
│                                   #   relabelling drawn past its domain). Backups starburst_vs_aging.pre-p16c-btaxis.bak,
│                                   #   .pre-p16b-classes.bak.
│                                   #   SAME DAY (v41) Part 16b: a STATISTICS BLOCK per panel (PF7B_STATS, under the z label
│                                   #   where the panels are empty): N per class; AUC(kappa_*) of the DUSTY (f_dust >= 1e-4) vs
│                                   #   the other QGs of the anchor — no class, no cut (Part 0 mw_auc: 0.65 at z 0.3, 0.54-0.56
│                                   #   at z 0.7-2, p < 1e-3 / 0.04 / 0.005 / then n.s.); P(ordered) = the posterior probability
│                                   #   from the Beta draws at DUSTY_CUT that every class is dustier than the next more
│                                   #   spheroidal one (84 / 68 / 82 / 31 / 24 / 6 %); P(top > ref) 99.5-100 % everywhere (in
│                                   #   the CSV + printout). Spearman rho of kappa_* vs log f_dust over EVERYONE is ~0 at
│                                   #   every z (-0.08..+0.02) — the bulk sits near isotropic kappa_* with little dust, the
│                                   #   disky-dusty link is a property of the dusty TAIL — kept in the CSV/printout, NOT on the
│                                   #   figure. Backup starburst_vs_aging.pre-p16b-stats.bak.
│                                   #   SAME DAY (v42) Part 16b v3: the panel block is now the COUNTS — N_QG, dusty discs (B/T <=
│                                   #   PF7B_DISC_BT 0.85 & f_dust >= 1e-4) vs dusty spheroids (B/T > PF7B_SPH_BT 0.9): 58 vs 193
│                                   #   at z 0.3 ... 14 vs 37 at z 2 (the AUC / P(ordered) / rho_S table stays in the printout,
│                                   #   PF7B_STATS_FULL); the RATIO STRIP restyled as its own panel (tinted PF7B_RBG, fully
│                                   #   framed, a gap PF7B_RGAP above it via a nested GridSpec, 'ratio to B/T = 1' + the
│                                   #   quantity written in the first strip). Backup starburst_vs_aging.pre-p16b-restyle.bak.
│                                   #   SAME DAY (v43) Part 16b v4: the panel block is the MEDIAN log M_dust/M_gas of every class
│                                   #   (PF7B_DGR, class colours; m100: -4.7 / -5.0 / -5.2 / -5.2 at z 0.3 -> -2.7 / -3.0 / -3.2 /
│                                   #   -3.6 at z 2 — the discs are the dust-richest ISM at every z), no counts / tests on the
│                                   #   figure (all in the printout + CSV, PF7B_STATS_FULL); the 30 % level emphasised (PF7B_YMARK:
│                                   #   a dotted line + a light band below it, in the legend) and the probability axis per ROW
│                                   #   (PF7B_YMAX None: the low-z row stops at 60 %, the high-z row at 90 %); PF7B_RGAP 0.12.
│                                   #   Backup starburst_vs_aging.pre-p16b-dgr.bak.
│                                   #   SAME DAY (v44) Part 16b: the block is delta_DGR = log M_dust/M_H2 (M_H2 x HE, Part 1's
│                                   #   ldh2 convention, PF7B_DGR_HE), written 'median delta_DGR'; z label 17 pt (PF7B_ZFS), block
│                                   #   15 pt, moved down clear of it. m100: -4.6 / -4.9 / -5.0 / -5.1 at z 0.3 -> -2.7 / -2.9 /
│                                   #   -3.6 / -3.6 at z 2. Backup starburst_vs_aging.pre-p16b-dh2.bak.
│                                   #   SAME DAY (v45) Part 8: the FORMATION REDSHIFT on every row label (P8_ZFORM 'age' = z of
│                                   #   the mass-weighted formation time t_obs - <age>_M | 'half' = z at which the branch's own
│                                   #   log M* first reached half its observed value, from Part 1's series; both in the CSV
│                                   #   and the printout). z(t) read off the chain's own (t, z) pairs inside the chain, else
│                                   #   astropy Planck15 z_at_value (the repo's cosmology). Mock harness updated (series carry
│                                   #   z, F carries age). Backup starburst_vs_aging.pre-p8zform.bak.
│                                   #   SAME DAY (v46) BROKEN CHAINS, found by that z_form (the user: 'z_form is after t_jet' —
│                                   #   (116, 105) in m25 has t_form 2 Gyr AFTER its own SFT): the anchor row is a log M* 10.25,
│                                   #   2.2 Gyr-old galaxy while the snapshot before it on its chain is a 10.83 galaxy aged 4.6
│                                   #   Gyr — the progenitor link jumped to the host, so t_jet / t_SFT were the host's. Part 1
│                                   #   now measures the largest one-step FALL of log M* along every branch (F lms_jump, t_jump),
│                                   #   flags F broken (P1_JUMP 0.5 dex) and P1_DROP_BROKEN (default on) removes them before F /
│                                   #   T / R are aligned: m100 7 branches (5 galaxies, 0.2 %), m50 0, m25 6 branches (2 of 82
│                                   #   galaxies; (116, 105) and (134, 110) of the selection). Peel regions 712 / 1039 (were 714
│                                   #   / 1039). A mass-weighted t_form AFTER t_jet is otherwise legitimate (14 of 33 m25
│                                   #   members: the jet leg lasts Gyr and the galaxy keeps forming stars through it); after the
│                                   #   drop none has t_form after its SFT. Backup starburst_vs_aging.pre-p1broken.bak.
│                                   #   SAME DAY (v47) Part 8 MATCHED PAIRS (the user: 'the sources must have similar mass and
│                                   #   age at observed'): P8_MATCH (on) draws the P8_MATCH_SEED group ('sel', the dusty QGs) by
│                                   #   P8_ORDER inside the windows, then gives every seed the closest partner of the other group
│                                   #   in log M* / age / z AT THE ANCHOR (distance = sum of |d| / P8_MATCH_TOL (0.25 dex, 0.75
│                                   #   Gyr, 0.15); no shared chains; the best is taken even outside the tolerances and the
│                                   #   printout says so) -> PAIR8 + a printed pair table; False = the old independent draw.
│                                   #   Verified in the same pass: every row-label number (log M*, age, z_form, lfd, kappas) is
│                                   #   the anchor's own catalogue value, never the z=0 descendant's. Backup
│                                   #   starburst_vs_aging.pre-p8match.bak.
│                                   #   SAME DAY (v48) Part 8 P8_ORDER = 'pairs' (the new default): THE TIGHTEST PAIRS — every
│                                   #   dusty x dust-poor pair of the pool ranked by its distance in log M* / age / z at the
│                                   #   anchor (P8_MATCH_TOL), the P8_NPER tightest drawn (P8_SKIP skips down the ranking), no
│                                   #   galaxy and no shared chain twice, P8_NGASMIN the only resolution floor; 'res' + P8_MATCH
│                                   #   = the previous seed-and-partner rule. Backup starburst_vs_aging.pre-p8pairs.bak.
│                                   #   SAME DAY (v49) Part 8 P8_RGB_SCALE = 'grid' (default): the composite's white point is
│                                   #   ONE per channel over every drawn panel (P8_PMAX percentile of the channel's sigma_map
│                                   #   pixels, P8_DYN decades below), Screen-blended in the cell from the same surface-density
│                                   #   maps the physical figures use — a z=0 descendant with ONE gas particle no longer
│                                   #   stretches it to white and hides its stars (the user's report); 'panel' = the
│                                   #   renderer's per-panel autoscale. Backup starburst_vs_aging.pre-p8rgbgrid.bak.
│                                   #   SAME DAY (v50) P8_RGB_SCALE is PER CHANNEL: {star: 'panel', gas: 'grid', dust: 'grid'} —
│                                   #   on the grid scale the stars of a spread-out z=0 descendant (2x its anchor's R50) fell
│                                   #   below the one decade the star channel shows (the user: 'the stellar component is too
│                                   #   faint'); the stars have no lone-particle problem, so they keep the per-panel top while
│                                   #   gas and dust stay on the grid. Backup starburst_vs_aging.pre-p8rgbstar.bak.
│                                   #   SAME DAY (v51) Part 8 text ALMOST DOUBLED: P8_FS 1.5 -> 2.8, the panels grow with it
│                                   #   (P8_PANEL_IN 4.4 in, rows 1.12 x that; were 2.9 / 3.05 in hard-coded), the row-label
│                                   #   block at P8_LABEL_LS 1.25 spacing (1.6 overflowed the row); the panel ratio is written
│                                   #   as the symbol alone, delta_DGR (= log M_dust / (HE x M_H2), Part 16b's definition; 'dg'
│                                   #   -> delta_DG). Verified on the mock harness (paper_save now writes the PNG to h8/out).
│                                   #   Backup starburst_vs_aging.pre-p8fs.bak.
│                                   #   SAME DAY (v52) Part 8 P8_ZFORM = 'first' (the user's choice): z_form on the row label =
│                                   #   the redshift of the chain's FIRST snapshot, when the progenitor first exists (m25
│                                   #   selection 4.2-9.0, median 5.9). The mass-weighted one ('age') was VERIFIED RIGHT three
│                                   #   ways (Planck15 to 3 decimals, the chain's (t, z), the chain's own SFR integrated: age
│                                   #   within 0.1 Gyr of the catalogue's) but is the MEAN of a history and lands after the SF
│                                   #   peak for a galaxy that kept forming stars — a contradiction beside that column. 'age'
│                                   #   / 'half' / 'first' all in the CSV. Backup starburst_vs_aging.pre-p8zfirst.bak.
│                                   #   2026-09-19 (v53) LABELS: Part 10c (the m100 peel) — PFP_CBLAB 'counts normalized by
│                                   #   peak at epoch' and PFP_REGLAB ['dusty branch', 'dust-poor branch'] naming the two
│                                   #   peeled regions in the legend AND the in-panel notes (the '(stage 1)' / '(stage 2)'
│                                   #   suffixes dropped); Part 12b (ba_paper_onset_m25) — PF3B_EVSHORT names the events by
│                                   #   symbol alone (t_jet / t_q,end / t_obs, no descriptions) under PF3B_LEGTITLE 'markers:
│                                   #   median + 16th to 84th percentile'; the ' (no clock position)' suffix removed from
│                                   #   Parts 12 and 12b. The x axis was ALREADY t - t_q,start since v36 — the cluster's
│                                   #   ba_paper_onset*.png are from 09-15, before that rename, so re-run Parts 12 / 12b.
│                                   #   Backup starburst_vs_aging.pre-p10c-p12b-labels.bak.
│                                   #   2026-10-12 PART 8 SINGLES (P8_SINGLES, default on): every panel of every gallery
│                                   #   figure (rgb + each P8_MAPS map) also saved alone -> ba_gallery_singles_<box>/
│                                   #   <figure>_<group>_a<anchor>_g<gal>_<col><stage>.png (decorated as in the grid, ruler
│                                   #   on every panel; P8_SINGLES_FMT / _DECOR / _TITLE), <name>_raw.png = the bare pixel
│                                   #   array on the grid's colour scale (P8_SINGLES_RAW), cbar_<figure>.png, and
│                                   #   ba_gallery_singles_<box>.csv = one row per figure x galaxy x stage (files, galaxy
│                                   #   numbers gal_* + row_text, epoch, frame/member totals, panel D/G, R50, scale_*). The
│                                   #   panel decoration is one helper (_decor8) shared by grid and singles; the grids are
│                                   #   byte-identical to before (mock harness). Backup starburst_vs_aging.pre-p8singles.bak.
│                                   #   2026-10-12 PART 8s (cells p8s-md/p8s, after the gallery): the powderday SEDs of every
│                                   #   gallery epoch, dust_on (dusty_simdust) + dust_off (nodust_1e-12), looked up first in
│                                   #   the m25 campaign (output/cis25/sed_quenched_regions: only its anchors exist there —
│                                   #   3 of the 24 epochs: 134/68, 116/4, 110/17) then in output/cis25/sed_ba_gallery ->
│                                   #   OUT/ba_gallery_seds_<box>/sed_<group>_a<anchor>_g<gal>_<col><stage>.{fits,csv}
│                                   #   (rest-frame nu L_nu [erg/s] + MC error, 5 apertures x 4 sightlines x 2 arms, NaN
│                                   #   for an arm not run), the index ba_gallery_seds_<box>.csv (sources, L_bol, L_IR,
│                                   #   L_dust = int_{>3um}(on-off), A_V, A_FUV at P8S_AP with the sightline 16-84), the
│                                   #   grid ba_gallery_seds_<box> + single panels. P8S_STAGE = True (cluster, once):
│                                   #   Stage-0 cutouts (campaign radius/ptypes, overwrite False) + MakeSED masters of both
│                                   #   arms (campaign paramf, job names bag_*) for the missing epochs -> submit
│                                   #   sed_ba_gallery/{dusty_simdust,nodust_1e-12}/powderday_sed_out/submit_all_snaps.sh.
│                                   #   Tested on the real 116/4 pair (hyperion in a scratch venv); staging tested with
│                                   #   stubs. Backup starburst_vs_aging.pre-p8s-seds.bak.
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
│                                   #   5d (after 5c) = THE SURVIVOR TEST of the old fully-dosed tail (age
│                                   #   >= 2 Gyr, E_x >= 1 Gyr: tail vs control at fixed dose + age, the
│                                   #   5c puzzle): x_coup(t) rebuilt per galaxy from B's histories (gate
│                                   #   verbatim), duty-cycle anatomy (t_since_x, f_x_late, gap_max,
│                                   #   n_epi), gas re-accretion (regas = log mgas_anchor / min mgas
│                                   #   post-SFT, t_regas) and the shielding proxies (gate_at_sft, w_pre,
│                                   #   e_x_pre, lag_gate), AUC / MW per mechanism + the centrals-only
│                                   #   repeat (satellite chain inheritance); once B's histories carry
│                                   #   mdust (Part 6 HIST_COLS + rebuild) the cell auto-adds the direct
│                                   #   SURVIVOR / REGROWER split (net_lfd vs rebound_lfd at 0.5 dex) and
│                                   #   the dust-route panel -> paper_xray_survivor_anatomy{.png,.pdf,.csv};
│                                   #   5e (after 5c / the 5d survivor cell) = THE SPECIFICITY OF THE 5c AUC
│                                   #   TEST (every catalogue
│                                   #   property scored for the dusty tail in blocks dose / kinematics /
│                                   #   structure / BH / population / gas + a drawn random control; raw AUC
│                                   #   and AUC inside the (age, log M*) quantile cells with a within-cell
│                                   #   label-permutation p, hand-rolled Benjamini-Hochberg FDR per box —
│                                   #   what survives at fixed age + mass; s50nox / s50noagn run the same
│                                   #   battery as channel-off controls, their dose rows fill with the
│                                   #   would-be dose once B builds the nox histories ->
│                                   #   paper_xray_auc_battery{.png,.pdf,.csv}); the nox run also joins the
│                                   #   Section 5 dust distributions as a full outline histogram (not only
│                                   #   the median line) and the 5b grids as its own all-quenched dashed
│                                   #   track (NOX_BOX);
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
├── peel_collage_kinms_model.py     # LOCAL, kinms-env: rebuilds the KinMS best-fit model cube of an ALMA-C11
│                                   #   source from the fitted table (Dynamics/kinms_out/<id>_out.fits; the
│                                   #   fitter set up as in kinms.ipynb: trim_32 cubelet, spectral_trim 25-89,
│                                   #   the skySampler clouds, nrings = the V_i rows) and writes the data + model
│                                   #   moment maps on the fitter's grid (x East, y North, 0.14"/pix), the beam,
│                                   #   the fitted centre / Vsys / inc / PA (receding side), the ring V's with an
│                                   #   `ring_ok` flag (rings beyond the clouds sit at the prior ceiling) ->
│                                   #   output/peel_collage/<id>_kinms_model.npz.  The fit is NOT redone.
├── peel_collage_hc3.py             # LOCAL, base env: the "peel" collage — a square N-up E-left field (--ext 16 kpc
│                                   #   half-size) cut into three 120° slices meeting at the fitted kinematic
│                                   #   centre, the SAME field through three lenses: JWST = the pure Lupton RGB of
│                                   #   the DJA NIRCam F115W/F277W/F444W cutouts (the 60" ones of Cycle 5/Data/
│                                   #   cutouts made by imaging_v1/cutouts.py, else the 20" Nextcloud dja ones;
│                                   #   core-normalised, --obs-gamma; no contours; hc3 has NO MIRI coverage —
│                                   #   F770W masked in COSMOS-Web, every F770W cutout a "no cutout found" stub,
│                                   #   also on the workstation), ALMA =
│                                   #   the KinMS best-fit model velocity field as it comes out of the model cube
│                                   #   (nothing convolved or smoothed beyond KinMS's own beam; a 5 %-of-peak
│                                   #   moment-0 mask, thin iso-velocity lines, --vel-contours 0 removes them),
│                                   #   SIMBA = the source's rank-1 companion (agn_states_companions.csv of
│                                   #   agn_states_dusty_vs_pure Part 27; its reduced cutout, local mirror or the
│                                   #   cluster mount) seen from the observed line of sight (Part 27's _view27 on
│                                   #   the --frame disc frame, default dust_L; i/PA from the table or --inc kin;
│                                   #   turned by 180° about the LOS so its receding side matches the KinMS PA)
│                                   #   as its log Sigma_dust SURFACE DENSITY with the dust-weighted gas flow as
│                                   #   streamlines coloured by v_LOS on the same RdBu scale as the ALMA slice.
│                                   #   --renderer sphviewer (default when it imports) = simbanator's own
│                                   #   ReducedRender (rendering.py loaded standalone, no yt) on the imposed frame
│                                   #   (set_evecs): the notebooks' sigma_map / velocity_field / weighted_map;
│                                   #   py-sphviewer comes from the cluster pd39 env's py_sphviewer + pykdtree
│                                   #   eggs (py3.9 x86_64) put on sys.path from output/peel_collage/
│                                   #   _sphviewer_egg/ (copied 2026-09-07) or from the cluster mount;
│                                   #   --renderer splat = own Gaussian kernels (sigma = --kernel 0.3 x the
│                                   #   --knn-th neighbour distance, signed weights kept).  Annotations are
│                                   #   outreach-style (the THESIS COVER): "Optical+mid-IR observations",
│                                   #   "Dynamics", "Cosmological simulations" WRAPPED around the galaxy, letter
│                                   #   by letter along a ring outside the fade ellipse (--label-r, --label-size,
│                                   #   --label-angles; upright, reading left to right), one kpc bar on the JWST
│                                   #   slice's own sky, plain seams — no title, colour bars, beam or
│                                   #   contours on the image.  --page white (default): the sky fades into a
│                                   #   white page outside an ELLIPSE hugging the galaxy (--fade-a 16.5 kpc along
│                                   #   the source's PA, --fade-q 0.52; fully white from --fade-r1 1.3 x it), the
│                                   #   frame is cropped to that ellipse's bounding box (landscape; --no-crop =
│                                   #   the square --ext field), black text, seams white on the sky / black on
│                                   #   the page — little ink to print; --page black = the all-black version.
│                                   #   --rot sets the seams
│                                   #   (105 = JWST holds the
│                                   #   whole East half, ALMA below with both velocity signs, SIMBA upper-right),
│                                   #   --tag names a variant -> output/peel_collage/peel_collage_<id>[tag].{png,
│                                   #   pdf} + _layers.png (the three layers uncut, the alignment check).
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

Simulation paths live in `config/simulations.json` at the repository root: a plain, visible JSON file you can edit
by hand, never hardcoded in scripts. It is machine-specific, so it is not committed; start from
`config/simulations.example.json` (see `config/README.md`). `SIMBANATOR_CONFIG=/path/file.json` points elsewhere,
and the old hidden `~/.simbanator/config.json` is still read when the visible file does not exist
(`sb.config_path()` tells you which file is in use).

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

Example `config/simulations.json`:

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
