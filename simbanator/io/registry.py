"""The shared products: every file one notebook (or job) writes and another reads.

One entry per product::

    name: dict(path=..., producer=..., depends_on=(...), params=(...), stale=..., note=...)

- ``path``       template under the output root (:func:`simbanator.io.paths.output_dir`); fields in
                 braces are filled by :func:`simbanator.io.products.path` (defaults in TAG_DEFAULTS)
- ``producer``   the notebook part or job that writes it (named in every "missing" error)
- ``depends_on`` the products it is built from (staleness propagates down these links)
- ``params``     the convention parameters its content depends on; writers stamp their values
                 (:func:`products.stamp`) and :func:`products.check` compares them with CONVENTIONS
- ``stale``      a known reason the existing files are out of date (decision D3: flagged, rebuilt
                 later in one pass); cleared by a rebuild stamped with the current conventions

Edit this file when a product is added, renamed or moved. ``docs/products.md`` is generated from it.
"""

# The conventions currently in force (restructure decisions, docs/restructure_plan.md).
CONVENTIONS = {
    "keep_zero": False,            # D1: SFR = 0 snapshots are below resolution, dropped by the quench finder
    "smooth_window": 1,            # quench finder boxcar (lesson 2026-09: 3 erased the fast mode)
    "require_accretion": True,     # D2: an f_Edd = 0 BH is dormant, jet weight 0
    "frame": "pos@evecs",          # principal-frame coordinates = pos @ evecs (columns = axes)
    "h2_recipe": "caesar-v1",      # build_profiles_job.H2_RECIPE (2026-08-27 fix)
}

TAG_DEFAULTS = {"variant": "", "cls": ""}

_D1 = "D1: built with the SFR = 0 snapshots kept (keep_zero=True)"
_D2 = "D2: x_coup / jet weights gave an f_Edd = 0 BH weight 1 (require_accretion=False)"
_FRAME = "frame bug: profiles measured in pos @ evecs.T (wrong plane); rebuild"

PRODUCTS = {
    # ── pipeline: merger trees, histories, particle files (shared by every project) ────────────
    "progen_links": dict(
        path="{sim}/progen_links/{catalog}",
        producer="find_progen_m25_job.py (submit_find_progen_m25.sh)",
        note="caesar catalogues with tree_data, when the shared ones are read-only (cis25)"),
    "progenitors_anchor": dict(
        path="{sim}/progenitors/progenitors_anchor_{tag}.fits",
        producer="histories.trees (anchor progenitor index; powderday_flux / quench_mode / residual_dust Part 1)",
        depends_on=("progen_links",)),
    "progenitors_most_mass": dict(
        path="{sim}/progenitors/progenitors_most_mass.fits",
        producer="histories.trees (z = 0 progenitor index; residual_dust_quenching I.3)",
        depends_on=("progen_links",)),
    "history_anchor": dict(
        path="{sim}/caesar_sfh/history_anchor_{tag}.hdf5",
        producer="histories.properties save_history_to_hdf5 (powderday_flux / quench_mode / residual_dust Part 1)",
        depends_on=("progenitors_anchor",)),
    "bh_history_anchor": dict(
        path="{sim}/caesar_sfh/bh_history_anchor_{tag}.hdf5",
        producer="histories BH history (powderday_flux / quench_mode Part 1)",
        depends_on=("progenitors_anchor",)),
    "history_passive_z0": dict(
        path="{sim}/caesar_sfh/history_passive_z0.hdf5",
        producer="residual_dust_quenching I.3 (§1)",
        depends_on=("progenitors_most_mass",)),
    "bh_history_passive_z0": dict(
        path="{sim}/caesar_sfh/bh_history_passive_z0.hdf5",
        producer="residual_dust_quenching I.3 (§1)",
        depends_on=("progenitors_most_mass",)),
    "dust_profile_plan": dict(
        path="{sim}/caesar_sfh/prof_{plan}/dust_profile_plan_{plan}.hdf5",
        producer="the notebook that owns the plan tag: kstracks / kstracks_postsft -> ks_tracks_quenched_m25 "
                 "Part 2; boxcomp -> box_resolution_comparison Part 2; the sputtering / almamatch plans -> "
                 "agn_states_dusty_vs_pure Parts 19, 20, 27",
        depends_on=("history_anchor",), params=("keep_zero",)),
    "dust_profile_plan_cis100": dict(
        path="cis100/caesar_sfh/dust_profile_plan.hdf5",
        producer="residual_dust_quenching §5 (the original plan, no tag)",
        depends_on=("history_passive_z0",), params=("keep_zero",)),
    "dust_profiles": dict(
        path="{sim}/caesar_sfh/{kind}_allcrit.hdf5",
        producer="build_profiles_job.py + merge_profiles.py (kind = dust_profiles / cgm_profiles / cgm_temperature)",
        depends_on=("dust_profile_plan_cis100",), params=("h2_recipe", "frame")),
    "reduced_particles": dict(
        path="{sim}/reduced_particles/snap_{snap:03d}/{prefix}_snap{snap:03d}_gal{gal:06d}.h5",
        producer="build_reduced_particles_job.py (submit_reduced_particles.sh <plan>)",
        depends_on=("dust_profile_plan",), params=("h2_recipe",)),
    "filtered_particles": dict(
        path="{sim}/filtered_particles/snap_{snap:03d}/{prefix}_snap{snap:03d}_gal{gal:06d}.h5",
        producer="particles.cutouts (powderday_flux Part 4 / analogues RT notebooks)"),

    # ── m25 sample, RT, attenuation (powderday_flux_quenched_m25) ──────────────────────────────
    "powderday_selection": dict(
        path="{sim}/tables/powderday_quenched_selection{variant}.fits",
        producer="powderday_flux_quenched_m25 Parts 2-3",
        depends_on=("history_anchor", "bh_history_anchor"),
        params=("keep_zero", "require_accretion"), stale=f"{_D1}; {_D2}"),
    "agn_classifier_windows": dict(
        path="{sim}/tables/agn_classifier_windows.fits",
        producer="powderday_flux_quenched_m25 Part 2d",
        depends_on=("history_anchor", "bh_history_anchor"),
        params=("keep_zero", "require_accretion"), stale=f"{_D1}; {_D2}"),
    "agn_classifier_tracks": dict(
        path="{sim}/tables/agn_classifier_tracks.fits",
        producer="powderday_flux_quenched_m25 Part 2d",
        depends_on=("history_anchor", "bh_history_anchor"),
        params=("keep_zero", "require_accretion"), stale=f"{_D1}; {_D2}"),
    "attenuation_vs_ism": dict(
        path="{sim}/tables/attenuation_vs_ism.fits",
        producer="powderday_flux_quenched_m25 Part 7a",
        depends_on=("powderday_selection", "filtered_particles")),
    "cigale_region_results": dict(
        path="{sim}/tables/cigale_region_results.fits",
        producer="powderday_flux_quenched_m25 Part 7f",
        depends_on=("powderday_selection",)),
    "annulus_ism_truth": dict(
        path="{sim}/tables/annulus_ism_truth.fits",
        producer="powderday_flux_quenched_m25 Part 8 (8a)",
        depends_on=("powderday_selection", "filtered_particles")),
    "annulus_av_allincl": dict(
        path="{sim}/tables/annulus_av_allincl.fits",
        producer="powderday_flux_quenched_m25 Part 8",
        depends_on=("powderday_selection",)),
    "annulus_kinematics": dict(
        path="{sim}/tables/annulus_kinematics.fits",
        producer="powderday_flux_quenched_m25 Part 8h",
        depends_on=("powderday_selection", "filtered_particles"), params=("frame",)),
    "aperture_truth": dict(
        path="{sim}/tables/aperture_truth.fits",
        producer="powderday_flux_quenched_m25 Part 7e",
        depends_on=("powderday_selection", "filtered_particles")),
    "reduced_profiles_cache": dict(
        path="{sim}/tables/reduced_profiles_cache_{tag}.hdf5",
        producer="quench_mode_vs_sigma_gas Part 2 (Sigma-cache)",
        depends_on=("reduced_particles",), params=("frame",), stale=_FRAME),

    # ── KS tracks (ks_tracks_quenched_m25) ───────────────────────────────────────────────────
    "ks_track_epochs": dict(
        path="{sim}/ks_tracks/ks_track_epochs.fits",
        producer="ks_tracks_quenched_m25 Part 1",
        depends_on=("powderday_selection", "history_anchor"), params=("keep_zero",), stale=_D1),
    "ks_track_measurements": dict(
        path="{sim}/ks_tracks/ks_track_measurements.fits",
        producer="ks_tracks_quenched_m25 Part 3",
        depends_on=("ks_track_epochs", "reduced_particles")),
    "ks_stage_kinematics": dict(
        path="{sim}/ks_tracks/ks_stage_kinematics.fits",
        producer="ks_tracks_quenched_m25 Part 3b",
        depends_on=("ks_track_epochs", "reduced_particles"), params=("frame",)),
    "ks_stage_budget": dict(
        path="{sim}/ks_tracks/ks_stage_budget.fits",
        producer="ks_tracks_quenched_m25 Part 3c",
        depends_on=("ks_track_measurements",)),
    "ks_track_histories": dict(
        path="{sim}/ks_tracks/ks_track_histories.fits",
        producer="ks_tracks_quenched_m25 Part 5",
        depends_on=("ks_track_epochs", "history_anchor")),
    "ks_tracks": dict(
        path="{sim}/ks_tracks{cls}/ks_tracks.fits",
        producer="ks_tracks_quenched_m25 Part 4 (cls = '' for the quench_window classifier, else _<CLASS_TAG>)",
        depends_on=("ks_track_measurements", "agn_classifier_windows")),
    "ks_galaxies": dict(
        path="{sim}/ks_tracks{cls}/ks_galaxies.fits",
        producer="ks_tracks_quenched_m25 Part 5",
        depends_on=("ks_tracks", "ks_track_histories")),
    "ks_postsft_map": dict(
        path="{sim}/ks_tracks{cls}/ks_postsft_map.csv",
        producer="ks_tracks_quenched_m25 Part 2",
        depends_on=("ks_track_measurements",)),

    # ── box comparison: catalogue-level ISM prediction (paper_ism_prediction_boxes) ───────────
    "ism_prediction_catalogue": dict(
        path="box_resolution/ism_prediction/ism_prediction_catalogue.fits",
        producer="paper_ism_prediction_boxes Part 1"),
    "ism_prediction_histories": dict(
        path="box_resolution/ism_prediction/ism_prediction_histories_{box}.hdf5",
        producer="paper_ism_prediction_boxes Part 6 (build_histories)",
        depends_on=("ism_prediction_catalogue",)),
    "ism_prediction_agn_classes": dict(
        path="box_resolution/ism_prediction/ism_prediction_agn_classes.csv",
        producer="paper_ism_prediction_boxes Part 6",
        depends_on=("ism_prediction_histories",),
        params=("keep_zero", "require_accretion"), stale=f"{_D1}; {_D2}"),
    "ism_prediction_xray_exposure": dict(
        path="box_resolution/ism_prediction/ism_prediction_xray_exposure.csv",
        producer="paper_ism_prediction_boxes Part 9",
        depends_on=("ism_prediction_agn_classes", "ism_prediction_histories")),
    "agn_states_kin_histories": dict(
        path="box_resolution/agn_states/agn_states_kin_histories_{box}.hdf5",
        producer="agn_states_dusty_vs_pure Part 36 (kinematic sidecar)",
        depends_on=("ism_prediction_agn_classes",), params=("frame",)),
    "kappa_m25_branch_ids": dict(
        path="box_resolution/kappa_selection/kappa_m25_branch_ids.csv",
        producer="dusty_qgs_kappa_selection Part 9",
        depends_on=("powderday_selection", "ism_prediction_agn_classes")),
    "kappa_m25_cgm_nodes": dict(
        path="box_resolution/kappa_selection/kappa_m25_cgm_nodes.csv",
        producer="dusty_qgs_kappa_selection Part 13",
        depends_on=("kappa_m25_branch_ids",)),
}
