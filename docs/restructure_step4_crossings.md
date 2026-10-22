# Restructure step 4: values that will cross a notebook boundary

Inventory (2026-10-22) of the names a notebook section computes in memory and a section assigned to a
*different* project reads later, using the triage section map (cell ranges -> project) and an AST pass
over the cluster copies (names a cell loads but never binds; setup / config sections and paths excluded,
since the paths go through `io.products` and the knobs into the project parameter files).

When step 7 splits the notebooks, every **data** row must be written as a product (with an `io.registry`
entry) by the section that computes it, or the reading section must recompute it; **style** rows move to
the project parameter file or `projects/<p>/_helpers.py`; **leak** rows are loop / scratch variables
that a later cell reads by accident of execution order, to be made explicit (or removed) at the split.

Notation: `defining section [its project] -> reading project`. The classification is by name pattern:
check each row when its notebook is split.

## Data (become products or are recomputed): 69

| notebook | name | crossing |
|---|---|---|
| agn_states_dusty_vs_pure | `A23` | agn:23[divisions]->ism_distribution |
| agn_states_dusty_vs_pure | `ANC22` | agn:22[divisions]->ism_distribution; agn:22[divisions]->ism_history |
| agn_states_dusty_vs_pure | `CUT28` | agn:28[ism_history]->divisions; agn:28[ism_history]->ism_distribution |
| agn_states_dusty_vs_pure | `DOSE` | agn:2[ism_history]->pipeline |
| agn_states_dusty_vs_pure | `DOSE_ALL` | agn:2[ism_history]->ism_distribution |
| agn_states_dusty_vs_pure | `EPO22` | agn:22[divisions]->ism_distribution; agn:22[divisions]->ism_history |
| agn_states_dusty_vs_pure | `FIG1_22` | agn:22[divisions]->ism_history; agn:22[divisions]->observables |
| agn_states_dusty_vs_pure | `FR28` | agn:28[ism_history]->divisions |
| agn_states_dusty_vs_pure | `FR29` | agn:29[divisions]->ism_history |
| agn_states_dusty_vs_pure | `FR31` | agn:31[ism_history]->divisions |
| agn_states_dusty_vs_pure | `G3` | agn:2[ism_history]->pipeline |
| agn_states_dusty_vs_pure | `GRP22` | agn:22[divisions]->ism_distribution; agn:22[divisions]->ism_history; agn:22[divisions]->observables |
| agn_states_dusty_vs_pure | `GRPS` | agn:2[ism_history]->pipeline |
| agn_states_dusty_vs_pure | `INBOX26` | agn:26[observables]->ism_history |
| agn_states_dusty_vs_pure | `MEA22` | agn:22[divisions]->ism_history |
| agn_states_dusty_vs_pure | `P32_DOSE` | agn:32[divisions]->ism_distribution |
| agn_states_dusty_vs_pure | `S32` | agn:32[divisions]->ism_distribution |
| agn_states_dusty_vs_pure | `_CH22` | agn:22[divisions]->ism_history |
| analogues_specphot_almac11_rt | `union` | a11rt:2[observables]->pipeline |
| analogues_specphot_cis100_rt_agn | `PREFLIGHT_OK` | a100agn:1[observables]->pipeline |
| box_resolution_comparison | `EXTRACT` | box:2[pipeline]->ism_distribution |
| box_resolution_comparison | `SAMPLE` | box:1[ism_distribution]->pipeline |
| dusty_qgs_kappa_selection | `ANC6` | kap:6[divisions]->ism_distribution |
| dusty_qgs_kappa_selection | `BR9` | kap:9[ism_history]->ism_distribution; kap:9[ism_history]->pipeline |
| dusty_qgs_kappa_selection | `CH9` | kap:9[ism_history]->ism_distribution |
| dusty_qgs_kappa_selection | `CLS9` | kap:9[ism_history]->ism_distribution; kap:9[ism_history]->pipeline |
| dusty_qgs_kappa_selection | `EPO6` | kap:6[divisions]->ism_distribution |
| dusty_qgs_kappa_selection | `GRP6` | kap:6[divisions]->ism_distribution |
| dusty_qgs_kappa_selection | `HC` | kap:12[ism_distribution]->ism_history |
| dusty_qgs_kappa_selection | `HOLE12` | kap:12[ism_distribution]->ism_history |
| dusty_qgs_kappa_selection | `KCLS_OF` | kap:2[divisions]->ism_distribution; kap:2[divisions]->ism_history |
| dusty_qgs_kappa_selection | `KEEP9` | kap:9[ism_history]->ism_distribution; kap:9[ism_history]->pipeline |
| dusty_qgs_kappa_selection | `KROT` | kap:2[divisions]->ism_history |
| dusty_qgs_kappa_selection | `NOD12` | kap:12[ism_distribution]->ism_history |
| dusty_qgs_kappa_selection | `ST12` | kap:12[ism_distribution]->ism_history |
| dusty_qgs_kappa_selection | `T9` | kap:9[ism_history]->ism_distribution |
| dusty_qgs_kappa_selection | `Z10` | kap:10[pipeline]->ism_history |
| dusty_qgs_kappa_selection | `_CH6` | kap:6[divisions]->ism_distribution |
| dusty_qgs_kappa_selection | `_gcut` | kap:6[divisions]->ism_distribution |
| ks_tracks_quenched_m25 | `ANCHOR_PROPS` | ks:4[ism_distribution]->divisions |
| ks_tracks_quenched_m25 | `TRACKS` | ks:4[ism_distribution]->divisions |
| noxray_analogues | `C6J` | nox:6d-c34[divisions]->ism_history |
| noxray_analogues | `GN6` | nox:6[divisions]->ism_history |
| noxray_analogues | `KS7` | nox:7[ism_distribution]->ism_history |
| noxray_analogues | `X6L` | nox:6d-c38[divisions]->ism_history |
| paper_figures_quenched_m25 | `AZ` | pfig:2[ism_distribution]->observables |
| paper_figures_quenched_m25 | `GALS` | pfig:2[ism_distribution]->ism_history; pfig:2[ism_distribution]->observables |
| paper_figures_quenched_m25 | `GK` | pfig:2[ism_distribution]->ism_history; pfig:2[ism_distribution]->observables |
| paper_figures_quenched_m25 | `TRACKS` | pfig:2[ism_distribution]->ism_history |
| paper_ism_prediction_boxes | `BOX_GROUPS` | ism:2[ism_history]->pipeline |
| paper_ism_prediction_boxes | `CLS` | ism:6[pipeline]->ism_history |
| paper_ism_prediction_boxes | `Q` | ism:1[pipeline]->ism_history |
| paper_ism_prediction_boxes | `QM` | ism:1[pipeline]->divisions; ism:1[pipeline]->ism_history |
| paper_ism_prediction_boxes | `ROWS_BOXES` | ism:2[ism_history]->pipeline |
| paper_ism_prediction_boxes | `ROWS_P4` | ism:4[ism_history]->pipeline |
| paper_xray_classes_m25 | `GALS` | xray:2[observables]->ism_distribution; xray:2[observables]->ism_history |
| paper_xray_classes_m25 | `GK` | xray:2[observables]->ism_distribution; xray:2[observables]->ism_history |
| paper_xray_classes_m25 | `P9_BOXES` | xray:5[ism_history]->divisions |
| paper_xray_classes_m25 | `QM` | xray:5[ism_history]->divisions |
| paper_xray_classes_m25 | `TRACKS` | xray:2[observables]->ism_history |
| powderday_flux_quenched_m25 | `G8` | pdf:8[observables]->divisions; pdf:8[observables]->ism_history |
| powderday_flux_quenched_m25 | `P8` | pdf:8[observables]->divisions; pdf:8[observables]->ism_history |
| powderday_flux_quenched_m25 | `_G9` | pdf:8h[ism_history]->divisions |
| powderday_flux_quenched_m25 | `_GKEY` | pdf:8[observables]->ism_history |
| powderday_flux_quenched_m25 | `_R_MID` | pdf:8[observables]->divisions; pdf:8[observables]->ism_history |
| powderday_flux_quenched_m25 | `_T1` | pdf:8[observables]->ism_history |
| powderday_flux_quenched_m25 | `_TT` | pdf:8[observables]->ism_history |
| quench_mode_vs_sigma_gas | `PROFILE_CACHE` | qms:2-c18[pipeline]->ism_distribution; qms:2-c18[pipeline]->ism_history |
| residual_dust_quenching | `overdens` | rdq:8[ism_history]->divisions |

## Labels, colours, column lists (parameter file / _helpers): 38

| notebook | name | crossing |
|---|---|---|
| dusty_qgs_kappa_selection | `_KZ6` | kap:6[divisions]->ism_distribution |
| dusty_qgs_kappa_selection | `_SLAB12` | kap:12[ism_distribution]->ism_history |
| dusty_qgs_kappa_selection | `_SSHORT12` | kap:12[ism_distribution]->ism_history |
| ks_tracks_quenched_m25 | `KS_BAND` | ks:4[ism_distribution]->divisions |
| ks_tracks_quenched_m25 | `KS_REF` | ks:4[ism_distribution]->divisions |
| ks_tracks_quenched_m25 | `REGION_COLOR` | ks:4[ism_distribution]->divisions |
| noxray_analogues | `GCOL7` | nox:7[ism_distribution]->ism_history |
| paper_figures_quenched_m25 | `AV_DEF` | pfig:3[observables]->ism_history |
| paper_figures_quenched_m25 | `OBS_S` | pfig:2[ism_distribution]->observables |
| paper_ism_prediction_boxes | `CM` | ism:2[ism_history]->pipeline |
| paper_ism_prediction_boxes | `FONT` | ism:2[ism_history]->divisions; ism:2[ism_history]->pipeline |
| paper_ism_prediction_boxes | `NORM` | ism:2[ism_history]->pipeline |
| paper_ism_prediction_boxes | `SEL_TXT` | ism:2[ism_history]->divisions; ism:2[ism_history]->pipeline |
| paper_ism_prediction_boxes | `_P6_REF` | ism:6[pipeline]->divisions |
| paper_ism_prediction_boxes | `_VARS` | ism:3[divisions]->pipeline |
| paper_ism_prediction_boxes | `_VARS6` | ism:6[pipeline]->divisions |
| paper_ism_prediction_boxes | `_VARS8` | ism:8[divisions]->ism_history |
| paper_ism_prediction_boxes | `_cof` | ism:2[ism_history]->pipeline |
| paper_xray_classes_m25 | `AV_DEF` | xray:2[observables]->ism_history |
| paper_xray_classes_m25 | `FONT` | xray:5[ism_history]->divisions |
| paper_xray_classes_m25 | `KS_BAND` | xray:2[observables]->ism_distribution |
| paper_xray_classes_m25 | `KS_REF` | xray:2[observables]->ism_distribution |
| paper_xray_classes_m25 | `OBS_S` | xray:2[observables]->ism_distribution |
| paper_xray_classes_m25 | `SEL_TXT` | xray:5[ism_history]->divisions |
| paper_xray_classes_m25 | `_SF` | xray:2[observables]->ism_distribution |
| paper_xray_classes_m25 | `_TQ` | xray:2[observables]->ism_distribution |
| powderday_flux_quenched_m25 | `T7_LABS` | pdf:8h[ism_history]->divisions |
| powderday_flux_quenched_m25 | `_CLS8_COL` | pdf:8h[ism_history]->divisions |
| powderday_flux_quenched_m25 | `_CLS8_NAME` | pdf:8h[ism_history]->divisions |
| powderday_flux_quenched_m25 | `_LSCR` | pdf:8h[ism_history]->divisions |
| powderday_flux_quenched_m25 | `_PROF_LABS` | pdf:8[observables]->ism_history |
| powderday_flux_quenched_m25 | `_T7_NAME` | pdf:8h[ism_history]->divisions |
| starburst_vs_aging | `P1D_COL` | sba:1b[divisions]->ism_distribution; sba:1b[divisions]->ism_history |
| starburst_vs_aging | `PFC` | sba:18[ism_history]->ism_distribution; sba:2[divisions]->ism_history |
| starburst_vs_aging | `WIN_LAB` | sba:1[pipeline]->divisions |
| starburst_vs_aging | `WIN_TEX` | sba:1[pipeline]->divisions |
| starburst_vs_aging | `_BX1` | sba:1[pipeline]->divisions; sba:1[pipeline]->ism_distribution; sba:1[pipeline]->ism_history |
| starburst_vs_aging | `_EVTEX` | sba:1[pipeline]->divisions |

## Leaked scratch variables (make explicit): 27

| notebook | name | crossing |
|---|---|---|
| agn_states_dusty_vs_pure | `e` | agn:17[ism_history]->observables; agn:9[ism_history]->ism_distribution |
| dusty_qgs_kappa_selection | `_cov12` | kap:12[ism_distribution]->ism_history |
| dusty_qgs_kappa_selection | `_ghi` | kap:6[divisions]->ism_distribution |
| dusty_qgs_kappa_selection | `_glo` | kap:6[divisions]->ism_distribution |
| dusty_qgs_kappa_selection | `_inv12` | kap:12[ism_distribution]->ism_history |
| dusty_qgs_kappa_selection | `_tsft9` | kap:9[ism_history]->ism_distribution; kap:9[ism_history]->pipeline |
| dusty_qgs_kappa_selection | `e` | kap:4b[ism_history]->ism_distribution |
| noxray_analogues | `present6j` | nox:6d-c34[divisions]->ism_history |
| paper_figures_quenched_m25 | `_av` | pfig:3[observables]->ism_history |
| paper_figures_quenched_m25 | `_cl` | pfig:3[observables]->ism_history |
| paper_figures_quenched_m25 | `_lm` | pfig:3[observables]->ism_history |
| paper_xray_classes_m25 | `_av` | xray:2[observables]->ism_history |
| paper_xray_classes_m25 | `_cl` | xray:2[observables]->ism_distribution; xray:2[observables]->ism_history |
| paper_xray_classes_m25 | `_hc` | xray:2[observables]->ism_distribution; xray:2[observables]->ism_history |
| paper_xray_classes_m25 | `_lm` | xray:2[observables]->ism_history |
| paper_xray_classes_m25 | `_stage_rank` | xray:2[observables]->ism_distribution |
| paper_xray_classes_m25 | `_tq_gk` | xray:2[observables]->ism_distribution |
| powderday_flux_quenched_m25 | `_c8` | pdf:8[observables]->ism_history |
| powderday_flux_quenched_m25 | `_gals8` | pdf:8[observables]->divisions; pdf:8[observables]->ism_history |
| powderday_flux_quenched_m25 | `_kin_ap` | pdf:8h[ism_history]->divisions |
| powderday_flux_quenched_m25 | `_sn8` | pdf:8[observables]->ism_history |
| powderday_flux_quenched_m25 | `_snaps8` | pdf:8[observables]->ism_history |
| powderday_flux_quenched_m25 | `_z_of_snap` | pdf:8[observables]->ism_history |
| residual_dust_quenching | `_sij` | rdq:8[ism_history]->divisions |
| starburst_vs_aging | `_e` | sba:8[ism_distribution]->observables |
| starburst_vs_aging | `_u` | sba:1b[divisions]->ism_distribution |
| starburst_vs_aging | `e` | sba:4[observables]->ism_distribution |
