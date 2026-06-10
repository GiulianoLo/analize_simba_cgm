# Restructuring plan — `residual_dust_quenching.ipynb`

*Drafted 2026-06-10. Nothing in the notebook has been changed by this plan.*

The notebook (112 cells) interleaves three altitudes — infrastructure (build/cache cells),
machinery (plotters, registries), and science — and splits its core story (dust survival)
across §3f–3j, §4a, §5c–5e and §8g, with the AGN/CGM apparatus wedged in between.
Three plans, increasing in depth. **A** is pure cell *moves* (cached outputs survive, no
re-run needed). **B** adds code consolidation (edited cells need one cluster re-run).
**C** is the paper-prep split into module + focused notebooks.

---

## Ground rules / constraints

- **Moving** a cell preserves its cached outputs; **editing or deleting** loses them, and
  outputs can only be regenerated on the cluster.
- Execution order must stay topologically valid. The load-bearing shared state:

  | defined in | symbols | consumed by |
  |---|---|---|
  | §0/§1 | config, `P`, `galaxy_ids`, `snaps_arr`, `t_cosmic_yr`, `redshift` | everything |
  | §3 (cells 12–13) | `records`, `cols`, `is_fast/is_slow`, `mbin`, `STAGES` | everything after |
  | §3b (cell 16) | `_t_to_z`, `_z_to_t` | §3c, §3e–§3h, §4b-iii |
  | §3f (cell 24) | `tg`, `zg_asc`, `QT`, `retain`, `dustyR`, `RETAIN_THR`, `hmbin` | §3g–§3i |
  | §4 (cell 37) | `gather_stage`, `DIAGS` | §4b-ii, §7b, §8 setup, §8b, §8g, §8i |
  | §4b (cell 43/44) | `BH`, `bh_stage`, `Eagn`, `dMbh`, `mdot_qt`, `EPS_R`, `c2_erg_per_Msun` | §4b-i…iv, §7b, §7j, §8a–§8e |
  | §5·pre (cell 52) | `_PROG_INDEX` | §5a, §7a, §7L, §7N, §8c·pre |
  | §5b (cell 57) | `DP`, `DP_RATIO`, `DP_STAGES` | §5c–5e, §8f |
  | §7 setup (cell 63) | `hmbin` (canonical), `HMASS_BIN_*`, `violin_stage_g`, `median_track_g` | §7b–7p, §7N |
  | §7a (cell 65) | `SAT`, `DIAGS_SAT`, `_sij` | §7b, §7f, §7i |
  | §7i (cell 77) | `frac_stage_g` | §7j |
  | §7L (cell 79) | `t_infall`, `cen_full`, `nsat_full` | §8c, §8d, §8h |
  | §7N (cell 81) | `overdens`, `_sij` | §8a, §8b |
  | §8c (cell 94) | `t_agn`, `t_env`, `_estack`, `_track`, `t_inc` | §8d–§8i |
  | §8e (cell 98) | `PROP_REGISTRY`, `_reg`, `event_stack`, `TRIGGERS` | §8f–§8i |
  | §8g (cell 103) | `dust_split` | §8h |
  | §10a | `DQ` | §10b |

- `hmbin` is (re)defined identically in cells 20, 22, 24, 34, 63 (same edges everywhere).
  Under Plan A the duplicates are left alone — they are idempotent — so any move order
  among them is safe. They are the first dedupe target of Plan B.

---

## Plan A — reorder in place (recommended first step; zero re-run)

Target: 8 thematic parts. § numbers are **kept as identifiers** (no renumbering, so all
"Needs §…" cross-references stay true); a new TOC cell maps parts → sections. New cells
added: 1 TOC + 8 part-header markdown cells (pure markdown, no code).

Cell indices below are the **current** positions (0-based, 112 cells).

### Part I — Setup & simulation-level products
*Everything that depends only on the simulation, not on the sample.*

| move | cells | content | dependency note |
|---|---|---|---|
| keep | 0–4 | intro, §0 config, §0b plot style | — |
| keep | 5–8 | §1 progenitors + history build/load | — |
| **up** | 52 | §5·pre `_PROG_INDEX` | needs only §1 (`galaxy_ids`, `snaps_arr`) — safe |
| **up** | 43 | §4b BH-history **build** (gated `BUILD_BH`) | needs only `galaxy_ids`/`snaps_arr`; defines `BH_HIST_PATH`, `EPS_R` used later — safe |

### Part II — Sample, anchors, split
| move | cells | content | note |
|---|---|---|---|
| keep | 9–10 | §2 z=0 passive sample | |
| keep | 11–14 | §3 critical points, fast/slow split, FITS export | |
| keep | 15–16 | §3b timing distributions (defines `_t_to_z`) | |
| **up** | 63 | §7·setup — canonical halo-mass bins + generalized plotters | needs only `P`, `cols`, `records`, labels (§3) — safe; placing it here makes `hmbin` available "officially" from the start |

### Part II-b — Anchor-dependent cached products
*One block of build/load cells; every later science cell only reads memory.*

| move | cells | content | dependency note |
|---|---|---|---|
| **up** | 53–57 | §5 profile config, helpers, 5a·plan, 5a build (gated), 5b load | config needs `STAGES` (§3 ✓) |
| **up** | 44 | §4b BH-history **load** + per-galaxy AGN metrics (`Eagn`, `dMbh`, `mdot_qt`) | needs `records` ✓ |
| **up** | 64–65 | §7a satellite catalogue build/load (`SAT`) | needs `_PROG_INDEX` ✓, and `hmbin`/`NHBINS` from cell 63 ✓ (now Part II) |
| **up** | 71–72 | §7g markdown + CGM-temperature product load (`CT`) | pure load |
| **up** | 93 | §8c·pre local-overdensity history (gated/cached) | needs `_PROG_INDEX` ✓; uses `SPHERE_R` via `globals().get(..., 2.0)` so order vs §7N is safe |
| **up** | 109–110 | §10 markdown + §10a census build/load (`DQ`) | needs only `snaps_arr`/`t_cosmic_yr` ✓ — *or* leave §10a beside §10b in Part VII; either is valid. Recommendation: keep §10 intact in Part VII (fewer split sections), i.e. **don't** move these two. |

### Part III — When and how do they quench?
| cells | content |
|---|---|
| 17–18 | §3c process timescales (intervals) |
| 19–20 | §3d phase durations by stellar/halo mass |
| 21–22 | §3e phase occupancy over cosmic time |

(unchanged order, just grouped under one header)

### Part IV — Does dust survive quenching? (the core result)
| move | cells | content | note |
|---|---|---|---|
| keep | 23–24 | §3f fast fraction of quenched & still-dusty | defines `tg/QT/retain/dustyR/RETAIN_THR` |
| keep | 25–26 | §3g dust timescale (quench clock) | |
| keep | 27–28 | §3h SF-peak premise | |
| keep | 33–35 | §3j threshold-free dust survival | independent of §3f internals |
| keep | 29–32 | §3i synthesis: P(dusty\|quiescent), logistic, τ_dust | after §3f ✓ |
| **down** | 36–40 | §4 DIAGS + 4a/4b/4c/4d reservoir distributions & tracks | `DIAGS` (cell 37) must precede Part V ✓ |
| → Appendix | 41 | §4e example sSFR tracks (QC, not narrative) | optional — see Appendix |
| **down** | 58–61 | §5c/5d/5d-gas/5e dust & gas extension | needs `DP` (Part II-b ✓) |

*Note:* §8g (dusty-vs-non-dusty event stacks) belongs to this story thematically but
depends on the §8c/§8e engine; it stays in Part V with a one-line pointer here.

### Part V — What drives it? AGN vs environment
| move | cells | content | note |
|---|---|---|---|
| keep | 42 | §4b markdown (AGN cross-check rationale) | build/load cells already in I / II-b |
| keep | 45–46 | §4b-i stage tracks, §4b-ii dust-vs-E_AGN correlation | |
| → Appendix | 47–48 | §4b-iii full BH histories (shown **degenerate**; superseded by 4b-iv) | self-contained — safe to relocate |
| keep | 49–50 | §4b-iv quenching-clock BH view | `_stack` defined locally ✓ |
| keep | 62 | §7 markdown | setup cell 63 already in Part II |
| keep | 66–70 | §7b–7f CGM reservoir + satellites across stages | needs `DIAGS` (Part IV ✓), `SAT` (II-b ✓) |
| keep | 73 | §7h CGM temperature tracks | `CT` loaded in II-b ✓ |
| keep | 74–75 | §7p CGM Σ(R) profiles (load+plots are one cell — keep intact) | |
| keep | 76–77 | §7i central/satellite (defines `frac_stage_g`) | before §7j ✓ |
| keep | 78–79 | §7L cluster infall (defines `t_infall`, `cen_full`, `nsat_full`) | before §8c ✓ |
| keep | 80–81 | §7N local overdensity (defines `overdens`) | before §8a ✓ |
| keep | 82–83 | §7j BH/jet mode at critical points | after 77 ✓ |
| → Appendix | 84–85 | §7k M*–M_halo grids (the notebook itself says these washed out the signal) | self-contained |
| keep | 86–98 | §8 setup, 8a, 8b, 8c (md+code), 8d, 8e registry | 8c·pre already in II-b |
| → Appendix | 99 | `!python plot_cgm_stats.py` utility call | unrelated to §8 flow |
| keep | 100–105 | §8f, §8g, §8h | order 8g→8h preserved (`dust_split`) ✓ |
| keep | 106–107 | §8i kinematics | |

### Part VI — Interpretation
| cells | content |
|---|---|
| 108 | §9 (already moved here in the previous pass) |

### Part VII — Observables & number counts
| cells | content |
|---|---|
| 109–111 | §10 intro, §10a census, §10b abundances + N(>S) |

### Appendix — superseded, negative & QC material
*Moved, not deleted — outputs preserved; each gets a one-line "why it's here" note in the
appendix header (added text only in the new header cell, not in the moved cells).*

| cells | content | reason |
|---|---|---|
| 47–48 | §4b-iii pooled BH histories | demonstrated degenerate; 4b-iv supersedes |
| 84–85 | §7k median-in-bin mass grids | least-sensitive readout; §8 supersedes |
| 41 | §4e example sSFR tracks | per-galaxy QC, not narrative |
| 99 | `plot_cgm_stats.py` call | standalone utility |

**Effort:** one scripted pass (same id-keyed JSON approach as the previous edits), ~30 min
including an automated dependency lint (scan each cell's free names against the
definition order). **Re-run cost: zero.**

---

## Plan B — consolidate + renumber (after A; one cluster re-run)

Everything in A, plus code edits. Each edited cell loses its cached output, and the
dependency chain means the practical cost is **one full "Run All" on the cluster**.

1. **Canonical halo bins.** Keep the cell-63 definition (`HMASS_BIN_EDGES/LABELS`,
   `NHBINS`, `hmbin`) as the single source; in cells 20, 22, 24, 34 delete the local
   redefinitions (`HM_EDGES/_hmbin`, `HMASS_EDGES_E/hmbin_e`, `HMASS_EDGES/hmbin`,
   `HEDG/hb`) and alias to the canonical names.
2. **One `_track_start` helper.** Cells 18, 20, 22 each define the same function
   (`_track_start_yr` / `_tstart_yr` / `_tstart_gyr`); define once next to `records` (§3).
3. **Plotter dedupe.** §4's `violin_by_stage`/`median_track` are special cases of §7's
   `violin_stage_g`/`median_track_g`; keep only the generalized pair (defined in Part II)
   and call them with `mbin`/`MASS_BIN_LABELS` in §4.
4. **Trim the figure walls.** In §8e/§8f/§8g/§8h/§8i demo calls, cut the trigger list from
   7 to the informative 3 (`t_env`, `t_AGN`, `t_QT`; keep `t_SFpeak` only in the
   wide-window kinematics call) and split the 9-property §8g call into two 4-row calls.
5. **Renumber sections** to match the new part order and update every "Needs §…" comment.
   Proposed map (old → new): §3b→§2.3 · §3c/3d/3e→§3.1–3.3 · §3f–3j→§4.1–4.5 ·
   §4→§4.6 · §5c–5e→§4.7 · §4b→§5.1 · §7→§5.2 · §7i/7L/7N/7j→§5.3–5.6 ·
   §8→§5.7 · §9→§6 · §10→§7 · appendix→§A.1–A.4.
   This is the churn-heavy step — ~60 markdown/comment touch-points — and is only worth
   doing together with items 1–4 since the re-run is shared.

---

## Plan C — module + focused notebooks (paper-prep)

The natural decomposition exists already because every heavy intermediate is a cached
HDF5/FITS product.

1. **Extract `rdq_core.py`** (repo root or `simbanator.workflows.residual_dust`):
   config block, history loader, `build_records`/`build_split`, `gather_stage`/`DIAGS`
   builder, halo/stellar bin definitions, the three generalized plotters, `_wilson`,
   `_binned_median`, and the §8c/§8e clock-stack engine (`_estack`, `event_stack`,
   `PROP_REGISTRY`, `TRIGGERS`). The critical-points FITS (cell 14) already persists
   `records` + `CLASS`, so notebooks can also *load* the sample instead of recomputing.
2. **Split into five notebooks**, each opening with `from rdq_core import ...` + product
   loads only:
   - `rdq_01_sample_products.ipynb` — Parts I, II, II-b (all builds, QC plots)
   - `rdq_02_quenching_timescales.ipynb` — Part III
   - `rdq_03_dust_survival.ipynb` — Part IV (+ §8g, since the engine lives in the module)
   - `rdq_04_drivers_agn_environment.ipynb` — Part V
   - `rdq_05_observables_counts.ipynb` — Parts VI–VII
3. **Contract between notebooks** = the cached files only:
   `history_passive_z0.hdf5`, `residual_dust_critical_points.fits`,
   `bh_history_passive_z0.hdf5`, `dust_profiles_allcrit.hdf5`, `satellites_allcrit.hdf5`,
   `cgm_temperature_allcrit.hdf5`, `cgm_profiles_allcrit.hdf5`,
   `env_history_passive_z0.hdf5`, `dq_census_allsnaps.hdf5`.
   Missing piece: `t_infall`/`cen_full`/`nsat_full` (§7L) and `overdens` (§7N) are
   currently in-memory only → persist them (one small HDF5 each) so notebook 04 doesn't
   depend on re-running its own §7L/§7N and notebook 03's §8g doesn't depend on 04.
4. **Cost:** module extraction is mostly mechanical (the functions are already
   self-contained); full re-run of all five notebooks once; README "Package layout"
   section updated (per repo convention).

---

## Recommended sequence

1. **Plan A now** — immediate readability, zero risk, zero re-run.
2. **Plan B** bundled with the *next* science change that already forces a cluster
   re-run (it shares the same "Run All").
3. **Plan C** when the paper figures freeze — the module then doubles as the
   reproducibility layer for the publication.
