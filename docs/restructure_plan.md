# Restructure plan: steps 4-9

Agreed target (2026-10-21): `simbanator/` organised by task; notebooks grouped in `pipeline/`
(shared cluster products), `projects/{ism_distribution, ism_history, divisions, observables,
visuals}` and `archive/`. Placement rule: used once -> stays in its notebook; reused inside one
project -> `projects/<p>/_helpers.py`; used by two projects -> library. The per-function and
per-section decisions live in the triage page (https://claude.ai/artifact/5y7gDUGPC3TL3WWPxPiuBb).

Done: steps 1-2 (repo clean-up, task layout, commit 70e87a3) and step 3 (notebook functions
lifted into the library, commits a01a845..f652e68, see `restructure_step3.md`). All work stays
on branch `restructure` until step 9.

## Decisions (2026-10-22)

| id | decision | affects |
|---|---|---|
| D1 | SFR = 0 snapshots do NOT count: a zero SFR is below resolution, not a measurement. Library default `keep_zero=False` (flipped 2026-10-22). Checked on the 2026-09-02 variants table (`qt_smoothing_variants.csv`): at `smooth_window=1` dropping the zeros keeps the log(tau_q/t_H) bimodality (dBIC cis100 +1560 vs +1465 kept; same modes and trough); the smoothing, not the zeros, is what decides the fast mode | step 5 |
| D2 | A non-accreting BH (f_Edd = 0) is DORMANT: `jet_weight(require_accretion=True)` (already the library default). The x_coup products of paper_ism_prediction Part 6 used the other convention and change when rebuilt | step 5 |
| D3 | No rebuilds now. The frame-bug products (`reduced_profiles_cache_*`, box_resolution Part 3) and the D1/D2-affected products are flagged stale in the registry and rebuilt later, in one pass | step 5, rebuild pass |
| D4 | NO default division. The divisions project holds named, optional schemes (e.g. by X-ray injection, by AGN activation); an analysis opts into one by name. The label table has one column per scheme and no default | step 6 |
| D5 | The uncommitted Aug-Sep work is what the final analysis used: it is integrated into the package. The cluster copies of the four modified notebooks are newer than the local ones and are the source of truth (see step 4) | step 4 |
| D6 | `restructure` merges into `main` only after the full reshaping, the regenerated outputs and the checks | step 9 |

## Step 4: products registry, repo root, job scripts

Status (2026-10-22): done except the cluster deploy (see the last bullet).
- Triage coverage: all 883 triaged names checked; the only notebook definitions the triage never saw are
  `_double_exp18` / `_fit_double_exp18` (dusty_qgs, single use: stay) and 4 in sph_visual (archived).
- The 4 old test failures fixed: `Simba(data_dir=...)` now loads the bundled snap -> z map (every SIMBA box
  shares it); the ks_tracks test fixture writes the 2026-09-02 `nh` / `zgas` fields.
- `io.paths.repo_root / output_dir / data_file` (config `"paths"` section optional); the task modules'
  `getcwd()` defaults now go through `output_dir()`.
- `io.registry` (38 products, CONVENTIONS) + `io.products` (path / require / read / dependents /
  stamp / provenance / check / table); the D1-D3 stale flags are on the affected entries.
- In-memory crossings: `docs/restructure_step4_crossings.md` (69 data, 38 style, 27 leaked names).
- Jobs: `build_profiles_job` imports gas_components / nh_from_density / gas_temperature / header_units /
  frames from the library (checked bit-identical against the committed copy); the reduced job imports the
  library directly; `find_progen_m25_job` reads the cis25 paths from the config; outputs via `output_dir`.
  `submit_*.sh` use `$SLURM_SUBMIT_DIR` and `$CONDA_BASE` (default ~/miniconda3). `submit_flux_extraction.sh`
  calls a `run_extract_flux.py` that no longer exists (legacy, archived in step 8).
- `pyproject.toml` 0.3.0 (+pandas, tqdm; astroquery in [sed]; package data = rt/masters), `setup.py` a
  shim, `environment.yml` for pd39.
- Not yet: writers do not call `products.stamp` (step 5, when the notebooks switch to the library);
  deploy of the package + jobs to the cluster (diff first).

- D5, done 2026-10-22 (uncommitted): every notebook whose cluster copy was newer was pulled
  (agn_states_dusty_vs_pure, box_resolution_comparison, dust_attenuation_vs_bt_z03, ks_tracks,
  noxray_analogues, paper_figures, paper_ism_prediction_boxes, paper_xray_classes,
  powderday_flux, starburst_vs_aging; the local copies are in `backups/notebooks/*.pre-clusterpull.bak.ipynb`).
  Code differences were knob values and commented-out titles only; dusty_qgs's cluster copy equals
  the committed one. Still to do: triage and lift the functions the triage has not seen
  (dusty_qgs Part 19: `_km19`, `_logrank19`, `_cross19`, ...; noxray Part 6 additions), and commit
  with the local-only work (analogues_specphot_cis100/_rt/cis25, peel_collage_*.py, the almac11
  orientation table, tests/test_rendering_frame.py).
- Fix the 4 tests that already failed before the restructure (test_io snapshot->z mapping x2,
  test_ks_tracks vel/backfill field sets x2).
- `io.paths.repo_root()` (from the config, not the working directory) and `io.paths.data_file(name)`
  for `obs_data/`, filter `.res` files and the RT masters. Required before any notebook moves:
  20 notebooks use `getcwd()`, 7 import `ks_tracks_lib` from the root, 9 read `obs_data/`
  relatively, 2 read `ALMA_band6.res` from the root.
- `io.products` registry: one entry per shared product with
  `name -> path template, producer (notebook part or job), depends_on, key parameters`.
  `products.path(name, **tags)`, `products.read(name, ...)`, and a missing file names its producer.
- Provenance: every writer stamps git hash, library version and the key parameters
  (`keep_zero`, `require_accretion`, gate, frame convention) into the file attributes;
  `products.check(name)` compares them with the current defaults and lists stale descendants
  through `depends_on`.
- Inventory the values that only live in memory but are read by another part
  (e.g. residual_dust's `t_infall`, `overdens`). Every one that crosses a future notebook boundary
  gets written as a product with a registry entry.
- Job scripts (`build_profiles_job.py`, `build_reduced_particles_job.py`, `find_progen_m25_job.py`)
  import the library (`build_profiles_job` still has its own `_components`) and resolve paths
  through the registry; the `submit_*.sh` wrappers lose their hard-coded root paths.
- `pandas` into the `pyproject.toml` dependencies; an `environment.yml` describing the pd39 kernel
  (and noting `statsmodels` as absent).

## Step 5: notebooks import the library (in place, no moves)

- Follows D1-D3. Each notebook switches from its own copies to `simbanator` imports, one notebook
  per commit, backed up first (`backups/notebooks/<nb>.pre-step5.bak.ipynb`).
- Product regression check: a cluster-side script re-runs the pure-build parts with the library
  and compares the new product against the existing one (columns, values to tolerance). Expected
  differences are only the ones D1-D3 introduce, and they are listed per product.
- No rebuilds (D3): the products D1-D3 change are marked stale through the registry and rebuilt
  later in one pass, with everything `depends_on` lists below them.
- Cluster copies: diff each one immediately before deploying, merge live knob edits.

## Step 6: the divisions project

- D4: `projects/divisions/` defines named schemes (X-ray injection, AGN activation, ...; the
  Comparison sections become schemes or comparisons of them) and writes one label table
  (registry entry `division_labels`: box, snap, gal_id, one column per scheme, NO default).
  Analyses are undivided unless they opt into a scheme by name.

## Step 7: move the notebooks into projects

- Order: `pipeline/` first (it produces what everyone reads), then the projects, then `archive/`.
- Split and merge the 13 live notebooks along the triage's section map with a script working on
  the notebook JSON (nbformat), so cell outputs survive; `git mv` for whole-notebook moves so
  history follows.
- Per project: `_helpers.py` for code reused inside it, `README.md` with its notebooks, the
  products it reads and the ones it writes.
- Parameters (user requirement, 2026-10-22): every knob a notebook uses (thresholds, bins, gates,
  `XRAY_FGAS_MAX`, the P18_* fit limits, overwrite switches, ...) lives in ONE visible, hand-editable
  parameter file per project (`projects/<p>/params.toml` or `.json`, one section per notebook),
  read at the top of each notebook; no knob values buried in analysis cells, no environment
  variables. The values themselves are NOT reviewed during the restructure: they are carried over
  exactly as the cluster copies have them, and the user sets them at the end.
- Cluster cutover per project: freeze the cluster copies, diff, merge knob edits, move both sides
  in the same session; no copies left at the old paths.

## Step 8: archive and remove the legacy

- `archive/`: the May-June notebooks, the Archive / Drop sections, `modules/`, `plot_cgm_stats.py`,
  the old flux-extraction submitters, `notebook_restructure_plan.md` (the June plan for
  residual_dust only).
- Remove the shims (`simbanator/{analysis,sed,utils,visualization,data}`, `ks_tracks_lib.py`)
  once no live notebook or job imports them (grep check).
- Root clean-up: filter `.res` files into the data folder, figures and maps into `docs/`,
  `test_*.ipynb` into `archive/` or `tests/`, `notebooks_md/` regenerated or dropped.

## Step 9: documentation, tests, merge

- `docs/products.md` (generated from the registry: product, producer, dependencies, run order;
  replaces the run-order cheat sheets), one `docs/conventions.md` (frames, units, members,
  apertures; merges the notebooks' convention cells), README package layout and project map.
- Tests: the existing `tests/` run in one command; a small CI job for the parts that need no
  cluster data.
- Merge `restructure` into `main` (D6).
