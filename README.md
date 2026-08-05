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
│   ├── sfh_fsps.py       # compute_sfh, bin_sfh, save_sfh, load_sfh – FSPS SFHs
│   ├── sfh_utils.py      # smooth_resample_sfh, recent_sfr – de-burst snapshot-cadence
│   │                     #   SFR tracks (Gaussian kernel + uniform resample);
│   │                     #   sfr_delayed_bq, fit_delayed_bq – CIGALE sfhdelayedbq
│   │                     #   form + bounded fit (shared by 7b′ and aperture truth)
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
│   ├── cigale.py         # CIGALE 2025.0 end-to-end: write_cigale_input (flux table →
│   │                     #   data_file, DB band names incl. Subaru Suprime-Cam — SVO
│   │                     #   broad g/r/i/z map to CIGALE g+/r+/i+/z+), prepare_run
│   │                     #   (pcigale.ini/.spec,
│   │                     #   replaces init+genconf; genconf-style docs as ini comments;
│   │                     #   fit_bands= manual fitted-band list, '<band>'/'<band>_err'
│   │                     #   mix ok — errors auto-paired, unselected bands predicted),
│   │                     #   describe_run (module/param/variable reminder, full or compact),
│   │                     #   check/run/read_results/plot_seds wrappers, compare_results
│   │                     #   (truth vs bayes.*: per-galaxy print, offset/NMAD stats,
│   │                     #   one-to-one panels → <run_dir>/out/simba_vs_cigale.fits+.png),
│   │                     #   plot_parameter_priors (per gridded param: fitted distribution
│   │                     #   vs prior nodes + extend/refine/trim advice → param_priors.*),
│   │                     #   write_slurm_array (one SLURM array task per prepared run dir;
│   │                     #   default re-fits with CIGALE-native timestamped out/ backups,
│   │                     #   skip_if_done=True for cheap resubmits),
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
│                                   #    _temperature, _halo_of)
├── build_reduced_particles_job.py  # SLURM worker: lean 100 kpc reduced particle files (ISM+CGM)
│                                   #   Batched snapshot I/O: _catalog_pass (per-galaxy candidate
│                                   #   lists, halo-cached), _gather (slab-streamed reads at the
│                                   #   sorted union of indices; skips unneeded slabs), _Ctx (lazy
│                                   #   per-snapshot column store serving galaxies from memory).
│                                   #   Extensible field producers backfill new fields from the
│                                   #   stored idx without redoing geometry; datasets are lzf.
│                                   #   Env: DUST_PLAN, REDUCED_RMAX_KPC, REDUCED_PREFIX,
│                                   #        REDUCED_OVERWRITE, REDUCED_GATHER_MB
├── submit_reduced_particles.sh     # sbatch wrapper (array over snapshots; DUST_PLAN per anchor)
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
└── powderday_flux_quenched_m25.ipynb # Powderday flux catalogs for quenched (0.2/τ, logM*>10,
                                    #   ngas>20) cis25 galaxies at z≈0.3/0.6/0.7/1/2, split by
                                    #   weak/strong AGN coupling over [SFT,QT]; per-anchor gated
                                    #   history+BH builds → sample stats → dust_on/off RT over
                                    #   5 log-spaced apertures (10–160 kpc, powderday patch) →
                                    #   per-aperture flux catalogs with MC errors
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
