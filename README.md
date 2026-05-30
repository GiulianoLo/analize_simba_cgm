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
│   ├── profiles.py       # radial_profile – surface-density / mean radial profiles
│   ├── quenching.py      # find_quenching_times, load_quenching_events
│   ├── history.py        # deprecated shim → sfh_caesar
│   └── sfh.py            # deprecated shim → sfh_fsps
├── sed/
│   ├── makesed.py        # MakeSED – Powderday setup + flux extraction (needs hyperion)
│   ├── flux_extraction.py# flux_extraction, get_svo_filters – SED → photometry
│   └── parameters_master.py / parameters_master-nodust.py
├── utils/
│   ├── geometry.py       # shrink_center, principal_axes, rotate_to_frame
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

# Single galaxy
extract_particles(cs=cs, simfile=snap_file, snap=snap,
                  galaxy_id=42, sim_name=sim.name)

# Spatial aperture
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
)
# Outputs: output/<sim>/sed/<run_tag>/sed_fluxes/all_galaxies_fluxes.fits
#          output/<sim>/sed/<run_tag>/sed_fluxes/all_xmean.fits
```

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
