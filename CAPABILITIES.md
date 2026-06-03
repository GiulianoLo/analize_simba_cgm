# simbanator — Scientific Capabilities

**simbanator** is a Python toolkit for the full analysis pipeline of SIMBA cosmological
hydrodynamical simulations, from raw Caesar HDF5 catalogs through radiative-transfer SED
modeling and multi-band photometry.  All capabilities described below are available in the
package at `simbanator/` and are exercised in the notebooks in the repository root.

---

## Pipeline overview

```
Caesar catalog (HDF5)
        │
        ├─► 1. Sample selection  ──────── environment, sSFR, dust fraction, M★
        │
        ├─► 2. Progenitor tracking ─────── merger-tree FITS across all snapshots
        │         │
        │         ├─► 3. Property histories ── SFR, M★, SFR, M_mol, Z, etc.
        │         │
        │         ├─► 4. Quenching times ───── t_peak, t_quench, t_rejuv
        │         │
        │         └─► 5. Merger detection ───── companion search, major/minor counts
        │
        ├─► 6. Particle extraction ─────── per-galaxy HDF5 subsets
        │         │
        │         ├─► 7. Radial profiles ─── surface density / mean properties
        │         │
        │         ├─► 8. Images & videos ─── SPH rendering, RGB composites, GIF
        │         │
        │         └─► 9. SED modeling ─────── Powderday + dust radiative transfer
        │                   │
        │                   ├─► at critical epochs (SFR peak, quench, gas dep.)
        │                   │
        │                   └─► 10. Flux extraction ── multi-band photometry (FITS)
        │
        └─► Geometry tools ──────────────── centering, principal axes, frame rotation
```

---

## 1. Galaxy sample selection

**Module:** `simbanator.io.simba` · `simbanator.analysis.sfh_caesar`  
**Notebooks:** `QG_evolution_selection.ipynb`, `cgm_main.ipynb`, `multiscale_figure.ipynb`

Any Caesar catalog loaded via `Simulation.load_catalog(snap)` exposes the full galaxy
population.  Any combination of Caesar galaxy properties — stellar mass, specific
star-formation rate (sSFR), molecular gas fraction, dust-to-stellar-mass ratio, environment
(cluster membership, satellite/central flag) — can be used to define a sample in a single
NumPy boolean mask.

A worked example from the quenched-dusty-galaxy selection (`QG_evolution_selection.ipynb`):

```python
sim  = Simulation("cis100")
cs   = sim.load_catalog(snap=129)                   # z ≈ 0.1

sfr  = np.array([g.sfr              for g in cs.galaxies])
m    = np.array([g.masses['stellar']for g in cs.galaxies])
dust = np.array([g.masses['dust']   for g in cs.galaxies])

ssfr = sfr / m
sel  = (ssfr < 2e-11) & (dust / m > 1e-4)          # quenched + dust-rich
ids  = np.array([g.GroupID for g in cs.galaxies])[sel]
```

**Environment-based selection** uses `findsatellites()` to identify galaxies within a
physical aperture around chosen centers, enabling field/cluster/CGM sub-samples.

```python
from simbanator.utils.search import findsatellites
neighbours = findsatellites(centers, sim, snap, r=30)   # r in kpc
```

---

## 2. Progenitor tracking (merger trees)

**Module:** `simbanator.analysis.progenitors`  
**Function:** `caesar_read_progen(ids, outname, snaplist, sb, output_dir)`  
**Output:** FITS table, shape *(N_galaxies × N_snapshots)*, each cell is the Caesar
row index of the galaxy's most-massive progenitor; −1 when absent.

The function walks the Caesar catalogs in reverse snapshot order (from z = 0 toward high
redshift) and records the identity of the most-massive progenitor at every output time.
This index table is the backbone for all history-based analyses described below.

```python
from simbanator.analysis.progenitors import caesar_read_progen

caesar_read_progen(
    ids       = galaxy_ids,
    outname   = "tracks.fits",
    snaplist  = list(range(151, 5, -1)),
    sb        = sim,
    output_dir= out.progenitors,
)
```

---

## 3. Galaxy property histories

**Module:** `simbanator.analysis.sfh_caesar`  
**Class:** `HDF5BuildHistory`  
**Notebooks:** `test_sfh_caesar.ipynb`, `sed_critical_epochs.ipynb`

`HDF5BuildHistory` reads any Caesar HDF5 scalar property along the progenitor chains built
in step 2 and returns a time series array of shape *(N_snapshots × N_galaxies)*.  Dot-path
notation maps directly to Caesar dataset paths
(`'masses.stellar'` → `galaxy_data/dicts/masses.stellar`).

**Supported properties (non-exhaustive):**

| Key | Description |
|---|---|
| `sfr` | Star-formation rate (M☉ yr⁻¹) |
| `masses.stellar` | Stellar mass (M☉) |
| `masses.H2` | Molecular hydrogen mass (M☉) |
| `masses.dust` | Dust mass (M☉) |
| `metallicities.mass_weighted` | Mass-weighted metallicity |
| `radii.stellar_half_mass` | Stellar half-mass radius (kpc/h) |

```python
from simbanator.analysis.sfh_caesar import HDF5BuildHistory

hist = HDF5BuildHistory(sim, cs_z0)
hist.get_history_indx(galaxy_ids, start_snap=149, end_snap=6)

z_dict, props = hist.get_property_history({
    'galaxy_data': ['sfr', 'masses.stellar', 'masses.H2', 'masses.dust']
})
# props['sfr'].shape == (n_snaps, n_gals)
```

The returned `z_dict` carries the redshift and cosmic time at every snapshot, ready for
direct use with `astropy.cosmology` interpolators.

**FSPS-based SFH recovery** is available in `simbanator.analysis.sfh_fsps` for
particle-level star-by-star formation histories derived by inverting FSPS stellar-population
models (`compute_sfh`, `bin_sfh`).

---

## 4. Quenching time measurement

**Module:** `simbanator.analysis.quenching`  
**Function:** `find_quenching_times(time_array, sfr_array, ...)`  
**Notebook:** `sed_critical_epochs.ipynb`

Given a time-resolved SFR track and a user-defined threshold, `find_quenching_times`
identifies:

| Quantity | Definition |
|---|---|
| `t_sfr_peak` | Time of maximum SFR along the track |
| `t_quench_start` | Time SFR first drops below the threshold |
| `t_quench_end` | Time at which quenching is persistent (no recovery) |
| `t_gas_depletion` | Time molecular gas mass drops below a detection limit |
| `t_rejuvenation` | Time of any subsequent SFR recovery above threshold |

```python
from simbanator.analysis.quenching import find_quenching_times

events = find_quenching_times(
    time_array = t_yr,          # cosmic time in yr (shape n_snaps)
    sfr_array  = sfr_track,     # M☉ yr⁻¹ (shape n_snaps)
    threshold  = 1e-1,          # sSFR threshold (yr⁻¹) — or absolute SFR
)
```

The results are saved to a FITS table per galaxy via `_save_quenching_fits()` and
reloaded with `load_quenching_events()`.

---

## 5. Merger detection and classification

**Module:** `simbanator.analysis.mergers`  
**Functions:** `process_galaxies_with_tracks`, `analyze_mergers`  
**Notebook:** `merger_tracking.ipynb`

`process_galaxies_with_tracks` performs a spatial search in each Caesar catalog for
companion galaxies within a sphere of radius *f × r_half* (default *f* = 5) centred on
the progenitor.  Companions above a stellar-mass threshold (default 10⁹ M☉) are stored
as `Progenitor` objects that carry mass ratio, separation, molecular gas, and dust mass.

```python
from simbanator.analysis.mergers import process_galaxies_with_tracks, analyze_mergers

galaxies = process_galaxies_with_tracks(
    track_fits_path      = "progenitors/tracks.fits",
    box_size             = 100.0,       # Mpc/h (periodic boundary handling)
    sb                   = sim,
    snaplist             = snaplist,
    search_radius_factor = 5.0,
    mass_threshold       = 1e9,         # M☉
)

major, minor = analyze_mergers(
    galaxies,
    array_size         = (len(snaplist), len(galaxies)),
    mass_threshold_maj = 0.25,          # mass ratio ≥ 0.25 → major
    mass_threshold_min = 0.10,          # 0.10–0.25 → minor
)
# major, minor: int arrays, shape (n_snaps, n_galaxies)
```

Merger rates are visualized as bar charts binned by cosmic phase
(`plot_merger_rate_by_phase`) and as trajectory plots with companion events overlaid
(`plot_neighborhood_track`).

---

## 6. Radial profiles

**Module:** `simbanator.analysis.profiles`  
**Function:** `radial_profile(...)`  
**Notebook:** `radii_darko.ipynb`

Computes radial profiles of any particle property for gas (PartType0) and stars
(PartType4) using yt's SPH rendering as the underlying interpolant.  Both *surface-density*
profiles (integrated column) and *mean* profiles (volume-averaged) are supported.

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
    radii = np.arange(0, 100, 2),   # kpc, bin edges
    dens  = True,                   # True → surface density; False → mean
)
# profiles["PartType0"]["Masses"] is a list, one entry per radial bin
```

---

## 7. Images and videos

**Module:** `simbanator.visualization.rendering`, `simbanator.visualization.animation`  
**Classes:** `RenderRGB`, `SingleRender`, `ParticleProjectionRender`  
**Function:** `create_animation`  
**Notebooks:** `render_rgb_images.ipynb`, `sph_visual.ipynb`, `multiscale_figure.ipynb`

### 7a. SPH particle rendering

Three rendering classes wrap `py-sphviewer` (SPH-kernel smoothed projections) and `yt`
(adaptive mesh refinement) for publication-quality galaxy images:

| Class | Description |
|---|---|
| `RenderRGB` | Multi-component RGB blend: gas (blue), stars (white/yellow), dust (red) with logarithmic colour stretch |
| `SingleRender` | Single-component map; optionally overlays velocity-field streamlines |
| `ParticleProjectionRender` | Simple 3-panel (xy / xz / yz) scatter projection for quick inspection |

All renderers accept a `region` flag to render only the spatially extracted particles (from
step 6) rather than the full snapshot box.

```python
from simbanator.visualization.rendering import RenderRGB

r = RenderRGB(
    snapfile     = snap_file,
    catfile      = cat_file,
    galaxy_index = 42,
    components   = ['PartType0', 'PartType4'],
    ifdust       = True,
    dim          = 512,
)
```

### 7b. Multi-scale figures

`multiscale_figure.ipynb` produces a 3-level zoom figure connecting the full SIMBA
simulation volume (25 Mpc) → the CGM (~200 kpc) → individual galaxy structure (~20 kpc),
with gas, stellar, and dust components rendered separately per scale and assembled into a
single composite figure.

### 7c. Animations

`create_animation` turns any sequence of (x, y) coordinate arrays — e.g., galaxy
positions or property tracks over time — into a GIF with colour-coded points and fading
trails.

```python
from simbanator.visualization.animation import create_animation

create_animation(
    x       = positions[:, :, 0],   # shape (n_frames, n_points)
    y       = positions[:, :, 1],
    outfile = "galaxy_tracks.gif",
    fps     = 30,
)
```

---

## 8. SED modeling with dust radiative transfer

**Module:** `simbanator.sed.makesed`  
**Class:** `MakeSED`  
**Notebooks:** `test_powderday.ipynb`, `QG_evolution_selection.ipynb`, `multiscale_figure.ipynb`

`MakeSED` manages the full **Powderday** radiative-transfer pipeline.  Powderday couples
FSPS stellar-population synthesis to the HyperionRT 3-D Monte-Carlo dust code,
self-consistently computing attenuation by the interstellar dust distribution read directly
from the hydrodynamic particle data.

### Workflow

| Step | Method | Description |
|---|---|---|
| 1. Target selection | `selection_gals(snaps, galaxyIDs)` | Record snapshot / galaxy pairs in an HDF5 selection file |
| 2. Parameter files | `create_master(where, subset_type, partition, prefix)` | Write per-galaxy Powderday `parameters_master.py` files and SLURM job scripts |
| 3. Run on cluster | *(SLURM)* | Submit generated job scripts; Powderday writes `*.rtout.sed` files |
| 4. Inspect SEDs | `plotsed(snap, gal)` | Plot the full SED (rest-frame wavelength vs. flux) |
| 5. Extract fluxes | `extract_flux_batch(...)` | Convolve SEDs with filter curves → FITS tables |

Two parameter templates ship with the package:

- `parameters_master.py` — full dust physics (emission + attenuation)
- `parameters_master-nodust.py` — stellar emission only (no dust)

Both templates are written automatically to the relevant snapshot directories during
`create_master()`.

**Output directory tree:**
```
output/<sim>/sed/<run_tag>/
├── powderday_sed_out/
│   └── snap_<NNN>/
│       └── gal_<GGGGGG>/
│           ├── snap<NNN>.galaxy<GGGGGG>.rtin
│           └── snap<NNN>.galaxy<GGGGGG>.rtout.sed
├── target_selection/<selection>.h5
├── sed_fluxes/
│   ├── all_galaxies_fluxes.fits
│   └── all_xmean.fits
└── sed_plots/snap_<NNN>/gal_<GGGGGG>.png
```

---

## 9. SEDs at critical evolutionary epochs

**Module:** `simbanator.analysis.quenching`, `simbanator.analysis.sfh_caesar`,
`simbanator.sed.makesed`  
**Notebook:** `sed_critical_epochs.ipynb`

This workflow combines steps 3–4–8 to extract SEDs at the physically motivated epochs
of each galaxy's evolutionary history.  The four epochs targeted are:

| Epoch | Definition |
|---|---|
| **SFR peak** | Time of maximum star-formation rate along the PCHIP-interpolated SFR track |
| **Quench start** | First time SFR falls below the quenching threshold |
| **Quench end** | Time at which quenching becomes persistent (no rejuvenation) |
| **Gas depletion** | Time molecular gas mass falls below the detection limit |

For each galaxy the notebook:
1. Builds a continuous SFR / M★ / M_mol track via `HDF5BuildHistory`.
2. Interpolates with a PCHIP spline (preserves monotonicity without oscillation).
3. Identifies the four epoch times with `find_quenching_times`.
4. Maps each epoch to the nearest available Caesar snapshot.
5. Retrieves the progenitor GroupID at that snapshot.
6. Calls `extract_particles` to isolate the progenitor's particles.
7. Runs Powderday on those particles to produce the epoch SED.

The diagnostic figure produced at step 2–3 shows SFR, M★, and M_mol as continuous
curves with the four critical times marked, providing a visual sanity-check before
the (computationally expensive) SED runs are submitted.

---

## 10. Photometric flux extraction

**Module:** `simbanator.sed.flux_extraction`, `simbanator.utils.svo_filters`  
**Functions:** `flux_extraction`, `get_svo_filters`, `download_svo_filters`  
**Notebook:** `test_powderday.ipynb`

Filter transmission curves are fetched from the **Spanish Virtual Observatory (SVO)
Filter Profile Service** and convolved with each Powderday SED to produce photometric
fluxes.  The default instrument set covers UV through far-infrared:

| Facility | Instrument | Bands |
|---|---|---|
| GALEX | — | FUV, NUV |
| HST | WFC3 | UV–optical |
| JWST | NIRCam | NIR |
| SDSS | — | *u g r i z* |
| 2MASS | — | *J H K_s* |
| WISE | — | W1–W4 |
| Spitzer | IRAC / MIPS | 3.6–160 μm |
| Herschel | PACS / SPIRE | 70–500 μm |
| JCMT | SCUBA-2 | 850 μm |

Flux extraction integrates the SED over each filter passband, applies a redshift shift, and
writes the results to FITS:

```python
fluxes_fits, xmean_fits = makesed.extract_flux_batch(
    snaps      = snap_array,
    gals       = gal_array,
    facility   = ["HST",   "JWST",   "Spitzer", "Herschel"],
    instrument = ["WFC3",  "NIRCam", "IRAC",    "SPIRE"],
    wave_unit  = "micron",
    findx      = 0,              # inclination index (0 = face-on)
)
```

The output FITS file `all_galaxies_fluxes.fits` contains one row per (galaxy, snapshot)
pair with columns for every requested filter, plus a companion file `all_xmean.fits`
recording the effective wavelength of each filter convolution.

Filter curves can be downloaded once and reused:

```python
import simbanator as sb
sb.download_default_svo_filters(output_dir="filters/")    # SDSS + GALEX
sb.download_svo_filter("Herschel/SPIRE.PSW", "filters/")  # single filter
```

---

## Geometry and coordinate utilities

**Module:** `simbanator.utils.geometry`

Three utility functions handle the coordinate transformations required before
rendering or profile computation:

| Function | Description |
|---|---|
| `shrink_center(pos, masses)` | Iterative shrinking-sphere centre-of-mass finder; converges to the potential minimum of the stellar distribution |
| `principal_axes(pos, masses)` | Mass-weighted PCA of particle positions; returns inertia tensor, eigenvalues, eigenvectors, and axis ratios |
| `rotate_to_frame(pos, inclination, evecs)` | Rotate particle coordinates to the face-on (i = 0°) or edge-on (i = 90°) frame defined by the principal axes |

---

## Unit and chemical conversions

**Module:** `simbanator.utils.conversions`

| Function | Description |
|---|---|
| `Z_to_OH12(Z)` | Metallicity (mass fraction) → 12 + log(O/H) using solar Z = 0.0142 |
| `Dust_to_Metal(M_dust, M_H2, abundance)` | Dust-to-metal ratio following De Vis et al. (2019) |

---

## Configuration and output path management

**Modules:** `simbanator.io.config`, `simbanator.io.paths`

Simulation file paths are registered once per machine in `~/.simbanator/config.json` using
`add_simulation()` / `remove_simulation()`.  `OutputPaths` provides a lazy directory
manager that mirrors the analysis structure:

| Property | Directory |
|---|---|
| `out.progenitors` | `output/<sim>/progenitors/` |
| `out.filtered_snap(snap)` | `output/<sim>/filtered_particles/snap_<NNN>/` |
| `out.histories` | `output/<sim>/histories/` |
| `out.radial_profiles` | `output/<sim>/radial_profiles/` |
| `out.sed` | `output/<sim>/sed/` |
| `out.plots` | `output/<sim>/plots/` |

Directories are created on first access; no directory creation is needed in analysis scripts.

---

## Dependencies

| Group | Packages | Required for |
|---|---|---|
| Core | `numpy` `scipy` `astropy` `h5py` `matplotlib` `unyt` | all functionality |
| `[sed]` | `hyperion` `caesar` | SED modeling (steps 8–10) |
| `[full]` | `yt` `caesar` `py-sphviewer` `fsps` `svo_filters` | rendering, radial profiles, FSPS SFH |

Heavy optional imports are lazy — `import simbanator` succeeds without `yt` or
`py-sphviewer`; the relevant submodule raises an `ImportError` only when it is explicitly
imported.

---

*simbanator v0.2.0 — MIT License*
