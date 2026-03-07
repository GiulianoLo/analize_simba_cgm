# Simbanator

A Python toolkit for analysing [SIMBA](http://simba.roe.ac.uk/) cosmological simulations.  
Simbanator wraps **Caesar** catalogues, **Powderday** SED modelling, and **py-sphviewer** rendering into a single, installable package.

## Features

| Subpackage | What it does |
|---|---|
| `simbanator.io` | Resolve SIMBA file paths, load Caesar catalogues, manage output directories |
| `simbanator.analysis` | Build star-formation / metallicity / attenuation histories, radial profiles, progenitor trees, particle filtering |
| `simbanator.visualization` | Multi-panel history plots, scatter-plot animations, SPH RGB rendering, velocity-stream videos |
| `simbanator.sed` | Drive Powderday runs: write parameter files, launch cluster / local jobs |
| `simbanator.data` | Convert Caesar HDF5 catalogues to FITS tables |
| `simbanator.utils` | Unit conversions (Z → 12+log(O/H), dust-to-metal), satellite search, RAM monitoring |

## Installation

```bash
# Clone the repository
git clone https://github.com/GiulianoLo/analize_simba_cgm.git
cd analize_simba_cgm

# Core install (numpy, scipy, astropy, matplotlib, h5py, unyt, Pillow, psutil)
pip install -e .

# Full install – adds yt, caesar, py-sphviewer (conda recommended for these)
pip install -e ".[full]"

# SED modelling – adds hyperion + caesar
pip install -e ".[sed]"

# Development extras
pip install -e ".[dev]"
```

> **Note:** `yt`, `caesar`, and `py-sphviewer` are much easier to install via conda.  
> Install them in your conda environment first, then `pip install -e .` for the rest.

## Quick Start

```python
import simbanator as sb

# ---------- I/O ----------
simba = sb.Simba(machine="local", box=100)       # resolve paths for the 100 Mpc box
paths = sb.SavePaths(simba, snap=151)             # create output directories for snap 151

# ---------- Histories ----------
bh = sb.CaesarBuildHistory(simba, obj_type="galaxy", idx=0)
bh.build_history(props=["stellar_mass", "sfr"])   # build SFH for galaxy 0

# ---------- Particle filtering ----------
filtered = sb.filter_particles_by_obj(
    simba, snap=151, obj_type="galaxy", idx=0
)

# ---------- Radial profiles ----------
from simbanator.analysis import radial_profile
rp = radial_profile(simba, snap=151, obj_type="galaxy", idx=0,
                     field="temperature", n_bins=50)

# ---------- Visualization ----------
plots = sb.HistoryPlots(simba, snap=151, obj_type="galaxy", idx=0)
plots.plot_history(prop="stellar_mass")

# ---------- Rendering (requires py-sphviewer) ----------
from simbanator.visualization import RenderRGB
renderer = RenderRGB(simba, snap=151, galaxy_idx=0)
renderer.render()

# ---------- SED modelling (requires hyperion) ----------
from simbanator.sed import MakeSED
sed = MakeSED(simba, snap=151, galaxy_idx=0)
sed.run()
```

## Package Structure

```
simbanator/
├── __init__.py            # public API
├── io/
│   ├── simba.py           # Simba – path resolution & catalogue loading
│   └── paths.py           # SavePaths – output directory management
├── analysis/
│   ├── history.py         # CaesarBuildHistory, BuildHistory
│   ├── particles.py       # filter_particles_by_obj, filter_by_aperture
│   ├── profiles.py        # radial_profile
│   └── progenitors.py     # caesar_read_progen, read_progen
├── visualization/
│   ├── plots.py           # HistoryPlots
│   ├── animation.py       # create_animation
│   └── rendering.py       # RenderRGB, SingleRender
├── sed/
│   ├── makesed.py         # MakeSED – Powderday driver
│   ├── parameters_master.py
│   ├── parameters_model.py
│   └── *.sh               # cluster / local job scripts
├── data/
│   └── convert.py         # convert_hdf5_fits
└── utils/
    ├── conversions.py     # Z_to_OH12, Dust_to_Metal
    ├── search.py          # findsatellites
    └── debug.py           # print_ram_usage
```

## Requirements

| Dependency | Required | Notes |
|---|---|---|
| numpy, scipy, astropy, matplotlib | yes | Core scientific stack |
| h5py | yes | HDF5 file I/O |
| unyt | yes | Unit-aware arrays |
| Pillow | yes | Image processing |
| psutil | yes | RAM monitoring |
| yt | optional (`[full]`) | Used by radial profiles & rendering |
| caesar | optional (`[full]`/`[sed]`) | SIMBA galaxy/halo catalogues |
| py-sphviewer | optional (`[full]`) | SPH particle rendering |
| hyperion | optional (`[sed]`) | Dust radiative transfer for SED |

## Migration from `modules/`

The old `modules/` directory is kept for reference but is **not** part of the installed package.  
All imports should now use `simbanator`:

```python
# Old
from modules.io_paths.simba import Simba
from modules.anal_func.build_history import caesarBuildHistory

# New
from simbanator.io import Simba
from simbanator.analysis import CaesarBuildHistory
# or simply:
import simbanator as sb
simba = sb.Simba(...)
```

## License

MIT
