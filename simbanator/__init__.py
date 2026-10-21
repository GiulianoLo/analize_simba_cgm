"""
simbanator – A Python toolkit for analysing cosmological simulations.

Quick start::

    import simbanator as sb

    # One-time setup (stores paths in config/simulations.json):
    sb.add_simulation("simba_m100n1024",
                      data_dir="/data/SIMBA/box100",
                      catalog_dir="/data/SIMBA/box100/Groups")

    # Then use anywhere:
    sim = sb.Simba(box=100)          # reads from config
    out = sb.OutputPaths(sim.name)   # ./output/simba_m100n1024/<task>/

For heavy-dependency features (yt, sphviewer, caesar), install extras::

    pip install simbanator[full]
"""

__version__ = "0.3.0"

# ---------------------------------------------------------------------------
# Core I/O  (always available)
# ---------------------------------------------------------------------------
from .io.simba import Simulation, Simba
from .io.paths import OutputPaths
from .io.config import add_simulation, remove_simulation, list_simulations, config_path

# ---------------------------------------------------------------------------
# Histories (trees, properties, SFHs, events)
# ---------------------------------------------------------------------------
from .histories.properties import HDF5BuildHistory
from .histories.trees import caesar_read_progen, read_progen
from .histories.mergers import (
    Progenitor,
    Galaxy,
    process_galaxies_with_tracks,
    analyze_mergers,
)

# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------
from .particles.cutouts import extract_particles
from .particles.profiles import radial_profile
from .particles.frames import shrink_center, principal_axes, rotate_to_frame
from .particles.grains import Z_to_OH12, Dust_to_Metal

# ---------------------------------------------------------------------------
# I/O helpers and photometry filters
# ---------------------------------------------------------------------------
from .io.catalogs import findsatellites
from .io.sysinfo import print_ram_usage
from .io.tables import convert_hdf5_fits
from .photometry.filters import (
    DEFAULT_SVO_FILTERS,
    download_svo_filter,
    download_svo_filters,
    download_default_svo_filters,
)

# ---------------------------------------------------------------------------
# Plotting (lightweight)
# ---------------------------------------------------------------------------
from .plotting.plots import HistoryPlots
from .plotting.animation import create_animation

# Modules with heavy dependencies load on first use, or import them explicitly:
#   from simbanator.plotting.rendering import RenderRGB, SingleRender   (sphviewer, yt)
#   from simbanator.histories.sfh_fsps import compute_sfh               (yt, fsps)
#   from simbanator.rt import MakeSED                                    (caesar, hyperion)
#   from simbanator.photometry import flux_extraction                    (hyperion)

__all__ = [
    # io
    "Simulation", "Simba", "OutputPaths",
    "add_simulation", "remove_simulation", "list_simulations",
    # analysis
    "HDF5BuildHistory",
    "extract_particles",
    "radial_profile",
    "caesar_read_progen", "read_progen",
    "Progenitor", "Galaxy",
    "process_galaxies_with_tracks", "analyze_mergers",
    # utils
    "Z_to_OH12", "Dust_to_Metal", "findsatellites", "print_ram_usage",
    "shrink_center", "principal_axes", "rotate_to_frame",
    "DEFAULT_SVO_FILTERS", "download_svo_filter", "download_svo_filters",
    "download_default_svo_filters",
    # data
    "convert_hdf5_fits",
    # visualization
    "HistoryPlots", "create_animation",
]
