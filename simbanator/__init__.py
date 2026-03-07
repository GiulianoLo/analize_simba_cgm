"""
simbanator – A Python toolkit for analysing cosmological simulations.

Quick start::

    import simbanator as sb

    # One-time setup (stores paths in ~/.simbanator/config.json):
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
from .io.config import add_simulation, remove_simulation, list_simulations

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
from .analysis.history import CaesarBuildHistory, BuildHistory
from .analysis.particles import filter_particles_by_obj, filter_by_aperture
from .analysis.profiles import radial_profile
from .analysis.progenitors import caesar_read_progen, read_progen
from .analysis.sfh import bin_sfh, save_sfh, load_sfh

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
from .utils.conversions import Z_to_OH12, Dust_to_Metal
from .utils.search import findsatellites
from .utils.debug import print_ram_usage

# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------
from .data.convert import convert_hdf5_fits

# ---------------------------------------------------------------------------
# Visualization (lightweight)
# ---------------------------------------------------------------------------
from .visualization.plots import HistoryPlots
from .visualization.animation import create_animation

# Heavy visualization (yt / sphviewer) available via explicit import:
#   from simbanator.visualization.rendering import RenderRGB, SingleRender

# FSPS-based SFH recovery (requires fsps):
#   from simbanator.analysis.sfh import compute_sfh

# SED modeling (powderday) available via explicit import:
#   from simbanator.sed.makesed import MakeSED

__all__ = [
    # io
    "Simulation", "Simba", "OutputPaths",
    "add_simulation", "remove_simulation", "list_simulations",
    # analysis
    "CaesarBuildHistory", "BuildHistory",
    "filter_particles_by_obj", "filter_by_aperture",
    "radial_profile",
    "caesar_read_progen", "read_progen",
    "bin_sfh", "save_sfh", "load_sfh",
    # utils
    "Z_to_OH12", "Dust_to_Metal", "findsatellites", "print_ram_usage",
    # data
    "convert_hdf5_fits",
    # visualization
    "HistoryPlots", "create_animation",
]
