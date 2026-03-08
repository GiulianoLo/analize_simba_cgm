"""
analize_simba_cgm - Analysis tools for SIMBA simulation CGM data.

Usage:
    import analize_simba_cgm as ascgm

    sb = ascgm.Simba(machine='cis', size=100)
    paths = ascgm.SavePaths()
"""

__version__ = "0.1.0"

# Core I/O
from .io_paths import Simba, SavePaths

# Analysis functions
from .anal_func.build_history import caesarBuildHistory, BuildHistory
from .anal_func.filter_particles import extract_particles
from .anal_func.radial_profile import radial_profile
from .anal_func.read_progenitors import caesar_read_progen, read_progen

# Conversion utilities
from .anal_func.side_functions.conversions import Z_to_OH12, Dust_to_Metal
from .anal_func.side_functions.search import findsatellites

# Data objects
from .data_objects.convert_hdf5_fits import convert_hdf5_fits

# Visualization
from .visualize.simple_plots import HistoryPlots
from .visualize.animation import create_animation

# Debugging
from .debugging.debug_funct import print_ram_usage
