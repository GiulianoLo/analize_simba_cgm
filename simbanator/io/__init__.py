"""I/O: simulation handles, output paths, the simulation config, tables and catalogues.

- ``simba``     Simulation / Simba handles (paths, snapshots, redshifts)
- ``paths``     OutputPaths, the structured output-directory manager
- ``config``    the visible simulation config (config/simulations.json)
- ``tables``    HDF5 -> FITS conversion (table helpers to follow)
- ``catalogs``  catalogue searches (caesar readers to follow)
- ``sysinfo``   memory-usage printout
"""
from .simba import Simulation, Simba
from .paths import OutputPaths
from .config import (
    add_simulation,
    remove_simulation,
    list_simulations,
    load_config,
    config_path,
    CONFIG_FILE,
)
from .tables import convert_hdf5_fits
from .catalogs import findsatellites

__all__ = [
    "Simulation", "Simba", "OutputPaths",
    "add_simulation", "remove_simulation", "list_simulations",
    "load_config", "config_path", "CONFIG_FILE",
    "convert_hdf5_fits", "findsatellites",
]
