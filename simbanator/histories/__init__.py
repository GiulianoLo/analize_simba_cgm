"""Everything along a galaxy's branch through time.

- ``trees``       progenitor tracks from the caesar merger trees
- ``mergers``     merger detection and classification along the tracks
- ``properties``  per-snapshot property histories (mass, gas, dust, H2, Z, ...)
- ``sfh``         star-formation histories: smoothing, archaeological / region SFHs, delayed-bq fits
- ``sfh_fsps``    FSPS-based SFH recovery (needs yt + fsps; import it explicitly)
- ``events``      quenching events (SFT / QT), critical points and clocks
"""
from .trees import caesar_read_progen, read_progen, progen_tree_file
from .mergers import Progenitor, Galaxy, process_galaxies_with_tracks, analyze_mergers
from .properties import HDF5BuildHistory, find_property_threshold_crossings_from_hdf5
from .sfh import (smooth_resample_sfh, recent_sfr, sfr_delayed_bq, fit_delayed_bq,
                  build_mfrac_lookup, mfrac_of, archaeological_sfh, projected_region_sfh)
from .events import find_quenching_times, load_quenching_events

__all__ = [
    "caesar_read_progen", "read_progen", "progen_tree_file",
    "Progenitor", "Galaxy", "process_galaxies_with_tracks", "analyze_mergers",
    "HDF5BuildHistory", "find_property_threshold_crossings_from_hdf5",
    "smooth_resample_sfh", "recent_sfr", "sfr_delayed_bq", "fit_delayed_bq",
    "build_mfrac_lookup", "mfrac_of", "archaeological_sfh", "projected_region_sfh",
    "find_quenching_times", "load_quenching_events",
]
