"""Analysis functions for building histories, filtering particles, and computing profiles."""

from .sfh_caesar import HDF5BuildHistory, find_property_threshold_crossings_from_hdf5
from .particles import extract_particles
from .profiles import radial_profile
from .progenitors import caesar_read_progen, read_progen
from .mergers import (
    Progenitor,
    Galaxy,
    process_galaxies_with_tracks,
    analyze_mergers,
)
from .sfh_fsps import *
from .sfh_utils import (smooth_resample_sfh, recent_sfr,
                        sfr_delayed_bq, fit_delayed_bq,
                        build_mfrac_lookup, mfrac_of,
                        archaeological_sfh, projected_region_sfh)

__all__ = [
    "HDF5BuildHistory",
    "find_property_threshold_crossings_from_hdf5",
    "smooth_resample_sfh", "recent_sfr",
    "build_mfrac_lookup", "mfrac_of",
    "archaeological_sfh", "projected_region_sfh",
    "extract_particles",
    "radial_profile",
    "caesar_read_progen", "read_progen",
    "Progenitor", "Galaxy",
    "process_galaxies_with_tracks", "analyze_mergers",
]
