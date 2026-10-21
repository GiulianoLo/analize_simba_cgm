"""Old package name, kept so existing imports work. The code now lives in simbanator.histories
and simbanator.particles."""

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


def __getattr__(name):
    # The FSPS module needs yt + fsps: load it only when one of its names is asked for,
    # so the rest of this (old) package imports without them.
    import importlib
    mod = importlib.import_module("simbanator.histories.sfh_fsps")
    try:
        return getattr(mod, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
