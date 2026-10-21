"""Photometry from the radiative-transfer SEDs, and SED fitting with CIGALE.

- ``filters``     SVO filter curves (download and load)
- ``fluxes``      flux extraction, A_V, dust luminosity, annular fluxes (needs hyperion; loaded on first use)
- ``cigale``      CIGALE inputs, runs, collection and truth comparison (import explicitly)
- ``dl2014_fit``  dl2014 emissivity fit, run as a script
"""
from .filters import (DEFAULT_SVO_FILTERS, download_svo_filter, download_svo_filters,
                      download_default_svo_filters)

_LAZY = {n: "fluxes" for n in ("flux_extraction", "get_svo_filters", "load_local_filters", "resolve_filter_ids",
                                 "magTo_mJy", "mJyToMag", "convolveFilterWithSED", "attenuation_mag",
                                 "dust_luminosity", "annular_flux_table")}

def __getattr__(name):
    # Heavy modules load on first use, so importing this package stays light.
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"{__name__}.{sub}"), name)

