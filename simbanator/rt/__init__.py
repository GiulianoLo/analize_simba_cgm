"""Radiative transfer with powderday.

- ``makesed``  MakeSED: target selection files, job staging, SED reading (needs caesar + hyperion; loaded on first use)
- ``masters/``  the powderday parameter masters and the cluster setup scripts. MakeSED resolves a bare
  ``paramf`` name inside this folder.
"""
import os as _os

MASTERS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "masters")

_LAZY = {n: "makesed" for n in ("MakeSED", "list_sed_apertures", "read_selection_centers", "flatten_results")}

def __getattr__(name):
    # Heavy modules load on first use, so importing this package stays light.
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"{__name__}.{sub}"), name)

