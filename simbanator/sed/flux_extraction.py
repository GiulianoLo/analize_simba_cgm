"""Moved to ``simbanator.photometry.fluxes``.

This file only keeps old imports working (``simbanator.sed.flux_extraction``): it hands back the new
module itself, so every name, private ones included, is the same object.
New code should import from ``simbanator.photometry.fluxes``.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("simbanator.photometry.fluxes")
