"""Moved to ``simbanator.photometry.filters``.

This file only keeps old imports working (``simbanator.utils.svo_filters``): it hands back the new
module itself, so every name, private ones included, is the same object.
New code should import from ``simbanator.photometry.filters``.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("simbanator.photometry.filters")
