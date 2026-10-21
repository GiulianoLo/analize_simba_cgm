"""Moved to ``simbanator.plotting.plots``.

This file only keeps old imports working (``simbanator.visualization.plots``): it hands back the new
module itself, so every name, private ones included, is the same object.
New code should import from ``simbanator.plotting.plots``.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("simbanator.plotting.plots")
