"""Moved to ``simbanator.plotting.animation``.

This file only keeps old imports working (``simbanator.visualization.animation``): it hands back the new
module itself, so every name, private ones included, is the same object.
New code should import from ``simbanator.plotting.animation``.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("simbanator.plotting.animation")
