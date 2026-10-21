"""Moved to ``simbanator.photometry.dl2014_fit``.

This file only keeps old imports working (``simbanator.sed.dl2014_fit``): it hands back the new
module itself, so every name, private ones included, is the same object.
New code should import from ``simbanator.photometry.dl2014_fit``.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("simbanator.photometry.dl2014_fit")
