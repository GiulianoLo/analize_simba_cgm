"""Moved to ``simbanator.io.catalogs``.

This file only keeps old imports working (``simbanator.utils.search``): it hands back the new
module itself, so every name, private ones included, is the same object.
New code should import from ``simbanator.io.catalogs``.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("simbanator.io.catalogs")
