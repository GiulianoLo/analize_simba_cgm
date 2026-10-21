"""Moved to ``simbanator.rt.makesed``.

This file only keeps old imports working (``simbanator.sed.makesed``): it hands back the new
module itself, so every name, private ones included, is the same object.
New code should import from ``simbanator.rt.makesed``.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("simbanator.rt.makesed")
