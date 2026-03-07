"""Backward-compatibility shim — use ``sfh_caesar`` instead."""
import warnings
warnings.warn(
    "simbanator.analysis.history has been renamed to simbanator.analysis.sfh_caesar. "
    "Update your imports.", DeprecationWarning, stacklevel=2)
from .sfh_caesar import CaesarBuildHistory, BuildHistory  # noqa: F401
