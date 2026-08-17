"""Backward-compatibility shim — use ``sfh_fsps`` instead."""
import warnings
warnings.warn(
    "simbanator.analysis.sfh has been renamed to simbanator.analysis.sfh_fsps. "
    "Update your imports.", DeprecationWarning, stacklevel=2)
from .sfh_fsps import compute_sfh, bin_sfh, load_sfh_file  # noqa: F401

