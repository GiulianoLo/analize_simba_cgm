"""Shared analysis products: checked reads of the files the pipeline writes.

Step 4 of the restructure turns this into a registry (product name -> path template and the
notebook or job that produces it). For now it holds the readers the notebooks share.
"""

import os

import numpy as np

from .tables import read_table, to_frame


def need(path, made_by=None):
    """``path`` if it exists, otherwise ``FileNotFoundError`` naming what produces it."""
    if not os.path.exists(path):
        hint = f": run {made_by} first" if made_by else ""
        raise FileNotFoundError(f"{path} is missing{hint}")
    return path


def read_frame(path, cols=None, made_by=None, string_cols=("incl", "aperture")):
    """A product table as a pandas DataFrame (checked with :func:`need`; see :func:`.tables.to_frame`)."""
    return to_frame(read_table(need(path, made_by)), cols, string_cols=string_cols)


def read_selection(path, made_by=None):
    """A selection catalogue -> ``(table, snap array, gal_id array)``."""
    sel = read_table(need(path, made_by))
    return sel, np.asarray(sel["snap"], int), np.asarray(sel["gal_id"], int)
