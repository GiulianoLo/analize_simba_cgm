"""File integrity checks for simulation and product files."""

import os

import h5py


def hdf5_status(path, required=()):
    """State of an HDF5 file, without raising.

    Returns ``'ok'``, ``'missing'``, ``'no-perm'`` (exists but unreadable, not corrupt),
    ``'empty'`` (0 bytes), ``'no-keys'``, ``'no:<group>'`` (a ``required`` group is absent), or
    ``'corrupt:<ErrorType>'`` (truncated or half-downloaded files raise on open).
    """
    if not os.path.exists(path):
        return "missing"
    if not os.access(path, os.R_OK):
        return "no-perm"
    if os.path.getsize(path) == 0:
        return "empty"
    try:
        with h5py.File(path, "r") as f:
            if not set(f.keys()):
                return "no-keys"
            for g in required:
                if g not in f:
                    return f"no:{g}"
        return "ok"
    except Exception as e:
        return f"corrupt:{type(e).__name__}"
