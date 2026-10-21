"""Spatial search utilities for Caesar catalogs."""

import numpy as np


def findsatellites(pos_c, sb, snap, r=30):
    """Find satellite galaxies within radius *r* of given centres.

    Parameters
    ----------
    pos_c : array-like, shape (N, 3)
        Centre positions (one per target galaxy).
    sb : :class:`~simbanator.io.simba.Simba`
        Simba path manager.
    snap : int
        Snapshot number.
    r : float
        Search radius in kpc.

    Returns
    -------
    dict
        ``{index: array_of_satellite_GroupIDs}``
    """
    ids_out = {}
    cs = sb.get_caesar(snap)
    h = cs.simulation.scale_factor
    ids = np.asarray([i.GroupID for i in cs.galaxies])
    pos_l = np.asarray([i.pos for i in cs.galaxies])

    for key, p in enumerate(pos_c):
        dist = np.sqrt(
            (pos_l[:, 0] - p[0]) ** 2
            + (pos_l[:, 1] - p[1]) ** 2
            + (pos_l[:, 2] - p[2]) ** 2
        )
        ids_out[key] = ids[dist < r]
    return ids_out
