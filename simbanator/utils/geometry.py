"""Geometry utilities: centering, principal axes, and frame rotations.

These are general-purpose functions useful for orienting galaxies
(or any particle distribution) into face-on / edge-on frames.

Example::

    from simbanator.utils.geometry import shrink_center, principal_axes, rotate_to_frame

    center = shrink_center(positions, masses=masses)
    _, evals, evecs, _ = principal_axes(positions - center, masses=masses)
    pos_oriented = rotate_to_frame(positions, center, evecs)
"""

import numpy as np


def shrink_center(pos, masses=None, fraction=0.5, min_r=0.1, max_iter=50):
    """Iterative shrinking-sphere centre finder.

    Starting from the (mass-weighted) mean position, the algorithm
    repeatedly shrinks a sphere by *fraction* and recomputes the
    centre of the enclosed particles until the radius drops below
    *min_r* or *max_iter* iterations are reached.

    Parameters
    ----------
    pos : array-like, shape (N, 3)
        Particle positions.
    masses : array-like, shape (N,), optional
        Particle masses for weighting.  If *None*, uniform weights.
    fraction : float
        Factor by which the radius shrinks each iteration (0 < f < 1).
    min_r : float
        Stop when the radius is smaller than this value (same units
        as *pos*).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    center : ndarray, shape (3,)
        Estimated centre coordinates.
    """
    pos = np.asarray(pos, dtype=float)
    if masses is not None:
        masses = np.asarray(masses, dtype=float)
        center = (pos * masses[:, None]).sum(axis=0) / masses.sum()
    else:
        center = pos.mean(axis=0)

    r = np.max(np.linalg.norm(pos - center, axis=1))

    for _ in range(max_iter):
        if r <= min_r:
            break
        d = np.linalg.norm(pos - center, axis=1)
        inside = d <= r
        if inside.sum() < 10:
            break
        if masses is None:
            center = pos[inside].mean(axis=0)
        else:
            w = masses[inside]
            center = (pos[inside] * w[:, None]).sum(axis=0) / w.sum()
        r *= fraction

    return center


def principal_axes(pos, masses=None):
    """Mass-weighted principal-component analysis on positions.

    Parameters
    ----------
    pos : array-like, shape (N, 3)
        Particle positions (ideally already centred).
    masses : array-like, shape (N,), optional
        Particle masses.  Uniform weights if *None*.

    Returns
    -------
    center : ndarray, shape (3,)
        Mass-weighted centroid.
    eigenvalues : ndarray, shape (3,)
        Eigenvalues in **descending** order.
    eigenvectors : ndarray, shape (3, 3)
        Columns are the principal axes (in original-frame
        coordinates), ordered to match *eigenvalues*.
    covariance : ndarray, shape (3, 3)
        The weighted covariance matrix.
    """
    pos = np.asarray(pos, dtype=float)
    if masses is None:
        masses = np.ones(pos.shape[0], dtype=float)
    masses = np.asarray(masses, dtype=float)

    wsum = masses.sum()
    center = (pos * masses[:, None]).sum(axis=0) / wsum
    posc = pos - center

    cov = (posc * masses[:, None]).T @ posc / wsum

    evals, evecs = np.linalg.eigh(cov)  # ascending order
    idx = np.argsort(evals)[::-1]
    return center, evals[idx], evecs[:, idx], cov


def rotate_to_frame(pos, center, evecs):
    """Project positions into a principal-axis frame.

    Parameters
    ----------
    pos : array-like, shape (N, 3)
        Particle positions in the original coordinate system.
    center : array-like, shape (3,) or scalar 0
        Centre to subtract before rotating.  Pass ``0`` (or an
        array of zeros) if positions are already centred.
    evecs : ndarray, shape (3, 3)
        Rotation matrix whose **columns** are the target frame axes
        expressed in the original coordinate system (as returned by
        :func:`principal_axes`).

    Returns
    -------
    pos_rotated : ndarray, shape (N, 3)
        Positions in the new frame.
    """
    pos = np.asarray(pos, dtype=float)
    center = np.asarray(center, dtype=float)
    if center.ndim == 0 and center == 0:
        posc = pos
    else:
        posc = pos - center
    return posc @ evecs
