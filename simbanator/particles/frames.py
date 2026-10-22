"""Geometry utilities: centering, principal axes, and frame rotations.

These are general-purpose functions useful for orienting galaxies
(or any particle distribution) into face-on / edge-on frames.

Example::

    from simbanator.particles.frames import shrink_center, principal_axes, rotate_to_frame

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


def sightline_unit_vectors(theta_deg, phi_deg):
    """LOS unit vectors for (theta, phi) viewing angles [degrees].

    Matches Hyperion's peeling-direction convention (theta from +z,
    phi from +x): the same THETA/PHI lists as the powderday parameter
    masters give the vectors along which the SEDs were peeled.

    Returns ndarray of shape (n_angles, 3).
    """
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.column_stack([np.sin(th) * np.cos(ph),
                            np.sin(th) * np.sin(ph),
                            np.cos(th)])


def projected_radius(pos, nhat):
    """Distance of each position from the axis through the origin along *nhat*.

    This is the image-plane (projected) radius of a particle as seen along
    the line of sight *nhat* — the geometry of a Hyperion SED aperture
    (a circle in the image plane, full depth along the LOS).

    Parameters
    ----------
    pos : (N, 3) array — positions relative to the aperture centre.
    nhat : (3,) unit vector — the line of sight.

    Returns
    -------
    (N,) array of projected radii (same units as *pos*).
    """
    pos = np.asarray(pos, float)
    para = pos @ np.asarray(nhat, float)
    return np.sqrt(np.clip(np.einsum("ij,ij->i", pos, pos) - para ** 2,
                           0.0, None))


# ═════════════════════════════════════════════════════════════════════════════
# Moved from ks_tracks_lib.py (the K-S track measurements), unchanged
# ═════════════════════════════════════════════════════════════════════════════

def face_on_R(pos, evecs):
    """Face-on cylindrical radius [kpc] in the stored stellar principal frame.

    `evecs` columns are the principal axes (descending eigenvalue) exactly as
    simbanator.utils.geometry.principal_axes returns them; columns 0-1 span the disc plane.
    """
    pos = np.asarray(pos, float)
    if pos.size == 0:
        return np.zeros(0)
    proj = pos @ np.asarray(evecs, float)
    return np.hypot(proj[:, 0], proj[:, 1])


# ═════════════════════════════════════════════════════════════════════════════
# Radii in the stored frame, gas selections, spin vectors
# ═════════════════════════════════════════════════════════════════════════════

def reduced_radius(pos, evecs, geom="cyl"):
    """Radius of each particle: 3-D (``geom="3d"``) or face-on cylindrical in the stored principal frame (``"cyl"``).

    The principal axes are the COLUMNS of ``evecs`` (:func:`principal_axes`), so the face-on plane
    is ``(pos @ evecs)[:, :2]``. Two older notebooks used ``pos @ evecs.T``, which projects onto the
    wrong frame.
    """
    pos = np.asarray(pos, float)
    if geom == "3d":
        return np.sqrt(np.sum(pos ** 2, axis=1))
    return face_on_R(pos, evecs)


def select_gas(red, mass_key, geom="cyl", region="all", member_only=False):
    """``(R, mass, mask)`` of the gas of a reduced file for one mass column (``m_H2``, ``m_dust``, ...).

    ``region``: ``"all"``, ``"ism"`` (star-forming gas, SFR > 0) or ``"cgm"`` (SFR = 0).
    ``member_only`` keeps the CAESAR members. ``(None, None, None)`` when the column (or the
    member flag, if asked for) is missing.
    """
    g = red.get("gas") if red else None
    if g is None or mass_key not in g or (member_only and "member" not in g):
        return None, None, None
    R = reduced_radius(g["pos"], red["evecs"], geom)
    m = np.asarray(g[mass_key], float)
    sfr = np.asarray(g.get("sfr", np.zeros(len(m))), float)
    reg = np.asarray(g["member"], bool) if member_only else np.ones(len(m), bool)
    if region == "ism":
        reg = reg & (sfr > 0)
    elif region == "cgm":
        reg = reg & (sfr == 0)
    return R, m, reg


def spin_unit(L=None, alpha=None, beta=None):
    """Unit angular-momentum vectors ``(..., 3)`` from the L vectors, or from the caesar (ALPHA, BETA) angles.

    The angles are taken in radians, BETA polar from z and ALPHA azimuthal (check against the
    caesar build). None when neither is given.
    """
    if L is not None:
        L = np.asarray(L, float)
        if L.shape[-1] == 3:
            nrm = np.linalg.norm(L, axis=-1, keepdims=True)
            with np.errstate(all="ignore"):
                return np.where(nrm > 0, L / nrm, np.nan)
    if alpha is None or beta is None:
        return None
    a, b = np.asarray(alpha, float), np.asarray(beta, float)
    sb = np.sin(b)
    return np.stack([sb * np.cos(a), sb * np.sin(a), np.cos(b)], axis=-1)
