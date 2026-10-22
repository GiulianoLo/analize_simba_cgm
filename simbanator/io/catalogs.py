"""Caesar catalogues read directly with h5py (no caesar import), plus snapshot / redshift helpers.

Every reader takes the catalogue file path, so it works with any simulation handle
(``sim.get_caesar_file(snap)``) or a path from the config.
"""

import os
import warnings

import h5py
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


# ═════════════════════════════════════════════════════════════════════════════
# Direct catalogue reads
# ═════════════════════════════════════════════════════════════════════════════

def catalog_path(name, snap):
    """Catalogue file of snapshot ``snap`` for the configured simulation ``name``."""
    from .config import get_simulation_config
    cfg = get_simulation_config(name)
    cat_dir = cfg.get("catalog_dir") or os.path.join(cfg["data_dir"], "Groups")
    return os.path.join(cat_dir, cfg.get("file_format", "{snap:03d}.hdf5").format(snap=int(snap)))


def file_prefix(name):
    """The run prefix of a configured simulation's catalogue names (``m25n512``, ``m100n1024``)."""
    from .config import get_simulation_config
    return get_simulation_config(name).get("file_format", "").split("_{")[0]


def has_catalog(path):
    """True when the catalogue file exists (False for a path that cannot even be built)."""
    try:
        return os.path.exists(path)
    except (TypeError, ValueError):
        return False


def dataset(f, name, group="galaxy_data", dtype=np.float64):
    """``group/dicts/<name>`` or ``group/<name>`` from an open catalogue, as an array; None when neither exists.

    Caesar stores most quantities under ``dicts/`` (``masses.stellar``) and a few flat
    (``sfr``, ``pos``); this tries both, in that order.
    """
    for p in (f"{group}/dicts/{name}", f"{group}/{name}"):
        if p in f:
            return np.asarray(f[p][:], dtype=dtype)
    return None


def sim_attributes(path):
    """``simulation_attributes`` of a catalogue: redshift, scale factor, h, Omega_m, box size (comoving kpc)."""
    with h5py.File(path, "r") as f:
        a = f["simulation_attributes"].attrs
        out = dict(z=float(a["redshift"]), a=float(a["scale_factor"]))
        for key, attr in (("h", "hubble_constant"), ("omega_matter", "omega_matter"), ("boxsize", "boxsize")):
            if attr in a:
                out[key] = float(a[attr])
    return out


def read_galaxy_columns(path, columns, index=None, group="galaxy_data", missing="nan"):
    """Galaxy datasets of one catalogue, by output name.

    Parameters
    ----------
    path : str
        Catalogue file.
    columns : dict
        ``{out_name: dataset}`` where ``dataset`` is a name (``"masses.stellar"``, tried under
        ``dicts/`` and flat) or a list of candidate names, the first one present wins.
    index : array of int, optional
        Galaxy indices (= GroupID = row) to keep. Out-of-range indices give NaN.
    missing : {"nan", "raise", "skip"}
        What to do with a column none of whose candidates exists: NaN-filled, raise
        ``KeyError``, or leave it out of the result.

    Returns
    -------
    dict of numpy arrays, plus ``"_missing"``: the list of output names that were not found.
    A missing file gives all-NaN columns (and a warning) unless ``missing="raise"``.
    """
    idx = None if index is None else np.asarray(index, int)
    if not os.path.exists(path):
        if missing == "raise":
            raise FileNotFoundError(path)
        warnings.warn(f"no catalogue at {path}: its columns stay NaN", stacklevel=2)
        n = 0 if idx is None else len(idx)
        return {**{k: np.full(n, np.nan) for k in columns}, "_missing": list(columns)}
    out, miss = {}, []
    with h5py.File(path, "r") as f:
        for key, cands in columns.items():
            cands = [cands] if isinstance(cands, str) else list(cands)
            arr = next((a for a in (dataset(f, c, group, dtype=None) for c in cands) if a is not None), None)
            if arr is None:
                miss.append(key)
                if missing == "raise":
                    raise KeyError(f"{path}: none of {cands} in {group}")
                if missing == "skip":
                    continue
                n = len(idx) if idx is not None else (dataset(f, "masses.stellar", group).shape[0]
                                                      if dataset(f, "masses.stellar", group) is not None else 0)
                out[key] = np.full(n, np.nan)
                continue
            if idx is not None:
                ok = (idx >= 0) & (idx < len(arr))
                vals = np.full((len(idx),) + arr.shape[1:], np.nan)
                vals[ok] = arr[idx[ok]]
                arr = vals
            out[key] = arr
    out["_missing"] = miss
    return out


def parent_halo_columns(path, columns, index=None):
    """Datasets of each galaxy's parent halo (``halo_data``), aligned to the galaxies.

    ``columns`` maps output names to halo datasets (``"virial_quantities.m200c"``). Galaxies
    without a parent halo (index -1), or a halo index out of range, get NaN. Also returns
    ``"parent"``, the parent-halo index of every selected galaxy.
    """
    with h5py.File(path, "r") as f:
        parent = np.asarray(f["galaxy_data/parent_halo_index"][:], np.int64)
        halo = {k: dataset(f, d, group="halo_data") for k, d in columns.items()}
    if index is not None:
        idx = np.asarray(index, int)
        p = np.full(len(idx), -1, np.int64)
        ok = (idx >= 0) & (idx < len(parent))
        p[ok] = parent[idx[ok]]
        parent = p
    out = {"parent": parent}
    for k, arr in halo.items():
        vals = np.full(len(parent), np.nan)
        if arr is not None:
            ok = (parent >= 0) & (parent < len(arr))
            vals[ok] = arr[parent[ok]]
        out[k] = vals
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Snapshots, redshift and time
# ═════════════════════════════════════════════════════════════════════════════

def nearest_snapshot(value, snaps, grid):
    """The snapshot whose ``grid`` value (redshift or cosmic time) is closest to ``value``.

    Returns ``(snap, grid_value)``, or ``(None, nan)`` for a non-finite ``value``.
    """
    snaps, grid = np.asarray(snaps), np.asarray(grid, float)
    if not np.isfinite(value):
        return None, np.nan
    j = int(np.nanargmin(np.abs(grid - value)))
    return int(snaps[j]), float(grid[j])


def z_tag(z):
    """A redshift as a file-name tag: 0.3 -> ``'z0p3'``, 1 -> ``'z1'``."""
    return ("z%g" % z).replace(".", "p")


def time_redshift_maps(cosmology=None, zmax=20.0, n=4000):
    """Fast vectorised converters ``(z_to_t, t_to_z)`` [Gyr], interpolated on a log-spaced redshift grid.

    ``cosmology`` defaults to the package cosmology (Planck15, as in :mod:`simbanator.io.simba`).
    """
    if cosmology is None:
        from astropy.cosmology import Planck15 as cosmology
    zg = np.concatenate([[0.0], np.logspace(-4, np.log10(zmax), n)])
    tg = cosmology.age(zg).value
    def z_to_t(z):
        return np.interp(z, zg, tg)
    def t_to_z(t):
        return np.interp(t, tg[::-1], zg[::-1])
    return z_to_t, t_to_z


def softening_pkpc(eps_ckpc_h, h, a):
    """A comoving softening length [ckpc/h] in physical kpc at scale factor ``a``."""
    return eps_ckpc_h / h * a


# ═════════════════════════════════════════════════════════════════════════════
# Moved from ks_tracks_lib.py (the K-S track measurements), unchanged
# ═════════════════════════════════════════════════════════════════════════════

def make_a_to_t(cosmo, n=4096, a_min=0.02):
    """Callable a_form (scale factor) -> cosmic time [Gyr] on an interpolation grid
    (the Part-7e recipe of powderday_flux_quenched_m25: np.interp on 4096 nodes)."""
    a_grid = np.linspace(a_min, 1.0, int(n))
    t_grid = np.asarray(cosmo.age(1.0 / a_grid - 1.0).value, float)

    def a_to_t(a):
        a = np.asarray(a, float)
        return np.interp(np.clip(a, a_grid[0], 1.0), a_grid, t_grid)
    a_to_t.a_grid, a_to_t.t_grid = a_grid, t_grid
    return a_to_t
