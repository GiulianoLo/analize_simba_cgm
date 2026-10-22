"""Build progenitor index tables from Caesar merger trees.

Provides two approaches:

- :func:`caesar_read_progen` — reads directly from Caesar HDF5 catalogs.
- :func:`read_progen` — reads from pre-converted FITS catalogs.
"""

import os
import gc
import numpy as np
import h5py
from astropy.io import fits
from astropy.table import Table

from ..io.paths import OutputPaths
from ..io.paths import output_dir as _output_root


def progen_tree_file(sb, snap):
    """Return the file holding ``tree_data/progen_galaxy_star`` for *snap*, or None.

    Prefers the Caesar catalog itself (cis50/cis100 ship the trees in-catalog); falls
    back to a sidecar under ``output/<sim>/progen_links/`` written by
    ``find_progen_m25_job.py`` when the shared catalogs are read-only.
    """
    cat = sb.get_caesar_file(int(snap))
    try:
        with h5py.File(cat, 'r') as f:
            if 'tree_data/progen_galaxy_star' in f:
                return cat
    except OSError:
        pass
    side = os.path.join(str(_output_root()), sb.name, 'progen_links',
                        os.path.basename(cat))
    try:
        with h5py.File(side, 'r') as f:
            if 'tree_data/progen_galaxy_star' in f:
                return side
    except OSError:
        pass
    return None


def caesar_read_progen(ids, outname, snaplist, sb, output_dir=None):
    """Build a FITS progenitor table from Caesar catalogs.

    Loops through Caesar files and writes a table of the most-massive
    progenitor index for each galaxy across snapshots.

    Parameters
    ----------
    ids : array-like
        Galaxy GroupIDs at the base snapshot.
    outname : str
        Output FITS filename.
    snaplist : list of int
        Snapshots to process.
    sb : :class:`~simbanator.io.simba.Simba`
        Simba path manager.
    """
    progenid_dict = {i: [] for i in snaplist}

    for i in np.sort(snaplist)[::-1]:
        cs = sb.get_caesar(i)
        allids = np.asarray([galaxy.GroupID for galaxy in cs.galaxies])
        del cs
        gc.collect()

        try:
            fname = progen_tree_file(sb, i)     # catalog tree_data, else progen_links sidecar
            if fname is None:
                raise KeyError('tree_data/progen_galaxy_star')
            with h5py.File(fname, 'r') as hf:
                progenid = np.asarray(hf['tree_data']['progen_galaxy_star'])[:, 0]
        except (KeyError, OSError):
            # no merger tree at this snapshot (e.g. the earliest epoch — no previous galaxies
            # in the volume). Treat as "no progenitors here" and carry on.
            progenid_dict[i] = np.empty((0, 2), dtype=int)
            continue
        mask = progenid != -1
        progenid_dict[i] = np.column_stack((allids[mask], progenid[mask]))

    # Anchor at the LATEST snapshot (START_SNAP) — the epoch where the input `ids`
    # are defined and from which get_history_indx traces backward. Using the lowest
    # snapshot here keyed the table at END_SNAP (high z), collapsing the sample to
    # galaxies that already existed then (massive only; the low-mass end was missing).
    base_snapshot = np.sort(snaplist)[-1]
    base_data = progenid_dict[base_snapshot]
    if base_data.size == 0:
        # Writing an empty FITS here would silently collapse every downstream
        # selection to zero galaxies (empty valid_ids -> empty histories).
        raise RuntimeError(
            f"No 'tree_data/progen_galaxy_star' for the base snapshot {base_snapshot} — "
            f"neither in the Caesar catalog ({sb.get_caesar_file(int(base_snapshot))}) nor "
            f"as a sidecar in output/{sb.name}/progen_links/. Merger trees were never "
            f"computed for this box — run find_progen_m25_job.py (writes the sidecars), "
            f"then rebuild the progenitor table."
        )
    base_groupids = base_data[:, 0]

    progenid_table = np.full((len(base_groupids), len(snaplist) + 1), -1, dtype=int)
    progenid_table[:, 0] = base_groupids

    for i, snap in enumerate(np.sort(snaplist)):
        snap_data = progenid_dict[snap]
        if snap_data.size == 0:                          # snapshot had no merger tree -> leave -1 (no progenitor)
            continue
        snap_ids = snap_data[:, 0]
        snap_progens = snap_data[:, 1]
        # O(N log N) vectorised lookup via searchsorted
        sorter = np.argsort(snap_ids)
        sorted_ids = snap_ids[sorter]
        pos = np.searchsorted(sorted_ids, base_groupids)
        pos = np.clip(pos, 0, len(sorted_ids) - 1)
        found = sorted_ids[pos] == base_groupids
        progenid_table[found, i + 1] = snap_progens[sorter[pos[found]]]

    if output_dir is None:
        output_dir = os.path.join(str(_output_root()), sb.name, 'progenitors')
    os.makedirs(output_dir, exist_ok=True)

    col1 = fits.Column(name='GroupID', format='K', array=progenid_table[:, 0])
    columns = [col1]
    for i, snap in enumerate(np.sort(snaplist)):
        col = fits.Column(name=f'{snap}', format='K', array=progenid_table[:, i + 1])
        columns.append(col)

    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.writeto(os.path.join(output_dir, outname), overwrite=True)


def _get_fits(sb, snap, fitsdir):
    """Return the FITS filename for a given snapshot."""
    filename = sb.get_caesar_file(snap)
    basename = os.path.splitext(os.path.basename(filename))[0] + '.fits'
    return os.path.join(fitsdir, basename)


def read_progen(ids, outname, snaplist, sb, fitsdir, output_dir=None):
    """Build a progenitor table from converted FITS catalogs.

    Parameters
    ----------
    ids : array-like
        Galaxy GroupIDs at the starting snapshot.
    outname : str
        Output FITS filename.
    snaplist : list of int
        Snapshots to process (highest first).
    sb : :class:`~simbanator.io.simba.Simba`
        Simba path manager.
    fitsdir : str
        Directory with per-snapshot FITS files.
    """
    progenid_dict = {str(i): [] for i in snaplist}
    first_iteration = True
    snaplist = np.sort(snaplist)[::-1]
    progenid = None

    for idx, snap in enumerate(snaplist):
        fname = _get_fits(sb, snap, fitsdir)
        print(f'Processing current snap: {fname}')
        with fits.open(fname) as hdul:
            hf = hdul[1].data
            groupids = hf['GroupID'].astype(int)
            if first_iteration:
                progenid = hf['descend_galaxy_star'].astype(int)
                progenid_dict['GroupID'] = groupids
                progenid_dict[str(snap)] = groupids
                first_iteration = False
            else:
                currprog = hf['descend_galaxy_star'].astype(int)
                valid = progenid > -1
                temp = np.full(len(progenid), -1, dtype=int)
                temp[valid] = groupids[progenid[valid]]
                progenid[valid] = currprog[progenid[valid]]
                progenid[~valid] = -1
                progenid_dict[str(snap)] = temp

    table = Table(progenid_dict)

    if output_dir is None:
        output_dir = os.path.join(str(_output_root()), sb.name, 'progenitors')
    os.makedirs(output_dir, exist_ok=True)
    print('Saving...')
    table.write(os.path.join(output_dir, outname), overwrite=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main branches: progenitor / descendant links, anchor histories
# ═════════════════════════════════════════════════════════════════════════════

def main_progenitors(sb, snap):
    """Catalogue index at ``snap - 1`` of the most massive progenitor of every galaxy at ``snap`` (-1 = none).

    Read from ``tree_data/progen_galaxy_star`` in the catalogue or its ``progen_links`` sidecar
    (:func:`progen_tree_file`); None when neither exists.
    """
    tf = progen_tree_file(sb, snap)
    if tf is None:
        return None
    with h5py.File(tf, "r") as f:
        p = f["tree_data/progen_galaxy_star"][:]
    return (p[:, 0] if p.ndim == 2 else p).astype(np.int64)


def main_descendants(sb, snap_from, idx):
    """Catalogue index at ``snap_from + 1`` of the galaxy whose most massive progenitor is ``idx`` (NaN if none).

    When several galaxies share that progenitor, the most massive one is taken. Returns None
    when ``snap_from + 1`` has no merger-tree links at all.
    """
    s_to = int(snap_from) + 1
    prog = main_progenitors(sb, s_to)
    if prog is None:
        return None
    ms = None
    cat = sb.get_caesar_file(s_to)
    if os.path.exists(cat):
        with h5py.File(cat, "r") as f:
            if "galaxy_data/dicts/masses.stellar" in f:
                ms = np.asarray(f["galaxy_data/dicts/masses.stellar"][:], float)
    order = np.argsort(ms) if (ms is not None and len(ms) == len(prog)) else np.arange(len(prog))
    desc = {}
    for j in order:                        # ascending mass: the last write per progenitor is the most massive
        if prog[j] >= 0:
            desc[int(prog[j])] = int(j)
    idx = np.asarray(idx, float)
    out = np.full(len(idx), np.nan)
    ok = np.isfinite(idx) & (idx >= 0)
    out[ok] = [desc.get(int(i), np.nan) for i in idx[ok]]
    return out


def load_anchor_history(path, cosmology=None):
    """One anchor's property history (written by HDF5BuildHistory) -> dict.

    Keys: ``galaxy_ids``, ``snaps_arr``, ``redshift``, ``t_cosmic_yr`` and ``P`` (every dataset
    under ``properties``, shape ``(n_snap, n_gal)``, row 0 = the anchor). ``cosmology`` defaults
    to Planck15.
    """
    if cosmology is None:
        from astropy.cosmology import Planck15 as cosmology
    H = {"P": {}}
    with h5py.File(path, "r") as f:
        H["galaxy_ids"] = f["metadata/galaxy_ids"][:]
        H["snaps_arr"] = f["metadata/snapshots"][:]
        H["redshift"] = f["redshift/Redshift"][:]
        f["properties"].visititems(
            lambda name, obj: H["P"].__setitem__(name, obj[:]) if isinstance(obj, h5py.Dataset) else None)
    H["t_cosmic_yr"] = cosmology.age(H["redshift"]).value * 1e9
    return H


def progenitor_index_matrix(sb, anchor_snap, prog_file, galaxy_ids, snaps):
    """``(n_snap, n_gal)`` catalogue index of each galaxy's main progenitor, one row per entry of ``snaps``.

    Loads the anchor catalogue with caesar (needs caesar) and walks the progenitor links with
    :class:`~simbanator.histories.properties.HDF5BuildHistory`.
    """
    import gc
    from .properties import HDF5BuildHistory
    cs0 = sb.load_catalog(snap=anchor_snap)
    hP = HDF5BuildHistory(sb, cs0, progfilename=prog_file)
    hP.get_history_indx(galaxy_ids, int(np.max(snaps)), int(np.min(snaps)))
    M = np.vstack([hP.history_indx[str(s)] for s in snaps])
    del cs0, hP
    gc.collect()
    return M


def align_by_ids(arrays, cached_ids, current_ids, axis=0, name="", verbose=True):
    """Re-index arrays built for ``cached_ids`` onto ``current_ids`` along ``axis`` (missing ids -> NaN).

    Arrays whose length along ``axis`` is not ``len(cached_ids)`` are returned untouched.
    Returns ``(arrays, n_matched)``; a mismatch prints a warning naming ``name``.
    """
    cached_ids = np.asarray(cached_ids).astype(np.int64); current_ids = np.asarray(current_ids).astype(np.int64)
    if len(cached_ids) == len(current_ids) and np.array_equal(cached_ids, current_ids):
        return arrays, len(current_ids)
    pos = {int(g): j for j, g in enumerate(cached_ids)}
    sel = np.array([pos.get(int(g), -1) for g in current_ids]); found = sel >= 0
    out = {}
    for k, a in arrays.items():
        a = np.asarray(a)
        if a.ndim <= axis or a.shape[axis] != len(cached_ids):
            out[k] = a; continue
        shape = list(a.shape); shape[axis] = len(current_ids)
        b = np.full(shape, np.nan, float)
        sl = [slice(None)] * b.ndim; sl[axis] = np.where(found)[0]
        b[tuple(sl)] = np.take(a, sel[found], axis=axis)
        out[k] = b
    if verbose:
        print(f"[warn] {name}: cached product holds {len(cached_ids)} galaxies but the sample has {len(current_ids)} "
              f"({int(found.sum())} matched, {int((~found).sum())} -> NaN). Rebuild it to cover the full sample.")
    return out, int(found.sum())
