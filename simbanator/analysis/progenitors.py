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
        fname = cs.data_file
        del cs
        gc.collect()

        with h5py.File(fname, 'r') as hf:
            progenid = np.asarray(hf['tree_data']['progen_galaxy_star'])[:, 0]
            mask = progenid != -1
            progenid = progenid[mask]
            allids = allids[mask]
        progenid_dict[i] = np.column_stack((allids, progenid))

    base_snapshot = np.sort(snaplist)[0]
    base_data = progenid_dict[base_snapshot]
    base_groupids = base_data[:, 0]

    progenid_table = np.full((len(base_groupids), len(snaplist) + 1), -1, dtype=int)
    progenid_table[:, 0] = base_groupids

    for i, snap in enumerate(np.sort(snaplist)):
        snap_data = progenid_dict[snap]
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
        output_dir = os.path.join(os.getcwd(), 'output', sb.name, 'progenitors')
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
        output_dir = os.path.join(os.getcwd(), 'output', sb.name, 'progenitors')
    os.makedirs(output_dir, exist_ok=True)
    print('Saving...')
    table.write(os.path.join(output_dir, outname), overwrite=True)
