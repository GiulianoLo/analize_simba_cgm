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

from ..io.paths import SavePaths


def caesar_read_progen(ids, outname, snaplist, sb):
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
        for j, groupid in enumerate(base_groupids):
            idx = np.where(snap_data[:, 0] == groupid)[0]
            if len(idx) > 0:
                progenid_table[j, i + 1] = snap_data[idx[0], 1]

    paths = SavePaths()
    output_dir = paths.create_subdir(
        paths.get_filetype_path('fits'), 'progenitors_files'
    )

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
    new_filename = filename.split('/')[-1].split('.')[0] + '.fits'
    return os.path.join(fitsdir, new_filename)


def read_progen(ids, outname, snaplist, sb, fitsdir):
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
                temp = []
                for i in range(len(progenid)):
                    if progenid[i] > -1:
                        temp.append(groupids[progenid[i]])
                        progenid[i] = currprog[progenid[i]]
                    else:
                        temp.append(-1)
                        progenid[i] = -1
                progenid_dict[str(snap)] = np.array(temp)

    table = Table(progenid_dict)

    paths = SavePaths()
    output_dir = paths.create_subdir(
        paths.get_filetype_path('fits'), 'progenitors_files'
    )
    print('Saving...')
    table.write(os.path.join(output_dir, outname), overwrite=True)
