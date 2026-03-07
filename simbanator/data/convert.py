"""Convert HDF5 Caesar catalogs to FITS tables."""

import os
import numpy as np
import h5py
from astropy.io import fits

from ..io.paths import SavePaths


def _adapt_keys(path):
    """Convert an HDF5 dataset path to a human-readable column name.

    Example: ``'galaxy_data/dicts/masses.stellar'`` → ``'stellar_masses'``
    """
    final = path.split('/')[-1]
    if '.' in final:
        parts = final.split('.')
        final = '_'.join(reversed(parts))
    return final


def _read_main_tree(path):
    return path.split('/')[0]


def _empty_fits(filename, fitspath):
    """Create an empty FITS file with a primary HDU and empty BINTABLE."""
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(fits.ColDefs([])),
    ])
    new_filename = filename.split('/')[-1].split('.')[0] + '.fits'
    new_filename = os.path.join(fitspath, new_filename)
    hdul.writeto(new_filename, overwrite=True)
    return new_filename


def _list_all_datasets(file):
    """Return all dataset paths inside an HDF5 file."""
    paths = []
    file.visititems(lambda name, node: paths.append(name) if isinstance(node, h5py.Dataset) else None)
    return paths


def _process_hdf5(file_path, fitspath, ignore, only, verbose=0):
    """Convert one HDF5 Caesar catalog to a FITS table."""
    fits_filename = _empty_fits(file_path, fitspath)
    print(f'Saved FITS in: {fits_filename}')

    with h5py.File(file_path) as f:
        hdul = fits.open(fits_filename, mode='update')
        dataset_paths = _list_all_datasets(f)

        for dataset_path in dataset_paths:
            skip = False
            h5col = f[dataset_path]
            tree = _read_main_tree(dataset_path)
            column_name = _adapt_keys(dataset_path)

            if ignore:
                for ign in ignore:
                    if ign in dataset_path:
                        skip = True
                        break
            if only:
                for onl in only:
                    if onl not in dataset_path:
                        skip = True
                        break
            if skip:
                continue

            if tree == 'halo_data':
                if verbose:
                    print('Copying Halo')
                selgal = f['galaxy_data']['parent_halo_index']
                h5col = np.asarray(h5col)[selgal]
                column_name = 'halo_' + column_name

            if tree == 'tree_data':
                if verbose:
                    print('Copying tree')
                progenid = f['tree_data']['progen_galaxy_star'][:, 0]
                new_col = fits.Column(name='progen_galaxy_star', array=progenid, format='E')
                hdul[1].columns.add_col(new_col)
                break

            if len(np.asarray(h5col).shape) > 1:
                for i in range(np.asarray(h5col).shape[1]):
                    new_col = fits.Column(name=f'{column_name}_{i}', array=np.asarray(h5col)[:, i], format='E')
                    hdul[1].columns.add_col(new_col)
            else:
                new_col = fits.Column(name=column_name, array=np.asarray(h5col), format='E')
                hdul[1].columns.add_col(new_col)

        hdul.flush()
        hdul.close()


def convert_hdf5_fits(sb, snaprange, ignore=None, only=None, verbose=0):
    """Convert Caesar HDF5 catalogs to FITS for a range of snapshots.

    Parameters
    ----------
    sb : :class:`~simbanator.io.simba.Simba`
        Simba path manager.
    snaprange : iterable of int
        Snapshot numbers to convert.
    ignore : list of str, optional
        HDF5 path substrings to skip.
    only : list of str, optional
        If set, only include datasets whose paths contain these substrings.
    verbose : int
        Verbosity level.
    """
    if ignore is None:
        ignore = []
    if only is None:
        only = []

    paths = SavePaths()
    fitspath = paths.create_subdir(
        paths.get_filetype_path('fits'), 'converted_from_hdf5'
    )

    for snap in snaprange:
        file_path = sb.get_caesar_file(snap)
        print(f'Processing: {file_path}')
        _process_hdf5(file_path, fitspath, ignore, only, verbose)
