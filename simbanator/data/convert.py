"""Convert HDF5 Caesar catalogs to FITS tables."""

import os
import gc
import numpy as np
import h5py
from astropy.io import fits




def _adapt_keys(path):
    """Convert an HDF5 dataset path to a human-readable column name.

    Example: ``'galaxy_data/dicts/masses.stellar'`` → ``'stellar_masses'``
    """
    final = path.split('/')[-1]
    if '.' in final:
        parts = final.split('.')
        final = '_'.join(reversed(parts))
    return final


def _full_path_name(path):
    """Generate a unique column name from the full HDF5 path.

    Joins all path components with ``'__'`` and applies the dot-reversal
    convention to the leaf name.  Truncated to 68 chars (FITS limit).

    Example: ``'galaxy_data/dicts/masses.stellar'``
             → ``'galaxy_data__dicts__stellar_masses'``
    """
    parts = path.split('/')
    leaf = parts[-1]
    if '.' in leaf:
        leaf = '_'.join(reversed(leaf.split('.')))
    name = '__'.join(parts[:-1] + [leaf])
    return name[:68]  # FITS TTYPEn hard limit


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


# def _process_hdf5(file_path, fitspath, ignore, only, verbose=0):
#     """Convert one HDF5 Caesar catalog to a FITS table."""
#     fits_filename = _empty_fits(file_path, fitspath)
#     print(f'Saved FITS in: {fits_filename}')

#     with h5py.File(file_path) as f:
#         hdul = fits.open(fits_filename, mode='update')
#         dataset_paths = _list_all_datasets(f)

#         for dataset_path in dataset_paths:
#             skip = False
#             h5col = f[dataset_path]
#             tree = _read_main_tree(dataset_path)
#             column_name = _adapt_keys(dataset_path)

#             if ignore:
#                 for ign in ignore:
#                     if ign in dataset_path:
#                         skip = True
#                         break
#             if only:
#                 for onl in only:
#                     if onl not in dataset_path:
#                         skip = True
#                         break
#             if skip:
#                 continue

#             if tree == 'halo_data':
#                 if verbose:
#                     print('Copying Halo')
#                 selgal = f['galaxy_data']['parent_halo_index']
#                 h5col = np.asarray(h5col)[selgal]
#                 column_name = 'halo_' + column_name

#             if tree == 'tree_data':
#                 if verbose:
#                     print('Copying tree')
#                 progenid = f['tree_data']['progen_galaxy_star'][:, 0]
#                 new_col = fits.Column(name='progen_galaxy_star', array=progenid, format='E')
#                 hdul[1].columns.add_col(new_col)
#                 break

#             if len(np.asarray(h5col).shape) > 1:
#                 for i in range(np.asarray(h5col).shape[1]):
#                     new_col = fits.Column(name=f'{column_name}_{i}', array=np.asarray(h5col)[:, i], format='E')
#                     hdul[1].columns.add_col(new_col)
#             else:
#                 new_col = fits.Column(name=column_name, array=np.asarray(h5col), format='E')
#                 hdul[1].columns.add_col(new_col)

#         hdul.flush()
#         hdul.close()


def _fits_format(dtype):
    """Map a numpy dtype to the appropriate FITS column format string."""
    kind = np.dtype(dtype).kind
    itemsize = np.dtype(dtype).itemsize
    if kind == 'f':
        return 'D' if itemsize >= 8 else 'E'
    if kind in ('i', 'u'):
        return 'K' if itemsize >= 8 else 'J'
    return 'E'  # fallback for booleans / other kinds


def _iter_columns(f, dataset_paths, ignore, only, selgal, n_galaxies, n_halos, verbose):
    """Yield ``(column_name, dataset_path, dtype, shape, is_halo, sub_col)``
    for every output column, without loading any data.

    Datasets whose leading dimension does not match *n_galaxies* (or *n_halos*
    for ``halo_data``) are silently skipped — this guards against accidentally
    pulling in per-particle arrays that would make the skeleton tens of TiB.
    """
    seen_names = set()

    def _unique_name(candidate, dataset_path):
        """Return *candidate* if unused, else fall back to the full-path name."""
        if candidate not in seen_names:
            return candidate
        fallback = _full_path_name(dataset_path)
        if fallback in seen_names:
            # Last resort: append a numeric suffix
            n = 1
            while f'{fallback}_{n}' in seen_names:
                n += 1
            fallback = f'{fallback}_{n}'
        if verbose:
            print(f'  Name collision: "{candidate}" → "{fallback}" (path: {dataset_path})')
        return fallback

    for dataset_path in dataset_paths:
        if any(ign in dataset_path for ign in ignore):
            continue
        if only and not any(onl in dataset_path for onl in only):
            continue

        tree = _read_main_tree(dataset_path)
        ds = f[dataset_path]
        raw_shape = ds.shape
        dtype = ds.dtype

        # Skip scalar datasets or anything with no rows
        if len(raw_shape) == 0:
            if verbose:
                print(f'  Skipping scalar dataset: {dataset_path}')
            continue

        if tree == 'tree_data':
            if dataset_path == 'tree_data/progen_galaxy_star':
                name = _unique_name('progen_galaxy_star', dataset_path)
                seen_names.add(name)
                out_shape = (raw_shape[0],) if ds.ndim == 2 else raw_shape
                yield name, dataset_path, dtype, out_shape, False, None
            continue

        column_name = _adapt_keys(dataset_path)
        is_halo = tree == 'halo_data' and selgal is not None

        if is_halo:
            # halo arrays must have exactly n_halos rows
            if n_halos is not None and raw_shape[0] != n_halos:
                if verbose:
                    print(f'  Skipping {dataset_path}: '
                          f'rows {raw_shape[0]} != n_halos {n_halos}')
                continue
            out_len = len(selgal)
            column_name = 'halo_' + column_name
        else:
            # galaxy/tree arrays must have exactly n_galaxies rows
            if n_galaxies is not None and raw_shape[0] != n_galaxies:
                if verbose:
                    print(f'  Skipping {dataset_path}: '
                          f'rows {raw_shape[0]} != n_galaxies {n_galaxies}')
                continue
            out_len = raw_shape[0]

        if len(raw_shape) == 2:
            for i in range(raw_shape[1]):
                candidate = f'{column_name}_{i}'
                name = _unique_name(candidate, dataset_path)
                seen_names.add(name)
                yield name, dataset_path, dtype, (out_len,), is_halo, i
        else:
            name = _unique_name(column_name, dataset_path)
            seen_names.add(name)
            yield name, dataset_path, dtype, (out_len,), is_halo, None


def _load_column(f, dataset_path, column_name, is_halo, selgal):
    """Load and return a single output array (remapped if halo)."""
    tree = _read_main_tree(dataset_path)
    ds = f[dataset_path]

    if tree == 'tree_data':
        raw = ds[:]
        return raw[:, 0] if raw.ndim == 2 else raw

    data = ds[:]

    if is_halo:
        n_halos = data.shape[0]
        valid = (selgal >= 0) & (selgal < n_halos)
        is_float = np.issubdtype(data.dtype, np.floating)
        out_dtype = float if is_float else data.dtype
        fill = np.nan if is_float else -1
        if data.ndim == 1:
            result = np.full(len(selgal), fill, dtype=out_dtype)
            result[valid] = data[selgal[valid]]
        else:
            result = np.full((len(selgal),) + data.shape[1:], fill, dtype=out_dtype)
            result[valid] = data[selgal[valid]]
        return result

    return data


def _process_hdf5(file_path, fitspath, ignore, only, verbose=0):
    """Convert one HDF5 Caesar catalog to a FITS table.

    Uses a two-pass strategy to keep peak RAM to a single column at a time:

    1. Scan dataset shapes / dtypes without loading data → pre-allocate a
       zero-filled FITS file on disk.
    2. Re-open the FITS file with ``memmap=True`` and write each column
       directly into the memory-mapped buffer, freeing the array immediately
       afterwards.
    """
    basename = os.path.splitext(os.path.basename(file_path))[0] + '.fits'
    fits_filename = os.path.join(fitspath, basename)
    print(f"Processing {file_path}")

    # ------------------------------------------------------------------
    # Pass 1 — collect metadata, build zero-filled FITS skeleton on disk
    # ------------------------------------------------------------------
    with h5py.File(file_path, 'r') as f:
        dataset_paths = _list_all_datasets(f)
        selgal = (
            f['galaxy_data/parent_halo_index'][:]
            if 'galaxy_data/parent_halo_index' in f else None
        )

        # Determine expected row counts to guard against per-particle datasets
        n_galaxies = (
            len(f['galaxy_data/GroupID'])
            if 'galaxy_data/GroupID' in f else None
        )
        n_halos = (
            len(f['halo_data/GroupID'])
            if 'halo_data/GroupID' in f else None
        )
        if verbose:
            print(f'  n_galaxies={n_galaxies}, n_halos={n_halos}')

        col_meta = list(_iter_columns(
            f, dataset_paths, ignore, only, selgal,
            n_galaxies, n_halos, verbose
        ))

    # Build ColDefs with zero-length placeholder arrays (no data in RAM)
    col_defs = [
        fits.Column(name=name, format=_fits_format(dtype),
                    array=np.zeros(shape, dtype=dtype))
        for name, _path, dtype, shape, _halo, _sub in col_meta
    ]
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(col_defs)])
    hdul.writeto(fits_filename, overwrite=True)
    hdul.close()
    del col_defs, hdul
    gc.collect()

    if verbose:
        print(f'  Skeleton written ({len(col_meta)} columns)')

    # ------------------------------------------------------------------
    # Pass 2 — fill each column via memmap; one column in RAM at a time
    # ------------------------------------------------------------------
    with h5py.File(file_path, 'r') as f:
        selgal = (
            f['galaxy_data/parent_halo_index'][:]
            if 'galaxy_data/parent_halo_index' in f else None
        )
        n_galaxies = (
            len(f['galaxy_data/GroupID'])
            if 'galaxy_data/GroupID' in f else None
        )
        n_halos = (
            len(f['halo_data/GroupID'])
            if 'halo_data/GroupID' in f else None
        )

        with fits.open(fits_filename, mode='update', memmap=True) as hdul:
            seen_paths = {}   # dataset_path → loaded ndarray
            for col_idx, (name, dataset_path, _dtype, _shape, is_halo, sub_col) in enumerate(col_meta):
                if verbose:
                    print(f'  [{col_idx + 1}/{len(col_meta)}] Writing {name}')

                # Load the raw array once per dataset_path; reuse for each
                # sub-column of a 2-D dataset.
                if dataset_path not in seen_paths:
                    seen_paths[dataset_path] = _load_column(
                        f, dataset_path, name, is_halo, selgal
                    )

                raw = seen_paths[dataset_path]

                if sub_col is not None:
                    hdul[1].data[name][:] = raw[:, sub_col]
                    # Free after the last sub-column
                    if sub_col == raw.shape[1] - 1:
                        del seen_paths[dataset_path]
                        gc.collect()
                else:
                    hdul[1].data[name][:] = raw
                    del seen_paths[dataset_path]
                    gc.collect()

            hdul.flush()

    print(f'Saved FITS in: {fits_filename}')



def convert_hdf5_fits(sb, snaprange, ignore=None, only=None, verbose=0, output_dir=None):
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
        ignore = ['particle_data']
    else:
        ignore = list(ignore)
        if 'particle_data' not in ignore:
            ignore.append('particle_data')

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'output', 'converted_catalogs')
    os.makedirs(output_dir, exist_ok=True)

    for snap in snaprange:
        file_path = sb.get_caesar_file(snap)
        print(f'Processing: {file_path}')
        _process_hdf5(file_path, output_dir, ignore, only, verbose)
