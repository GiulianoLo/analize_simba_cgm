import os
import gc
import numpy as np
import h5py
from astropy.io import fits
from scipy.interpolate import interp1d, splrep, splev
from astropy.cosmology import Planck15 as _default_cosmo

from ..visualization.plots import HistoryPlots


def _resolve_h5_path(family, propr):
    """Map a ``(family, property)`` pair to the HDF5 dataset path.

    Caesar stores properties as e.g.::

        galaxy_data/sfr
        galaxy_data/dicts/masses.stellar
        galaxy_data/dicts/masses.gas
        halo_data/dicts/masses.total
        halo_data/virial_quantities/r200

    *propr* can be given as:
    - a full HDF5 path: ``'galaxy_data/dicts/masses.stellar'``
    - a dotted shorthand: ``'masses.stellar'`` (searches ``dicts/`` first,
      then the family group root)
    - a plain name: ``'sfr'``

    Parameters
    ----------
    family : str
        ``'galaxy_data'`` or ``'halo_data'``.
    propr : str
        Property name or HDF5 path.

    Returns
    -------
    str
        Full HDF5 dataset path.
    """
    if '/' in propr:
        return propr
    if '.' in propr:
        return f'{family}/dicts/{propr}'
    return f'{family}/{propr}'


def _read_h5_property(h5file, h5path, indices):
    """Read an HDF5 dataset for specific row indices only.

    Handles unsorted indices, duplicates, and big-endian dtypes safely.
    If the dataset is 2-D (e.g. positions, velocities), returns the full
    sub-matrix ``(len(indices), ncols)``.
    """
    ds = h5file[h5path]
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return np.array([], dtype=ds.dtype)
    # h5py requires indices to be sorted and unique, but we want to preserve duplicates and order
    unique_idx, inverse = np.unique(indices, return_inverse=True)
    unique_data = ds[unique_idx]
    # Reconstruct the output array to match the original indices (including duplicates)
    data = unique_data[inverse]
    return data


def _unwrap_positions(positions, boxsize):
    import numpy as np

    # positions shape: (n_snap, n_gal, ndim)  -> ndim = 3 for (x,y,z)
    unwrapped = positions.copy()

    # A galaxy is valid at a snapshot if NONE of its coords are NaN
    valid_mask = ~np.isnan(unwrapped).any(axis=-1)

    n_snap, n_gal, ndim = unwrapped.shape

    for gal in range(n_gal):
        for i in range(1, n_snap):

            if valid_mask[i, gal] and valid_mask[i - 1, gal]:

                delta = unwrapped[i, gal] - unwrapped[i - 1, gal]

                # Apply periodic correction per dimension
                delta[delta > 0.5 * boxsize] -= boxsize
                delta[delta < -0.5 * boxsize] += boxsize

                # Reconstruct continuous trajectory
                unwrapped[i, gal] = unwrapped[i - 1, gal] + delta

            # else: leave as NaN (or unchanged)

    return unwrapped


class HDF5BuildHistory:
    """Track galaxy properties directly from Caesar HDF5 files via h5py.

    This class **never** imports ``caesar`` and **never** requires a
    FITS conversion step.  Each snapshot file is opened, only the
    requested datasets are read at the requested row indices, and
    the file handle is immediately closed — peak RAM is minimal.

    Parameters
    ----------
    sb : :class:`~simbanator.io.simba.Simulation`
        Simulation handle (used for file paths and snap→z mapping).
    progfilename : str
        Progenitor index FITS file name (same table used by
        :class:`BuildHistory`).
    progen_dir : str, optional
        Directory containing the progenitor file.

    Examples
    --------
    >>> hist = HDF5BuildHistory(sim)
    >>> hist.get_history_indx([42, 99], 130, 149)
    >>> z, props = hist.get_property_history({
    ...     'galaxy_data': ['masses.stellar', 'sfr'],
    ...     'halo_data':   ['masses.total'],
    ... })
    """

    def __init__(self, sb, cs, progfilename='progenitors_most_mass.fits',
                 progen_dir=None):
        if progen_dir is None:
            progen_dir = os.path.join(os.getcwd(), 'output', sb.name, 'progenitors')
        self.progen_file = os.path.join(progen_dir, progfilename)
        self.history_indx = None
        self.sb = sb
        self.cs = cs
        self.z = {}
        self.propr = {}
        self.galaxy_ids = None

    # ── Progenitor index read ────────────────────────────────────────

    def get_history_indx(self, ids, start_snap, end_snap):
        """Retrieve progenitor index chains for *ids* from start_snap to end_snap.

        Traverses backward using the progenitor FITS file.
        """
        ids = np.atleast_1d(ids)
        self.galaxy_ids = np.asarray(ids, dtype=np.int64)
        with fits.open(self.progen_file) as hdul:
            data = hdul[1].data
            id_column = np.asarray(data['GroupID'])
            col_names = hdul[1].columns.names

            # Build lookup: GroupID -> row index
            id_lookup = {int(gid): idx for idx, gid in enumerate(id_column)}
            missing = [int(i) for i in ids if int(i) not in id_lookup]
            if missing:
                raise ValueError(f"GroupIDs not found in progen file: {missing}")

            # Starting row indices in FITS for the input GroupIDs
            current_idx = np.array([id_lookup[int(i)] for i in ids])
            n_gal = len(ids)

            # Snapshots in backward order
            snaps = [str(snap) for snap in range(start_snap, end_snap - 1, -1)]
            n_snaps = len(snaps)

            # Initialize output array
            indices = np.full((n_snaps, n_gal), np.nan, dtype=float)
            
            # Start with the input GroupIDs (these are row indices for start_snap)
            indices[0, :] = current_idx

            # Follow progenitor chain
            for i in range(1, n_snaps):
                prev_idx = indices[i - 1, :]
                col_snap = snaps[i - 1]
                for j in range(n_gal):
                    if np.isnan(prev_idx[j]):
                        indices[i, j] = np.nan
                    else:
                        idx = int(prev_idx[j])
                        if 0 <= idx < len(data):
                            val = data[col_snap][idx]
                            indices[i, j] = val if val >= 0 else np.nan
                        else:
                            indices[i, j] = np.nan

            # Build dict mapping snapshot string -> indices
            self.history_indx = {snap: indices[i, :] for i, snap in enumerate(snaps)}
            # for snap, ind in self.history_indx.items():
            #     print(snap, ind)

        return self.history_indx
    # ── Property read ────────────────────────────────────────────────

    def get_property_history(self, propr_dicts, verbose=0):
        """Read galaxy/halo properties at each snapshot from HDF5 files.

        Parameters
        ----------
        propr_dicts : dict
            ``{'galaxy_data': ['masses.stellar', 'sfr', ...],
               'halo_data':   ['masses.total', ...]}``

            Keys are the HDF5 top-level group names (``galaxy_data`` or
            ``halo_data``).  Values are lists of property names; each is
            resolved to an HDF5 path by :func:`_resolve_h5_path`.
        verbose : int
            Print progress messages.

        Returns
        -------
        tuple
            ``(redshift_dict, property_dict)``
        """
        if self.history_indx is None:
            raise RuntimeError("Call get_history_indx() first.")

        indx_dict = self.history_indx
        redshiftl = []
        snapshotl = []
        propr_out = {
            propr: []
            for family in propr_dicts
            for propr in propr_dicts[family]
        }

        for snap_str in indx_dict:
            snap = int(snap_str)
            h5path = self.sb.get_caesar_file(snap)
            if verbose:
                print(f'  snap {snap}: {h5path}')

            # Convert to array and mask missing progenitors
            galaxy_idx = np.atleast_1d(np.asarray(indx_dict[snap_str]))
            scalar_index = np.ndim(indx_dict[snap_str]) == 0

            # Mask valid progenitors (not NaN)
            valid_mask = ~np.isnan(galaxy_idx)
            galaxy_idx_valid = galaxy_idx[valid_mask].astype(int)
            assert np.issubdtype(galaxy_idx_valid.dtype, np.integer), f"Non-integer indices found: {galaxy_idx_valid}"

            # print(f'    {len(galaxy_idx_valid)} valid progenitors at snap {snap} (masked {len(galaxy_idx) - len(galaxy_idx_valid)})')
            # print(f'    Valid indices: {galaxy_idx_valid} (dtype: {galaxy_idx_valid.dtype})')
            with h5py.File(h5path, 'r') as f:
                if ('simulation_attributes' in f
                        and 'redshift' in f['simulation_attributes'].attrs):
                    z_val = float(f['simulation_attributes'].attrs['redshift'])
                else:
                    z_val = self.sb.get_z_from_snap(snap)
                redshiftl.append(z_val)
                snapshotl.append(snap)

                for family, properties in propr_dicts.items():
                    if family == 'halo_data':
                        parent_path = 'galaxy_data/parent_halo_index'
                        if parent_path not in f:
                            raise KeyError(f'{parent_path} not found in {h5path}')
                        parent_halo = np.asarray(f[parent_path][:], dtype=np.int64)
                        # Map only valid galaxy indices to their parent halos
                        idx = parent_halo[galaxy_idx_valid]
                    else:
                        idx = galaxy_idx_valid

                    for propr in properties:
                        ds_path = _resolve_h5_path(family, propr)
                        if ds_path not in f:
                            raise KeyError(
                                f'Dataset {ds_path} not found in {h5path}.'
                            )

                        valid_values = _read_h5_property(f, ds_path, idx)

                        # Build full output array, NaN where progenitor is missing
                        if valid_values.ndim == 1:
                            out = np.full(len(galaxy_idx), np.nan)
                        else:
                            out = np.full(
                                (len(galaxy_idx), valid_values.shape[1]), np.nan
                            )
                        out[valid_mask] = valid_values

                        if scalar_index:
                            out = out[0]
                        propr_out[propr].append(out)

        for propr in propr_out:
            propr_out[propr] = np.asarray(propr_out[propr])

        redshift = {
            'Redshift': np.asarray(redshiftl),
            'Snapshot': np.asarray(snapshotl, dtype=np.int32)}
        self.z = redshift
        self.propr = propr_out

        # Unwrap positions for periodic box crossing if present
        boxsize = self.cs.simulation.boxsize.value
        if 'pos' in self.propr:
            print(f'Unwrapping positions for idx {galaxy_idx} across snapshots...')
            self.propr['pos'] = _unwrap_positions(self.propr['pos'], boxsize)

        if verbose:
            n_snap = len(redshiftl)
            n_prop = sum(len(v) for v in propr_dicts.values())
            print(f'  Done: {n_snap} snapshots, {n_prop} properties')

        return redshift, propr_out

    # ── Interpolation ────────────────────────────────────────────────

    def propr_from_z(self, propr, z, indx=None, interpstyle='linear'):
        """Interpolate a stored property at an arbitrary redshift.

        Parameters
        ----------
        propr : str
            Property name (must exist in :attr:`propr`).
        z : float or array-like
            Target redshift(s).
        indx : int or array-like, optional
            Galaxy index/indices within the tracked sample.
        interpstyle : str
            ``scipy.interpolate.interp1d`` kind (default ``'linear'``).
        """
        if not self.z or propr not in self.propr:
            raise RuntimeError(
                "No history loaded. Call get_property_history() first."
            )
        zs = self.z['Redshift']
        vals = self.propr[propr]
        if indx is not None and vals.ndim > 1:
            vals = vals[:, indx]
        interp_func = interp1d(zs, vals, kind=interpstyle, axis=0)
        return interp_func(z)

    # ── Plotting ─────────────────────────────────────────────────────

    def plot_history(self, zlist, cosmo=None, propr=None, denom=None,
                     indx=0, outname='test.png', interpolate=None):
        """Quick diagnostic plot of a stored property history.

        Parameters
        ----------
        zlist : array-like
            Redshifts to mark on the top axis.
        cosmo : astropy cosmology, optional
            Cosmology used for age calculation.  Defaults to Planck15.
        propr : str, optional
            Property to plot.  Defaults to the first loaded.
        denom : str, optional
            If set, plot ``propr / denom`` (e.g. sSFR).
        indx : int
            Galaxy index within the tracked sample.
        outname : str
            Output file name.
        interpolate : bool, optional
            If set, draw a smooth interpolated line.
        """
        if cosmo is None:
            cosmo = _default_cosmo
        x = cosmo.age(self.z['Redshift']).value
        key = propr if propr is not None else next(iter(self.propr))
        y = self.propr[key]
        if y.ndim > 1:
            y = y[:, indx]
        if denom is not None:
            d = self.propr[denom]
            if d.ndim > 1:
                d = d[:, indx]
            y = y / d
        h = HistoryPlots(x, y, 1, 1, figsize=(15, 10))
        h.z_on_top(zlist, cosmo)
        if interpolate is not None:
            h.interpolate_plot(num_points=100, kind='linear')
        else:
            h.plot()
        h.save(outname=outname, sim_name=self.sb.name)

    # ── Save ─────────────────────────────────────────────────────────

    def save_history_to_hdf5(self, filename):
        """Save the loaded redshift and property histories to an HDF5 file in output/<sim_name>/caesar_sfh/.

        Parameters
        ----------
        filename : str
            Output HDF5 file name (just the file name, not a path).
        """
        import os
        if not self.z or not self.propr:
            raise RuntimeError(
                "No history loaded. Run get_property_history() first."
            )
        # Ensure output/<sim_name>/caesar_sfh directory exists
        out_dir = os.path.join(os.getcwd(), 'output', self.sb.name, 'caesar_sfh')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        with h5py.File(out_path, 'w') as f:
            for key, arr in self.z.items():
                f.create_dataset(f"redshift/{key}", data=arr)
            for key, arr in self.propr.items():
                f.create_dataset(f"properties/{key}", data=arr)
            if self.galaxy_ids is not None:
                f.create_dataset("metadata/galaxy_ids", data=self.galaxy_ids)
            if 'Snapshot' in self.z:
                f.create_dataset("metadata/snapshots", data=np.asarray(self.z['Snapshot'], dtype=np.int32))
        return out_path


def find_property_threshold_crossings_from_hdf5(
    hdf5_path,
    property_name,
    threshold,
    fits_output=None,
    interp_factor=20,
    spline_order=3,
    cosmo=None,
):
    """Find threshold crossings for a stored property history and save to FITS.

    Reads an HDF5 history file produced by :meth:`HDF5BuildHistory.save_history_to_hdf5`,
    interpolates each galaxy property curve with a B-spline, finds all threshold
    crossings, and writes one FITS table containing the crossing points.

    Parameters
    ----------
    hdf5_path : str
        Path to history HDF5 file.
    property_name : str
        Property dataset name under ``properties/``.
    threshold : float
        Threshold value used for crossing detection.
    fits_output : str, optional
        Output FITS path. Defaults to
        ``output/<sim_name>/threshold_crossings/property_threshold_crossings.fits``
        inferred from *hdf5_path*.
    interp_factor : int
        Number of dense interpolation samples per original sample.
    spline_order : int
        B-spline order (1 to 5). Reduced automatically when data are sparse.
    cosmo : astropy cosmology, optional
        Cosmology used to convert redshift to cosmic time. Defaults to Planck15.

    Returns
    -------
    tuple
        ``(fits_output_path, n_crossings)``.
    """
    if cosmo is None:
        cosmo = _default_cosmo

    if fits_output is None:
        history_dir = os.path.dirname(os.path.abspath(hdf5_path))
        sim_root = os.path.dirname(history_dir)
        fits_output = os.path.join(
            sim_root,
            'threshold_crossings',
            'property_threshold_crossings.fits',
        )

    with h5py.File(hdf5_path, 'r') as f:
        prop_path = f'properties/{property_name}'
        if prop_path not in f:
            raise KeyError(f'Dataset not found: {prop_path}')
        if 'redshift/Redshift' not in f:
            raise KeyError('Dataset not found: redshift/Redshift')

        prop = np.asarray(f[prop_path])
        redshift = np.asarray(f['redshift/Redshift'])
        if 'metadata/galaxy_ids' in f:
            galaxy_ids = np.asarray(f['metadata/galaxy_ids'])
        else:
            n_gal = prop.shape[1] if prop.ndim > 1 else 1
            galaxy_ids = np.arange(n_gal, dtype=np.int64)

    if prop.ndim == 1:
        prop = prop[:, None]
    if prop.shape[0] != redshift.shape[0]:
        raise ValueError(
            f'Mismatched snapshot dimension: property has {prop.shape[0]} rows, '
            f'redshift has {redshift.shape[0]} entries.'
        )

    cosmic_time_yr = cosmo.age(redshift).value * 1e9

    out_galaxy_id = []
    out_crossing_id = []
    out_time = []
    out_value = []

    for gal_idx in range(prop.shape[1]):
        y = np.asarray(prop[:, gal_idx], dtype=float)
        valid = np.isfinite(cosmic_time_yr) & np.isfinite(y)
        if np.sum(valid) < 2:
            continue

        t = cosmic_time_yr[valid]
        y = y[valid]

        sort_idx = np.argsort(t)
        t = t[sort_idx]
        y = y[sort_idx]

        t_unique, unique_idx = np.unique(t, return_index=True)
        y_unique = y[unique_idx]
        if len(t_unique) < 2:
            continue

        k = int(np.clip(spline_order, 1, 5))
        k = min(k, len(t_unique) - 1)
        if k < 1:
            continue

        tck = splrep(t_unique, y_unique, s=0, k=k)
        n_dense = max(len(t_unique) * int(max(interp_factor, 2)), len(t_unique) + 1)
        t_dense = np.linspace(t_unique[0], t_unique[-1], n_dense)
        y_dense = np.asarray(splev(t_dense, tck), dtype=float)

        d = y_dense - threshold
        crossing_count = 0

        for i in range(len(t_dense) - 1):
            d1, d2 = d[i], d[i + 1]
            t1, t2 = t_dense[i], t_dense[i + 1]
            y1, y2 = y_dense[i], y_dense[i + 1]

            has_crossing = ((d1 < 0 and d2 > 0) or (d1 > 0 and d2 < 0))
            if has_crossing:
                frac = -d1 / (d2 - d1)
                t_cross = t1 + frac * (t2 - t1)
                y_cross = y1 + frac * (y2 - y1)
            elif d1 == 0:
                t_cross = t1
                y_cross = y1
            else:
                continue

            if out_time and abs(t_cross - out_time[-1]) < 1e-6 and out_galaxy_id[-1] == galaxy_ids[gal_idx]:
                continue

            crossing_count += 1
            out_galaxy_id.append(int(galaxy_ids[gal_idx]))
            out_crossing_id.append(crossing_count)
            out_time.append(float(t_cross))
            out_value.append(float(y_cross))

    out_dir = os.path.dirname(fits_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if len(out_time) == 0:
        cols = fits.ColDefs([
            fits.Column(name='GALAXY_ID', array=np.array([], dtype=np.int64), format='K'),
            fits.Column(name='CROSS_ID', array=np.array([], dtype=np.int32), format='J'),
            fits.Column(name='TIME_YR', array=np.array([], dtype=np.float64), format='D'),
            fits.Column(name='PROPERTY_VALUE', array=np.array([], dtype=np.float64), format='D'),
        ])
    else:
        cols = fits.ColDefs([
            fits.Column(name='GALAXY_ID', array=np.asarray(out_galaxy_id, dtype=np.int64), format='K'),
            fits.Column(name='CROSS_ID', array=np.asarray(out_crossing_id, dtype=np.int32), format='J'),
            fits.Column(name='TIME_YR', array=np.asarray(out_time, dtype=np.float64), format='D'),
            fits.Column(name='PROPERTY_VALUE', array=np.asarray(out_value, dtype=np.float64), format='D'),
        ])

    events_hdu = fits.BinTableHDU.from_columns(cols, name='CROSSINGS')
    events_hdu.header['PROPKEY'] = property_name
    events_hdu.header['THRESH'] = float(threshold)
    hdul = fits.HDUList([fits.PrimaryHDU(), events_hdu])
    hdul.writeto(fits_output, overwrite=True)

    return fits_output, len(out_time)
