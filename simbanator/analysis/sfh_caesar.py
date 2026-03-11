"""Build galaxy property histories along merger trees using Caesar catalogs.

Provides three classes:

- :class:`CaesarBuildHistory` — reads histories via the Caesar Python API
  (slow; instantiates full galaxy/halo objects).
- :class:`BuildHistory` — reads histories from pre-converted FITS catalogs.
- :class:`HDF5BuildHistory` — reads histories directly from Caesar HDF5
  files via ``h5py`` (fastest; no Caesar import, no conversion step).
"""

import os
import gc
import numpy as np
import h5py
from astropy.io import fits
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as _default_cosmo

from ..visualization.plots import HistoryPlots


# -----------------------------------------------------------------------
# Caesar-based history (reads Caesar .hdf5 catalogs on the fly)
# -----------------------------------------------------------------------

class CaesarBuildHistory:
    """Track galaxy properties across snapshots via Caesar catalogs.

    Parameters
    ----------
    sb : :class:`~simbanator.io.simba.Simulation`
        Simulation handle (used for paths, redshifts, catalog loading).
    progfilename : str
        Progenitor index FITS file name.
    progen_dir : str, optional
        Directory containing the progenitor file.  Defaults to
        ``./output/<sim>/progenitors/``.
    """

    def __init__(self, sb, progfilename='progenitors_most_mass.fits',
                 progen_dir=None):
        if progen_dir is None:
            progen_dir = os.path.join(os.getcwd(), 'output', 'progenitors')
        self.progen_file = os.path.join(progen_dir, progfilename)
        self.history_indx = None
        self.sb = sb
        self.z = {}
        self.propr = {}

    def get_history_indx(self, id, start_snap, end_snap):
        """Retrieve progenitor indices for *id* between two snapshots.

        Parameters
        ----------
        id : int
            Galaxy GroupID at the starting snapshot.
        start_snap, end_snap : int or str
            Snapshot range (column names in the FITS table).

        Returns
        -------
        dict
            ``{snap_str: index}`` mapping.
        """
        with fits.open(self.progen_file) as hdul:
            data = hdul[1].data
            id_column = data['GroupID']
            col_names = hdul[1].columns.names

            matches = np.where(np.asarray(id_column) == id)[0]
            if len(matches) == 0:
                raise ValueError(f"ID {id} not found in the progen.fits file.")
            row_index = int(matches[0])

            try:
                start_col_index = col_names.index(str(start_snap))
                end_col_index = col_names.index(str(end_snap))
            except ValueError as e:
                raise ValueError(f"Snapshot name not found: {e}")

            if start_col_index > end_col_index:
                raise ValueError("start_snap should be less than or equal to end_snap")

            self.history_indx = {}
            for col_name in col_names[start_col_index:end_col_index + 1]:
                self.history_indx[col_name] = data[row_index][col_name]

            return self.history_indx

    def get_property_history(self, propr_dicts):
        """Evaluate galaxy/halo properties at each snapshot in the history.

        Parameters
        ----------
        propr_dicts : dict
            ``{'galaxies': ['masses.stellar', ...], 'halos': [...]}``

        Returns
        -------
        tuple
            ``(redshift_dict, property_dict)`` where each value maps
            property names to numpy arrays.
        """
        redshiftl = []
        propr_out = {propr: [] for key in propr_dicts for propr in propr_dicts[key]}
        indx_dict = self.history_indx

        for snap in indx_dict:
            cs = self.sb.get_caesar(int(snap))
            for mainattr, properties in propr_dicts.items():
                if mainattr == 'galaxies':
                    _o = cs.galaxies
                elif mainattr == 'halos':
                    _o = cs.halos
                else:
                    raise KeyError('propr_dicts keys must be "galaxies" or "halos"')

                for propr in properties:
                    if '.' in propr:
                        attr, sub_attr = propr.split('.')
                        values = np.asarray([getattr(i, attr)[sub_attr] for i in _o])[indx_dict[snap]]
                    else:
                        values = np.asarray([getattr(i, propr) for i in _o])[indx_dict[snap]]
                    propr_out[propr].append(values)

            redshiftl.append(cs.simulation.redshift)
            del cs
            gc.collect()

        for propr in propr_out:
            propr_out[propr] = np.asarray(propr_out[propr])

        redshift = {'Redshift': np.asarray(redshiftl)}
        self.z = redshift
        self.propr = propr_out
        return redshift, propr_out

    def _get_snap_data(self, snap, family, propr, indx):
        """Return property values for specific objects at a given snapshot."""
        cs = self.sb.get_caesar(snap)
        if family == 'galaxies':
            _o = cs.galaxies
        elif family == 'halos':
            _o = cs.halos
        else:
            raise KeyError('family must be "galaxies" or "halos"')
        return np.asarray([getattr(i, propr) for i in _o])[indx]

    def propr_from_z(self, propr, z, interpstyle='linear'):
        """Interpolate a stored property at an arbitrary redshift.

        Uses all loaded snapshots for a smooth fit.  Call
        :meth:`get_property_history` before using this method.

        Parameters
        ----------
        propr : str
            Property name (must be present in :attr:`propr`).
        z : float or array-like
            Target redshift(s).
        interpstyle : str
            ``scipy.interpolate.interp1d`` *kind* parameter (default
            ``'linear'``).

        Returns
        -------
        float or ndarray
            Interpolated property value(s).
        """
        if not self.z or propr not in self.propr:
            raise RuntimeError(
                "No history loaded. Call get_property_history() first."
            )
        zs = self.z['Redshift']
        vals = self.propr[propr]
        interp_func = interp1d(zs, vals, kind=interpstyle)
        return interp_func(z)

    def plot_history(self, zlist, cosmo=None, propr=None, outname='test.png',
                     interpolate=None):
        """Quick diagnostic plot of the stored history.

        Parameters
        ----------
        zlist : array-like
            Redshifts to mark on the top axis.
        cosmo : astropy cosmology, optional
            Cosmology used for age calculation.  Defaults to Planck15.
        propr : str, optional
            Property key from :attr:`propr` to plot.  Defaults to the first
            loaded property.
        outname : str
            Output file name.
        interpolate : bool, optional
            If set, draw a smooth interpolated line instead of raw data points.
        """
        if cosmo is None:
            cosmo = _default_cosmo
        x = self.z['Redshift']
        key = propr if propr is not None else next(iter(self.propr))
        y = self.propr[key]
        h = HistoryPlots(x, y, 1, 2)
        h.z_on_top(zlist, cosmo)
        if interpolate is not None:
            h.interpolate_plot(num_points=100, kind='linear')
        else:
            h.plot()
        h.save(outname=outname)


# -----------------------------------------------------------------------
# FITS-based history (reads from pre-converted FITS catalogs)
# -----------------------------------------------------------------------

class BuildHistory:
    """Track galaxy properties from pre-converted FITS catalogs.

    Parameters
    ----------
    sb : :class:`~simbanator.io.simba.Simba`
        Simba path manager.
    fitsdir : str
        Directory containing per-snapshot FITS files.
    progfilename : str
        Progenitor index FITS file name.
    """

    def __init__(self, sb, fitsdir, progfilename='progenitors_most_mass.fits',
                 progen_dir=None):
        if progen_dir is None:
            progen_dir = os.path.join(os.getcwd(), 'output', 'progenitors')
        self.progen_file = os.path.join(progen_dir, progfilename)
        self.history_indx = None
        self.sb = sb
        self.fitsdir = fitsdir
        self.z = {}
        self.propr = {}

    def get_fits(self, snap):
        """Return the FITS file path for a given snapshot."""
        filename = self.sb.get_caesar_file(snap)
        basename = os.path.splitext(os.path.basename(filename))[0] + '.fits'
        return os.path.join(self.fitsdir, basename)

    def get_history_indx(self, id, start_snap, end_snap):
        """Retrieve progenitor indices for multiple IDs between two snapshots.

        Parameters
        ----------
        id : array-like
            Galaxy GroupIDs.
        start_snap, end_snap : int or str
            Snapshot range.

        Returns
        -------
        dict
            ``{snap_str: indices_array}`` mapping.
        """
        with fits.open(self.progen_file) as hdul:
            data = hdul[1].data
            id_column = data['GroupID']
            col_names = hdul[1].columns.names

            id_lookup = {int(gid): idx for idx, gid in enumerate(np.asarray(id_column))}
            missing = [i for i in id if i not in id_lookup]
            if missing:
                raise ValueError(f"GroupIDs not found in progen file: {missing}")
            row_index = np.array([id_lookup[i] for i in id])

            try:
                start_col_index = col_names.index(str(start_snap))
                end_col_index = col_names.index(str(end_snap))
            except ValueError as e:
                raise ValueError(f"Snapshot name not found: {e}")

            if start_col_index > end_col_index:
                raise ValueError("start_snap should be less than or equal to end_snap")

            self.history_indx = {}
            for col_name in col_names[start_col_index:end_col_index + 1]:
                self.history_indx[col_name] = data[col_name][row_index]

            return self.history_indx

    def get_property_history(self, propr_dicts):
        """Evaluate properties at each snapshot from FITS files.

        Parameters
        ----------
        propr_dicts : list of str
            Column names to extract from the FITS files.

        Returns
        -------
        dict
            ``{property_name: 2D np.ndarray}`` where rows are snapshots.
        """
        propr_out = {key: [] for key in propr_dicts}
        indx_dict = self.history_indx
        redshiftl = []

        print(f'Number of snapshots: {len(indx_dict)}')

        for snap in indx_dict:
            fitsname = self.get_fits(int(snap))
            print(f'Opening {fitsname}')
            try:
                with fits.open(fitsname) as file:
                    f = file[1].data
                    redshiftl.append(self.sb.get_z_from_snap(snap))
                    for prop in propr_dicts:
                        if prop in f.columns.names:
                            propr_out[prop].append(np.asarray(f[prop])[indx_dict[snap]])
                        else:
                            print(f'Warning: Property {prop} not found in {fitsname}')
            except Exception as e:
                print(f'Error processing snapshot {snap}: {e}')

        for prop in propr_dicts:
            propr_out[prop] = np.array(propr_out[prop])

        redshift = {'Redshift': np.asarray(redshiftl)}
        self.z = redshift
        self.propr = propr_out
        return redshift, propr_out

    def _get_snap_data(self, snap, propr, indx):
        """Return property values from a FITS file for a given snapshot."""
        fitsname = self.get_fits(int(snap))
        with fits.open(fitsname) as f:
            return f[1].data[propr][indx]

    def propr_from_z(self, propr, z, indx=None, interpstyle='linear'):
        """Interpolate a stored property at an arbitrary redshift.

        Uses all loaded snapshots for a smooth fit.  Call
        :meth:`get_property_history` before using this method.

        Parameters
        ----------
        propr : str
            Property column name.
        z : float or array-like
            Target redshift(s).
        indx : int or array-like, optional
            Galaxy column index/indices.  If ``None``, returns values for
            all tracked galaxies.
        interpstyle : str
            ``scipy.interpolate.interp1d`` *kind* parameter (default
            ``'linear'``).
        """
        if not self.z or propr not in self.propr:
            raise RuntimeError(
                "No history loaded. Call get_property_history() first."
            )
        zs = self.z['Redshift']
        vals = self.propr[propr]          # shape (n_snaps, n_galaxies)
        if indx is not None:
            vals = vals[:, indx]
        interp_func = interp1d(zs, vals, kind=interpstyle, axis=0)
        return interp_func(z)

    def plot_history(self, zlist, cosmo=None, propr='sfr', denom=None,
                     indx=0, outname='test.png', interpolate=None):
        """Quick diagnostic plot of a stored property history.

        Parameters
        ----------
        zlist : array-like
            Redshifts to mark on the top axis.
        cosmo : astropy cosmology, optional
            Cosmology used for age calculation.  Defaults to Planck15.
        propr : str
            Property column to plot (default ``'sfr'``).
        denom : str, optional
            If set, plot ``propr / denom`` (e.g. ``denom='stellar_masses'``
            gives sSFR).
        indx : int
            Galaxy index within the tracked sample.
        outname : str
            Output file name.
        interpolate : bool, optional
            If set, draw a smooth interpolated line instead of raw points.
        """
        if cosmo is None:
            cosmo = _default_cosmo
        x = cosmo.age(self.z['Redshift']).value
        y = self.propr[propr][:, indx]
        if denom is not None:
            y = y / self.propr[denom][:, indx]
        h = HistoryPlots(x, y, 1, 1, figsize=(15, 10))
        h.z_on_top(zlist, cosmo)
        if interpolate is not None:
            h.interpolate_plot(num_points=100, kind='linear')
        else:
            h.plot()
        h.save(outname=outname)


# -----------------------------------------------------------------------
# HDF5-native history (reads Caesar .hdf5 directly via h5py — fastest)
# -----------------------------------------------------------------------

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
    # Already a full path → use as is
    if '/' in propr:
        return propr
    # Dotted shorthand → look under dicts
    if '.' in propr:
        return f'{family}/dicts/{propr}'
    # Plain name → direct child
    return f'{family}/{propr}'


def _read_h5_property(h5file, h5path, indices):
    """Read an HDF5 dataset for specific row indices only.

    If the dataset is 2-D (e.g. positions, velocities), returns the full
    sub-matrix ``(len(indices), ncols)``.
    """
    ds = h5file[h5path]
    # h5py fancy-indexing requires sorted indices
    sort_order = np.argsort(indices)
    sorted_idx = indices[sort_order]
    data = ds[sorted_idx]
    # Restore original order
    inv = np.empty_like(sort_order)
    inv[sort_order] = np.arange(len(sort_order))
    return data[inv]


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

    def __init__(self, sb, progfilename='progenitors_most_mass.fits',
                 progen_dir=None):
        if progen_dir is None:
            progen_dir = os.path.join(os.getcwd(), 'output', 'progenitors')
        self.progen_file = os.path.join(progen_dir, progfilename)
        self.history_indx = None
        self.sb = sb
        self.z = {}
        self.propr = {}

    # ── Progenitor index read ────────────────────────────────────────

    def get_history_indx(self, ids, start_snap, end_snap):
        """Retrieve progenitor indices for *ids* between two snapshots.

        Parameters
        ----------
        ids : int or array-like
            Galaxy GroupID(s) at the starting snapshot.
        start_snap, end_snap : int
            Snapshot range (inclusive).

        Returns
        -------
        dict
            ``{snap_str: index_or_indices}`` mapping.
        """
        ids = np.atleast_1d(ids)
        with fits.open(self.progen_file) as hdul:
            data = hdul[1].data
            id_column = np.asarray(data['GroupID'])
            col_names = hdul[1].columns.names

            id_lookup = {int(gid): idx for idx, gid in enumerate(id_column)}
            missing = [int(i) for i in ids if int(i) not in id_lookup]
            if missing:
                raise ValueError(
                    f"GroupIDs not found in progen file: {missing}"
                )
            row_index = np.array([id_lookup[int(i)] for i in ids])

            try:
                start_col_index = col_names.index(str(int(start_snap)))
                end_col_index = col_names.index(str(int(end_snap)))
            except ValueError as e:
                raise ValueError(f"Snapshot name not found: {e}")
            if start_col_index > end_col_index:
                raise ValueError(
                    "start_snap should be less than or equal to end_snap"
                )

            self.history_indx = {}
            for col_name in col_names[start_col_index:end_col_index + 1]:
                vals = data[col_name][row_index]
                self.history_indx[col_name] = (
                    int(vals[0]) if len(ids) == 1 else vals
                )

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
        # Flatten all requested properties into one output dict
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

            galaxy_idx = indx_dict[snap_str]
            scalar_index = np.ndim(galaxy_idx) == 0
            galaxy_idx = np.atleast_1d(galaxy_idx)

            with h5py.File(h5path, 'r') as f:
                # Redshift from the HDF5 header
                if ('simulation_attributes' in f
                        and 'redshift' in f['simulation_attributes'].attrs):
                    z_val = float(
                        f['simulation_attributes'].attrs['redshift']
                    )
                else:
                    z_val = self.sb.get_z_from_snap(snap)
                redshiftl.append(z_val)

                for family, properties in propr_dicts.items():
                    # For halo_data we need the galaxy→halo mapping
                    if family == 'halo_data':
                        parent_path = 'galaxy_data/parent_halo_index'
                        if parent_path not in f:
                            raise KeyError(
                                f'{parent_path} not found in {h5path}'
                            )
                        parent_halo = f[parent_path][:]
                        halo_idx = parent_halo[galaxy_idx]
                    else:
                        halo_idx = None

                    for propr in properties:
                        ds_path = _resolve_h5_path(family, propr)
                        if ds_path not in f:
                            raise KeyError(
                                f'Dataset {ds_path} not found in '
                                f'{h5path}.'
                            )

                        idx = halo_idx if family == 'halo_data' else galaxy_idx
                        values = _read_h5_property(f, ds_path, idx)

                        if scalar_index:
                            values = values[0]
                        propr_out[propr].append(values)

        # Convert lists → arrays
        for propr in propr_out:
            propr_out[propr] = np.asarray(propr_out[propr])

        redshift = {'Redshift': np.asarray(redshiftl)}
        self.z = redshift
        self.propr = propr_out

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
        h.save(outname=outname)