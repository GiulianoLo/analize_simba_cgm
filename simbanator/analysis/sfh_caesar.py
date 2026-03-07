"""Build galaxy property histories along merger trees using Caesar catalogs.

Provides two classes:

- :class:`CaesarBuildHistory` — reads histories directly from Caesar catalogs.
- :class:`BuildHistory` — reads histories from pre-converted FITS catalogs.
"""

import os
import gc
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

from ..io.paths import OutputPaths
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
            out = OutputPaths(sb.name)
            progen_dir = out.progenitors
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

            row_index = None
            for index, value in enumerate(id_column):
                if value == id:
                    row_index = index
                    break
            if row_index is None:
                raise ValueError(f"ID {id} not found in the progen.fits file.")

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

    def propr_from_z(self, family, propr, z, indx, interpstyle='linear'):
        """Interpolate a property value at an arbitrary redshift.

        Parameters
        ----------
        family : str
            ``'galaxies'`` or ``'halos'``.
        propr : str
            Property name (Caesar attribute).
        z : float
            Target redshift.
        indx : array-like
            Object indices.
        interpstyle : str
            ``scipy.interpolate.interp1d`` *kind* parameter.

        Returns
        -------
        float or ndarray
            Interpolated property value(s).
        """
        import caesar as cs_mod
        nearest_snap = cs_mod.progen.z_to_snap(z, snaplist_file='Simba')[0]
        previous_snap = nearest_snap - 1
        next_snap = nearest_snap + 1

        prev_data = self._get_snap_data(previous_snap, family, propr, indx)
        next_data = self._get_snap_data(next_snap, family, propr, indx)

        prev_z = self.sb.get_z_from_snap(previous_snap)
        next_z = self.sb.get_z_from_snap(next_snap)

        interp_func = interp1d([prev_z, next_z], [prev_data, next_data], kind=interpstyle)
        return interp_func(z)

    def plot_history(self, zlist, cosmo, interpolate=None):
        """Quick diagnostic plot of the stored history."""
        x = list(self.z.values())[0]
        y = list(self.propr.values())[0]
        h = HistoryPlots(x, y, 1, 2)
        h.z_on_top(zlist, cosmo)
        if interpolate is not None:
            h.interpolate_plot(num_points=100, kind='linear')
        else:
            h.plot()
        h.save(outname='test.png')


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
            out = OutputPaths(sb.name)
            progen_dir = out.progenitors
        self.progen_file = os.path.join(progen_dir, progfilename)
        self.history_indx = None
        self.sb = sb
        self.fitsdir = fitsdir
        self.z = {}
        self.propr = {}

    def get_fits(self, snap):
        """Return the FITS file path for a given snapshot."""
        filename = self.sb.get_caesar_file(snap)
        new_filename = filename.split('/')[-1].split('.')[0] + '.fits'
        return os.path.join(self.fitsdir, new_filename)

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

            row_index = []
            for i in id:
                row_index.append(np.where(id_column == i)[0][0])

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
        temp_storage = {key: [] for key in propr_dicts}

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
                            values = f[prop]
                            indices = indx_dict[snap]
                            selected_values = values[indices]
                            temp_storage[prop].append(list(selected_values))
                        else:
                            print(f'Warning: Property {prop} not found in {fitsname}')
            except Exception as e:
                print(f'Error processing snapshot {snap}: {e}')

        for prop in propr_dicts:
            propr_out[prop] = np.array(temp_storage[prop])

        redshift = {'Redshift': np.asarray(redshiftl)}
        self.z = redshift
        self.propr = propr_out
        return propr_out

    def _get_snap_data(self, snap, propr, indx):
        """Return property values from a FITS file for a given snapshot."""
        fitsname = self.get_fits(int(snap))
        with fits.open(fitsname) as f:
            return f[1].data[propr][indx]

    def propr_from_z(self, propr, z, indx, interpstyle='linear'):
        """Interpolate a property value at an arbitrary redshift."""
        import caesar as cs_mod
        nearest_snap = cs_mod.progen.z_to_snap(z, snaplist_file='Simba')[0]
        previous_snap = nearest_snap - 1
        next_snap = nearest_snap + 1

        prev_data = self._get_snap_data(previous_snap, propr, indx)
        next_data = self._get_snap_data(next_snap, propr, indx)

        prev_z = self.sb.get_z_from_snap(previous_snap)
        next_z = self.sb.get_z_from_snap(next_snap)

        interp_func = interp1d([prev_z, next_z], [prev_data, next_data], kind=interpstyle)
        return interp_func(z)

    def plot_history(self, zlist, cosmo, indx=0, interpolate=None):
        """Quick diagnostic plot of stored sSFR history."""
        x = self.z['Redshift']
        x = cosmo.age(x).value
        y = (self.propr['sfr'] / self.propr['stellar_masses'])[:, indx]
        h = HistoryPlots(x, y, 1, 1, figsize=(15, 10))
        h.z_on_top(zlist, cosmo)
        if interpolate is not None:
            h.interpolate_plot(num_points=100, kind='linear')
        else:
            h.plot()
        h.save(outname='test.png')
