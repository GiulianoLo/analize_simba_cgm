

import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from ..visualize.simple_plots import HistoryPlots
from .. import SavePaths
import gc

class BuildHistory:
    def __init__(self, sb, fitsdir, progfilename='progenitors_most_mass.fits'):
        save_paths = SavePaths()
        self.progen_file = os.path.join(save_paths.get_filetype_path('fits'), 'progenitors_files', progfilename)
        self.history_indx = None
        self.sb = sb
        self.fitsdir = fitsdir
        self.z = {}
        self.propr = {}

    def get_fits(self, snap):
        filename = self.sb.get_caesar_file(snap)
        new_filename = filename.split('/')[-1].split('.')[0] + '.fits'
        return os.path.join(self.fitsdir, new_filename)

    def get_history_indx(self, id, start_snap, end_snap):
        with fits.open(self.progen_file) as hdul:
            data = hdul[1].data
            id_column = data['GroupID']
            col_names = hdul[1].columns.names
            row_index = [np.where(id_column == i)[0][0] for i in id]

            try:
                start_col_index = col_names.index(str(start_snap))
                end_col_index = col_names.index(str(end_snap))
            except ValueError as e:
                raise ValueError(f"Snapshot name not found: {e}")

            if start_col_index > end_col_index:
                raise ValueError("start_snap should be less than or equal to end_snap")

            self.history_indx = {col_name: data[col_name][row_index] for col_name in col_names[start_col_index:end_col_index + 1]}
            return self.history_indx

    def get_property_history(self, propr_dicts):
        propr_out = {key: [] for key in propr_dicts}
        indx_dict = self.history_indx
        redshiftl = []

        print(f'Number of snapshots: {len(indx_dict.keys())}')

        temp_storage = {key: [] for key in propr_dicts}

        for snap in indx_dict.keys():
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
                            print(f'Warning: Property {prop} not found in FITS file {fitsname}')
            except Exception as e:
                print(f'Error processing snapshot {snap}: {e}')

        for prop in propr_dicts:
            propr_out[prop] = np.array(temp_storage[prop])

        redshift = {'Redshift': np.asarray(redshiftl)}
        self.z = redshift
        self.propr = propr_out

        boxsize = self.sb.get_boxsize() if hasattr(self.sb, 'get_boxsize') else 100.0
        for coord in ['x', 'y', 'z']:
            if coord in self.propr:
                print(f'Unwrapping {coord}-positions')
                self.propr[coord] = self.unwrap_positions(self.propr[coord], boxsize)

        return propr_out

    def unwrap_positions(self, positions, boxsize):
        unwrapped = positions.copy()
        for i in range(1, len(positions)):
            delta = unwrapped[i] - unwrapped[i - 1]
            delta[delta > +0.5 * boxsize] -= boxsize
            delta[delta < -0.5 * boxsize] += boxsize
            unwrapped[i] = unwrapped[i - 1] + delta
        return unwrapped

    def _get_snap_data(self, snap, propr, indx):
        fitsname = self.get_fits(int(snap))
        with fits.open(fitsname) as f:
            propr = f[propr][indx]
        return propr

    def propr_from_z(self, propr, z, indx, interpstyle='BSpline'):
        import caesar
        nearest_snap = caesar.progen.z_to_snap(z, snaplist_file='Simba')[0]        
        previous_snap = nearest_snap - 1
        next_snap = nearest_snap + 1

        prev_data = self._get_snap_data(previous_snap, propr, indx)
        next_data = self._get_snap_data(next_snap, propr, indx)

        prev_z = self.sb.get_z_from_snap(previous_snap)
        next_z = self.sb.get_z_from_snap(next_snap)

        interp_func = interp1d([prev_z, next_z], [prev_data, next_data], kind=interpstyle)
        interpolated_value = interp_func(z)

        return interpolated_value

    def plot_history(self, zlist, cosmo, indx=0, interpolate=None):
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
