import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from ..visualize.simple_plots import HistoryPlots
from .. import SavePaths
import gc

class caesarBuildHistory:
    def __init__(self, sb, progfilename='progenitors_most_mass.fits'):
        """
        Initialize with the directory containing the progen.fits files.
        
        :param progen_directory: Path to the directory containing progen.fits files.
        """
        save_paths = SavePaths()
        self.progen_file = os.path.join(save_paths.get_filetype_path('fits'), 'progenitors_files', progfilename)
        self.history_indx = None
        self.sb = sb
        self.z = {}
        self.propr = {}

    def get_history_indx(self, id, start_snap, end_snap):
        """
        Retrieve the history for the given id between start_snap and end_snap.
        
        :param id: The ID for which to retrieve the history.
        :param start_snap: The starting snapshot index (as a string).
        :param end_snap: The ending snapshot index (as a string).
        
        :return: List containing the data for the specified id from start_snap to end_snap.
        """
        with fits.open(self.progen_file) as hdul:
            data = hdul[1].data
            id_column = data['GroupID']  # assuming the first column contains the IDs
            col_names = hdul[1].columns.names  # list of column names

            # Find the row corresponding to the given id
            row_index = None
            for index, value in enumerate(id_column):
                if value == id:
                    row_index = index
                    break
            if row_index is None:
                raise ValueError(f"ID {id} not found in the progen.fits file.")
            
            # Extract the columns from start_snap to end_snap
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
        """Gives the value of a certain property for each snapshot in the range indicated in get_history_indx.
    
           Attr:
              propr_dicts (dictionary): contains the list of properties to evaluate. Keys are the family of properties, either galaxies or halos.
              
           Return:
              dictionary with properties as keys and corresponding array of the property at each snapshot in values
        """
        redshiftl = []
        propr_out = {propr: [] for key in propr_dicts.keys() for propr in propr_dicts[key]}
        indx_dict = self.history_indx
        
        for snap in indx_dict.keys():
            cs = self.sb.get_caesar(int(snap))
            for mainattr, properties in propr_dicts.items():
                if mainattr == 'galaxies':
                    _o = cs.galaxies
                elif mainattr == 'halos':
                    _o = cs.halos
                else:
                    raise KeyError('Wrong attributes in propr_dicts dictionary keys: only galaxies and halos permitted')
                
                for propr in properties:
                    if '.' in propr:
                        attr, sub_attr = propr.split('.')
                        values = np.asarray([getattr(i, attr)[sub_attr] for i in _o])[indx_dict[snap]]
                    else:
                        values = np.asarray([getattr(i, propr) for i in _o])[indx_dict[snap]]
                    
                    propr_out[propr].append(values)
            
            redshiftl.append(cs.simulation.redshift)
            # remove caesar file to not overload memory
            del cs
            gc.collect()
        
        # Convert lists to numpy arrays
        for propr in propr_out.keys():
            propr_out[propr] = np.asarray(propr_out[propr])
        redshift = {'Redshift':np.asarray(redshiftl)}
        self.z = redshift
        self.propr = propr_out
        return redshift, propr_out

    def _get_snap_data(self, snap, family, propr, indx):
        """gives the values of a property given snapshot and galaxy index

           Attr:
               snap (int): snapshot number
               family (str): either galaxies or halos
               propr (str): property to evaluate
               indx (array-like): the IDs of the galaxies
           Return:
               array of properties. each entry is a property at a given index
        """
        cs = self.sb.get_caesar(snap)
        if family == 'galaxies': _o = cs.galaxies
        elif family == 'halos' : _o = cs.halos
        else: print('KeyError: Wrong attributes in propr_dicts dictionary keys: only galaxies and halos permitted')
        propr = np.asarray([getattr(_o, propr)])[indx]
        
        return propr
        
    
    def propr_from_z(self, family, propr, z, indx, interpstyle='BSpline'):
        """gives the value of a property at any redshift
        
           Attr:
               family (str): either galaxies or halos
               propr (str): property to evaluate
               z (float): the redshift to evaluate
               indx (array-like): the IDs of the galaxies
               interpstyle (str): the type of interpolation as stated in scipy.interpolate
           Return:
               the interpolated value
        
        """
        nearest_snap = caesar.progen.z_to_snap(z, snaplist_file='Simba')[0]        
        
        # Retrieve the snapshots around the nearest_snap
        previous_snap = nearest_snap - 1
        next_snap = nearest_snap + 1
        
        # Load property values for the previous and next snapshots
        prev_data = self._get_snapshot_data(previous_snap, family, propr, indx)
        next_data = self._get_snapshot_data(next_snap, family, propr, indx)
        
        # Define the interpolation function
        interp_func = interp1d([prev_z, next_z], [prev_data, next_data], kind=interpstyle)
        
        # Interpolate the property value at redshift z
        interpolated_value = interp_func(z)
        
        return interpolated_value

    def plot_history(self, zlist, cosmo, interpolate=None):
        x = self.z
        y = self.propr
        x = [x[i] for i in x.keys()][0]
        y = [y[i] for i in y.keys()][0]
        h = HistoryPlots(x, y, 1, 2)
        h.z_on_top(zlist, cosmo)
        if interpolate!=None:
            h.interpolate_plot(num_points=100, kind='linear')
        else:
            h.plot()
        h.save(outname='test.png')









import os
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits

class BuildHistory:
    def __init__(self, sb, fitsdir, progfilename='progenitors_most_mass.fits'):
        """
        Initialize with the directory containing the progen.fits files.
        
        :param progen_directory: Path to the directory containing progen.fits files.
        """
        save_paths = SavePaths()
        self.progen_file = os.path.join(save_paths.get_filetype_path('fits'), 'progenitors_files', progfilename)
        self.history_indx = None
        self.sb = sb
        self.fitsdir = fitsdir
        self.z = {}
        self.propr = {}

    def get_fits(self, snap):
        filename = self.sb.get_caesar_file(snap)
        # Define the new filename with the .fits extension
        new_filename = filename.split('/')[-1].split('.')[0] + '.fits'
        # Construct the full path for the new FITS file
        new_filename = os.path.join(self.fitsdir, new_filename)
        return new_filename
        

    def get_history_indx(self, id, start_snap, end_snap):
        """
        Retrieve the history for the given id between start_snap and end_snap.
        
        :param id: The ID for which to retrieve the history.
        :param start_snap: The starting snapshot index (as a string).
        :param end_snap: The ending snapshot index (as a string).
        
        :return: List containing the data for the specified id from start_snap to end_snap.
        """
        with fits.open(self.progen_file) as hdul:
            data = hdul[1].data
            id_column = data['GroupID']  # assuming the first column contains the IDs
            col_names = hdul[1].columns.names  # list of column names
            # Find the row corresponding to the given id
            

            row_index = []
            for i in id:
                row_index.append(np.where(id_column == i)[0][0])
            # Extract the columns from start_snap to end_snap
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
        """Gives the value of a certain property for each snapshot in the range indicated in get_history_indx.
    
           Attr:
              propr_dicts (dictionary): contains the list of properties to evaluate. Keys are the family of properties, either galaxies or halos.
              
           Return:
              dictionary with properties as keys and corresponding array of the property at each snapshot in values
        """
        # Initialize the dictionary to hold lists of lists for each property
        propr_out = {key: [] for key in propr_dicts}
        indx_dict = self.history_indx
        redshiftl = []
        
        print(f'Number of snapshots: {len(indx_dict.keys())}')
        
        # Initialize temporary storage to collect rows for each property
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
                            
                            if len(temp_storage[prop]) == 0:
                                # Initialize the list of lists if it's the first snapshot
                                temp_storage[prop] = [list(selected_values)]
                            else:
                                # Append the new row to the existing list of lists
                                temp_storage[prop].append(list(selected_values))
                        else:
                            print(f'Warning: Property {prop} not found in FITS file {fitsname}')
            
            except Exception as e:
                print(f'Error processing snapshot {snap}: {e}')
        
        # Convert lists of lists to 2D numpy arrays
        for prop in propr_dicts:
            propr_out[prop] = np.array(temp_storage[prop])
            
        redshift = {'Redshift':np.asarray(redshiftl)}
        self.z = redshift
        self.propr = propr_out
        return propr_out

    def _get_snap_data(self, snap, propr, indx):
        """gives the values of a property given snapshot and galaxy index

           Attr:
               snap (int): snapshot number
               family (str): either galaxies or halos
               propr (str): property to evaluate
               indx (array-like): the IDs of the galaxies
           Return:
               array of properties. each entry is a property at a given index
        """
        fitsname = self.get_fits(int(snap))
        with fits.open(fitsname) as f:
            propr = f[propr][indx]
        
        return propr
        
    
    def propr_from_z(self, propr, z, indx, interpstyle='BSpline'):
        """gives the value of a property at any redshift
        
           Attr:
               family (str): either galaxies or halos
               propr (str): property to evaluate
               z (float): the redshift to evaluate
               indx (array-like): the IDs of the galaxies
               interpstyle (str): the type of interpolation as stated in scipy.interpolate
           Return:
               the interpolated value
        
        """
        nearest_snap = caesar.progen.z_to_snap(z, snaplist_file='Simba')[0]        
        
        # Retrieve the snapshots around the nearest_snap
        previous_snap = nearest_snap - 1
        next_snap = nearest_snap + 1
        
        # Load property values for the previous and next snapshots
        prev_data = self._get_snapshot_data(previous_snap, family, propr, indx)
        next_data = self._get_snapshot_data(next_snap, family, propr, indx)
        
        # Define the interpolation function
        interp_func = interp1d([prev_z, next_z], [prev_data, next_data], kind=interpstyle)
        
        # Interpolate the property value at redshift z
        interpolated_value = interp_func(z)
        
        return interpolated_value

    def plot_history(self, zlist, cosmo, indx=0, interpolate=None):
        x = self.z['Redshift']
        x = cosmo.age(x).value
        y = (self.propr['sfr']/self.propr['stellar_masses'])[:,indx]
        h = HistoryPlots(x, y, 1, 1, figsize=(15, 10))
        h.z_on_top(zlist, cosmo)
        if interpolate!=None:
            h.interpolate_plot(num_points=100, kind='linear')
        else:
            h.plot()
        h.save(outname='test.png')
        
