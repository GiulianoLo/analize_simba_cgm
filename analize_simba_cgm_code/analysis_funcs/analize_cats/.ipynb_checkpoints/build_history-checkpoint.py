import os
from astropy.io import fits
from analize_simba_cgm_code.io_paths.savepaths import SavePaths
from scipy.interpolate import interp1d

class BuildHistory:
    def __init__(self, simfilename, progfilename, sb):
        """
        Initialize with the directory containing the progen.fits files.
        
        :param progen_directory: Path to the directory containing progen.fits files.
        """
        save_paths = SavePaths()
        self.progen_file = os.path.join(save_paths.get_filetype_path('fits'), 'progenitors_files', filename)
        self.history_indx = None
        self.sb = sb

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
            id_column = data.field(0)  # assuming the first column contains the IDs
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
                start_col_index = col_names.index(start_snap)
                end_col_index = col_names.index(end_snap)
            except ValueError as e:
                raise ValueError(f"Snapshot name not found: {e}")

            if start_col_index > end_col_index:
                raise ValueError("start_snap should be less than or equal to end_snap")

            self.history_indx = {}
            for col_name in col_names[start_col_index:end_col_index + 1]:
                self.history_indx[col_name] = data[row_index][col_name]

            return self.history_indx

    def get_proprty_history(self, propr_dicts):
        """gives the value of a certain property for each snapshot in the range indicated in get_history_indx

           Attr:
              propr_dicts (dictionary): contains the list of properties to evaluate. Keys are the family of properties, either galaxies or halos.
              
           Return:
              dictionary with properties as keys and corresponding array of the property at each snapshot in values
           
        """
        redshifts = []
        propr_out = {}
        indx_dict = self.history_indx
        for snap in indx_dict.keys():
            cs = self.sb.get_caesar(snap)
            for mainattr in propr_dicts.keys():
                if mainattr == 'galaxies': _o = cs.galaxies
                elif mainattr == 'halos' : _o = cs.halos
                else: print('KeyError: Wrong attributes in propr_dicts dictionary keys: only galaxies and halos permitted')
            for propr in propr_dicts[mainattr]:
                propr_out[propr] = np.asarray(getattr(_o, propr)[indx_dict[snap])
            redshifts.append(self.sb.get_redshifts[int(snap)])
            # remove caesar file to not overload memory
            del cs
            gc.collect()
        return redshifts, propr_out

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
