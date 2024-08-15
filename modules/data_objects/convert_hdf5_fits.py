import os
import numpy as np
import h5py
from astropy.table import Table
from astropy.io import fits
import re
from ..io_paths.savepaths import SavePaths

def _adapt_keys(path):
    '''Takes the keys to HDF5 groups and datasets and returns it as a human-readable
       name. e.g., the key 'galaxy_data/dicts/masses.stellar' becomes 'stellar_masses'.
    
    Args:
        path (str): The original HDF5 dataset path.
    
    Returns:
        str: The human-readable name.
    '''
    components = path.split('/')
    final_component = components[-1]
    if '.' in final_component:
        parts = final_component.split('.')
        final_component = '_'.join(reversed(parts))
    return final_component

def _read_main_tree(path):
    components = path.split('/')
    start_component = components[0]
    return start_component


def _empty_fits(filename, fitspath):
    # Create a PrimaryHDU object which represents the primary HDU (Header Data Unit)
    primary_hdu = fits.PrimaryHDU()
    
    # Create an empty BINTABLE HDU (with no columns)
    coldefs = fits.ColDefs([])  # Empty column definitions
    bintable_hdu = fits.BinTableHDU.from_columns(coldefs)
    
    # Create an HDUList object containing both the primary HDU and the empty BINTABLE HDU
    hdul = fits.HDUList([primary_hdu, bintable_hdu])
    
    # Define the new filename with the .fits extension
    new_filename = filename.split('/')[-1].split('.')[0] + '.fits'
    
    # Construct the full path for the new FITS file
    new_filename = os.path.join(fitspath, new_filename)
    
    # Write the HDUList to a new FITS file
    hdul.writeto(new_filename, overwrite=True)
    
    return new_filename


def _list_all_datasets(file):
    # Open the HDF5 file in read mode
    dataset_paths = []

    # Define a recursive function to iterate through groups and datasets
    def visit_func(name, node):
        if isinstance(node, h5py.Dataset):
            dataset_paths.append(name)

    # Visit all nodes in the file and apply visit_func
    file.visititems(visit_func)

    return dataset_paths



def _process_hdf5(file_path, fitspath, ignore, chunk=10, verbose=0):
    # create the empty fits file
    fits_filename=  _empty_fits(file_path, fitspath)
    print(f'Saved fits in: {fits_filename}')
    # open the hdf5 file
    with h5py.File(file_path) as f:
        hdul =  fits.open(fits_filename, mode='update')
        dataset_paths = _list_all_datasets(f)
        # write the dataset into the fits
        for dataset_path in dataset_paths:
            flag = False
            h5col = f[dataset_path]
            #print(dataset_path, h5col.shape)
            tree = _read_main_tree(dataset_path)
            # create column name
            column_name = _adapt_keys(dataset_path)
            for ign in ignore:
                if ign in dataset_path:
                    flag = True
                    break
            if flag==True:
                continue
            if tree=='halo_data':
                if verbose==1:
                    print('Copying Halo')
                selgal = f['galaxy_data']['parent_halo_index']                  
                h5col = np.asarray(h5col)[selgal]
                column_name = 'halo_'+column_name  
            if tree=='tree_data':
                if verbose==1:
                    print('Copying tree')
                progenid = f['tree_data']['progen_galaxy_star'][:, 0]
                h5col = progenid
                new_col = fits.Column(name=column_name, array=h5col, format='E')
                hdul[1].columns.add_col(new_col)
                column_name = 'progen_galaxy_star'
                break
            #with fits.open(fits_filename, mode='update') as hdul:
            # handle multiple D dataselts
            if len(h5col.shape) > 1:
                for i in range(3):
                    # Create a new column
                    new_col = fits.Column(name=f'{column_name}_{i}', array=h5col[:, i], format='E')
                    hdul[1].columns.add_col(new_col)
            else:       
                # Create a new column
                new_col = fits.Column(name=column_name, array=h5col, format='E')
                hdul[1].columns.add_col(new_col)
        # Save changes
        hdul.flush()
        hdul.close()


def convert_hdf5_fits(sb, snaprange, ignore, verbose=0, overwrite=False):
    # crete the path to save fits
    save_paths = SavePaths()
    fitspath = save_paths.get_filetype_path('fits')
    fitspath = save_paths.create_subdir(fitspath, 'converted_from_hdf5')
    # iterate over snapshots
    for snap in snaprange:
        # write the path to the catalog file
        file_path = sb.get_caesar_file(snap)
        # create the fits file
        print(f'Processing : {file_path}')
        _process_hdf5(file_path, fitspath, ignore, verbose)
        


