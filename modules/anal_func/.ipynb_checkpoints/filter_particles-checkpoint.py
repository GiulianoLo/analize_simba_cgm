"""Reduces the size of particles files for better handling.
   At the moment the filtered hdf5 is not completely recovered from the original and the code is slow.
   As a result I cannot access the reduced files using pygadgetreader (from narayanan github), but yt works. 
   A chencking on the saving part is requirement to avoid that the original saving function enforse some attributes by itself.
"""

import numpy as np
import h5py
import os
import unyt
from unyt import unyt_array, kpc
from ..io_paths.savepaths import SavePaths

def copy_skeleton(src_filename, dst_filename):
    """Copy the structure of the original hdf5 without saving in it any actual data
       This ensures the reduced hdf5 can at least be interpreted by yt
    """
    with h5py.File(src_filename, 'r') as src_file:
        with h5py.File(dst_filename, 'w') as dst_file:
            def copy_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    dst_file.create_group(name)
                elif isinstance(obj, h5py.Dataset):
                    # Create an empty dataset with the same shape and dtype
                    dst_file.create_dataset(name, shape=obj.shape, dtype=obj.dtype)
                # Copy attributes if any
                for key, value in obj.attrs.items():
                    dst_file[name].attrs[key] = value

            src_file.visititems(copy_structure)


def filter_particles_by_obj(cs, simfile, snap, selection, oidx, verbose=0, overwrite=True, ignore_fields=[], keyword=None):
    """Opens the particles file in hdf5 format and saves the selected subset of particles.

       Args:
           cs (caesar load object): the caesar object of the type ds = sb.get_sim_file(snap) where sb is the simba class
           snap (int): the snapshot identifier
           selection (str): either 'galaxy' or 'halo', to select particles based on halo or galaxy indexing from caesar catalog
           oidx (list): the list of ids of galaxies or halos whose particles we want to extract
           verbose (int): to print info of the processing
           overwrite (bool): to overwrite existing filtered files; if False, requires a keyword to attach to the output naming
           ignore_fields (list of str): list of fields to ignore when writing the filtered file from the original
           keyword (str or None): keyword to append to the output file name if overwrite is False
    """
    # Instantiate SavePaths
    paths = SavePaths()

    # Determine the output directory for HDF5 files
    output_dir = paths.get_filetype_path('hdf5')
    output_dir = paths.create_subdir(output_dir, 'filtered_files')

    out_folder = f'{output_dir}/snap_{snap}'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    if selection not in ['galaxy', 'halo']:
        raise ValueError('Invalid object type selection: must be galaxy or halo')

    if not overwrite and keyword is None:
        raise ValueError('Keyword must be provided if overwrite is False')

    with h5py.File(simfile, 'r') as input_file:
        header = input_file['Header']
        for obj in oidx:
            if overwrite:
                output = f'{out_folder}/subset_{obj:06.0f}.h5'
            else:
                output = f'{out_folder}/subset_{obj:06.0f}_{keyword}.h5'
                
            copy_skeleton(simfile, output)
            with h5py.File(output, 'a') as output_file:
                # input_file.copy('Header', output_file)
                for ptype in ['PartType0', 'PartType1', 'PartType4', 'PartType5']:
                    if ptype in input_file:  # check if particle type is present
                        if verbose > 0:
                            print('Processing:', ptype)

                        if selection == 'galaxy':
                            _o = cs.galaxies[obj]
                        if selection == 'halo':
                            _o = cs.halos[obj]

                        if ptype == 'PartType0':
                            plist = _o.glist
                        elif ptype == 'PartType4':
                            plist = _o.slist
                        elif ptype == 'PartType1':
                            plist = _o.glist
                        elif ptype == 'PartType5':
                            plist = _o.bhlist
                        else:
                            if verbose > 0:
                                print('No compatible particle type specified')
                            continue

                        for k in input_file[ptype]:
                            if k in ignore_fields:
                                if verbose > 1:
                                    print(k, 'skipped...')
                                continue

                            if verbose > 0:
                                print(ptype, k)

                            # Extract and filter the dataset
                            temp_dset = input_file[ptype][k][:]
                            if verbose > 1:
                                print(temp_dset.shape)

                            filtered_dset = temp_dset[plist]
                                
                            
                            del output_file[ptype][k]
                            output_file[ptype][k] = filtered_dset

                        # Update header information for the number of particles
                        temp = output_file['Header'].attrs['NumPart_ThisFile']
                        temp[int(ptype[8:])] = len(plist)
                        output_file['Header'].attrs['NumPart_ThisFile'] = temp

                        temp = output_file['Header'].attrs['NumPart_Total']
                        temp[int(ptype[8:])] = len(plist)
                        output_file['Header'].attrs['NumPart_Total'] = temp

                        if verbose > 0:
                            print("Updated Header:", dict(output_file['Header'].attrs))

        print('Finished with particle filters')


def apply_boundary_conditions(pos, box_size):
    return (pos + box_size / 2) % box_size - box_size / 2

def filter_by_aperture(cs, simfile, snap, center, radius, selection=None, verbose=0, overwrite=True, ignore_fields=[], keyword=None):
    """Filters particles in an HDF5 file by selecting those within a spherical aperture.

       Args:
           ds (caesar load object): the caesar object of the type ds = sb.get_sim_file(snap) where sb is the simba class
           snap (int): the snapshot identifier
           center (array-like or int): the [x, y, z] position of the center of the aperture or an object ID
           radius (float): the radius of the spherical aperture in Kpc
           selection (str or None): 'galaxy' or 'halo' if center is an object ID
           verbose (int): verbosity level for logging information
           overwrite (bool): to overwrite existing filtered files; if False, requires a keyword to attach to the output naming
           ignore_fields (list of str): list of fields to ignore when writing the filtered file from the original
           keyword (str or None): keyword to append to the output file name if overwrite is False
    """
    if verbose>0:
        print(f'Analizing particle file {simfile}')
    # scale factor to convert from comoving to physical distanca units
    a = cs.simulation.scale_factor
    box_size = 25000#cs.simulation.boxsize.value
    
    # Instantiate SavePaths
    paths = SavePaths()

    # Determine the output directory for HDF5 files
    output_dir = paths.get_filetype_path('hdf5')
    output_dir = paths.create_subdir(output_dir, 'filtered_part_files')

    out_folder = f'{output_dir}/snap_{snap}'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    if not overwrite and keyword is None:
        raise ValueError('Keyword must be provided if overwrite is False')

    if not overwrite:
        output = f'{out_folder}/particles_within_aperture_{keyword}.h5'
    else:
        output = f'{out_folder}/particles_within_aperture.h5'

    if isinstance(center, (list, np.ndarray)) and len(center) == 3:
        # Center is provided as a coordinate
        center_coord = np.array(center)
    elif isinstance(center, int) and selection in ['galaxy', 'halo']:
        # Center is an object ID, retrieve coordinates based on selection
        if selection == 'galaxy':
            center_coord = cs.galaxies[center].pos
        elif selection == 'halo':
            center_coord = cs.halos[center].pos
        else:
            raise ValueError('Invalid selection type. Must be "galaxy" or "halo".')
    else:
        raise ValueError('Center must be a list/array with 3 positions or an object ID with a valid selection.')

    with h5py.File(simfile, 'r') as input_file:
        header = input_file['Header']
        if verbose>0:
            print('Start copy the file structure')
        copy_skeleton(simfile, output)
        if verbose>0:
            print('Finish copy the file structure')
        with h5py.File(output, 'a') as output_file:
            for ptype in ['PartType0', 'PartType1', 'PartType4', 'PartType5']:
                if ptype in input_file:  # check if particle type is present
                    if verbose > 0:
                        print('Selecting particles for :', ptype)
                    # Get particle positions
                    pos_key = 'Coordinates'
                    if pos_key not in input_file[ptype]:
                        if verbose > 0:
                            print(f'{pos_key} not found in {ptype}')
                        continue

                    pos = input_file[ptype][pos_key]#[:]
                    if verbose>0:
                        print(f'Using an aperture of {radius} Kpc ---> {radius/a} cKpc centered at {center}')
                    # Apply periodic boundary conditions to calculate distances correctly
                    distances = np.sqrt((pos[:,0]-center_coord[0])**2+(pos[:,1]-center_coord[1])**2+(pos[:,2]-center_coord[2])**2)
                    # Filter particles within the radius
                    if isinstance(radius, list):
                        if verbose>0:
                            print('Found anulus')
                        mask = np.all([distances>=radius[0]/a, distances<=radius[1]/a], axis=0)
                    else:
                        if verbose>0:
                            print('Found single radius')
                        mask = distances <= radius/a
                        
                    filtered_positions = pos[mask,:]
                    if len(filtered_positions[:,0])<1:
                        print('SelectionError: No particles have been found within the aperture')
                        continue
                    
                    # Handle other fields
                    for k in input_file[ptype]:
                        if k in ignore_fields:
                            if verbose > 1:
                                print(k, 'skipped...')
                            continue

                        if verbose > 0:
                            print(ptype, k)

                        # Extract and filter the dataset
                        temp_dset = input_file[ptype][k][:]
                        if verbose > 1:
                            print(temp_dset.shape)

                        filtered_dset = temp_dset[mask]
                        
                        # Create filtered datasets in output file
                        if k in output_file[ptype]:
                            del output_file[ptype][k]
                        output_file[ptype][k] = filtered_dset

                    # Update header information for the number of particles
                    temp = output_file['Header'].attrs['NumPart_ThisFile']
                    temp[int(ptype[8:])] = len(filtered_positions)
                    output_file['Header'].attrs['NumPart_ThisFile'] = temp

                    temp = output_file['Header'].attrs['NumPart_Total']
                    temp[int(ptype[8:])] = len(filtered_positions)
                    output_file['Header'].attrs['NumPart_Total'] = temp

                    if verbose > 0:
                        print("Updated Header:", dict(output_file['Header'].attrs))

        print('Finished with aperture filter')


