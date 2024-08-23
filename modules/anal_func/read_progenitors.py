import numpy as np
import h5py
from astropy.io import fits
from modules.io_paths.savepaths import SavePaths
import gc
import os
from astropy.table import Table

def caesar_read_progen(ids, outname, snaplist, sb):
    """Loop all the caesar files and write a fits file containing the indexes of the most massive progenitor for each galaxy

       Args:
           ids (array-like): arrays at snap 151 for which we want to extract the history
           outname (str): the name of the output fits file
           snaplist (list): the list of snapshots to use
           sb (simba class): necessary to use caesar
        
    """
    # Initialize a dictionary to store progenid data for each snapshot
    progenid_dict = {i: [] for i in snaplist}
    
    # Get GroupIDs and progenid from each snapshot
    for i in np.sort(snaplist)[::-1]:
        cs = sb.get_caesar(i)
        allids = np.asarray([galaxy.GroupID for galaxy in cs.galaxies])
        fname = cs.data_file
        # remove caesar file to not overload memory
        del cs
        gc.collect()
        
        with h5py.File(fname, 'r') as hf:
            progenid = np.asarray(hf['tree_data']['progen_galaxy_star'])[:, 0]
            # Mask to exclude -1 progenids
            mask = progenid != -1
            progenid = progenid[mask]
            allids = allids[mask]
        
        progenid_dict[i] = np.column_stack((allids, progenid))
    
    # Extract GroupIDs from the first snapshot (snaplist[0] should be 151 if passed in the right order)
    base_snapshot = np.sort(snaplist)[0]
    base_data = progenid_dict[base_snapshot]
    base_groupids = base_data[:, 0]
    
    # Initialize an array to store the result
    progenid_table = np.full((len(base_groupids), len(snaplist) + 1), -1, dtype=int)
    progenid_table[:, 0] = base_groupids
    
    # Fill the table with progenid data for each snapshot
    for i, snap in enumerate(np.sort(snaplist)):
        snap_data = progenid_dict[snap]
        for j, groupid in enumerate(base_groupids):
            idx = np.where(snap_data[:, 0] == groupid)[0]
            if len(idx) > 0:
                progenid_table[j, i + 1] = snap_data[idx, 1]
                
    # Instantiate SavePaths
    paths = SavePaths()
    # Determine the output directory for HDF5 files
    output_dir = paths.get_filetype_path('fits')
    output_dir = paths.create_subdir(output_dir, 'progenitors_files')
    
    # Create a FITS table with appropriate columns
    col1 = fits.Column(name='GroupID', format='K', array=progenid_table[:, 0])
    columns = [col1]
    for i, snap in enumerate(np.sort(snaplist)):
        col = fits.Column(name=f'{snap}', format='K', array=progenid_table[:, i + 1])
        columns.append(col)
    
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.writeto(os.path.join(output_dir, outname), overwrite=True)





import os
import numpy as np
from astropy.io import fits
import random

def get_fits(sb, snap, fitsdir):
    filename = sb.get_caesar_file(snap)
    new_filename = filename.split('/')[-1].split('.')[0] + '.fits'
    new_filename = os.path.join(fitsdir, new_filename)
    return new_filename


def read_progen(ids, outname, snaplist, sb, fitsdir):
    # Initialize a dictionary to store GroupID data for each snapshot
    progenid_dict = {str(i): [] for i in snaplist}
    # Start with the first snapshot (which should be 151)
    first_iteration = True
    snaplist = np.sort(snaplist)[::-1]
    for idx, snap in enumerate(snaplist):  # Process snapshots in reverse order
        fname = get_fits(sb, snap, fitsdir)
        print(f'Processing current snap: {fname}')
        with fits.open(fname) as hdul:
            hf = hdul[1].data
            groupids = hf['GroupID'].astype(int)
            if first_iteration:
                #match_ids = np.isin(groupids, ids)
                progenid = hf['descend_galaxy_star'].astype(int)#[match_ids]
                progenid_dict['GroupID'] = groupids
                progenid_dict[str(snap)] = groupids #[match_ids]
                first_iteration = False
            else:
                currprog = hf['descend_galaxy_star'].astype(int)
                temp = []
                for i in range(len(progenid)):
                    if i>-1:
                        temp.append(groupids[progenid[i]])
                        progenid[i] = currprog[progenid[i]]
                    else:
                        temp.append(-1)
                        progenid[i] = -1
                        
                progenid_dict[str(snap)] = np.array(temp)
    

    table = Table(progenid_dict)

    # Instantiate SavePaths (assuming SavePaths is defined elsewhere in your code)
    print('Saving...')
    paths = SavePaths()
    output_dir = paths.get_filetype_path('fits')
    output_dir = paths.create_subdir(output_dir, 'progenitors_files')

    table.write(os.path.join(output_dir, outname), overwrite=True)


