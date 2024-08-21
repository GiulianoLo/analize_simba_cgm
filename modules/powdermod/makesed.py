"""
purpose: to set up slurm files and model *.py files from the
positions written by caesar_cosmology_npzgen.py for a cosmological
simulation.

Updated by Chris Lovell for cosma machine (Durham) 12/10/19
"""
import sys
import os
import numpy as np
import h5py
from subprocess import call
import caesar
import json
from shutil import copyfile
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from hyperion.model import ModelOutput
from astropy.cosmology import Planck13
from astropy import units as u
from astropy import constants


from modules.io_paths.savepaths import SavePaths

class MakeSED:
    def __init__(self, sb,  nnodes, model_run_name, hydro_dir_base, preselect, selection_file, COSMOFLAG=0):
        # initialize the properties for the parameters master
        self.sb = sb
        self.nnodes = nnodes
        self.model_run_name = model_run_name
        self.COSMOFLAG = COSMOFLAG
        # initialize saving class
        paths = SavePaths()
        # initialize main output directory
        base = paths.base_output_dir
        # subdirectory for the SED is in hdf5 (powderday output format)
        self.hydro_dir_base = hydro_dir_base # where the simulated particle files to use are
        output_dir = paths.get_filetype_path('hdf5')
        self.model_dir_base = paths.create_subdir(output_dir, 'powderday_sed_out') # the output directory
        # other input directories for parameters_masters
        self.hydro_outputfile = self.sb.output_file # directory with the file with the conversion factor from snap to z
        self.selection_file = selection_file # hdf5 file with the galaxy coordinates and id from caesar file
        self.preselect = preselect
        

    def selection_gals(self, snaps, galaxyID):
        """Given a file with galaxy selections (that contains galaxy ID, halo ID, and galaxy position),
           the input file is a txt, and the output is an HDF5 file.
        """
        # Define path for output file with the selection
        paths = SavePaths()
        output_dir = paths.get_filetype_path('hdf5')
        output_dir = paths.create_subdir(output_dir, 'target_selection_for_SED')
        filepath = os.path.join(output_dir, self.selection_file+'.h5')
        
        # Check if the file exists; if not, create an empty HDF5 file
        if not os.path.exists(filepath):
            with h5py.File(filepath, 'w') as hf:
                pass  # Create an empty HDF5 file

        from collections import defaultdict
        # Data containers for each snap
        data_dict = defaultdict(lambda: {'galaxy_GroupID': [], 'halo_GroupID': [], 'code_coods': []})
        
        for snap in snaps:
            cs = self.sb.get_caesar(snap)
            a = cs.simulation.scale_factor
            coods_code = np.array([g.pos.in_units('code_length').value for g in cs.galaxies])
            hidx = np.array([g.parent_halo_index for g in cs.galaxies])
            
            # Accumulate data for each snap
            data_dict[snap]['galaxy_GroupID'].extend(galaxyID)
            data_dict[snap]['halo_GroupID'].extend(hidx[galaxyID])
            data_dict[snap]['code_coods'].extend(coods_code[galaxyID])
        
        # After all snaps have been processed, write to the file
        with h5py.File(filepath, 'a') as hf:
            for snap, data in data_dict.items():
                group_name = f'snap{snap:03}'
                hf.create_group(group_name)
                hf.create_dataset(f'{group_name}/galaxy_GroupID', data=np.array(data['galaxy_GroupID']))
                hf.create_dataset(f'{group_name}/halo_GroupID', data=np.array(data['halo_GroupID']))
                hf.create_dataset(f'{group_name}/code_coods', data=np.array(data['code_coods']))
                print('Inserted aggregated data for:', group_name)

        
        
        # for snap in snaps:
        #     cs = self.sb.get_caesar(snap)
        #     a = cs.simulation.scale_factor
        #     coods_code = np.array([g.pos.in_units('code_length').value for g in cs.galaxies])
        #     print('coods_code+++++++++++', coods_code)
        #     #coods_code = np.array([g.pos for g in cs.galaxies])
        #     hidx = np.array([g.parent_halo_index for g in cs.galaxies])
            
    
        #     with h5py.File(filepath, 'a') as hf:
        #         group_name = f'snap{snap:03}'
        #         hf.create_group(group_name)
        #         hf.create_dataset(f'{group_name}/galaxy_GroupID', data=galaxyID)
        #         hf.create_dataset(f'{group_name}/halo_GroupID', data=hidx[galaxyID])
        #         hf.create_dataset(f'{group_name}/code_coods', data=coods_code[galaxyID])
        #         print('originaaaaaaaaaaaaaaal', coods_code[galaxyID])



    
    def create_master(self, where):
        """Uses the input files for the selection to initialize the parameters_master and parameters_models.
           This function automatically reads the selection file and creates a subdirectory for each snapshot 
           in the hydro_dir_base and model_dir_base, and for each snap, the folder corresponding to the galaxy ID.
        """
        # Define path for output file with the selection
        paths = SavePaths()
        output_dir = paths.get_filetype_path('hdf5')
        output_dir = paths.create_subdir(output_dir, 'target_selection_for_SED')
        filepath = os.path.join(output_dir, self.selection_file+'.h5')
    
        with h5py.File(filepath, 'r') as hf: 
            snaps = list(hf.keys())[::-1]
            snaps = [int(snap[4:]) for snap in snaps]
            scalefactor = np.loadtxt(self.hydro_outputfile)
    
            for n, snap in enumerate(snaps):
                # Load galaxy information from HDF5 file
                _dat = hf[f'snap{snap:03}']
                ids = hf[f'snap{snap:03}/galaxy_GroupID'][:]
                hidx = hf[f'snap{snap:03}/halo_GroupID'][:]
                pos = hf[f'snap{snap:03}/code_coods'][:]
                print('=====================', f'snap{snap:03}/code_coods', pos)
    
                # Create subdirectories in hydro_dir_base and model_dir_base for the current snapshot
                hydro_dir = os.path.join(self.hydro_dir_base, f'snap_{snap:03}')
                model_dir = os.path.join(self.model_dir_base, f'snap_{snap:03}')
                os.makedirs(hydro_dir, exist_ok=True)
                os.makedirs(model_dir, exist_ok=True)
    
                # Determine redshift and CMB temperature for the current snapshot
                redshift = (1. / scalefactor[n]) - 1.
                tcmb = 2.73 * (1. + redshift)
    
                N = len(ids)
                if N > 0:
                    # Write job submission script and setup files for each galaxy
                    for i, nh in enumerate(ids):
                        # Only write job submission script once for the first galaxy
                        job_flag = 1 if i == 0 else 0
    
                        # Create subdirectory in model_dir for each galaxy
                        model_dir_remote = os.path.join(model_dir, f'gal_{nh}')
                        os.makedirs(model_dir_remote, exist_ok=True)
    
                        xpos, ypos, zpos = pos[i]
                        print('Processing galaxy: ', nh, ' in snap ', snap)
                        print('Initializing position: ', pos[i])
    
                        # Prepare the command for the bash file to create the parameters master
                        if where == 'local':
                            setupfile = os.path.join(os.getcwd(), 'modules', 'powdermod', 'cosmology_setup_all_local.pc39.sh')
                        elif where == 'cluster':
                            setupfile = os.path.join(os.getcwd(), 'modules', 'powdermod', 'cosmology_setup_all_cluster.cis.sh')
                            
                        cmd = (
                            f"{setupfile} {self.nnodes} {model_dir} {hydro_dir} {self.model_run_name} {self.COSMOFLAG} "
                            f"{model_dir_remote} {hydro_dir} {xpos} {ypos} {zpos} {nh} {int(snap)} {tcmb} {i} "
                            f"{job_flag} {N-1} {hidx[i]}"
                        )
                        # Run the command to set up the parameters master
                        call(cmd, shell=True)
    
                    # Save the IDs used for sanity check
                    np.savetxt(os.path.join(model_dir, 'ids.txt'), ids, fmt='%i')
    
                    # Copy the parameters_master.py file to the output directory
                    paramfile = os.path.join(os.getcwd(), 'modules', 'powdermod', 'parameters_master.py')
                    copyfile(paramfile, os.path.join(model_dir, 'parameters_master.py'))


    def plotsed(self, snap, gal, show=False):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        run = os.path.join(self.model_dir_base, f'snap_{snap}', f'gal_{gal}', f'snap{snap}.galaxy{gal:06}.rtout.sed')
        z = self.sb.get_z_from_snap(snap)
        m = ModelOutput(run)
        wav,flux = m.get_sed(inclination='all',aperture=-1)
        wav  = np.asarray(wav)*u.micron #wav is in micron
        wav *= (1.+z)
        
        flux = np.asarray(flux)*u.erg/u.s
        dl = Planck13.luminosity_distance(z)
        dl = dl.to(u.cm)
            
        flux /= (4.*3.14*dl**2.)
            
        nu = constants.c.cgs/(wav.to(u.cm))
        nu = nu.to(u.Hz)
        
        flux /= nu
        flux = flux.to(u.mJy)
        
        for i in range(flux.shape[0]):
            ax.loglog(wav,flux[i,:])
        
        ax.set_xlabel(r'$\lambda$ [$\mu$m]')
        ax.set_ylabel('Flux (mJy)')
        #ax.set_ylim([1,1e8])
        ax.set_xlim(0.05,15000)
        print('saving')

        # saving
        paths = SavePaths()
        output_dir = paths.get_filetype_path('plot')
        output_dir = paths.create_subdir(output_dir, 'sed_out')
        output_dir = paths.create_subdir(output_dir, f'snap_{snap}')
        filepath = os.path.join(output_dir, f'gal_{gal}' +'.png')
        plt.savefig(filepath, bbox_inches='tight')
        if show:
            plt.show()
        

