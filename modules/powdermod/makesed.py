"""
purpose: to set up slurm files and model *.py files from the
positions written by caesar_cosmology_npzgen.py for a cosmological
simulation.

Updated by Chris Lovell for cosma machine (Durham) 12/10/19
"""
import sys
import numpy as np
import h5py
from subprocess import call
import caesar
import json
from shutil import copyfile

from simba import simba
from modules.io_paths.savepaths import SavePaths

class MakeSED:
    def __init__(self, sb,  nnodes, model_run_name, hydro_dir_base, selection_file, COSMOFLAG=0):
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

    def selection_gals(self):
        """given the file with the galaxy selections (that contains galaxy ID, halo ID and galaxy position)
           the input file is a txt 
           the output is an hdf5 file
        """
        self.sb = sb
        # define path for output file with the selection
        paths = SavePaths()
        output_dir = paths.get_filetype_path('txt')
        output_dir = paths.create_subdir(output_dir, 'target_selection_for_SED')
        filepath = os.path.join(output_dir, self.selection_file)
        # load selection file
        dat = np.loadtxt(filepath, skiprows=1, dtype=int)
        snaps = dat[:,0]
        z0_id = dat[:,1]
        gal_id = dat[:,2]
        halo_id = dat[:,3]

        for snap in np.unique(snaps):
            cs = sb.get_caesar(f'{snap:03}')
            a = cs.simulation.scale_factor
                coods_code = np.array([g.pos.in_units('code_length').value for g in cs.galaxies])
                hidx = np.array([g.parent_halo_index for g in cs.galaxies])
                idx_arr = gal_id[np.where(snaps == snap)[0]]
                z0_arr = z0_id[np.where(snaps == snap)[0]]
                with h5py.File(filepath, 'a') as hf:
                    hf.create_group(f'snap{snap:03}')
                    hf.create_dataset(f'snap{snap:03}/galaxy_GroupID', data=idx_arr)
                    hf.create_dataset(f'snap{snap:03}/z0_ID', data=z0_arr)
                    hf.create_dataset(f'snap{snap:03}/halo_GroupID', data=hidx[idx_arr])
                    hf.create_dataset(f'snap{snap:03}/code_coods', data=coods_code[idx_arr])


    
    def create_master(self):
        """ uses the input files for the selection to initialize the parameters_master and parameters_models
            all the needed files are initialized with the class
            folders are already in the structure of the code
            this function automatically reads the selection file and create a subdirectory for each snapshot in the hydro_dir_base
            and for each snap the folder corresponding to the galaxy ID 
        """

        with h5py.File(self.selection_file, 'r') as hf: 
            snaps = list(hf.keys())[::-1]
            snaps = [int(snap[4:]) for snap in snaps]
            snaps_to_redshift = {}
            scalefactor = np.loadtxt(self.hydro_outputfile)
            for n, snap in enumerate(snaps):
                snaps_to_redshift['{:03.0f}'.format(n)] = (1./scalefactor[n])-1.
                _dat = hf[f'snap{snap:03}']
                ids =  hf[f'snap{snap:03}/galaxy_GroupID'][:]
                hidx = hf[f'snap{snap:03}/halo_GroupID'][:]
                pos =  hf[f'snap{snap:03}/code_coods'][:]
                # subfolder in the hydro_dir_base for each snapshot
                hydro_dir = f'{hydro_dir_base}/snap_{snap:03}'
                hydro_dir_remote = hydro_dir
                # subfolder in the output directory, one for each snapshot
                model_dir = f'{model_dir_base}/snap_{snap:03}'
                # determine redshift and CMB temperature for a given snapshot
                redshift = snaps_to_redshift[f'{snap:03}']
                tcmb = 2.73*(1.+redshift)

                N = len(ids)
                if N > 0:
                    # write for each galaxy a job
                    for i,nh in enumerate(ids):
                        # only write job submission script once...
                        if i == 0: job_flag = 1
                        else: job_flag = 0
                        # create sudirectory in output folder for each ID
                        model_dir_remote = model_dir+'/gal_%s'%nh
                        xpos = pos[i][0]
                        ypos = pos[i][1]
                        zpos = pos[i][2]
                        # create the inputs for the bash file to create the parameters master
                        cmd = "./cosmology_setup_all_cluster.cis.sh "+str(nnodes)+' '+model_dir+' '+hydro_dir+' '+model_run_name+' '+str(COSMOFLAG)+\
                        ' '+model_dir_remote+' '+hydro_dir_remote+' '+str(xpos)+' '+str(ypos)+' '+str(zpos)+' '+str(nh)+' '+str(int(snap))+' '+\
                        str(tcmb)+' '+str(i)+' '+str(job_flag)+' '+str(N-1)+' '+str(hidx[i])
                        # run command for the parameters master
                        call(cmd,shell=True)
                    # just save the ID used for sanity check
                    np.savetxt(model_dir+'/ids.txt',ids,fmt='%i')
                    # put the rewritten parameters_master in the output directory
                    copyfile('parameters_master.py', model_dir+'/parameters_master.py')
