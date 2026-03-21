"""Powderday SED generation wrapper.

Requires optional heavy dependencies: ``hyperion``, ``caesar``.
Install with ``pip install simbanator[full]``.
"""

import sys
import os
import numpy as np
import h5py
from shutil import copyfile
from collections import defaultdict

import caesar
import json
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from hyperion.model import ModelOutput
from astropy.cosmology import Planck13
from astropy import units as u
from astropy import constants




class MakeSED:
    """Create and run Powderday SED models for selected galaxies.

    Parameters
    ----------
    sb : :class:`~simbanator.io.simba.Simba`
        Simba path manager.
    nnodes : int
        Number of compute nodes.
    model_run_name : str
        Name tag for the model run.
    hydro_dir_base : str
        Base directory for simulation particle files.
    preselect : bool
        Whether galaxy pre-selection is used.
    selection_file : str
        Name of the HDF5 file with target galaxy info.
    COSMOFLAG : int
        Cosmological simulation flag (0 or 1).
    """

    def __init__(self, sb, nnodes, model_run_name, hydro_dir_base,
                 preselect, selection_file, COSMOFLAG=0, output_dir=None):
        self.sb = sb
        self.nnodes = nnodes
        self.model_run_name = model_run_name
        self.COSMOFLAG = COSMOFLAG
        self.hydro_dir_base = hydro_dir_base
        self.preselect = preselect
        self.selection_file = selection_file

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'output', 'sed')
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.model_dir_base = os.path.join(output_dir, 'powderday_sed_out')
        os.makedirs(self.model_dir_base, exist_ok=True)

    def selection_gals(self, snaps, galaxyID):
        """Write galaxy selection info to an HDF5 file.

        Parameters
        ----------
        snaps : list of int
            Snapshot numbers, one per selected galaxy.
        galaxyID : array-like of int
            Galaxy GroupIDs to select, paired with *snaps*.
        """
        filepath = os.path.join(self.output_dir, 'target_selection', self.selection_file + '.h5')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if not os.path.exists(filepath):
            with h5py.File(filepath, 'w') as hf:
                pass

        snaps = np.asarray(snaps, dtype=int)
        galaxyID = np.asarray(galaxyID, dtype=int)

        if snaps.shape != galaxyID.shape:
            raise ValueError(
                "snaps and galaxyID must have the same shape (pairwise mapping: snaps[i] -> galaxyID[i])."
            )

        grouped_ids = defaultdict(list)
        for snap, gid in zip(snaps, galaxyID):
            grouped_ids[int(snap)].append(int(gid))

        data_dict = defaultdict(lambda: {
            'galaxy_GroupID': [], 'halo_GroupID': [], 'code_coods': []
        })

        for snap, ids_for_snap in grouped_ids.items():
            cs = self.sb.get_caesar(snap)
            coods_code = np.array([g.pos.in_units('code_length').value for g in cs.galaxies])
            hidx = np.array([g.parent_halo_index for g in cs.galaxies])
            ids_for_snap = np.asarray(ids_for_snap, dtype=int)
            data_dict[snap]['galaxy_GroupID'].extend(ids_for_snap)
            data_dict[snap]['halo_GroupID'].extend(hidx[ids_for_snap])
            data_dict[snap]['code_coods'].extend(coods_code[ids_for_snap])

        with h5py.File(filepath, 'a') as hf:
            for snap, data in data_dict.items():
                grp = f'snap{snap:03}'
                if grp in hf:
                    del hf[grp]
                hf.create_group(grp)
                hf.create_dataset(f'{grp}/galaxy_GroupID', data=np.array(data['galaxy_GroupID']))
                hf.create_dataset(f'{grp}/halo_GroupID', data=np.array(data['halo_GroupID']))
                hf.create_dataset(f'{grp}/code_coods', data=np.array(data['code_coods']))
                print(f'Written data for {grp}')

    def create_master(self, where, subset_type='plist', radius=None):
        """Generate Powderday parameter files and batch scripts.

        Two job types are generated: ``master.snap{snap}.job`` (GNU parallel)
        and ``master.snap{snap}.array.job`` (Slurm array). See PARALLELIZATION.md
        for detailed parallelization strategy comparison and recommendations.

        Parameters
        ----------
        where : str
            ``'local'`` or ``'cluster'``.
        subset_type : str
            Particle subset method: ``'plist'`` (particle list) or
            ``'region'`` (spherical aperture).
        radius : float, optional
            Aperture radius (required when *subset_type* is ``'region'``).
        """
        if where not in {'local', 'cluster'}:
            raise ValueError("where must be 'local' or 'cluster'")
        if subset_type not in {'plist', 'region'}:
            raise ValueError("subset_type must be 'plist' or 'region'")
        if subset_type == 'region' and radius is None:
            raise ValueError("radius is required when subset_type='region'")

        filepath = os.path.join(self.output_dir, 'target_selection', self.selection_file + '.h5')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Selection file not found: {filepath}")

        pkg_dir = os.path.dirname(__file__)
        paramfile = os.path.join(pkg_dir, 'parameters_master.py')
        cluster_job_files = []
        local_run_files = []

        with h5py.File(filepath, 'r') as hf:
            snaps = sorted([int(s[4:]) for s in hf.keys()])

            for snap in snaps:
                ids = hf[f'snap{snap:03}/galaxy_GroupID'][:]
                pos = hf[f'snap{snap:03}/code_coods'][:]

                hydro_dir = os.path.join(self.hydro_dir_base, f'snap_{snap:03}')
                model_dir = os.path.join(self.model_dir_base, f'snap_{snap:03}')
                os.makedirs(hydro_dir, exist_ok=True)
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(os.path.join(model_dir, 'out'), exist_ok=True)
                os.makedirs(os.path.join(model_dir, 'err'), exist_ok=True)

                redshift = self.sb.get_z_from_snap(snap)
                tcmb = 2.73 * (1. + redshift)
                N = len(ids)

                if N == 0:
                    continue

                for i, nh in enumerate(ids):
                    model_dir_remote = os.path.join(model_dir, f'gal_{nh}')
                    os.makedirs(model_dir_remote, exist_ok=True)
                    xpos, ypos, zpos = pos[i]

                    model_file = os.path.join(model_dir, f'snap{snap}_{nh}.py')
                    with open(model_file, 'w', encoding='utf-8') as f:
                        f.write('#Snapshot Parameters\n')
                        f.write('snapshot_num = ' + str(int(snap)) + '\n')
                        f.write('galaxy_num = ' + str(int(nh)) + '\n\n')

                        f.write("galaxy_num_str = '{:06.0f}'.format(galaxy_num)\n")
                        f.write("snapnum_str = '{:03.0f}'.format(snapshot_num)\n\n")

                        f.write(f"hydro_dir = '{hydro_dir}/'\n")
                        if subset_type == 'plist':
                            f.write("snapshot_name = f'subset_snap{snapnum_str}_gal'+galaxy_num_str+'.h5'\n\n")
                        else:
                            f.write(
                                "snapshot_name = f'region_snap{snapnum_str}_r"
                                + str(radius)
                                + "_gal'+galaxy_num_str+'.h5'\n\n"
                            )

                        f.write('#where the files should go\n')
                        f.write(f"PD_output_dir = '{model_dir_remote}/'\n")
                        f.write("Auto_TF_file = 'snap'+snapnum_str+'.logical'\n")
                        f.write("Auto_dustdens_file = 'snap'+snapnum_str+'.dustdens'\n\n")

                        f.write('#===============================================\n')
                        f.write('#FILE I/O\n')
                        f.write('#===============================================\n')
                        f.write("inputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'.rtin'\n")
                        f.write("outputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'.rtout'\n\n")

                        f.write('#===============================================\n')
                        f.write('#GRID POSITIONS\n')
                        f.write('#===============================================\n')
                        f.write(f'x_cent = {xpos}\n')
                        f.write(f'y_cent = {ypos}\n')
                        f.write(f'z_cent = {zpos}\n\n')

                        f.write('#===============================================\n')
                        f.write('#CMB INFORMATION\n')
                        f.write('#===============================================\n')
                        f.write(f'TCMB = {tcmb}\n')

                if where == 'cluster':
                    walltime = '0-08:00' if N > 1000 else '1-00:00'
                    
                    # Create main parallel job script
                    qsubfile = os.path.join(model_dir, f'master.snap{snap}.job')
                    with open(qsubfile, 'w', encoding='utf-8') as f:
                        f.write('#! /bin/bash\n')
                        f.write(f'#SBATCH --job-name={self.model_run_name}.snap{snap}\n')
                        f.write(f'#SBATCH --output=out/pd.master.snap{snap}.%N.%j.o\n')
                        f.write(f'#SBATCH --error=err/pd.master.snap{snap}.%N.%j.e\n')
                        f.write('#SBATCH --mail-type=FAIL\n')
                        f.write(f'#SBATCH -t {walltime}\n')
                        f.write('#SBATCH --nodes=1\n')
                        f.write('#SBATCH --ntasks=1\n')
                        f.write('#SBATCH --cpus-per-task=8\n')
                        f.write('#SBATCH --mem=30G\n')
                        f.write('\n')
                        f.write('module purge\n')
                        f.write('module load git/2.20.1 gcc/9.3.0 openmpi/4.0.5 hdf5/1.12.1\n\n')
                        f.write('conda activate pd-env\n\n')
                        f.write('PD_FRONT_END="/mnt/home/glorenzon/powderday/pd_front_end.py"\n')
                        f.write('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n')
                        f.write('cd "$SCRIPT_DIR"\n\n')
                        f.write('export OMP_NUM_THREADS=1\n')
                        f.write('set -e\n\n')
                        f.write('echo "Starting parallel processing of galaxies for snap{snap} at $(date)"\n')
                        f.write('echo "Using CPU_COUNT=${SLURM_CPUS_PER_TASK:-8}"\n\n')
                        f.write('# Process galaxies in parallel using GNU parallel\n')
                        f.write('cat ids.txt | parallel --no-notice --halt soon,fail=1 ')
                        f.write(f'--line-buffer -j ${{SLURM_CPUS_PER_TASK:-8}} ')
                        f.write('"mkdir -p gal_{{}}/logs && ')
                        f.write(f'python $PD_FRONT_END . parameters_master snap{snap}_{{}} ')
                        f.write(f'> gal_{{}}/snap{snap}_{{}}.LOG 2>&1"\n\n')
                        f.write('PARALLEL_EXIT=$?\n')
                        f.write('if [ $PARALLEL_EXIT -ne 0 ]; then\n')
                        f.write('  echo "WARNING: Some galaxy processing failed (exit code: $PARALLEL_EXIT)"\n')
                        f.write('fi\n\n')
                        f.write('echo "Galaxy processing complete at $(date)"\n')
                        f.write('echo "Job summary:"\n')
                        f.write('sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,Elapsed,ExitCode,MaxRSS,CPUTime,SystemCPU,ReqMem\n')
                        f.write('exit $PARALLEL_EXIT\n')
                    cluster_job_files.append(qsubfile)
                    
                    # Optional: Create individual task array job for even better parallelization
                    task_array_file = os.path.join(model_dir, f'master.snap{snap}.array.job')
                    with open(task_array_file, 'w', encoding='utf-8') as f:
                        f.write('#! /bin/bash\n')
                        f.write(f'#SBATCH --job-name={self.model_run_name}.snap{snap}.array\n')
                        f.write(f'#SBATCH --output=out/pd.snap{snap}.%a.%j.o\n')
                        f.write(f'#SBATCH --error=err/pd.snap{snap}.%a.%j.e\n')
                        f.write(f'#SBATCH --array=1-{N}%16\n')  # Run max 16 jobs in parallel
                        f.write('#SBATCH --mail-type=FAIL\n')
                        f.write(f'#SBATCH -t {walltime}\n')
                        f.write('#SBATCH --nodes=1\n')
                        f.write('#SBATCH --ntasks=1\n')
                        f.write('#SBATCH --cpus-per-task=1\n')
                        f.write('#SBATCH --mem=8G\n')
                        f.write('\n')
                        f.write('module purge\n')
                        f.write('module load git/2.20.1 gcc/9.3.0 openmpi/4.0.5 hdf5/1.12.1\n\n')
                        f.write('conda activate pd-env\n\n')
                        f.write('PD_FRONT_END="/mnt/home/glorenzon/powderday/pd_front_end.py"\n')
                        f.write('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n')
                        f.write('cd "$SCRIPT_DIR"\n\n')
                        f.write('set -e\n\n')
                        f.write('ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ids.txt)\n')
                        f.write('echo "Processing galaxy $ID (task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}) at $(date)"\n\n')
                        f.write(f'python $PD_FRONT_END . parameters_master snap{snap}_$ID > gal_$ID/snap{snap}_$ID.LOG 2>&1\n\n')
                        f.write('echo "Galaxy $ID complete at $(date)"\n')
                    cluster_job_files.append(task_array_file)
                else:
                    runfile = os.path.join(model_dir, 'run_local.sh')
                    with open(runfile, 'w', encoding='utf-8') as f:
                        f.write('#!/bin/bash\n')
                        f.write('set -e\n')
                        f.write('echo "Starting local Powderday batch for snap {snap}..."\n')
                        f.write('echo "Using $(nproc) CPU cores"\n\n')
                        f.write('export OMP_NUM_THREADS=1\n')
                        f.write(f'cat {os.path.join(model_dir, "ids.txt")} | parallel --no-notice ')
                        f.write('--halt soon,fail=1 --line-buffer -j 0 ')
                        f.write('"mkdir -p gal_{{}}/logs && ')
                        f.write('python /home/lorenzong/powderday/pd_front_end.py . parameters_master ')
                        f.write(f'snap{snap}_{{}} > gal_{{}}/snap{snap}_{{}}.LOG 2>&1"\n\n')
                        f.write('PARALLEL_EXIT=$?\n')
                        f.write('if [ $PARALLEL_EXIT -eq 0 ]; then\n')
                        f.write('  echo "✓ Local batch complete successfully"\n')
                        f.write('else\n')
                        f.write('  echo "✗ Some processing failed (check logs)"\n')
                        f.write('fi\n')
                        f.write('exit $PARALLEL_EXIT\n')
                    os.chmod(runfile, 0o755)
                    local_run_files.append(runfile)

                np.savetxt(os.path.join(model_dir, 'ids.txt'), ids, fmt='%i')
                copyfile(paramfile, os.path.join(model_dir, 'parameters_master.py'))

        if where == 'cluster' and cluster_job_files:
            submit_all = os.path.join(self.model_dir_base, 'submit_all_snaps.sh')
            with open(submit_all, 'w', encoding='utf-8') as f:
                f.write('#!/bin/bash\n')
                f.write('set -e\n')
                f.write('echo "Submitting all snapshot Slurm jobs in parallel..."\n')
                f.write('SUBMITTED_JOBS=()\n\n')
                for job_file in cluster_job_files:
                    f.write(f'echo "Submitting {job_file}"\n')
                    f.write(
                        f'cd "{os.path.dirname(job_file)}" && '
                        f'JOB_ID=$(sbatch -p INTEL_HASWELL "{os.path.basename(job_file)}" '
                        f'| awk \'{{print $NF}}\') && '
                        f'SUBMITTED_JOBS+=($JOB_ID) && '
                        f'echo "  -> Job ID: $JOB_ID"\n'
                    )
                f.write('\necho ""\n')
                f.write('echo "All snapshot jobs submitted. Jobs: ${SUBMITTED_JOBS[@]}"\n')
                f.write('echo "Monitor progress with: squeue -u $USER -l"\n')
                f.write('echo "Cancel all with: scancel ${SUBMITTED_JOBS[@]}"\n')
            os.chmod(submit_all, 0o755)

        if where == 'local' and local_run_files:
            run_all_local = os.path.join(self.model_dir_base, 'run_all_local.sh')
            with open(run_all_local, 'w', encoding='utf-8') as f:
                f.write('#!/bin/bash\n')
                f.write('set -e\n')
                f.write('echo "Running all local snapshot batches (sequentially)..."\n')
                f.write('FAILED_SNAPS=()\n\n')
                for run_file in local_run_files:
                    f.write(f'echo "\n========== Running {os.path.dirname(run_file)} =========="\n')
                    f.write(f'if ! {run_file}; then\n')
                    f.write(f'  FAILED_SNAPS+=("{os.path.dirname(run_file)}")\n')
                    f.write(f'fi\n\n')
                f.write('echo "\\nSummary:"\n')
                f.write('if [ ${#FAILED_SNAPS[@]} -eq 0 ]; then\n')
                f.write('  echo "✓ All local snapshot batches completed successfully"\n')
                f.write('else\n')
                f.write('  echo "✗ Failed snapshots: ${FAILED_SNAPS[@]}"\n')
                f.write('  exit 1\n')
                f.write('fi\n')
            os.chmod(run_all_local, 0o755)

    def plotsed(self, snap, gal, show=False, ret=False):
        """Plot the SED output for a given galaxy at a given snapshot.

        Parameters
        ----------
        snap : int
            Snapshot number.
        gal : int
            Galaxy ID.
        show : bool
            Call ``plt.show()``.
        ret : bool
            If *True*, return ``(fig, ax)``.
        """
        fig, ax = plt.subplots()
        run = os.path.join(
            self.model_dir_base, f'snap_{snap}', f'gal_{gal}',
            f'snap{snap}.galaxy{gal:06}.rtout.sed',
        )
        z = self.sb.get_z_from_snap(snap)
        m = ModelOutput(run)
        wav, flux = m.get_sed(inclination='all', aperture=-1)
        wav = np.asarray(wav) * u.micron * (1. + z)

        flux = np.asarray(flux) * u.erg / u.s
        dl = Planck13.luminosity_distance(z).to(u.cm)
        flux /= (4. * np.pi * dl ** 2)

        nu = (constants.c.cgs / wav.to(u.cm)).to(u.Hz)
        flux = (flux / nu).to(u.mJy)

        for i in range(flux.shape[0]):
            ax.loglog(wav.value, flux[i, :].value)

        ax.set_xlabel(r'$\lambda\;[\mu\mathrm{m}]$')
        ax.set_ylabel('Flux (mJy)')
        ax.set_xlim(0.05, 15000)

        out = os.path.join(self.output_dir, 'sed_plots', f'snap_{snap}')
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, f'gal_{gal}.png'), bbox_inches='tight')

        if show:
            plt.show()
        if ret:
            return fig, ax

    def monitor_jobs(self, username=None):
        """Print monitoring commands for submitted cluster jobs.
        
        Parameters
        ----------
        username : str, optional
            Username for job filtering. If None, uses current user.
        """
        if username is None:
            username = os.environ.get('USER', 'unknown')
        
        print("Job monitoring commands:")
        print(f"  squeue -u {username} -l")
        print(f"  squeue -u {username} --state=R")
        print(f"\nLog directories:")
        print(f"  {os.path.join(self.model_dir_base, 'snap_*/out/')}")
        print(f"  {os.path.join(self.model_dir_base, 'snap_*/err/')}")
        print("\nSee PARALLELIZATION.md for detailed monitoring instructions.")


