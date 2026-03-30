"""Powderday SED generation wrapper.

Requires optional heavy dependencies: ``hyperion``, ``caesar``.
Install with ``pip install simbanator[full]``.
"""
import warnings
import sys
import os
import csv
import numpy as np
import h5py
import subprocess
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
from astropy.table import Table
from astropy.table import vstack

from simbanator.sed.flux_extraction import flux_extraction, get_svo_filters

def flatten_results(results, snap, gal):
    rows = []

    for fac in results:
        for inst in results[fac]:
            for f in results[fac][inst]:
                entry = results[fac][inst][f]

                rows.append({
                    'snap': snap,
                    'gal': gal,
                    'facility': fac,
                    'instrument': inst,
                    'filter': f,
                    'xmean': entry['xmean'],
                    'mJy': entry['mJy'],
                    'mag': entry['mag']
                })

    return rows



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
    selection_file : str
        Name of the HDF5 file with target galaxy info.
    COSMOFLAG : int
        Cosmological simulation flag (0 or 1).
    output_dir : str, optional
        Base output directory for SED products.
    run_tag : str, optional
        Subfolder label used to separate different runs (e.g., dust/no-dust).
        If None, ``model_run_name`` is used.
    """

    def __init__(self, sb, nnodes, model_run_name, hydro_dir_base,
                 selection_file, COSMOFLAG=0, output_dir=None,
                 run_tag=None):
        self.sb = sb
        self.nnodes = nnodes
        self.model_run_name = model_run_name
        self.COSMOFLAG = COSMOFLAG
        self.hydro_dir_base = hydro_dir_base
        self.selection_file = selection_file

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'output', sb.name, 'sed')
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        if run_tag is None:
            run_tag = model_run_name
        run_tag = str(run_tag).strip()
        if not run_tag:
            raise ValueError("run_tag must be a non-empty string")
        run_tag = run_tag.replace('/', '_').replace(' ', '_')
        self.run_tag = run_tag

        self.model_dir_base = os.path.join(output_dir, self.run_tag, 'powderday_sed_out')
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
        filepath = os.path.join(self.output_dir, self.run_tag, 'target_selection', self.selection_file + '.h5')
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

    def _resolve_hydro_dir_for_snap(self, snap):
        """Return the most likely filtered-particle directory for a snapshot.

        Supports common layouts under ``hydro_dir_base`` such as:
        - ``snap_XXX``
        - ``XXX``
        - ``snapXXX``
        - ``snap_XXX`` with non-padded integer folder names
        """
        snap_i = int(snap)
        snap3 = f"{snap_i:03d}"
        candidates = [
            os.path.join(self.hydro_dir_base, f'snap_{snap3}'),
            os.path.join(self.hydro_dir_base, snap3),
            os.path.join(self.hydro_dir_base, f'snap{snap3}'),
            os.path.join(self.hydro_dir_base, str(snap_i)),
            os.path.join(self.hydro_dir_base, f'snap_{snap_i}'),
            os.path.join(self.hydro_dir_base, f'snap{snap_i}'),
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return candidates[0]

    def create_master(self, where, subset_type='plist', radius=None, max_parallel_snaps=2,
                      galaxies_per_task=4, use_job_deps=False, partition='INTEL_HASWELL',
                      cluster_setup_template=None, prefix=None, paramf='parameters_master.py', snaps_to_run=None):
        """Generate Powderday parameter files and batch scripts.

        Parameters
        ----------
        where : str
            ``'local'`` or ``'cluster'``.
        subset_type : str
            Particle subset method: ``'plist'`` (particle list) or
            ``'region'`` (spherical aperture).
        radius : float, optional
            Aperture radius (required when *subset_type* is ``'region'``).
        max_parallel_snaps : int, optional
            Maximum number of snapshot jobs to run concurrently on the cluster.
            Default: 2. Set to None for unlimited.
        galaxies_per_task : int, optional
            Number of galaxies to process per parallel task within a job.
            Default: 4. Increase for better parallelization, decrease to reduce
            per-task memory usage.
        use_job_deps : bool, optional
            If True, chain jobs with SLURM dependencies so snapshots run sequentially.
            If False (default), all jobs can run in parallel (up to max_parallel_snaps).
        partition : str, optional
            SLURM partition to submit jobs to. Default: 'INTEL_HASWELL'.
        cluster_setup_template : str, optional
            Path to the cluster setup shell script template. If None, defaults
            to ``simbanator/sed/cosmology_setup_all_cluster.cis.sh``.
        prefix : str, optional
            Optional filtered-file prefix used in snapshot naming (for example
            ``m100n1024``), generating names like
            ``snap_<prefix>_<snap3>_snap<snap>_gal<id>.h5``. If None, use
            default local names (``subset_snap...`` / ``region_snap...``).

        Notes
        -----
        This method only prepares files for later execution (e.g. via Slurm).
        It does not execute Powderday jobs during setup.
        """
        if where not in {'local', 'cluster'}:
            raise ValueError("where must be 'local' or 'cluster'")
        if subset_type not in {'plist', 'region'}:
            raise ValueError("subset_type must be 'plist' or 'region'")
        if subset_type == 'region' and radius is None:
            raise ValueError("radius is required when subset_type='region'")
        if where == 'cluster' and max_parallel_snaps is not None and max_parallel_snaps < 1:
            raise ValueError("max_parallel_snaps must be >= 1 or None")
        if prefix is not None:
            prefix = str(prefix).strip()
            if not prefix:
                raise ValueError("prefix must be a non-empty string when provided")

        filepath = os.path.join(self.output_dir, self.run_tag, 'target_selection', self.selection_file + '.h5')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Selection file not found: {filepath}")

        pkg_dir = os.path.dirname(__file__)
        paramfile = os.path.join(pkg_dir, paramf)
        if cluster_setup_template is None:
            cluster_setup_template = os.path.join(pkg_dir, 'cosmology_setup_all_cluster.cis.sh')
        if where == 'cluster' and not os.path.exists(cluster_setup_template):
            raise FileNotFoundError(
                f"Cluster setup template not found: {cluster_setup_template}"
            )
        cluster_job_files = []
        local_run_files = []

        with h5py.File(filepath, 'r') as hf:
            all_snaps = sorted([int(s[4:]) for s in hf.keys()])
            if snaps_to_run is not None:
                snaps = [s for s in all_snaps if s in np.atleast_1d(snaps_to_run)]
            else:
                snaps = all_snaps

            for snap in snaps:
                ids = hf[f'snap{snap:03}/galaxy_GroupID'][:]
                hidx = hf[f'snap{snap:03}/halo_GroupID'][:]
                pos = hf[f'snap{snap:03}/code_coods'][:]

                hydro_dir = self._resolve_hydro_dir_for_snap(snap)
                model_dir = os.path.join(self.model_dir_base, f'snap_{snap:03}')
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(os.path.join(model_dir, 'out'), exist_ok=True)
                os.makedirs(os.path.join(model_dir, 'err'), exist_ok=True)

                if not os.path.isdir(hydro_dir):
                    raise FileNotFoundError(
                        f"Filtered snapshot folder not found for snap {snap:03}. "
                        f"Looked under base: {self.hydro_dir_base}"
                    )

                redshift = self.sb.get_z_from_snap(snap)
                tcmb = 2.73 * (1. + redshift)
                N = len(ids)

                if N == 0:
                    continue

                for i, nh in enumerate(ids):
                    model_dir_remote = os.path.join(model_dir, f'gal_{nh}')
                    os.makedirs(model_dir_remote, exist_ok=True)
                    xpos, ypos, zpos = pos[i]

                    if where == 'cluster':
                        job_flag = 1 if i == 0 else 0
                        cmd = [
                            'bash',
                            cluster_setup_template,
                            str(self.nnodes),
                            model_dir,
                            hydro_dir,
                            self.model_run_name,
                            str(self.COSMOFLAG),
                            model_dir_remote,
                            hydro_dir,
                            str(xpos),
                            str(ypos),
                            str(zpos),
                            str(int(nh)),
                            str(int(snap)),
                            str(float(tcmb)),
                            str(i),
                            str(job_flag),
                            str(max(0, N - 1)),
                            str(int(hidx[i])),
                            '' if radius is None else str(radius),
                            subset_type,
                            '' if prefix is None else prefix,
                        ]
                        subprocess.run(cmd, check=True)
                        continue
                    model_file = os.path.join(model_dir, f'snap{snap:03}_{nh:06}.py')
                    with open(model_file, 'w', encoding='utf-8') as f:
                        f.write('#Snapshot Parameters\n')
                        f.write('snapshot_num = ' + str(int(snap)) + '\n')
                        f.write('galaxy_num = ' + str(int(nh)) + '\n\n')
                        
                        f.write("galaxy_num_str = '{:06d}'.format(int(galaxy_num))\n")
                        f.write("snapnum_str = '{:03d}'.format(int(snapshot_num))\n\n")


                        f.write(f"hydro_dir = '{hydro_dir}/'\n")
                        # if prefix is not None:
                        #     f.write(
                        #         "snapshot_name = f'snap_"
                        #         + prefix
                        #         + "_{snapnum_str}_snap{snapnum_str}_gal{galaxy_num_str}.h5'\n\n"
                        #     )
                        # elif subset_type == 'plist':
                        #     f.write("snapshot_name = f'subset_snap{snapnum_str}_gal'+galaxy_num_str+'.h5'\n\n")
                        # else:
                        #     f.write(
                        #         "snapshot_name = f'region_snap{snapnum_str}_r"
                        #         + str(radius)
                        #         + "_gal'+galaxy_num_str+'.h5'\n\n"
                        #     )

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
                    qsubfile = os.path.join(model_dir, f'master.snap{snap:03}.job')
                    if os.path.exists(qsubfile):
                        cluster_job_files.append(qsubfile)
                else:
                    runfile = os.path.join(model_dir, 'run_local.sh')
                    with open(runfile, 'w', encoding='utf-8') as f:
                        f.write('#!/bin/bash\n')
                        f.write('set -e\n')
                        f.write('echo "Starting local Powderday batch..."\n')
                        f.write(f'for id in $(cat {os.path.join(model_dir, "ids.txt")}); do\n')
                        f.write(
                            '  python /home/lorenzong/powderday/pd_front_end.py . parameters_master '
                            + f'snap{snap:03}_$id > gal_$id/snap{snap:03}_$id.LOG\n'
                        )
                        f.write('done\n')
                        f.write('echo "Local batch complete"\n')
                    os.chmod(runfile, 0o755)
                    local_run_files.append(runfile)

                np.savetxt(os.path.join(model_dir, 'ids.txt'), ids, fmt='%i')
                copyfile(paramfile, os.path.join(model_dir, 'parameters_master.py'))
        if where == 'cluster' and cluster_job_files:
            submit_all = os.path.join(self.model_dir_base, 'submit_all_snaps.sh')
            with open(submit_all, 'w', encoding='utf-8') as f:
                f.write('#!/bin/bash\n')
                f.write('set -e\n')
                f.write('echo "Submitting snapshot Slurm jobs with rate limiting..."\n')
                f.write(f'# Max parallel snapshots: {max_parallel_snaps if max_parallel_snaps else "unlimited"}\n')
                f.write(f'# Using job dependencies: {use_job_deps}\n\n')
                f.write(f'JOB_NAME_PREFIX="{self.model_run_name}.snap"\n')
                f.write('POLL_SECONDS=10\n\n')
                f.write('submit_job() {\n')
                f.write('    local job_file=$1\n')
                f.write('    local job_dir=$(dirname "$job_file")\n')
                f.write('    local job_name=$(basename "$job_file")\n')
                f.write('    echo "Submitting $job_name from $job_dir"\n')
                f.write(f'    local job_id=$(cd "$job_dir" && sbatch -p {partition} "$job_name" | sed -n "s/^Submitted batch job //p")\n')
                f.write('    echo "  Job ID: $job_id"\n')
                f.write('}\n\n')
                
                if use_job_deps:
                    f.write('# Chain jobs so snapshots run sequentially\n')
                    f.write('PREV_JOB_ID=""\n\n')
                    for i, job_file in enumerate(cluster_job_files):
                        dep_str = ' --dependency=afterok:$PREV_JOB_ID' if i > 0 else ''
                        f.write(f'echo "Submitting {job_file}"\n')
                        f.write(
                            f'cd "{os.path.dirname(job_file)}" && '
                            f'JOB_ID=$(sbatch{dep_str} -p {partition} {os.path.basename(job_file)} | sed -n "s/^Submitted batch job //p")\n'
                            f'PREV_JOB_ID=$JOB_ID\n'
                            f'echo "  Job ID: $JOB_ID"\n\n'
                        )
                else:
                    if max_parallel_snaps is None:
                        f.write('# Submit all snapshot jobs without queue throttling\n')
                    else:
                        f.write(f'# Submit with max {max_parallel_snaps} concurrent snapshot jobs\n')
                        f.write(f'MAX_PARALLEL={max_parallel_snaps}\n\n')
                        f.write('wait_for_slot() {\n')
                        f.write('    local running_count\n')
                        f.write('    running_count=$(squeue -u "$USER" --states=R,PD -h -n "${JOB_NAME_PREFIX}*" 2>/dev/null | wc -l)\n')
                        f.write('    while [ "$running_count" -ge "$MAX_PARALLEL" ]; do\n')
                        f.write('        echo "Queue full for this run ($running_count >= $MAX_PARALLEL). Waiting ${POLL_SECONDS}s..."\n')
                        f.write('        sleep "$POLL_SECONDS"\n')
                        f.write('        running_count=$(squeue -u "$USER" --states=R,PD -h -n "${JOB_NAME_PREFIX}*" 2>/dev/null | wc -l)\n')
                        f.write('    done\n')
                        f.write('}\n\n')

                    f.write('# Submit jobs\n')
                    for job_file in cluster_job_files:
                        if max_parallel_snaps is not None:
                            f.write('wait_for_slot\n')
                        f.write(f'submit_job "{job_file}"\n')
                
                f.write('echo "All snapshot jobs submitted."\n')
                f.write('echo "Monitor with: squeue -u $USER"\n')
            os.chmod(submit_all, 0o755)


        if where == 'local' and local_run_files:
            run_all_local = os.path.join(self.model_dir_base, 'run_all_local.sh')
            with open(run_all_local, 'w', encoding='utf-8') as f:
                f.write('#!/bin/bash\n')
                f.write('set -e\n')
                f.write('echo "Running all local snapshot batches..."\n')
                for run_file in local_run_files:
                    f.write(f'echo "Running {run_file}"\n')
                    f.write(
                        f'cd "{os.path.dirname(run_file)}" && '
                        f'./{os.path.basename(run_file)}\n'
                    )
                f.write('echo "All local snapshot batches completed."\n')
            os.chmod(run_all_local, 0o755)

    def plotsed(self, snap, gal, show=False, ret=False, retval=False):
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
        retval: bool
            If True, return frequency and fluxes
        """
        fig, ax = plt.subplots()
        run = os.path.join(
            self.model_dir_base, f'snap_{snap:03}', f'gal_{gal:06}',
            f'snap{snap:03}.galaxy{gal:06}.rtout.sed',
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

        out = os.path.join(self.output_dir, self.run_tag, 'sed_plots', f'snap_{snap:03}')
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, f'gal_{gal:06}.png'), bbox_inches='tight')

        if show:
            plt.show()
        if ret:
            return fig, ax
        if retval: 
            return nu, flux


    def extract_flux_single(self, snap, gal, facility, instrument,
                     filters=None, wave_unit='micron', findx=0, redshift=False):
    
        run = os.path.join(
            self.model_dir_base,
            f'snap_{snap:03}',
            f'gal_{gal:06}',
            f'snap{snap:03}.galaxy{gal:06}.rtout.sed',
        )
    
        z = self.sb.get_z_from_snap(snap)
    
        # --- Check file exists ---
        if not os.path.isfile(run):
            warnings.warn(f"[Missing SED] snap={snap:03}, gal={gal:06} → {run}")
        
        # --- Units ---
        wav = np.asarray(wav) * u.micron 
        if redshift:
            wav = wav* (1. + z)
    
        flux = np.asarray(flux) * u.erg / u.s
        dl = Planck13.luminosity_distance(z).to(u.cm)
        flux /= (4. * np.pi * dl**2)
    
        nu = (constants.c.cgs / wav.to(u.cm)).to(u.Hz)
        flux = (flux / nu).to(u.mJy)
    
        flux = flux[findx]
    
        # --- Compute fluxes ---
        results = flux_extraction(
            facility,
            instrument,
            wav,
            flux,
            filters=filters,
            wave_unit=wave_unit
        )
    
        # --- Flatten ---
        rows = flatten_results(results, snap, gal)
    
        # --- Output path ---
        outdir = os.path.join(self.output_dir, self.run_tag, 'sed_fluxes', f'snap_{snap:03}')
        os.makedirs(outdir, exist_ok=True)
    
        filename = os.path.join(outdir, f'snap{snap:03}_gal_{gal:06}_fluxes.fits')
    
        # --- Save ---
        table = Table(rows)
        table.write(filename, overwrite=True)
    
        return filename


    
    def extract_flux_batch(self, snaps, gals, facility, instrument,
                      filters=None, wave_unit='micron',
                      findx=0, redshift=False, funyt='mJy', outname=None):

        snaps = np.asarray(snaps)
        gals = np.asarray(gals)
    
        assert len(snaps) == len(gals), "snaps and gals must match in length"
    
        all_tables = []
        xmean_rows = []
    
        unique_snaps = np.unique(snaps)

        # setup_filters for the entire batch
        profiles = get_svo_filters(
            facility,
            instrument,
            filters=filters,
            wave_unit=wave_unit
        )
        
        for snap in unique_snaps:
    
            mask = snaps == snap
            snap_gals = gals[mask]
    
            z = self.sb.get_z_from_snap(snap)
    
            rows = []
    
            for gal in snap_gals:
    
                run = os.path.join(
                    self.model_dir_base,
                    f'snap_{snap:03}',
                    f'gal_{gal}',
                    f'snap{snap:03}.galaxy{gal:06}.rtout.sed',
                )
                # --- Check file exists ---
                if not os.path.isfile(run):
                    warnings.warn(f"[Missing SED] snap={snap:03}, gal={gal:06} → {run}")
                    continue
                
                # --- Try reading SED ---
                try:
                    m = ModelOutput(run)
                    wav, flux = m.get_sed(inclination='all', aperture=-1)
                except Exception as e:
                    warnings.warn(f"[SED read error] snap={snap:03}, gal={gal:06} → {e}")
                    continue
    
                # --- Units ---
                wav = np.asarray(wav) * u.micron
                if redshift:
                    wav = wav * (1. + z)
    
                flux = np.asarray(flux) * u.erg / u.s
                dl = Planck13.luminosity_distance(z).to(u.cm)
                flux /= (4. * np.pi * dl**2)
    
                nu = (constants.c.cgs / wav.to(u.cm)).to(u.Hz)
                flux = (flux / nu).to(u.mJy)
    
                flux = flux[findx]
    
                # --- Flux extraction ---
                results = flux_extraction(
                    facility,
                    instrument,
                    wav,
                    flux,
                    filters=filters,
                    wave_unit=wave_unit,
                    filter_list=profiles
                )
    
                # --- Build row (one galaxy) ---
                row = {
                    'gal_id_at_snap': gal,
                    'snap': snap,
                    'redshift': z
                }

                for fac, inst_dict in results.items():
                    for inst, filt_dict in inst_dict.items():
                        for filt_name, filt_data in filt_dict.items():
                            colname = f"{fac}.{inst}.{filt_name}"
                            row[colname] = filt_data[funyt]
                
                            xmean_rows.append({
                                'gal_id_at_snap': gal,
                                'snap': snap,
                                'filter': colname,
                                'xmean': filt_data.get('xmean', np.nan)
                            })
    
                rows.append(row)
    
            table = Table(rows)
            all_tables.append(table)

    
        # --- Combine all snapshots ---
        final_table = vstack(all_tables)
    
        # --- Output paths ---
        outdir = os.path.join(self.output_dir, self.run_tag, 'sed_fluxes')
        os.makedirs(outdir, exist_ok=True)
        if outname==None:
            flux_file = os.path.join(outdir, 'all_galaxies_fluxes.fits')
        else: 
            flux_file = os.path.join(outdir, outname)
        xmean_file = os.path.join(outdir, 'all_xmean.fits')
    
        final_table.write(flux_file, overwrite=True)
    
        xmean_table = Table(xmean_rows)
        xmean_table.write(xmean_file, overwrite=True)
    
        return flux_file, xmean_file

        