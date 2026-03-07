"""Powderday SED generation wrapper.

Requires optional heavy dependencies: ``hyperion``, ``caesar``.
Install with ``pip install simbanator[full]``.
"""

import sys
import os
import numpy as np
import h5py
from subprocess import call
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

from ..io.paths import SavePaths


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
                 preselect, selection_file, COSMOFLAG=0):
        self.sb = sb
        self.nnodes = nnodes
        self.model_run_name = model_run_name
        self.COSMOFLAG = COSMOFLAG
        self.hydro_dir_base = hydro_dir_base
        self.preselect = preselect
        self.selection_file = selection_file

        paths = SavePaths()
        self.hydro_outputfile = self.sb.output_file
        output_dir = paths.get_filetype_path('hdf5')
        self.model_dir_base = paths.create_subdir(output_dir, 'powderday_sed_out')

    def selection_gals(self, snaps, galaxyID):
        """Write galaxy selection info to an HDF5 file.

        Parameters
        ----------
        snaps : list of int
            Snapshot numbers.
        galaxyID : array-like of int
            Galaxy GroupIDs to select.
        """
        paths = SavePaths()
        output_dir = paths.create_subdir(
            paths.get_filetype_path('hdf5'), 'target_selection_for_SED'
        )
        filepath = os.path.join(output_dir, self.selection_file + '.h5')

        if not os.path.exists(filepath):
            with h5py.File(filepath, 'w') as hf:
                pass

        data_dict = defaultdict(lambda: {
            'galaxy_GroupID': [], 'halo_GroupID': [], 'code_coods': []
        })

        for snap in snaps:
            cs = self.sb.get_caesar(snap)
            coods_code = np.array([g.pos.in_units('code_length').value for g in cs.galaxies])
            hidx = np.array([g.parent_halo_index for g in cs.galaxies])
            data_dict[snap]['galaxy_GroupID'].extend(galaxyID)
            data_dict[snap]['halo_GroupID'].extend(hidx[galaxyID])
            data_dict[snap]['code_coods'].extend(coods_code[galaxyID])

        with h5py.File(filepath, 'a') as hf:
            for snap, data in data_dict.items():
                grp = f'snap{snap:03}'
                hf.create_group(grp)
                hf.create_dataset(f'{grp}/galaxy_GroupID', data=np.array(data['galaxy_GroupID']))
                hf.create_dataset(f'{grp}/halo_GroupID', data=np.array(data['halo_GroupID']))
                hf.create_dataset(f'{grp}/code_coods', data=np.array(data['code_coods']))
                print(f'Written data for {grp}')

    def create_master(self, where):
        """Generate Powderday parameter files and job scripts.

        Parameters
        ----------
        where : str
            ``'local'`` or ``'cluster'``.
        """
        paths = SavePaths()
        output_dir = paths.create_subdir(
            paths.get_filetype_path('hdf5'), 'target_selection_for_SED'
        )
        filepath = os.path.join(output_dir, self.selection_file + '.h5')

        # Locate shell scripts relative to *this* file
        pkg_dir = os.path.dirname(__file__)
        if where == 'local':
            setupfile = os.path.join(pkg_dir, 'cosmology_setup_all_local.pc39.sh')
        elif where == 'cluster':
            setupfile = os.path.join(pkg_dir, 'cosmology_setup_all_cluster.cis.sh')
        else:
            raise ValueError("where must be 'local' or 'cluster'")

        paramfile = os.path.join(pkg_dir, 'parameters_master.py')

        with h5py.File(filepath, 'r') as hf:
            snaps = sorted([int(s[4:]) for s in hf.keys()])
            scalefactor = np.loadtxt(self.hydro_outputfile)

            for n, snap in enumerate(snaps):
                ids = hf[f'snap{snap:03}/galaxy_GroupID'][:]
                hidx = hf[f'snap{snap:03}/halo_GroupID'][:]
                pos = hf[f'snap{snap:03}/code_coods'][:]

                hydro_dir = os.path.join(self.hydro_dir_base, f'snap_{snap:03}')
                model_dir = os.path.join(self.model_dir_base, f'snap_{snap:03}')
                os.makedirs(hydro_dir, exist_ok=True)
                os.makedirs(model_dir, exist_ok=True)

                redshift = (1. / scalefactor[n]) - 1.
                tcmb = 2.73 * (1. + redshift)
                N = len(ids)

                for i, nh in enumerate(ids):
                    job_flag = 1 if i == 0 else 0
                    model_dir_remote = os.path.join(model_dir, f'gal_{nh}')
                    os.makedirs(model_dir_remote, exist_ok=True)
                    xpos, ypos, zpos = pos[i]

                    cmd = (
                        f"{setupfile} {self.nnodes} {model_dir} {hydro_dir} "
                        f"{self.model_run_name} {self.COSMOFLAG} "
                        f"{model_dir_remote} {hydro_dir} {xpos} {ypos} {zpos} "
                        f"{nh} {int(snap)} {tcmb} {i} {job_flag} {N - 1} {hidx[i]}"
                    )
                    call(cmd, shell=True)

                np.savetxt(os.path.join(model_dir, 'ids.txt'), ids, fmt='%i')
                copyfile(paramfile, os.path.join(model_dir, 'parameters_master.py'))

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

        paths = SavePaths()
        out = paths.create_subdir(
            paths.create_subdir(paths.get_filetype_path('plot'), 'sed_out'),
            f'snap_{snap}',
        )
        fig.savefig(os.path.join(out, f'gal_{gal}.png'), bbox_inches='tight')

        if show:
            plt.show()
        if ret:
            return fig, ax
