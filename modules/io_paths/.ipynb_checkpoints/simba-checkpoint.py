import glob
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from astropy import constants

import caesar


class Simba:
    def __init__(self, machine='PC39BP4', size=25):
        self.snaps = np.array([str(s).zfill(3) for s in np.arange(6, 152)[::-1]])
        if machine == 'cis':
            if size == 100:
                self.sim_directory = '/mnt/home/glorenzon/SIMBA_catalogs/simba100_snaps/'
                self.name_form = 'm100n1024_%03d.hdf5'
            elif size == '50_full':
                self.sim_directory = '/mnt/home/glorenzon/SIMBA_catalogs/simba50_snaps/full/'
                self.name_form = 'm50n512_%03d.hdf5'
            elif size == '50_noagn':
                self.sim_directory = '/mnt/home/glorenzon/SIMBA_catalogs/simba50_snaps/noagn/'
                self.name_form = 'm50n512_%03d.hdf5'
            else:
                print('Invalid box-size for SIMBA simulation: redirected to high-res')
                self.sim_directory = '/media/lorenzong/Data1/simba_hig_res/'
                self.name_form = 'm25n512_%03dd.hdf5'
            self.output_file = '/mnt/home/glorenzon/simbanator/analize_simba_cgm/output/txt/convert_snap_z_factors/zsnap_map_caesar_box100.txt'
        elif machine == 'PC39BP4':
            if size == 25:
                self.sim_directory = '/media/lorenzong/Data1/simba_hig_res/'
                self.name_form = 'm25n512_%03d.hdf5'
            elif size == 50:
                print('No mid-resolution repository for now: redirected to high-res')
                self.sim_directory = '/media/lorenzong/Data1/simba_hig_res/'
                self.name_form = 'm50n512_%03d.hdf5'
            elif size == 100:
                self.sim_directory = '/media/lorenzong/Data1/SIMBA_catalogs/'
                self.name_form = 'm100n1024_%03d.hdf5'
            else:
                print('Invalid box-size for SIMBA simulation: redirected to high-res')
                self.sim_directory = '/media/lorenzong/Data1/simba_hig_res/'
                self.name_form = 'm25n512_%03d.hdf5'
            self.output_file = '/home/lorenzong/analize_simba_cgm/output/txt/convert_snap_z_factors/zsnap_map_caesar_box100.txt'
        else:
            print("`machine` not recognised, set to default (PC39BP4)")
            self.sim_directory = '/media/lorenzong/Data1/simba_hig_res/'
            self.output_file = '/home/lorenzong/analize_simba_cgm/output/txt/convert_snap_z_factors/zsnap_map_caesar_box100.txt'

        self.cs_directory = self.sim_directory+'/Groups/'
        self.cosmo = cosmo

        outs = np.loadtxt(self.output_file)
        self.zeds = np.array([1. / outs[int(snap)] - 1 for snap in self.snaps])

        self.filters = [
            'GALEX_FUV', 'GALEX_NUV', 'SDSS_u', 'SDSS_g', 'SDSS_r',
            'SDSS_i', 'SDSS_z', '2MASS_H', '2MASS_J', '2MASS_Ks',
            'PS1_y', 'PS1_z',
            'WISE_RSR_W1', 'WISE_RSR_W2',
            'WISE_RSR_W3', 'WISE_RSR_W4',
            'SPITZER_IRAC_36', 'SPITZER_IRAC_45',
            'SPITZER_IRAC_58', 'SPITZER_IRAC_80',
            'HERSCHEL_PACS_BLUE', 'HERSCHEL_PACS_GREEN',
            'HERSCHEL_PACS_RED',
            'HERSCHEL_SPIRE_PSW', 'HERSCHEL_SPIRE_PMW',
            'HERSCHEL_SPIRE_PLW',
            'JCMT_450', 'JCMT_850'
        ]

        self.filters_pretty = [
            'GALEX FUV', 'GALEX NUV', 'SDSS u', 'SDSS g', 'SDSS r',
            'SDSS i', 'SDSS z', '2MASS H', '2MASS J', '2MASS Ks',
            'PS1 y', 'PS1 z',
            'WISE 1', 'WISE 2',
            'WISE 3', 'WISE 4',
            'IRAC $3.6\mathrm{\mu m}$', 'IRAC $4.5\mathrm{\mu m}$',
            'IRAC $5.8\mathrm{\mu m}$', 'IRAC$ 8 \mathrm{\mu m}$',
            'HERSCHEL PACS BLUE', 'HERSCHEL PACS GREEN',
            'HERSCHEL PACS RED',
            'HERSCHEL SPIRE PSW', 'HERSCHEL SPIRE PMW',
            'HERSCHEL SPIRE PLW',
            'JCMT $450\mathrm{\mu m}$', 'JCMT $850\mathrm{\mu m}$'
        ]

    def get_sim_file(self, snap, snap_str=None):
        if snap_str is None:
            snap_str = 'snap_' + self.name_form
        return self.sim_directory + (snap_str % snap)

    def get_caesar_file(self, snap, fname=None, verbose=False):
        if fname is None:
            fname = self.name_form
        fname = self.cs_directory + (fname % snap)
        return fname

    def get_caesar(self, snap, fname=None, verbose=False):
        if fname is None:
            fname = self.name_form
        fname = self.cs_directory + (fname % snap)
        return caesar.load(fname)

    def get_redshifts(self):
        """
        Get the redshift values for each snapshot.

        Returns:
        np.array: Array of redshift values.
        """
        return self.zeds

    def get_z_from_snap(self, snap):
        """
        Get the redshift value for a given snapshot.

        Returns:
        float: the redshift of the snapshot
        """
        outs = np.loadtxt(self.output_file)
        return 1. / outs[int(snap)] - 1
