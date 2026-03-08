"""
Recover star-formation histories using FSPS stellar population models.

This module inverts present-day stellar masses to formation masses using
FSPS stellar evolution tracks and reconstructs a time-resolved SFH.

Typical usage
-------------

from sfh_fsps import compute_sfh, bin_sfh

sfh = compute_sfh(snapshot, caesar_file, cosmological=True)

time, sfr = bin_sfh(sfh, galaxy_id=0)

"""

import os
import pickle
import numpy as np
from tqdm.auto import tqdm


import yt
import fsps
from multiprocessing import Pool



# # ============================================================
# # FSPS MASS LOSS INVERSION
# # ============================================================

# def _formation_mass(age, metallicity, mass, fsps_ssp, solar_Z):
#     """Convert present stellar mass → formation mass using FSPS."""

#     mass = mass.in_units("Msun")

#     Z = max(float(metallicity), 1e-10)

#     fsps_ssp.params["logzsol"] = np.log10(Z / solar_Z)

#     mass_remaining = fsps_ssp.stellar_mass

#     initial_mass = np.interp(
#         np.log10(age * 1e9),
#         fsps_ssp.ssp_ages,
#         mass_remaining,
#     )

#     return mass / initial_mass


# def _recover_sfh_single(ages, masses, metals, simtime, fsps_ssp, solar_Z):
#     """Recover formation masses for a single galaxy."""

#     formation_masses = [
#         _formation_mass(a, z, m, fsps_ssp, solar_Z)
#         for a, z, m in zip(ages, metals, masses)
#     ]

#     formation_times = simtime - np.asarray(ages, dtype=float)

#     return np.asarray(formation_times), np.asarray(formation_masses)


# def _sfh_worker(args):

#     galaxy_index, slist, stellar_ages, stellar_masses, stellar_metals, solar_Z, simtime = args

#     ages = stellar_ages[slist]
#     masses = stellar_masses[slist]
#     metals = stellar_metals[slist]

#     formation_masses = []

#     fsps_ssp = fsps.StellarPopulation(zcontinuous=1)

#     for age, metallicity, mass in zip(ages, metals, masses):

#         mass = mass.in_units("Msun")

#         fsps_ssp.params["logzsol"] = np.log10(metallicity / solar_Z)

#         mass_remaining = fsps_ssp.stellar_mass

#         initial_mass = np.interp(
#             np.log10(age * 1e9),
#             fsps_ssp.ssp_ages,
#             mass_remaining,
#         )

#         formation_masses.append(mass / initial_mass)

#     formation_masses = np.array(formation_masses)
#     formation_times = np.array(simtime - ages, dtype=float)

#     return formation_times, formation_masses

# # ============================================================
# # MAIN SFH ROUTINE
# # ============================================================

# def compute_sfh(snapshot,
#                 caesar_file=None,
#                 cosmological=True,
#                 arepo=False,
#                 filtered=True,
#                 n_workers=8,
#                 max_galaxies=None):
#     """
#     Recover star formation histories from a simulation snapshot.

#     Parameters
#     ----------
#     snapshot : str
#         Path to yt-readable snapshot.
#     caesar_file : str
#         Path to CAESAR catalog (required if filtered=True).
#     cosmological : bool
#         Whether simulation uses cosmological scale factors.
#     arepo : bool
#         Use AREPO star particle naming.
#     filtered : bool
#         If True, compute SFH per CAESAR galaxy.
#         If False, compute SFH using all star particles.
#     n_workers : int
#         Number of multiprocessing workers.
#     max_galaxies : int
#         Optional limit for debugging.

#     Returns
#     -------
#     dict
#         {
#             "id": np.ndarray,
#             "tform": list of arrays (Gyr),
#             "massform": list of arrays (Msun)
#         }
#     """

#     import yt
#     import fsps
#     import numpy as np
#     from multiprocessing import Pool
#     from tqdm.auto import tqdm

#     if filtered and caesar_file is None:
#         raise ValueError("caesar_file required when filtered=True")

#     print("Loading snapshot")
#     ds = yt.load(snapshot)

#     # ------------------------------------------------------------
#     # AREPO star particle filter
#     # ------------------------------------------------------------

#     if arepo:

#         def _newstars(pfilter, data):
#             return data[(pfilter.filtered_type, "GFM_StellarFormationTime")] > 0

#         yt.add_particle_filter("newstars", function=_newstars, filtered_type="PartType4")
#         ds.add_particle_filter("newstars")

#     dd = ds.all_data()

#     # ------------------------------------------------------------
#     # Stellar ages
#     # ------------------------------------------------------------

#     if cosmological:

#         if not arepo:
#             scalefactor = dd[("PartType4", "StellarFormationTime")]
#         else:
#             scalefactor = dd[("newstars", "GFM_StellarFormationTime")].value

#         formation_z = (1.0 / scalefactor) - 1.0

#         yt_cosmo = yt.utilities.cosmology.Cosmology(
#             hubble_constant=0.68,
#             omega_lambda=0.7,
#             omega_matter=0.3,
#         )

#         stellar_formation_times = yt_cosmo.t_from_z(formation_z).in_units("Gyr")

#         simtime = yt_cosmo.t_from_z(ds.current_redshift).in_units("Gyr")

#         stellar_ages = (simtime - stellar_formation_times).in_units("Gyr")

#     else:

#         simtime = ds.current_time.in_units("Gyr").value

#         if arepo:

#             print("--------------------------------------------------")
#             print("WARNING: assuming stellar ages units = s*kpc/km")
#             print("--------------------------------------------------")

#             age = simtime - (
#                 ds.arr(dd[("newstars", "GFM_StellarFormationTime")], "s*kpc/km")
#                 .in_units("Gyr")
#             ).value

#         else:

#             age = simtime - (
#                 ds.arr(dd[("PartType4", "StellarFormationTime")], "Gyr")
#             ).value

#         age[age < 1e-3] = 1e-3

#         stellar_ages = age

#     # ------------------------------------------------------------
#     # Particle properties
#     # ------------------------------------------------------------

#     if not arepo:
#         stellar_masses = dd[("PartType4", "Masses")]
#         stellar_metals = dd[("PartType4", "metallicity")]
#     else:
#         stellar_masses = dd[("newstars", "Masses")]
#         stellar_metals = dd[("newstars", "GFM_Metallicity")]

#     # ------------------------------------------------------------
#     # FSPS model
#     # ------------------------------------------------------------

#     print("Loading FSPS")

#     fsps_ssp = fsps.StellarPopulation(
#         sfh=0,
#         zcontinuous=1,
#         imf_type=2,
#         zred=0.0,
#         add_dust_emission=False,
#     )

#     solar_Z = 0.0142

#     print(f"Simulation time: {float(simtime):.2f} Gyr")

#     # ------------------------------------------------------------
#     # Helper: compute SFH for one galaxy
#     # ------------------------------------------------------------

#     def get_sfh_particle_subset(stellar_ages, stellar_masses, stellar_metals):

#         formation_masses = []

#         for age, metallicity, mass in zip(
#             stellar_ages, stellar_metals, stellar_masses
#         ):

#             mass = mass.in_units("Msun")

#             fsps_ssp.params["logzsol"] = np.log10(metallicity / solar_Z)

#             mass_remaining = fsps_ssp.stellar_mass

#             initial_mass = np.interp(
#                 np.log10(age * 1e9),
#                 fsps_ssp.ssp_ages,
#                 mass_remaining,
#             )

#             massform = mass / initial_mass

#             formation_masses.append(massform)

#         formation_masses = np.array(formation_masses)

#         formation_times = np.array(simtime - stellar_ages, dtype=float)

#         return formation_times, formation_masses

#     # ------------------------------------------------------------
#     # CASE 1 — filtered (galaxy SFHs)
#     # ------------------------------------------------------------

#     if not filtered:
#         import caesar

#         print("Loading CAESAR catalog")
#         obj = caesar.load(caesar_file)

#         ids = [g.GroupID for g in obj.galaxies]

#         if max_galaxies:
#             ids = ids[:max_galaxies]

#         def get_sfh(galaxy_index):

#             gal = obj.galaxies[ids[galaxy_index]]

#             slist = gal.slist

#             ages = stellar_ages[slist]
#             masses = stellar_masses[slist]
#             metals = stellar_metals[slist]

#             return get_sfh_particle_subset(ages, masses, metals)

#         print("Computing galaxy SFHs")

#         galaxy_slists = [obj.galaxies[i].slist for i in ids]

#         args_list = [
#             (
#                 i,
#                 galaxy_slists[i],
#                 stellar_ages,
#                 stellar_masses,
#                 stellar_metals,
#                 solar_Z,
#                 float(simtime),
#             )
#             for i in range(len(ids))
#         ]

#         with Pool(n_workers) as p:

#             results = list(
#                 tqdm(
#                     p.imap(_sfh_worker, args_list),
#                     total=len(args_list),
#                 )
#             )

#         tforms, mforms = zip(*results)

#     # ------------------------------------------------------------
#     # CASE 2 — unfiltered (all particles)
#     # ------------------------------------------------------------

#     else:

#         print("Computing SFH using all star particles")

#         tform, massform = get_sfh_particle_subset(
#             stellar_ages,
#             stellar_masses,
#             stellar_metals,
#         )

#         ids = np.array([0])
#         tforms = [tform]
#         mforms = [massform]

#     # ------------------------------------------------------------
#     # Return structure
#     # ------------------------------------------------------------

#     return {
#         "id": np.array(ids),
#         "tform": list(tforms),
#         "massform": list(mforms),
#     }


# # ============================================================
# # SFH BINNING
# # ============================================================

# def bin_sfh(sfh_result, galaxy_id=0, bin_width=50):
#     """
#     Convert formation events into an SFR(t).

#     Parameters
#     ----------
#     sfh_result : dict or path
#     galaxy_id : int
#         CAESAR GroupID
#     bin_width : float
#         Myr

#     Returns
#     -------
#     time : Myr
#     sfr : Msun/yr
#     """

#     if isinstance(sfh_result, (str, os.PathLike)):
#         with open(sfh_result, "rb") as f:
#             sfh_result = pickle.load(f)

#     ids = np.asarray(sfh_result["id"])

#     idx = np.where(ids == galaxy_id)[0]

#     if len(idx) == 0:
#         raise ValueError(f"Galaxy {galaxy_id} not found")

#     idx = idx[0]

#     massform = np.asarray(sfh_result["massform"][idx])
#     tform = np.asarray(sfh_result["tform"][idx]) * 1000   # Gyr → Myr

#     tmin = tform.min()
#     tmax = tform.max()

#     bins = np.arange(tmin, tmax + bin_width, bin_width)

#     mass_hist, edges = np.histogram(
#         tform,
#         bins=bins,
#         weights=massform
#     )

#     dt = bin_width * 1e6

#     sfr = mass_hist / dt

#     time = 0.5 * (edges[:-1] + edges[1:])

#     return time, sfr


# # ============================================================
# # I/O HELPERS
# # ============================================================

# def save_sfh(result, path):

#     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

#     with open(path, "wb") as f:
#         pickle.dump(result, f)


# def load_sfh(path):

#     with open(path, "rb") as f:
#         return pickle.load(f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def compute_sfh(snapshot, csfile, FILTERED=True, COSMOLOGICAL=False, AREPO=False, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'output', 'fsps_sfh')
        
    os.makedirs(output_dir, exist_ok=True)
    outfile=os.path.join(output_dir, snapshot.split('/')[-1].split('.')[0]+'_sfh.pkl')
        
    print('Loading yt snapshot')
    ds = yt.load(snapshot)


    if AREPO: 
        def _newstars(pfilter,data):
            filter = data[(pfilter.filtered_type, "GFM_StellarFormationTime")] > 0
            return filter
            
        yt.add_particle_filter("newstars",function=_newstars,filtered_type='PartType4')
        ds.add_particle_filter("newstars")

    if COSMOLOGICAL:
        if FILTERED == False: 
            print('Quick loading caesar')
            obj = caesar.load(csfile)
            obj.yt_dataset = ds
            dd = obj.yt_dataset.all_data()
            print('Loading caesar-selected particle data')
        else:
            print('Loading ALL FILTERED particle data')
            dd = ds.all_data()
            
        if AREPO == False: 
            scalefactor = dd[("PartType4", "StellarFormationTime")]
        else:
            scalefactor = data[("newstars","GFM_StellarFormationTime")].value
        # Compute the age of all the star particles from the provided scale factor at creation
        formation_z = (1.0 / scalefactor) - 1.0
        yt_cosmo = yt.utilities.cosmology.Cosmology(hubble_constant=0.68, omega_lambda = 0.7, omega_matter = 0.3)
        stellar_formation_times = yt_cosmo.t_from_z(formation_z).in_units("Gyr")
        
        # Age of the universe right now
        simtime = yt_cosmo.t_from_z(ds.current_redshift).in_units("Gyr")
        stellar_ages = (simtime - stellar_formation_times).in_units("Gyr")
        
    else:
        dd = ds.all_data()
        #code ripped from powderday arepo and gadget front ends
        simtime = ds.current_time.in_units('Gyr')
        simtime = simtime.value
        if AREPO:
            print("------------------------------------------------------------------")
            print("WARNING WARNING WARNING:")
            print("Assuming units in stellar ages are s*kpc/km")
            print("if this is not true - please edit right under this warning message")
            print("------------------------------------------------------------------")
            age = simtime-(ds.arr(dd[("newstars","GFM_StellarFormationTime")],'s*kpc/km').in_units('Gyr')).value
            # make the minimum age 1 million years
            age[np.where(age < 1.e-3)[0]] = 1.e-3
            stellar_ages = age
        else:
            age = simtime-ds.arr(dd[('PartType4', 'StellarFormationTime')],'Gyr').value
            # make the minimum age 1 million years
            age[np.where(age < 1.e-3)[0]] = 1.e-3
            stellar_ages = age
        

    if AREPO == False:
        stellar_masses = dd[("PartType4", "Masses")]
        stellar_metals = dd[("PartType4", 'metallicity')]
    else:
        stellar_masses = dd[("newstars", "Masses")]
        stellar_metals = dd[('newstars', 'GFM_Metallicity')]

    print('Loading fsps')
    fsps_ssp = fsps.StellarPopulation(sfh=0,
                    zcontinuous=1,
                    imf_type=2,
                    zred=0.0, add_dust_emission=False)
    solar_Z = 0.0142

    print(f'simtime: {simtime:.1f}')
    x = 0
    final_massfrac, final_formation_times, final_formation_masses = [], [], []

    ids = []
    if COSMOLOGICAL:
        #get the galaxies from the caesar file
        for i in obj.galaxies:
            ids.append(i.GroupID)

        def get_sfh(galaxy):
            this_galaxy_stellar_ages = stellar_ages[obj.galaxies[ids[galaxy]].slist]
            this_galaxy_stellar_masses = stellar_masses[obj.galaxies[ids[galaxy]].slist]
            this_galaxy_stellar_metals = stellar_metals[obj.galaxies[ids[galaxy]].slist]
            this_galaxy_formation_masses = []
            for age, metallicity, mass in zip(this_galaxy_stellar_ages, this_galaxy_stellar_metals, this_galaxy_stellar_masses):
                mass = mass.in_units('Msun')
                fsps_ssp.params['logzsol'] = np.log10(metallicity/solar_Z)
                mass_remaining = fsps_ssp.stellar_mass
                initial_mass = np.interp(np.log10(age*1e9), fsps_ssp.ssp_ages, mass_remaining)
                massform = mass / initial_mass
                this_galaxy_formation_masses.append(massform)
            this_galaxy_formation_masses = np.array(this_galaxy_formation_masses)
            this_galaxy_formation_times = np.array(simtime - this_galaxy_stellar_ages, dtype=float)
            return this_galaxy_formation_times, this_galaxy_formation_masses

        with Pool(16) as p:
            out1, out2 = zip(*tqdm(p.imap(get_sfh, range(len(ids))), total=len(ids)))
            final_formation_times = out1
            final_formation_masses = out2

    elif COSMOLOGICAL and FILTERED:
        #get the galaxies from the caesar file
        for i in obj.galaxies:
            ids.append(i.GroupID)

        def get_sfh(galaxy):
            this_galaxy_stellar_ages = stellar_ages
            this_galaxy_stellar_masses = stellar_masses
            this_galaxy_stellar_metals = stellar_metals
            this_galaxy_formation_masses = []
            for age, metallicity, mass in zip(this_galaxy_stellar_ages, this_galaxy_stellar_metals, this_galaxy_stellar_masses):
                mass = mass.in_units('Msun')
                fsps_ssp.params['logzsol'] = np.log10(metallicity/solar_Z)
                mass_remaining = fsps_ssp.stellar_mass
                initial_mass = np.interp(np.log10(age*1e9), fsps_ssp.ssp_ages, mass_remaining)
                massform = mass / initial_mass
                this_galaxy_formation_masses.append(massform)
            this_galaxy_formation_masses = np.array(this_galaxy_formation_masses)
            this_galaxy_formation_times = np.array(simtime - this_galaxy_stellar_ages, dtype=float)
            return this_galaxy_formation_times, this_galaxy_formation_masses

        with Pool(16) as p:
            out1, out2 = zip(*tqdm(p.imap(get_sfh, range(len(ids))), total=len(ids)))
            final_formation_times = out1
            final_formation_masses = out2

    else:

        this_galaxy_stellar_ages = stellar_ages
        this_galaxy_stellar_masses = stellar_masses
        this_galaxy_stellar_metals = stellar_metals
        this_galaxy_formation_masses = []


        for i in tqdm(range(len(this_galaxy_stellar_ages))):
            age = this_galaxy_stellar_ages[i]
            metallicity = this_galaxy_stellar_metals[i]
            mass = this_galaxy_stellar_masses[i]
        #for age, metallicity, mass in tqdm(zip(this_galaxy_stellar_ages, this_galaxy_stellar_metals, this_galaxy_stellar_masses)):
            mass = mass.in_units('Msun')
            fsps_ssp.params['logzsol'] = np.log10(metallicity/solar_Z)
            mass_remaining = fsps_ssp.stellar_mass
            initial_mass = np.interp(np.log10(age*1e9), fsps_ssp.ssp_ages, mass_remaining)
            massform = mass / initial_mass
            this_galaxy_formation_masses.append(massform)
        this_galaxy_formation_masses = np.array(this_galaxy_formation_masses)
        this_galaxy_formation_times = np.array(simtime - this_galaxy_stellar_ages, dtype=float)
        final_formation_masses = this_galaxy_formation_masses
        final_formation_times = this_galaxy_formation_times

    with open(outfile, 'wb') as f:
        pickle.dump({
            'id': ids,
            'massform': final_formation_masses,
            'tform': final_formation_times,
        }, f)
    return {
        'id': ids,
        'massform': final_formation_masses,
        'tform': final_formation_times,
    }
    
    
    
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
matplotlib.rcParams.update({
    "savefig.facecolor": "w",
    "figure.facecolor" : 'w',
    "figure.figsize" : (10,8),
    "text.color": "k",
    "legend.fontsize" : 20,
    "font.size" : 30,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "axes.linewidth": 3,
    "xtick.color": "k",
    "ytick.color": "k",
    "xtick.labelsize" : 25,
    "ytick.labelsize" : 25,
    "ytick.major.size" : 12,
    "xtick.major.size" : 12,
    "ytick.major.width" : 2,
    "xtick.major.width" : 2,
    "font.family": 'STIXGeneral',
    "mathtext.fontset" : "cm"
})

binwidth = 3 #Myr


#this function is given to scipy.stats.binned_statistics and acts on the particle masses per bin to give total(Msun) / timebin
def get_massform(massform):
        return np.sum(massform) / (binwidth * 1e6)

def get_galaxy_SFH(file_, galaxy_id, save_plot=False, plot_path=None):

    dat = pd.read_pickle(file_)
    # Handle both single and multiple galaxy outputs
    ids = dat['id']
    massforms = dat['massform']
    tforms = dat['tform']
    # If ids is not a list/array, treat as single galaxy
    print(f"IDs found in data: {ids}")
    if isinstance(ids, (int, float, np.integer, np.floating)) or (isinstance(ids, list) and len(ids) == 1) or len(ids) == 0:
        print(f"Single galaxy found." )
        massform = np.array(massforms)
        tform = np.array(tforms) * 1000
    else:
        print(f"Multiple galaxies found. Extracting data for galaxy_id={galaxy_id}" )
        idx = np.where(np.asarray(ids) == galaxy_id)[0][0]
        massform = np.array(massforms[idx])
        tform = np.array(tforms[idx]) * 1000
    t_H = np.max(tform)
    bins = np.arange(100, t_H, binwidth)
    sfrs, bins, binnumber = scipy.stats.binned_statistic(tform, massform, statistic=get_massform, bins=bins)
    sfrs[np.isnan(sfrs)] = 0
    bincenters = 0.5 * (bins[:-1] + bins[1:])
    sfh = sfrs
    if save_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,8))
        plt.plot(bincenters, sfh, color='k')
        plt.ylabel('SFR [M$_{\odot}$/yr]')
        plt.xlabel('t$_H$ [Myr]')
        plt.grid(True)
        if plot_path is None:
            # Default plot path: same as file_ but .png
            plot_path = file_.replace('.pkl', f'_gal{galaxy_id}_sfh.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"SFH plot saved to {plot_path}")
    return bincenters, sfh
