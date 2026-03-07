"""Recover star-formation histories using FSPS stellar population models.

This module wraps `python-fsps <https://dfm.io/python-fsps/>`_ to invert
present-day stellar masses back to their formation masses, then bins the
result into a time-resolved SFH.

Requires the optional ``fsps`` dependency::

    pip install fsps

Typical usage::

    from simbanator.analysis.sfh_fsps import compute_sfh, bin_sfh

    # For a cosmological SIMBA snapshot:
    result = compute_sfh(snapshot, caesar_file, cosmological=True)
    times, sfr = bin_sfh(result, galaxy_id=0)
"""

import os
import pickle

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_sfh(snapshot, caesar_file=None, *,
                cosmological=True, arepo=False,
                galaxy_center=None, max_galaxies=None,
                n_workers=1):
    """Compute formation masses for star particles using FSPS.

    Parameters
    ----------
    snapshot : str
        Path to the simulation snapshot file (yt-readable).
    caesar_file : str, optional
        Path to the Caesar catalog (required if *cosmological* is True).
    cosmological : bool
        Whether the simulation is cosmological.
    arepo : bool
        Whether the simulation uses the AREPO code.
    galaxy_center : array-like, optional
        Galaxy centre in code units for idealised runs.
    max_galaxies : int, optional
        Limit the number of galaxies processed (useful for testing).
    n_workers : int
        Number of parallel workers (uses ``multiprocessing.Pool``).

    Returns
    -------
    dict
        ``{'id': list, 'massform': list[np.ndarray],
           'tform': list[np.ndarray]}``
        Each list element corresponds to one galaxy; each array element
        to a star particle.
    """
    import yt          # heavy – lazy
    import fsps         # heavy – lazy

    ds = yt.load(snapshot)

    # ----- AREPO star filter ------------------------------------------
    if arepo:
        def _newstars(pfilter, data):
            return data[(pfilter.filtered_type,
                         "GFM_StellarFormationTime")] > 0
        yt.add_particle_filter("newstars", function=_newstars,
                               filtered_type="PartType4")
        ds.add_particle_filter("newstars")

    # ----- Load ages, masses, metallicities ---------------------------
    if cosmological:
        import caesar  # heavy – lazy
        if caesar_file is None:
            raise ValueError("caesar_file is required for cosmological runs")
        obj = caesar.load(caesar_file)
        obj.yt_dataset = ds
        dd = obj.yt_dataset.all_data()

        if not arepo:
            scalefactor = dd[("PartType4", "StellarFormationTime")]
        else:
            scalefactor = dd[("newstars", "GFM_StellarFormationTime")].value

        formation_z = (1.0 / scalefactor) - 1.0
        yt_cosmo = yt.utilities.cosmology.Cosmology(
            hubble_constant=0.68, omega_lambda=0.7, omega_matter=0.3)
        stellar_formation_times = yt_cosmo.t_from_z(formation_z).in_units("Gyr")
        simtime = yt_cosmo.t_from_z(ds.current_redshift).in_units("Gyr")
        stellar_ages = (simtime - stellar_formation_times).in_units("Gyr")
    else:
        dd = ds.all_data()
        simtime = ds.current_time.in_units("Gyr").value
        if arepo:
            age = simtime - ds.arr(
                dd[("newstars", "GFM_StellarFormationTime")],
                "s*kpc/km").in_units("Gyr").value
        else:
            age = simtime - ds.arr(
                dd[("PartType4", "StellarFormationTime")],
                "Gyr").value
        age[age < 1.0e-3] = 1.0e-3
        stellar_ages = age

    star_type = "newstars" if arepo else "PartType4"
    stellar_masses = dd[(star_type, "Masses")]
    stellar_metals = dd[(star_type,
                         "GFM_Metallicity" if arepo else "metallicity")]

    # ----- FSPS SSP ---------------------------------------------------
    fsps_ssp = fsps.StellarPopulation(
        sfh=0, zcontinuous=1, imf_type=2,
        zred=0.0, add_dust_emission=False)
    solar_Z = 0.0142

    # ----- Per-galaxy SFH recovery ------------------------------------
    if cosmological:
        galaxy_ids = [g.GroupID for g in obj.galaxies]
        if max_galaxies is not None:
            galaxy_ids = galaxy_ids[:max_galaxies]

        def _get_sfh_for_galaxy(gal_idx):
            gal = obj.galaxies[gal_idx]
            ages_g = stellar_ages[gal.slist]
            masses_g = stellar_masses[gal.slist]
            metals_g = stellar_metals[gal.slist]
            return _recover_formation_masses(
                ages_g, masses_g, metals_g,
                fsps_ssp, solar_Z, float(simtime))

        if n_workers > 1:
            from multiprocessing import Pool
            with Pool(n_workers) as pool:
                results = pool.map(_get_sfh_for_galaxy, range(len(galaxy_ids)))
            formation_times = [r[0] for r in results]
            formation_masses = [r[1] for r in results]
        else:
            formation_times, formation_masses = [], []
            for idx in range(len(galaxy_ids)):
                t, m = _get_sfh_for_galaxy(idx)
                formation_times.append(t)
                formation_masses.append(m)
    else:
        galaxy_ids = [0]  # single galaxy
        t, m = _recover_formation_masses(
            stellar_ages, stellar_masses, stellar_metals,
            fsps_ssp, solar_Z,
            float(simtime) if np.ndim(simtime) == 0 else float(simtime.value))
        formation_times = [t]
        formation_masses = [m]

    return {
        "id": galaxy_ids,
        "massform": formation_masses,
        "tform": formation_times,
    }


def bin_sfh(sfh_result, galaxy_id=0, bin_width=3.0):
    """Bin recovered formation masses into a time-resolved SFR.

    Parameters
    ----------
    sfh_result : dict
        Output of :func:`compute_sfh` (or a pickle file path).
    galaxy_id : int
        Galaxy GroupID to extract.
    bin_width : float
        Bin width in Myr (default 3).

    Returns
    -------
    bin_centers : np.ndarray
        Bin centres in Myr.
    sfr : np.ndarray
        Star-formation rate in $M_\\odot\\,\\mathrm{yr}^{-1}$
        (mass formed per Myr in each bin, converted to per-year).
    """
    from scipy.stats import binned_statistic

    if isinstance(sfh_result, (str, os.PathLike)):
        import pandas as pd
        sfh_result = pd.read_pickle(sfh_result)

    ids = np.asarray(sfh_result["id"])
    idx = np.where(ids == galaxy_id)[0]
    if len(idx) == 0:
        raise KeyError(f"galaxy_id {galaxy_id} not found in SFH result "
                       f"(available: {ids.tolist()})")
    idx = idx[0]

    massform = np.asarray(sfh_result["massform"][idx], dtype=float)
    tform = np.asarray(sfh_result["tform"][idx], dtype=float) * 1000  # Gyr → Myr

    t_H = np.max(tform)
    bins = np.arange(100, t_H, bin_width)
    if len(bins) < 2:
        raise ValueError("Not enough time range to create bins")

    sfrs, edges, _ = binned_statistic(
        tform, massform, statistic="sum", bins=bins)
    sfrs[np.isnan(sfrs)] = 0.0

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers, sfrs


def save_sfh(sfh_result, path):
    """Save an SFH result dict to a pickle file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(sfh_result, f)


def load_sfh(path):
    """Load a previously saved SFH result dict."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _recover_formation_masses(ages, masses, metallicities,
                              fsps_ssp, solar_Z, simtime):
    """Use FSPS mass-remaining curve to invert present mass → formation mass."""
    formation_masses = []
    for age, metallicity, mass in zip(ages, metallicities, masses):
        mass_solar = float(mass.in_units("Msun") if hasattr(mass, "in_units")
                           else mass)
        Z_val = float(metallicity)
        if Z_val <= 0:
            Z_val = 1e-10
        fsps_ssp.params["logzsol"] = np.log10(Z_val / solar_Z)
        mass_remaining = fsps_ssp.stellar_mass
        age_val = float(age.value if hasattr(age, "value") else age)
        initial_mass = np.interp(
            np.log10(age_val * 1e9), fsps_ssp.ssp_ages, mass_remaining)
        formation_masses.append(mass_solar / initial_mass)

    formation_times = np.array(simtime - np.asarray(ages, dtype=float),
                               dtype=float)
    return formation_times, np.array(formation_masses)
