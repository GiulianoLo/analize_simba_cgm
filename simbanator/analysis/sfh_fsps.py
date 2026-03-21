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
import numpy as np

import yt
import numpy as np
import fsps
import os
import pickle
import time
from multiprocessing import Pool
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

try:
    import yt
    import yt.utilities.cosmology
except ImportError:
    raise ImportError("yt is required.")

try:
    import caesar
except ImportError:
    caesar = None  # Only needed for FILTERED=False


# ---------------------------------------------------------
# Physical constants / floors
# ---------------------------------------------------------
SOLAR_Z      = 0.0142
MIN_AGE_GYR  = 1e-3   # 1 Myr — minimum stellar age
MIN_Z        = 1e-4   # metallicity floor to avoid log(0)
MIN_FRAC     = 1e-6   # mass fraction floor to avoid div/0


# ---------------------------------------------------------
# Utility: strip yt/unyt units → plain numpy float64
# ---------------------------------------------------------
def _to_numpy(arr):
    """Strip yt/unyt quantities and return a plain numpy float64 array."""
    if hasattr(arr, 'value'):
        return np.asarray(arr.value, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


# ---------------------------------------------------------
# Build FSPS stellar mass fraction grid
# ---------------------------------------------------------
def build_fsps_grid(stellar_metals, n_metals=16, n_ages=64):
    """
    Build a 2D interpolation grid of FSPS remaining stellar mass
    fractions as a function of metallicity and log10(age/yr).

    Parameters
    ----------
    stellar_metals : np.ndarray
        Absolute metallicity values of all stellar particles.
    n_metals : int
        Number of metallicity grid points.
    n_ages : int
        Number of log-age grid points.

    Returns
    -------
    metal_grid    : np.ndarray  shape (n_metals,)
    log_age_grid  : np.ndarray  shape (n_ages,)
    mass_table    : np.ndarray  shape (n_metals, n_ages)
    """
    clipped = np.clip(stellar_metals, MIN_Z, None)

    zmin = clipped.min()
    zmax = clipped.max()

    # Guard against a degenerate single-metallicity snapshot
    if np.isclose(zmin, zmax):
        zmin *= 0.5
        zmax *= 2.0

    metal_grid   = np.logspace(np.log10(zmin), np.log10(zmax), n_metals)
    log_age_grid = np.linspace(6.0, 10.2, n_ages)   # log10(yr)

    sp = fsps.StellarPopulation(
        compute_vega_mags=False,
        zcontinuous=1,
        imf_type=1
    )

    mass_table = np.zeros((n_metals, n_ages))

    for i, z in enumerate(tqdm(metal_grid, desc="  FSPS metallicity loop")):
        sp.params["logzsol"] = np.log10(z / SOLAR_Z)
        for j, log_age in enumerate(log_age_grid):
            sp.params["tage"] = 10.0**log_age / 1e9
            mass_table[i, j] = max(float(sp.stellar_mass), MIN_FRAC)

    # Clip entire table for safety
    mass_table = np.clip(mass_table, MIN_FRAC, None)

    return metal_grid, log_age_grid, mass_table


# ---------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------
def sfh_worker(args):
    """
    Compute formation times and initial masses for a chunk of
    stellar particles using an FSPS interpolation grid.

    Parameters
    ----------
    args : tuple
        (chunk_indices, stellar_ages, stellar_masses,
         stellar_metals, simtime,
         metal_grid, log_age_grid, mass_table)

    Returns
    -------
    chunk_indices    : np.ndarray
    formation_times  : np.ndarray  (Gyr)
    formation_masses : np.ndarray  (Msun)
    """
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np

    (
        chunk_indices,
        stellar_ages,
        stellar_masses,
        stellar_metals,
        simtime,
        metal_grid,
        log_age_grid,
        mass_table
    ) = args

    interp = RegularGridInterpolator(
        (metal_grid, log_age_grid),
        mass_table,
        method='linear',
        bounds_error=False,
        fill_value=None     # extrapolate at grid edges
    )

    n = len(chunk_indices)
    formation_times  = np.empty(n)
    formation_masses = np.empty(n)

    for i, idx in enumerate(chunk_indices):

        # Clamp to physical range before interpolation
        age_gyr = max(float(stellar_ages[idx]),  MIN_AGE_GYR)
        metal   = max(float(stellar_metals[idx]), MIN_Z)
        mass    = float(stellar_masses[idx])

        log_age = np.log10(age_gyr * 1e9)   # Gyr → yr → log10

        frac = float(interp((metal, log_age)))

        # Enforce physical mass fraction — prevents negatives entirely
        if not np.isfinite(frac) or frac <= 0.0:
            frac = MIN_FRAC

        formation_masses[i] = mass / frac
        formation_times[i]  = simtime - age_gyr

    return chunk_indices, formation_times, formation_masses


# ---------------------------------------------------------
# Snapshot loader  (isolated, testable)
# ---------------------------------------------------------
def _load_snapshot_data(snapshot, COSMOLOGICAL, AREPO, FILTERED, csfile):
    """
    Load all required stellar particle arrays from a yt snapshot.

    Returns
    -------
    stellar_ages   : np.ndarray  (Gyr)
    stellar_masses : np.ndarray  (Msun)
    stellar_metals : np.ndarray  (absolute Z, clipped)
    simtime        : float       (Gyr)
    obj            : caesar object or None
    """

    print("Loading yt snapshot...")
    ds  = yt.load(snapshot)
    obj = None

    # --- AREPO star filter ---
    if AREPO:
        def _newstars(pfilter, data):
            return data[(pfilter.filtered_type, "GFM_StellarFormationTime")] > 0
        yt.add_particle_filter("newstars", function=_newstars, filtered_type='PartType4')
        ds.add_particle_filter("newstars")

    # --- Data container ---
    if COSMOLOGICAL and not FILTERED:
        if caesar is None:
            raise ImportError("caesar is required for FILTERED=False cosmological mode.")
        print("Loading caesar file...")
        obj = caesar.load(csfile)
        obj.yt_dataset = ds
        dd = obj.yt_dataset.all_data()
    else:
        dd = ds.all_data()

    # --- Stellar ages ---
    if COSMOLOGICAL:
        ptype = "newstars" if AREPO else "PartType4"
        field = "GFM_StellarFormationTime" if AREPO else "StellarFormationTime"
        scalefactor = _to_numpy(dd[(ptype, field)])

        formation_z = (1.0 / scalefactor) - 1.0
        cosmo = yt.utilities.cosmology.Cosmology(
            hubble_constant=0.68,
            omega_lambda=0.7,
            omega_matter=0.3
        )
        stellar_formation_times = _to_numpy(cosmo.t_from_z(formation_z).in_units("Gyr"))
        simtime       = float(_to_numpy(cosmo.t_from_z(ds.current_redshift).in_units("Gyr")))
        stellar_ages  = simtime - stellar_formation_times

    else:
        simtime = float(_to_numpy(ds.current_time.in_units('Gyr')))
        if AREPO:
            print("WARNING: Assuming stellar age units are s*kpc/km.")
            stellar_ages = simtime - _to_numpy(
                ds.arr(dd[("newstars", "GFM_StellarFormationTime")], 's*kpc/km').in_units('Gyr')
            )
        else:
            stellar_ages = simtime - _to_numpy(
                ds.arr(dd[('PartType4', 'StellarFormationTime')], 'Gyr')
            )

    stellar_ages = np.clip(stellar_ages, MIN_AGE_GYR, None)

    # --- Masses and metallicities ---
    ptype = "newstars" if AREPO else "PartType4"
    mfield  = "Masses"
    zfield  = "GFM_Metallicity" if AREPO else "metallicity"

    stellar_masses = _to_numpy(dd[(ptype, mfield)].in_units("Msun"))
    stellar_metals = np.clip(_to_numpy(dd[(ptype, zfield)]), MIN_Z, None)

    return stellar_ages, stellar_masses, stellar_metals, simtime, obj


# ---------------------------------------------------------
# Parallel compute path
# ---------------------------------------------------------
def _compute_parallel(
    stellar_ages, stellar_masses, stellar_metals,
    simtime, metal_grid, log_age_grid, mass_table,
    n_workers, n_chunks
):
    """Distribute all particles across workers and collect results."""
    n_particles   = len(stellar_ages)
    chunk_indices = np.array_split(np.arange(n_particles), n_chunks)

    args = [
        (chunk, stellar_ages, stellar_masses, stellar_metals,
         simtime, metal_grid, log_age_grid, mass_table)
        for chunk in chunk_indices
    ]

    print(f"Running {n_particles} particles over {n_chunks} chunks "
          f"with {n_workers} workers...")
    t0 = time.time()

    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(sfh_worker, args, chunksize=1),
                total=len(args),
                desc="Processing chunks"
            )
        )

    print(f"Parallel processing done in {time.time() - t0:.2f}s")

    formation_times  = np.empty(n_particles)
    formation_masses = np.empty(n_particles)

    for indices, tform, mform in results:
        formation_times[indices]  = tform
        formation_masses[indices] = mform

    return formation_times, np.clip(formation_masses, 0, None)


# ---------------------------------------------------------
# Caesar galaxy compute path
# ---------------------------------------------------------
def _compute_caesar_galaxies(
    obj,
    stellar_ages, stellar_masses, stellar_metals,
    simtime, metal_grid, log_age_grid, mass_table,
    n_workers, n_chunks
):
    """Run parallel SFH computation per caesar galaxy."""
    n_galaxies = len(obj.galaxies)
    print(f"Processing {n_galaxies} caesar galaxies...")

    all_tform = []
    all_mform = []

    for gal_idx in tqdm(range(n_galaxies), desc="Galaxies"):
        slist = obj.galaxies[gal_idx].slist

        if len(slist) == 0:
            all_tform.append(np.array([]))
            all_mform.append(np.array([]))
            continue

        gal_ages   = stellar_ages[slist]
        gal_masses = stellar_masses[slist]
        gal_metals = stellar_metals[slist]

        n_gal    = len(slist)
        n_ch     = min(n_chunks, n_gal)
        chunks   = np.array_split(np.arange(n_gal), n_ch)

        args = [
            (chunk, gal_ages, gal_masses, gal_metals,
             simtime, metal_grid, log_age_grid, mass_table)
            for chunk in chunks
        ]

        with Pool(n_workers) as pool:
            results = list(pool.imap(sfh_worker, args, chunksize=1))

        tform = np.empty(n_gal)
        mform = np.empty(n_gal)

        for indices, t, m in results:
            tform[indices] = t
            mform[indices] = m

        all_tform.append(tform)
        all_mform.append(np.clip(mform, 0, None))

    return all_tform, all_mform


# ---------------------------------------------------------
# Serial compute path (non-cosmological)
# ---------------------------------------------------------
def _compute_serial(
    stellar_ages, stellar_masses, stellar_metals,
    simtime, metal_grid, log_age_grid, mass_table
):
    """Single-threaded SFH computation for non-cosmological snapshots."""
    interp = RegularGridInterpolator(
        (metal_grid, log_age_grid),
        mass_table,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    n = len(stellar_ages)
    formation_times  = np.empty(n)
    formation_masses = np.empty(n)

    for i in tqdm(range(n), desc="Particles (serial)"):
        age_gyr = max(float(stellar_ages[i]),  MIN_AGE_GYR)
        metal   = max(float(stellar_metals[i]), MIN_Z)
        mass    = float(stellar_masses[i])

        frac = float(interp((metal, np.log10(age_gyr * 1e9))))

        if not np.isfinite(frac) or frac <= 0.0:
            frac = MIN_FRAC

        formation_masses[i] = mass / frac
        formation_times[i]  = simtime - age_gyr

    return formation_times, np.clip(formation_masses, 0, None)


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def compute_sfh(
    snapshot,
    csfile     = None,
    FILTERED   = True,
    COSMOLOGICAL = False,
    AREPO      = False,
    output_dir = None,
    sim_name   = None,
    n_workers  = 16,
    n_chunks   = 30
):
    
    
    """
    Compute star formation history for a simulation snapshot.

    Parameters
    ----------
    snapshot     : str   Path to snapshot file.
    csfile       : str   Path to caesar file (FILTERED=False only).
    FILTERED     : bool  True → use all particles; False → caesar galaxies.
    COSMOLOGICAL : bool  True → compute ages from scale factors.
    AREPO        : bool  True → use AREPO field names.
    output_dir   : str   Output directory (default: ``./output/<sim_name>/fsps_sfh/``).
    sim_name     : str   Simulation name for default output path inference.
    n_workers    : int   Number of parallel workers.
    n_chunks     : int   Number of particle chunks.

    Returns
    -------
    dict
        {'id': [...], 'tform': np.ndarray (Gyr), 'massform': np.ndarray (Msun)}
    """
    # --- Output setup ---
    if output_dir is None:
        if sim_name is None:
            sim_name = os.path.splitext(os.path.basename(snapshot))[0]
        output_dir = os.path.join(os.getcwd(), 'output', sim_name, 'fsps_sfh')
    os.makedirs(output_dir, exist_ok=True)

    snap_name = os.path.splitext(os.path.basename(snapshot))[0]
    outfile   = os.path.join(output_dir, f"{snap_name}_sfh.pkl")

    # --- Load particle data ---
    stellar_ages, stellar_masses, stellar_metals, simtime, obj = _load_snapshot_data(
        snapshot, COSMOLOGICAL, AREPO, FILTERED, csfile
    )

    print(f"Snapshot time : {simtime:.3f} Gyr")
    print(f"Stellar particles : {len(stellar_ages):,}")

    # --- Build FSPS grid ---
    print("Building FSPS interpolation grid...")
    t0 = time.time()
    metal_grid, log_age_grid, mass_table = build_fsps_grid(stellar_metals)
    print(f"FSPS grid ready in {time.time() - t0:.2f}s")

    # --- Compute ---
    if COSMOLOGICAL and FILTERED:
        tform, mform = _compute_parallel(
            stellar_ages, stellar_masses, stellar_metals,
            simtime, metal_grid, log_age_grid, mass_table,
            n_workers, n_chunks
        )
        ids = [0]

    elif COSMOLOGICAL and not FILTERED:
        if obj is None:
            raise ValueError("Caesar object not loaded. Check csfile path.")
        tform, mform = _compute_caesar_galaxies(
            obj,
            stellar_ages, stellar_masses, stellar_metals,
            simtime, metal_grid, log_age_grid, mass_table,
            n_workers, n_chunks
        )
        ids = list(range(len(obj.galaxies)))

    else:
        tform, mform = _compute_serial(
            stellar_ages, stellar_masses, stellar_metals,
            simtime, metal_grid, log_age_grid, mass_table
        )
        ids = [0]

    # --- Save ---
    result = {'id': ids, 'tform': tform, 'massform': mform}
    with open(outfile, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved → {outfile}")

    return result




    
# =============================================================
# sfh_extract.py
# Extracts and plots star formation histories from SFH pickles
# produced by sfh_compute.py.
# =============================================================

import pickle
import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "savefig.facecolor"  : "w",
    "figure.facecolor"   : "w",
    "figure.figsize"     : (10, 8),
    "text.color"         : "k",
    "legend.fontsize"    : 20,
    "font.size"          : 30,
    "axes.edgecolor"     : "k",
    "axes.labelcolor"    : "k",
    "axes.linewidth"     : 3,
    "xtick.color"        : "k",
    "ytick.color"        : "k",
    "xtick.labelsize"    : 25,
    "ytick.labelsize"    : 25,
    "ytick.major.size"   : 12,
    "xtick.major.size"   : 12,
    "ytick.major.width"  : 2,
    "xtick.major.width"  : 2,
    "font.family"        : "STIXGeneral",
    "mathtext.fontset"   : "cm"
})


# ---------------------------------------------------------
# File loader
# ---------------------------------------------------------
def load_sfh_file(file_):
    """
    Load a SFH pickle file produced by sfh_compute.py.

    Parameters
    ----------
    file_ : str

    Returns
    -------
    dict  with keys 'id', 'tform', 'massform'
    """
    with open(file_, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------
# Binning
# ---------------------------------------------------------
def bin_sfh(tform_myr, massform, binwidth_myr=300.0):
    """
    Bin stellar mass formation data into a SFR time series.

    Parameters
    ----------
    tform_myr   : np.ndarray  Formation times in Myr.
    massform    : np.ndarray  Formation masses in Msun.
    binwidth_myr: float       Bin width in Myr.

    Returns
    -------
    bincenters : np.ndarray   Bin centers in Myr.
    sfr        : np.ndarray   SFR in Msun/yr per bin. Always >= 0.
    """
    tform_myr = np.asarray(tform_myr, dtype=np.float64).ravel()
    massform  = np.clip(np.asarray(massform, dtype=np.float64).ravel(), 0, None)

    t_max = tform_myr.max()
    bins  = np.arange(100.0, t_max + binwidth_myr, binwidth_myr)

    sfr, bin_edges, _ = scipy.stats.binned_statistic(
        tform_myr,
        massform,
        statistic=lambda m: np.sum(m) / (binwidth_myr * 1e6),
        bins=bins
    )

    # Final clip — guaranteed no negative SFR
    sfr        = np.clip(sfr, 0.0, None)
    bincenters = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bincenters, sfr


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def plot_sfh(bincenters, sfr, file_=None, galaxy_id=0, plot_path=None):
    """
    Plot a star formation history curve and save to disk.

    Parameters
    ----------
    bincenters : np.ndarray   Bin centers in Myr.
    sfr        : np.ndarray   SFR in Msun/yr.
    file_      : str          Source data file (used for auto plot_path).
    galaxy_id  : int          Galaxy ID for filename labelling.
    plot_path  : str          Explicit save path (overrides auto).
    """
    if plot_path is None:
        if file_ is not None:
            plot_path = file_.replace('.pkl', f'_gal{galaxy_id}_sfh.png')
        else:
            plot_path = f'sfh_gal{galaxy_id}.png'

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(bincenters, sfr, color='k', linewidth=2)
    ax.set_ylabel(r'SFR [M$_{\odot}$ yr$^{-1}$]')
    ax.set_xlabel(r'$t_H$ [Myr]')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


# ---------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------
def get_galaxy_SFH(
    file_,
    galaxy_id    = 0,
    binwidth_myr = 300.0,
    save_plot    = False,
    plot_path    = None
):
    """
    Extract and optionally plot the SFH for a given galaxy.

    Parameters
    ----------
    file_        : str    Path to SFH pickle file.
    galaxy_id    : int    Galaxy ID to extract (ignored for single-galaxy files).
    binwidth_myr : float  SFR bin width in Myr.
    save_plot    : bool   If True, save a plot.
    plot_path    : str    Explicit plot save path.

    Returns
    -------
    bincenters : np.ndarray   Bin centers in Myr.
    sfr        : np.ndarray   SFR in Msun/yr (always >= 0).
    """
    dat      = load_sfh_file(file_)
    ids      = dat['id']
    massforms = dat['massform']
    tforms   = dat['tform']

    print(f"Galaxy IDs in file: {ids}")

    # --- Select galaxy ---
    is_single = (
        isinstance(ids, (int, float, np.integer, np.floating))
        or (hasattr(ids, '__len__') and len(ids) <= 1)
    )

    if is_single:
        print("Single galaxy detected.")
        massform  = np.asarray(massforms, dtype=np.float64).ravel()
        tform_myr = np.asarray(tforms,    dtype=np.float64).ravel() * 1000.0

    else:
        ids_arr = np.asarray(ids)
        match   = np.where(ids_arr == galaxy_id)[0]
        if len(match) == 0:
            raise ValueError(
                f"Galaxy ID {galaxy_id} not found. "
                f"Available IDs: {ids_arr.tolist()}"
            )
        idx = match[0]
        print(f"Extracting galaxy_id={galaxy_id} at index {idx}")
        massform  = np.asarray(massforms[idx], dtype=np.float64).ravel()
        tform_myr = np.asarray(tforms[idx],    dtype=np.float64).ravel() * 1000.0

    # --- Bin ---
    bincenters, sfr = bin_sfh(tform_myr, massform, binwidth_myr=binwidth_myr)

    # --- Plot ---
    if save_plot:
        plot_sfh(bincenters, sfr, file_=file_, galaxy_id=galaxy_id, plot_path=plot_path)

    return bincenters, sfr
