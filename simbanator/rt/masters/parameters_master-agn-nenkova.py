# coding=utf-8
import os

# ===============================================
# HOME INFORMATION
# ===============================================
# Set root to the parent directory containing powderday, hyperion-dust, etc.
root = os.environ.get('POWDERDAY_ROOT', os.path.expanduser('~'))

pd_source_dir = os.path.join(root, 'powderday') + '/'

# ===============================================
# RESOLUTION KEYWORDS
# ===============================================
oref = 0
n_ref = 8
zoom_box_len = 100  # kpc; box will be +/- zoom_box_len from centre
                    # (= the largest SED aperture and the Stage-0 cutout radius)
bbox_lim = 200      # kpc - initial bounding box (+/- bbox_lim)

# ===============================================
# PARALLELIZATION
# ===============================================
n_processes = 12
n_MPI_processes = 12

# ===============================================
# RT INFORMATION
# ===============================================
# Photon budget cut 1e7 -> 1e6 on 2026-07-31 (= the stock powderday default,
# ~/powderday/parameters_master.py:28-31). Frozen 1e7 copy for the validation
# run: snap_091/parameters_master_ref1e7.py
# 2026-08-03: 1e6 adopted for the WHOLE m25 quenched campaign (all three masters:
# dust_on / -nodust / -agn), so the three runs share one MC noise floor.
#
# WHY: with BH_SED=True these galaxies cost ~8x the AGN-off run, and 95% of that
# is the two raytracing passes. Measured on snap 091 / gal 1747, same galaxy both
# ways, cpu-hours:
#
#                        AGN-off    AGN-on
#     7 x Lucy             0.24       1.85
#     final iteration      0.10       0.81
#     raytracing sources   4.17      39.28   <-- dominant
#     raytracing dust      4.49      ~30     <-- dominant
#     TOTAL                9.0       ~73     (~7 h wall)
#
# The AGN is a central point source carrying L_AGN ~ 3e44 erg/s with the Hopkins
# template peaking in the far-UV (its nu grid runs down to 0.095 um), where dust
# opacity and albedo are highest -> many more scatterings per photon. That
# per-photon cost is intrinsic to having an AGN; the only levers are the photon
# count and the number of peel-off directions (see THETA/PHI below).
#
# Raytracing cost scales as n_photons x n_viewing_angles x n_freq_bins. Reaching
# half the photons cost ~33% of the pass time in all six raytracing/final blocks
# measured across both logs, so this cut buys at least its nominal 10x.
#
# CAVEAT: these photons set the SED shot noise, and the science product is the
# AGN-on minus AGN-off residual. set_uncertainties(True) is on in
# front_end_tools.py, so the noise is measurable in the .rtout.sed -- check it
# before trusting a faint-end residual.
n_photons_initial = 1.e6
n_photons_imaging = 1.e6
n_photons_raytracing_sources = 1.e6
n_photons_raytracing_dust = 1.e6   # full dust radiative transfer
n_photons_DIG = 1e6

FORCE_RANDOM_SEED = False
seed = -12345

# ===============================================
# DUST INFORMATION
# ===============================================
dustdir = os.path.join(root, 'hyperion-dust', 'dust_files') + '/'
dustfile = 'kmh94_3.1_hg.hdf5'
PAH = True

dust_grid_type = 'manual'    # 'manual' -> use SIMBA live dust masses from the snapshot
dusttometals_ratio = 0.4   # unused under 'manual' (kept at the simulation/reference value)
enforce_energy_range = False

SUBLIMATION = False
SUBLIMATION_TEMPERATURE = 1600.  # K

# Experimental Dust
otf_extinction = False
otf_extinction_log_min_size = -4  # micron
otf_extinction_log_max_size = 0   # micron

draine21_pah_model = True
draine21_pah_grid_write = True
dust_density = 2.4  # g/cm^3

# ===============================================
# STELLAR SEDS INFO
# ===============================================
FORCE_BINNED = True
max_age_direct = 1.e-2  # Gyr

imf_type = 1   # 0=salpeter, 1=chabrier, 2=kroupa
imf1 = 1.3
imf2 = 2.3
imf3 = 2.3
pagb = 1

add_agb_dust_model = True
alpha_enhacement = False

# ===============================================
# NEBULAR EMISSION INFO
# ===============================================
add_neb_emission = True
use_cloudy_tables = True

# ===============================================
# BIRTH CLOUD INFORMATION
# ===============================================
CF_on = False
birth_cloud_clearing_age = 0.01  # Gyr

# ===============================================
# Idealized Galaxy SED Parameters
# ===============================================
Z_init = 0
disk_stars_age = 8        # Gyr
bulge_stars_age = 8       # Gyr
disk_stars_metals = 12
bulge_stars_metals = 12

# ===============================================
# Stellar Ages and Metallicities
# ===============================================
N_STELLAR_AGE_BINS = 25

# ===============================================
# BLACK HOLE STUFF
# ===============================================
# AGN run: identical to parameters_master.py (dust_on) except BH_SED=True.
# Requires PartType5 (BH_Mass/BH_Mdot/Coordinates) in the Stage-0 cutouts.
# L_bol = BH_eta * BH_Mdot * c^2 per BH, spectrum from the chosen template,
# injected as point sources and attenuated by the same SIMBA live-dust grid.
BH_SED = True
BH_eta = 0.1
# 'Nenkova' = Hopkins+2007 template x CLUMPY torus transmission at ONE fixed
# inclination (nenkova_params i), injected isotropically — no viewing-angle
# physics, it only reshapes the input spectrum. Needs the CLUMPY database
# ~/powderday/agn_models/clumpy_models_201410_tvavg.hdf5 (1.2 GB,
# wget -c https://clumpy.org/downloads/clumpy_models_201410_tvavg.hdf5 — no www,
# broken cert there).
BH_model = "Nenkova"
BH_modelfile = os.path.join(root, 'powderday', 'agn_models', 'clumpy_models_201410_tvavg.hdf5')
# BH_var=False: use the SIMBA BH_Mdot as-is, so L_AGN stays consistent with the
# f_Edd-based weak/strong AGN split; True would rescale each BH by a random
# Hickox+2014 duty-cycle draw (and FORCE_RANDOM_SEED=False makes that irreproducible).
BH_var = False
nenkova_params = [5, 30, 90, 1.5, 30, 40]  # i=90: edge-on (type-2) CLUMPY view; single fixed-i transmission x Hopkins, injected isotropically

# ===============================================
# IMAGES AND SED
# ===============================================
NTHETA = 1
NPHI = 1
SED = True

# ── multi-aperture SEDs (read by the patched powderday; stock powderday ignores these) ──
# The patch calls image.set_aperture_range(SED_APERTURE_NAP, min*kpc, max*kpc) on the
# peeled-image conf, so the .rtout.sed carries SED_APERTURE_NAP log-spaced projected
# apertures; extract each with MakeSED.extract_flux_batch(..., aperture=i).
# 1->100 kpc with 5 apertures = 10^(k/2) ladder: 1, 3.16, 10, 31.6, 100 kpc
# (central -> outskirts).
SED_APERTURE_NAP = 5
SED_APERTURE_MIN_KPC = 1.
SED_APERTURE_MAX_KPC = 100.

SED_MONOCHROMATIC = False
FIX_SED_MONOCHROMATIC_WAVELENGTHS = False
SED_MONOCHROMATIC_min_lam = 0.3  # micron
SED_MONOCHROMATIC_max_lam = 0.4  # micron

IMAGING = False
filterdir = os.path.join(root, 'powderday', 'filters') + '/'
filterfiles = ['H2.filter']
npix_x = 512
npix_y = 512

IMAGING_TRANSMISSION_FILTER = False
filter_list = ['filters/STIS_clear.filter']
TRANSMISSION_FILTER_REDSHIFT = 3.1

# ===============================================
# GRID INFORMATION
# ===============================================
# 4 sightlines RESTORED on 2026-08-03 for the m25 quenched campaign: the m25
# pipeline has the inclination axis baked in (per-incl catalogs i{θ}p{φ},
# get_sed/extract_flux_batch findx = 0..3, Part 4b projected-annulus geometry),
# so all three runs (dust_on / dust_off / agn_on) use the same 4 angles.
#
# Cost note (from the 2026-07-31 snap 091 / gal 1747 benchmark): raytracing --
# 95% of an AGN-on run -- is linear in the number of viewing angles
# (front_end_tools.py:83 -> sed.set_viewing_angles(); hyperion peels off to every
# angle at every interaction). The 1-angle option below is a ~4x cut kept for
# single-sightline validation runs (used by the cis100 analogues AGN test, where
# the 4 sightlines agreed to <0.5%, part6g_single_galaxy.py:172-174).
MANUAL_ORIENTATION = True
THETA = [0, 45, 90, 135]   # 4 quasi-orthogonal sightlines (deg)
PHI = [0, 90, 180, 270]
# THETA = [0]   # single-sightline option -- ~4x cheaper, findx = 0 only
# PHI   = [0]

# ===============================================
# OTHER INFORMATION
# ===============================================
PAH_frac = {'usg': 0.0586, 'vsg': 0.1351, 'big': 0.8063}

# ===============================================
# DEBUGGING
# ===============================================
SOURCES_RANDOM_POSITIONS = False
SOURCES_IN_CENTER = False
STELLAR_SED_WRITE = False
SKIP_RT = False
SUPER_SIMPLE_SED = False
SKIP_GRID_READIN = False
CONSTANT_DUST_GRID = False
N_MASS_BINS = 1

FORCE_STELLAR_AGES = False
FORCE_STELLAR_AGES_VALUE = 0.05  # Gyr

FORCE_STELLAR_METALLICITIES = False
FORCE_STELLAR_METALLICITIES_VALUE = 0.012

SKIRT_DATA_DUMP = True
REMOVE_INPUT_SEDS = False
