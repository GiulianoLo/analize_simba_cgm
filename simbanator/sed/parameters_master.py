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
# 1e7 -> 1e6 on 2026-08-03 (uniform with parameters_master-agn.py, whose comments
# hold the cost benchmark): all three m25 runs share one MC noise floor. The MC
# error is propagated into the <filter>_err catalog columns -- check it in the
# Part 6 QC before trusting faint-band differentials.
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
BH_SED = False
BH_eta = 0.1
BH_model = "Nenkova"
BH_modelfile = os.path.join(root, 'powderday', 'agn_models', 'clumpy_models_201410_tvavg.hdf5')
BH_var = True
nenkova_params = [5, 30, 0, 1.5, 30, 40]

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
# 4 quasi-orthogonal sightlines (deg); galaxies are randomly oriented in the box,
# so these sample the per-galaxy orientation spread. get_sed/extract_flux_batch
# picks one with the inclination index findx = 0..3.
MANUAL_ORIENTATION = True
THETA = [0, 45, 90, 135]
PHI = [0, 90, 180, 270]

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
