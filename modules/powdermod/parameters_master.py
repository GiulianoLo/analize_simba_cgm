# coding=utf-8
#===============================================
#HOME INFORMATION
#===============================================
root = '/mnt/home/glorenzon'

pd_source_dir = root+'/powderday/'

#===============================================
#RESOLUTION KEYWORDS
#===============================================
oref = 0 #over refine factor - should typically be set to 0
n_ref = 8 #when n_particles > n_ref, octree refines further
zoom_box_len = 30 #kpc; so the box will be +/- zoom_box_len from the center
bbox_lim = 60 #kpc - this is the initial bounding box of the grid (+/- bbox_lim)
               #This *must* encompass all of the particles in the
               #simulation. 

#===============================================
#PARALLELIZATION
#===============================================

n_processes = 12 #number of pool processes to run
n_MPI_processes = 12 #number oF MPI processes to run

#===============================================
#RT INFORMATION
#===============================================
n_photons_initial = 1.e7
n_photons_imaging = 1.e7
n_photons_raytracing_sources = 1.e7
n_photons_raytracing_dust = 1.e7
n_photons_DIG = 1e7

FORCE_RANDOM_SEED = False
seed = -12345 #has to be an int, and negative.

#===============================================
#DUST INFORMATION 
#===============================================
dustdir = root+'/hyperion-dust/dust_files/' #location of your dust files
dustfile = 'kmh94_3.1_hg.hdf5' #'d03_3.1_6.0_A.hdf5'
PAH = True

dust_grid_type = 'manual' #needs to be in ['dtm','rr','manual']
dusttometals_ratio = 0.4
enforce_energy_range = False #False is the default;  ensures energy conservation

SUBLIMATION = False #do we automatically kill dust grains above the
                    #sublimation temperature; right now is set to fast
                    #mode
SUBLIMATION_TEMPERATURE = 1600. #K -- meaningliess if SUBLIMATION == False

#---------------------------------------------------------------
#Experimental Dust -- Note, these features are not fully vetted
#---------------------------------------------------------------

otf_extinction = False #flag for on the fly extinction.  If set, then we
                                    #ignore the dustdir/dustfile extinction information above. if
                                    #false, all otf_extinction* quantities are meaningless
otf_extinction_log_min_size = -4 #micron; must match what is set in the hydro simulation
otf_extinction_log_max_size = 0 #micron; must match what is set in the hydro simulation

draine21_pah_model = True
draine21_pah_grid_write = True #this will write the PAH spectrum out
                               #to the npz file for every cell in the
                               #grid.  note, this causes the grid*.npz
                               #files to be ~10 GB or more.

dust_density = 2.4 #units are g/cm**3.  this *only* is used in the OTF
                   #pah calculations, so if you do not have an active dust model, this will not be useful.

#===============================================
#STELLAR SEDS INFO
#===============================================
FORCE_BINNED = True #force SED binning

max_age_direct  = 1.e-2           # Age (in Gyr) below which stars will be directly added without binning (works only if FORCE_BINNED is False)

imf_type = 1 # FSPS imf types; 0 = salpeter, 1 = chabrier; 2 = kroupa; 3 and 4 (vandokkum/dave) not currently supported
imf1 = 1.3 # Logarithmic slope of the IMF over the range 0.08 < M < 0.5. Only used if imf_type=2. (Default: 1.3)
imf2 = 2.3 # Logarithmic slope of the IMF over the range 0.5 < M < 1.0. Only used if imf_type=2. (Default: 2.3)
imf3 = 2.3 # Logarithmic slope of the IMF over the range 1.0 < M < 120. Only used if imf_type=2. (Default: 2.3)

pagb = 1 #weight given to post agb stars# 1 is the default

add_agb_dust_model = True    # add circumstellar AGB dust model (100%); Villaume, Conroy & Jonson 2015

alpha_enhacement = False                    # If set, then the metallicity of star particles is set to [Fe/H] rather than the total metals. 
                                            # Since FSPS does not support non solar abundance ratios, this parameter can be used to mimic the 
                                            # hardening of the radiation field due to alpha-enhancement. (Default: False)

#===============================================
#NEBULAR EMISSION INFO
#===============================================
add_neb_emission = True #add nebular line emission from Cloudy Lookup tables (dev. by Nell Byler)

use_cloudy_tables = True                    # If True, CLOUDY look up tables (dev. by Nell Byler) will be used to calculate 
                                            # nebular emission. Note that these tables are only applicable for star particles with ages below 10 Myr.
                                            # If False, CLOUDY models are generated individually for each star particle. This mode provides greater flexibility 
                                            # and allows for customization by tuning various free parameters. To take advantage of this flexibility, you need to 
                                            # provide a separate parameter master file (see parameters_master_neb.py example file provided with the code) when running 
                                            # Powderday. If you don't provide this file, default values will be used instead.



#===============================================
#BIRTH CLOUD INFORMATION
#===============================================

CF_on = True               # if set to true, then we enable the Charlot & Fall birthcloud models 

birth_cloud_clearing_age = 0.01 # Gyr - stars with age <
                                # birth_cloud_clearing_age have
                                # charlot&fall birthclouds meaningless
                                # of CF_on  == False

#===============================================
# Idealized Galaxy SED Parameters
#===============================================
Z_init = 0 # force a metallicity increase in the newstar particles.
           # This is useful for idealized galaxies.  The units for this
           # are absolute (so enter 0.02 for solar).  Setting to 0
           # means you use the stellar metallicities as they come in
           # the simulation (more likely appropriate for cosmological
           # runs)


#Idealized Galaxy SED Parameters
disk_stars_age = 8 #Gyr ;meaningless if this is a cosmological simulation; note, if this is <= 7, then these will live in birth clouds
bulge_stars_age = 8 #Gyr ; meaningless if this is a cosmological simulation; note, if this is <= 7, then these will live in birth clouds
disk_stars_metals = 12 #in fsps metallicity units
bulge_stars_metals = 12 #in fsps metallicity units



#===============================================
# Stellar Ages and Metallicities
#===============================================
#bins for binning the stellar ages and metallicities for SED
#assignments in cases of many (where many ==
#>N_METALLICITY_BINS*N_STELLAR_AGE_BINS) stars; this is necessary for
#reduction of memory load; see manual for details.

N_STELLAR_AGE_BINS = 25

# metallicity_legend="/cosma7/data/dp004/dc-love2/codes/simba_cosmo_smgs/pday_setup/zlegend.dat" 
# metallicity_legend="/cosma7/data/dp004/dc-love2/codes/python-fsps/src/fsps/libfsps/ISOCHRONES/BPASS/zlegend.bpass_v2.2_bin_chab300.dat"

#===============================================
#BLACK HOLE STUFF
#===============================================

BH_SED = False #True
BH_eta = 0.1 #bhluminosity = BH_eta * mdot * c**2.
BH_model = "Nenkova"
BH_modelfile = root+"/powderday/agn_models/clumpy_models_201410_tvavg.hdf5"
# The Nenkova BH_modelfile can be downloaded here:
# https://www.clumpy.org/downloads/clumpy_models_201410_tvavg.hdf5
BH_var = True # Include time variations on BH luminosity (default Hickox+ 2014)

nenkova_params = [5,30,0,1.5,30,40] #Nenkova+ (2008) model parameters


#===============================================
#IMAGES AND SED
#===============================================

NTHETA = 1
NPHI = 1
SED = True

SED_MONOCHROMATIC = False
FIX_SED_MONOCHROMATIC_WAVELENGTHS = False #if set, then we only use
                                         #nlam wavelengths in the
                                         #range between min_lam and
                                         #max_lam
SED_MONOCHROMATIC_min_lam = 0.3 #micron
SED_MONOCHROMATIC_max_lam = 0.4 #micron



IMAGING = False
filterdir=root+'/powderday/filters/'
filterfiles = ['H2.filter']

npix_x = 512
npix_y = 512

#experimental and under development - not advised for use
IMAGING_TRANSMISSION_FILTER = False
filter_list = ['filters/STIS_clear.filter']
TRANSMISSION_FILTER_REDSHIFT = 3.1

#===============================================
#GRID INFORMATION  
#===============================================

MANUAL_ORIENTATION = False
THETA = 0
PHI = 0


#===============================================
#OTHER INFORMATION
#===============================================

PAH_frac = {'usg': 0.0586, 'vsg': 0.1351, 'big': 0.8063} # values will be normalized to 1

#===============================================
#DEBUGGING
#===============================================
SOURCES_RANDOM_POSITIONS = False
SOURCES_IN_CENTER = False
STELLAR_SED_WRITE = False
SKIP_RT = False #skip radiative transfer (i.e. just read in the grids and maybe write some diagnostics)
SUPER_SIMPLE_SED = False #just generate 1 oct of 100 pc on a side,
                         #centered on [0,0,0].  sources are added at
                         #random positions.
SKIP_GRID_READIN = False

CONSTANT_DUST_GRID = False #if set, then we don't create a dust grid by
                          #smoothing, but rather just make it the same
                          #size as the octree with a constant value of
                          #4e-20
                          
N_MASS_BINS = 1 #this is really just a place holder that exists in
                #some loops to be able to insert some code downstream
                #for spatially varying IMFs.  right now for speed best
                #to set to 1 as it doesn't actually do anything.

FORCE_STELLAR_AGES = False
FORCE_STELLAR_AGES_VALUE = 0.05# Gyr

FORCE_STELLAR_METALLICITIES = False
FORCE_STELLAR_METALLICITIES_VALUE = 0.012 #absolute values (so 0.013 ~ solar)

SKIRT_DATA_DUMP = True # if set, various data files useful for running SKIRT are saved.

REMOVE_INPUT_SEDS = False # If set, the hyperion input SED file is deleted after RT finishes