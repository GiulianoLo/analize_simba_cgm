import os
import numpy as np
from astropy.io import fits
from astropy.cosmology import LambdaCDM

from simbanator.sed.makesed import MakeSED
from simbanator.io.simba import Simulation

# --- SLURM task ID ---
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# --- CONFIG ---
chunk_size = 50  

cosmo = LambdaCDM(H0=68, Om0=0.3, Ode0=0.7, Ob0=0.048)

sim = Simulation('cis100')

hydro_dir_base = os.path.join(os.getcwd(), 'output', sim.name, 'filtered')
preselect = '/mnt/home/glorenzon/output/quenching_times/legacy_poststarbursts_snpa_id.txt'
selection_file = 'selection_simba100_poststarburst.hdf5'


makesed = MakeSED(
    sim,
    nnodes=1,
    model_run_name='PSBG_dust_on',
    hydro_dir_base=hydro_dir_base,
    selection_file=selection_file,
    preselect=preselect
)

# --- Load data ---
with fits.open('/mnt/home/glorenzon/analize_simba_cgm/output/cis100/quenching_times/forward_modeled_unique_sample.fits') as f:
    snaps = f[1].data['SNAPSHOT']
    ids   = f[1].data['GROUPID_SNAPSHOT']

# --- Split work ---
start = task_id * chunk_size
end   = (task_id + 1) * chunk_size

snaps_chunk = snaps[start:end]
ids_chunk   = ids[start:end]

print(f"[Task {task_id}] Processing galaxies {start} → {end}")

# --- Run ---
makesed.extract_flux_batch(
    snaps_chunk,
    ids_chunk,
    ['HST', 'JWST', 'Spitzer', 'Herschel'],
    ['WFC3', 'NIRCam', 'IRAC', 'SPIRE'],
    filters=None,
    wave_unit='micron',
    findx=0,
    outname=f'fluxes_task_{task_id:03}.fits'
)