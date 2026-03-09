from simbanator.io.simba import Simulation
from simbanator.analysis.particles import extract_particles

snap = 149

# Load your simulation (use the name you configured)
sim = Simulation("test_sim")  # Replace XX with your snapshot number

# Load the Caesar catalog for this snapshot
obj = sim.load_catalog(snap)
snap_path = sim.get_snapshot_file(snap)

from simbanator.analysis.sfh_fsps import compute_sfh, get_galaxy_SFH

# Set your file paths
snapshot = "filtered_snap" + str(snap) + ".h5"
caesar_file = sim.get_caesar_file(snap)

#Compute SFH for all galaxies in the snapshot
sfh_result = compute_sfh(
    snapshot,
    caesar_file,
    COSMOLOGICAL=True,
    FILTERED=True
)


get_galaxy_SFH('/home/lorenzong/analize_simba_cgm/output/fsps_sfh/filtered_snap' + str(snap) + '_sfh.pkl', galaxy_id=0, save_plot=True, binwidth_myr=3)