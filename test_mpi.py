from simbanator.io.simba import Simulation
from simbanator.analysis.particles import extract_particles

# Load your simulation (use the name you configured)
sim = Simulation("test_sim")  # Replace XX with your snapshot number

# Load the Caesar catalog for this snapshot
obj = sim.load_catalog(136)
snap_path = sim.get_snapshot_file(136)

from simbanator.analysis.sfh_fsps import compute_sfh

# Set your file paths
snapshot = "filtered_snap136.h5"
caesar_file = sim.get_caesar_file(136)

# Compute SFH for all galaxies in the snapshot
sfh_result = compute_sfh(
    snapshot,
    caesar_file,
    COSMOLOGICAL=True,
    FILTERED=True
)
