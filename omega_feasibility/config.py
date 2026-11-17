"""Knobs of the OMEGA feasibility check (one place; edit here, not in the scripts).

Snapshots were chosen from the snapshot headers (2026-11-17), identical in m100n1024 and m25n512:
    026 z=7.963 | 030 z=7.054 | 036 z=5.930 (035 is z=6.100) | 129 z=0.406
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# simbanator config names -> full run names (figures use the full names)
SIMS = {"cis25": "m25n512", "cis100": "m100n1024"}

# target redshift -> snapshot (headers, see above)
HIGHZ_SNAPS = {8: 26, 7: 30, 6: 36}
LOWZ_SNAP = 129

# resolution
N_STAR_MIN = 30            # star particles
N_YOUNG_MIN = 5            # star particles younger than T_YOUNG_MYR
T_YOUNG_MYR = 10.0
T_SFR_MYR = (10.0, 100.0)  # archaeological SFR windows
RES_FRACTION = 0.8         # a mass bin is "resolved" when >= this fraction of its galaxies pass
MASS_BIN_DEX = 0.25

# z~0.4 quiescent sample (selection.quenched_sample, without the m25 gas / dust cuts)
Q_MASS_FLOOR = 10.0        # log M*
Q_PASSIVE_FACTOR = 0.2     # sSFR < 0.2 / t_H  (the QT threshold of the quench finder)
# compact vs disky: stellar concentration C = r80/r20 (caesar radii), median split per box
CLASS_SPLIT = "median"

# solar reference: Asplund et al. 2009 (photospheric), number ratios
SOLAR = {"OH12": 8.69, "logCO": 8.43 - 8.69, "logNO": 7.83 - 8.69}
# SIMBA PartType0/Metallicity columns (Dave+19): Z, He, C, N, O, Ne, Mg, Si, S, Ca, Fe
METAL_COLS = {"Z": 0, "He": 1, "C": 2, "N": 3, "O": 4}
ATOMIC_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}

# observations (only from the user's file; nothing is typed in by hand)
OBS_CSV = os.environ.get("OMEGA_OBS_CSV", os.path.join(REPO, "omega_obs_z6-8.csv"))
OBS_COLUMNS = {  # our name -> column of the OMEGA table
    "id": "msaid", "z": "z", "logMstar": "logMstar", "logSFR": "logSFR_Ha",
    "OH12": "OH12_direct", "logCO": "logCO", "logNO": "logNO", "logNO_limit": "logNO_limit",
}
OBS_DZ_MATCH = 0.5   # an observed point is compared with the snapshot of the nearest target z if |dz| < this
KDE_FRACTION = 0.95  # "inside the SIMBA locus" = inside the 95 % KDE contour of the resolved analogues
KDE_NMIN = 10        # fewer resolved analogues than this -> no contour, overlap not computed

N_BOOT = 2000
SEED = 7


def out_dir(sim):
    d = os.path.join(REPO, "output", sim, "omega_feasibility")
    os.makedirs(d, exist_ok=True)
    return d


def combined_dir():
    d = os.path.join(REPO, "output", "omega_feasibility")
    os.makedirs(d, exist_ok=True)
    return d
