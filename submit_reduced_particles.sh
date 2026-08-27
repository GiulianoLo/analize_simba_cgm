#!/bin/bash
#SBATCH --job-name=reduced_particles
#SBATCH --output=logs/reduced_%A_%a.out
#SBATCH --error=logs/reduced_%A_%a.err

#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

#SBATCH --array=0-15

# --- environment ---
source /mnt/home/glorenzon/miniconda3/etc/profile.d/conda.sh
conda activate pd39

# --- go to working directory ---
cd /mnt/home/glorenzon/analize_simba_cgm

# --- plan written by the notebook cell "5a·plan" (SHARED with build_profiles_job.py) ---
# Run ONCE PER ANCHOR, pointing the job at that anchor's plan. Preferred form — the plan
# path as the script's FIRST ARGUMENT (sbatch passes script arguments to the job verbatim,
# independent of the login shell's environment / --export policy):
#   sbatch --array=0-15 submit_reduced_particles.sh \
#       output/<SIM_NAME>/caesar_sfh/prof_<tag>/dust_profile_plan_<tag>.hdf5
# The env form still works when the assignment is on the SAME command line as sbatch:
#   DUST_PLAN=output/<SIM_NAME>/caesar_sfh/prof_<tag>/dust_profile_plan_<tag>.hdf5 \
#     sbatch --array=0-15 submit_reduced_particles.sh
# (an assignment on its own line is a plain, un-exported shell variable -> the job never sees it,
#  which is how the 2026-08-27 kstracks arrays died on the guard below).
# REQUIRED (no default): a silent cis100 fallback processed the wrong sim when the
# env was forgotten. The plan file itself carries the sim name the job runs on.
DUST_PLAN="${1:-${DUST_PLAN:-}}"
if [ -z "$DUST_PLAN" ]; then
    echo "DUST_PLAN is not set: run  sbatch --array=0-15 submit_reduced_particles.sh output/<SIM_NAME>/caesar_sfh/prof_<tag>/dust_profile_plan_<tag>.hdf5" >&2
    exit 1
fi
if [ ! -f "$DUST_PLAN" ]; then
    echo "DUST_PLAN=$DUST_PLAN does not exist (relative to $(pwd)) -> build it first (notebook plan cell)" >&2
    exit 1
fi
export DUST_PLAN
echo "DUST_PLAN=$DUST_PLAN"

# --- aperture + naming (must match the notebook reduced-particle loader) ---
export REDUCED_RMAX_KPC=${REDUCED_RMAX_KPC:-100}   # ISM+CGM physical aperture [kpc]
# REDUCED_PREFIX: leave unset — the job derives it from the plan's sim (m100n1024 / m50n512 / ...)
if [ -n "${REDUCED_PREFIX:-}" ]; then export REDUCED_PREFIX; fi
export REDUCED_OVERWRITE=${REDUCED_OVERWRITE:-0}

# --- ONE pass per snapshot: lean reduced gas+star particle files in the aperture ---
python build_reduced_particles_job.py
