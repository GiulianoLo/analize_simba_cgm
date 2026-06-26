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
# Run ONCE PER ANCHOR, pointing DUST_PLAN at that anchor's plan:
#   DUST_PLAN=output/cis100/caesar_sfh/prof_<tag>/dust_profile_plan_<tag>.hdf5 \
#     sbatch --array=0-15 submit_reduced_particles.sh
export DUST_PLAN=${DUST_PLAN:-output/cis100/caesar_sfh/dust_profile_plan.hdf5}

# --- aperture + naming (must match the notebook reduced-particle loader) ---
export REDUCED_RMAX_KPC=${REDUCED_RMAX_KPC:-100}   # ISM+CGM physical aperture [kpc]
export REDUCED_PREFIX=${REDUCED_PREFIX:-m100n1024}
export REDUCED_OVERWRITE=${REDUCED_OVERWRITE:-0}

# --- ONE pass per snapshot: lean reduced gas+star particle files in the aperture ---
python build_reduced_particles_job.py
