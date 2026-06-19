#!/bin/bash
#SBATCH --job-name=profiles
#SBATCH --output=logs/profiles_%A_%a.out
#SBATCH --error=logs/profiles_%A_%a.err

#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

#SBATCH --array=0-15

# --- environment ---
source /mnt/home/glorenzon/miniconda3/etc/profile.d/conda.sh
conda activate pd39

# --- go to working directory ---
cd /mnt/home/glorenzon/analize_simba_cgm

# --- plan written by the notebook cell "5a·plan" (shared, unchanged) ---
export DUST_PLAN=output/cis100/caesar_sfh/dust_profile_plan.hdf5

# --- CGM radial extent (ISM extent comes from the plan, ~50 kpc) ---
export CGM_RMAX_KPC=300
export CGM_NBINS_R=30

# --- ONE pass per snapshot: ISM dust profiles + CGM Sigma(R) + CGM temperature ---
python build_profiles_job.py
