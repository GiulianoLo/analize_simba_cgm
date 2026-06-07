#!/bin/bash
#SBATCH --job-name=cgmprof
#SBATCH --output=logs/cgmprof_%A_%a.out
#SBATCH --error=logs/cgmprof_%A_%a.err

#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

#SBATCH --array=0-15

# --- environment ---
source ~/.bashrc
conda activate pd39

# --- go to working directory ---
cd /mnt/home/glorenzon/analize_simba_cgm

# --- reuse the dust-profile plan (same galaxies/stages/snapshots); CGM radial extent below ---
export DUST_PLAN=output/cis100/caesar_sfh/dust_profile_plan.hdf5
export CGM_RMAX_KPC=300
export CGM_NBINS_R=30

# --- CGM radial Sigma(R) profiles for this snapshot-chunk ---
python build_cgm_profiles_job.py
