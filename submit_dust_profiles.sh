#!/bin/bash
#SBATCH --job-name=dustprof
#SBATCH --output=logs/dustprof_%A_%a.out
#SBATCH --error=logs/dustprof_%A_%a.err

#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

#SBATCH --array=0-15

# --- environment ---
source ~/.bashrc
conda activate pd39

# --- go to working directory ---
cd /mnt/home/glorenzon/analize_simba_cgm

# --- plan written by the notebook cell "5a·plan" ---
export DUST_PLAN=output/cis100/caesar_sfh/dust_profile_plan.hdf5

# --- run one snapshot-chunk for this array task ---
python build_dust_profiles_job.py
