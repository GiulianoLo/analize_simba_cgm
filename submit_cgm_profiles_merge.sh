#!/bin/bash
#SBATCH --job-name=cgmprofmerge
#SBATCH --output=logs/cgmprofmerge_%j.out
#SBATCH --error=logs/cgmprofmerge_%j.err

#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# --- environment ---
source ~/.bashrc
conda activate pd39

# --- go to working directory ---
cd /mnt/home/glorenzon/analize_simba_cgm

export DUST_PLAN=output/cis100/caesar_sfh/dust_profile_plan.hdf5

python merge_cgm_profiles.py
