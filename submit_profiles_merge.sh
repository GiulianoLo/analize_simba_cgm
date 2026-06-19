#!/bin/bash
#SBATCH --job-name=profmerge
#SBATCH --output=logs/profmerge_%j.out
#SBATCH --error=logs/profmerge_%j.err

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# --- environment ---
source /mnt/home/glorenzon/miniconda3/etc/profile.d/conda.sh
conda activate pd39

# --- go to working directory ---
cd /mnt/home/glorenzon/analize_simba_cgm

export DUST_PLAN=output/cis100/caesar_sfh/dust_profile_plan.hdf5

# --- assemble the three product files (dust / cgm-profiles / cgm-temperature) ---
python merge_profiles.py
