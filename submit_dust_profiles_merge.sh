#!/bin/bash
#SBATCH --job-name=dustmerge
#SBATCH --output=logs/dustmerge_%j.out
#SBATCH --error=logs/dustmerge_%j.err

#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# --- environment ---
source ~/.bashrc
conda activate pd39

# --- go to working directory ---
cd /mnt/home/glorenzon/analize_simba_cgm

export DUST_PLAN=output/cis100/caesar_sfh/dust_profile_plan.hdf5

# --- assemble partials into dust_profiles_allcrit.hdf5 (schema matches §5a) ---
python merge_dust_profiles.py
