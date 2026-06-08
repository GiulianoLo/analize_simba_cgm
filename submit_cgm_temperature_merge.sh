#!/bin/bash
#SBATCH --job-name=cgmTmerge
#SBATCH --output=logs/cgmTmerge_%j.out
#SBATCH --error=logs/cgmTmerge_%j.err

#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# --- environment ---
source /mnt/home/glorenzon/miniconda3/etc/profile.d/conda.sh
conda activate pd39

# --- go to working directory ---
cd /mnt/home/glorenzon/analize_simba_cgm

export DUST_PLAN=output/cis100/caesar_sfh/dust_profile_plan.hdf5

python merge_cgm_temperature.py
