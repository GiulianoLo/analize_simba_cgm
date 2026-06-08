#!/bin/bash
#SBATCH --job-name=cgmT
#SBATCH --output=logs/cgmT_%A_%a.out
#SBATCH --error=logs/cgmT_%A_%a.err

#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

#SBATCH --array=0-15

# --- environment ---
source /mnt/home/glorenzon/miniconda3/etc/profile.d/conda.sh
conda activate pd39

# --- go to working directory ---
cd /mnt/home/glorenzon/analize_simba_cgm

# --- reuse the dust-profile plan (same galaxies/stages/snapshots) ---
export DUST_PLAN=output/cis100/caesar_sfh/dust_profile_plan.hdf5

# --- CGM temperature for this snapshot-chunk ---
python build_cgm_temperature_job.py
