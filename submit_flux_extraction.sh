#!/bin/bash
#SBATCH --job-name=sed_flux
#SBATCH --output=logs/sed_flux_%A_%a.out
#SBATCH --error=logs/sed_flux_%A_%a.err

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

#SBATCH --array=0-14

# --- environment ---
source ~/.bashrc
conda activate pd39  

# --- go to working directory ---
cd /mnt/home/glorenzon/analize_simba_cgm

# --- run ---
python run_extract_flux.py