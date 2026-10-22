#!/bin/bash
#SBATCH --job-name=sed_flux
#SBATCH --output=logs/sed_flux_%A_%a.out
#SBATCH --error=logs/sed_flux_%A_%a.err

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

#SBATCH --array=0-14

# --- environment ---
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate pd39  

# --- go to working directory ---
# submit from the repository root: logs/ and the job script are resolved there
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f run_extract_flux.py ]; then
    echo "run sbatch from the repository root (run_extract_flux.py not found in $(pwd))" >&2
    exit 1
fi

# --- run ---
python run_extract_flux.py