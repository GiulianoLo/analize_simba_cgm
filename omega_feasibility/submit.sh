#!/bin/bash
#SBATCH --job-name=omega_feas
#SBATCH --output=logs/omega_feas_%j.out
#SBATCH --error=logs/omega_feas_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# OMEGA feasibility catalogue (omega_feasibility/build_catalogue.py), then the light analysis.
# One task: m25n512 + m100n1024, snapshots 026/030/036 (particles, members only) and 129 (catalogue),
# main branches 129 -> 026 (catalogue columns). Expected ~1 h, < 8 GB; --mem is explicit because
# CR_Core does not reserve memory.
# Submit from the repository root:  sbatch omega_feasibility/submit.sh

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate pd39

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f omega_feasibility/build_catalogue.py ]; then
    echo "run sbatch from the repository root (omega_feasibility/ not found in $(pwd))" >&2
    exit 1
fi
mkdir -p logs

python -u -m omega_feasibility.build_catalogue --sims cis25 cis100 --stages highz lowz trees && \
python -u -m omega_feasibility.analyse
