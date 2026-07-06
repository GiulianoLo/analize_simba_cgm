#!/bin/bash
#SBATCH --job-name=progen_m25
#SBATCH --output=logs/progen_m25_%A_%a.out
#SBATCH --error=logs/progen_m25_%A_%a.err

#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

#SBATCH --array=0-3

# One-time merger-tree build for the SIMBA_25 catalogs (see find_progen_m25_job.py).
# The shared catalogs are file-level read-only, so links land in SIDECAR files under
# output/cis25/progen_links/. The snapshot range is split into 4 chunks whose bounds
# overlap by design: task i writes sidecars for snaps LO+1..HI, so the union covers
# snaps 20..151 exactly once — tasks touch disjoint sidecar files (parallel-safe),
# and finished sidecars are skipped on resubmit (resumable).

# --- environment (caesar 0.2b0 + pyGadgetReader live in pd39) ---
source /mnt/home/glorenzon/miniconda3/etc/profile.d/conda.sh
conda activate pd39

cd /mnt/home/glorenzon/analize_simba_cgm
mkdir -p logs

BOUNDS=(151 118 85 52 19)
HI=${BOUNDS[$SLURM_ARRAY_TASK_ID]}
LO=${BOUNDS[$((SLURM_ARRAY_TASK_ID + 1))]}

echo "task $SLURM_ARRAY_TASK_ID: progen pairs over snaps $HI -> $LO"
python find_progen_m25_job.py "$HI" "$LO"
