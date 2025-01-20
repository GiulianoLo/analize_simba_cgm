#!/bin/bash
#SBATCH --job-name=run_sfh
#SBATCH --output=run_sfh.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giuliano9797@gmail.com
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8000mb
#SBATCH --qos=normal

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/pd39

python /mnt/home/glorenzon/simbanator/analize_simba_cgm/bin_sfh.py --file "/mnt/home/glorenzon/sfh_test"