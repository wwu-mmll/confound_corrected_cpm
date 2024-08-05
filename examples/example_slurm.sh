#!/bin/bash

#SBATCH --job-name=cpm_slurm
#SBATCH --output=log_%a.log

#SBATCH --partition normal
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-10:00:00
#SBATCH --array=1-2

# add python
module load palma/2021a
module load Miniconda3

# activate conda env
eval "$(conda shell.bash hook)"
conda activate cpm

export PYTHONPATH=$PYTHONPATH:/scratch/tmp/wintern/cpm/cpm_python
python ../cpm/cpm_analysis.py --results-directory "/scratch/tmp/wintern/cpm/results/" --data-directory "./simulated_data/" --config-file "./config.pkl" --perm-run $SLURM_ARRAY_TASK_ID
