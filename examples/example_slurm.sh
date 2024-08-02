#!/bin/bash

#SBATCH --job-name=cpm_slurm
#SBATCH --output=log_%a.log

#SBATCH --partition normal
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-10:00:00
#SBATCH --array=1-1000

# add python
module load palma/2021a
module load Miniconda3

# activate conda env
eval "$(conda shell.bash hook)"
conda activate cpm

python cpm_analysis.py --results_directory "" --data_directory "" --config_file "" perm_run $SLURM_ARRAY_TASK_ID
