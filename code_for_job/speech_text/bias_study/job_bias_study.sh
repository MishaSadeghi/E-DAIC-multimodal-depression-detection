#!/bin/bash -l
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

pwd

module load python/3.10-anaconda
source activate venv
module load cuda/11.8.0

export HF_DATASETS_OFFLINE=1

# module load python/3.10-anaconda
# conda activate myenv
# # module load cuda/11.1.0
# module load cuda/11.6.1

srun python bias_study.py


    