#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

pwd

module load python/3.10-anaconda
source activate mypenv
# module load cuda/11.6.1

srun python chatgpt_ensemble_with_fine_tuning_deproberta.py


