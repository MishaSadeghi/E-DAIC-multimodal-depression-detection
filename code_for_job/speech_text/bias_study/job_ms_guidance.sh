#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

pwd

module load python/3.10-anaconda
source activate myenv
module load cuda/11.6.1

srun python ms_guidance.py


