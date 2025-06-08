#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

pwd

module load python/3.10-anaconda
source activate myenv
# module load cuda/11.6.1

srun python gpt_deproberta_fine_tuned.py


