#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

module load python/3.10-anaconda
source activate my2ndenv
module load cuda/11.6.1

srun python audio_based_training.py


