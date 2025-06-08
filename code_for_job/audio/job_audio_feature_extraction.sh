#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=30

pwd

module load python/3.10-anaconda
source activate myenv

srun python audio_based_training.py


