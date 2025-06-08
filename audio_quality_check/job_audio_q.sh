#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100


module load python/3.10-anaconda
module load cuda/11.8.0
source activate nisqa

srun python test.py

