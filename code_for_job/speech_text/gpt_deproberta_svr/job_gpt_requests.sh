#!/bin/bash -l

#SBATCH --mail-user=misha.sadeghi@fau.de
#SBATCH --mail-type=ALL --time=8:00:00
#SBATCH --gres=gpu:1

pwd

module load python/3.10-anaconda
source activate myenv
module load cuda/11.6.1

srun python parallel-chatgpt-request.py