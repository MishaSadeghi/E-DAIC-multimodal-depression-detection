#!/bin/bash -l
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100


module load python/3.10-anaconda
module load cuda/11.8.0
source activate myenv

srun python LSTM_enhance.py