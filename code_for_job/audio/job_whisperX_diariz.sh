#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

module load python/3.10-anaconda
source activate whisperxenv
module load cuda/11.8
module load cudnn/8.6.0.163-11.8  

srun python whisperX_diarization.py


