#!/bin/bash -l
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080

pwd

module load python/3.10-anaconda
module load cuda/11.8.0
source activate myenv

srun python visual_feature_extraction_LSTM.py