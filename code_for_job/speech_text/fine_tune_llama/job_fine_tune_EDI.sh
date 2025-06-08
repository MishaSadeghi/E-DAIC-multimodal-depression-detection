#!/bin/bash -l
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

pwd

module load python/3.10-anaconda
module load cuda/11.8.0
source activate myenv
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

srun python EDI_dataset_finetune_.py