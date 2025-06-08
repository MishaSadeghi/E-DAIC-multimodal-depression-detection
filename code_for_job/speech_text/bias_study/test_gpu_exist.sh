#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080

pwd

module load python/3.10-anaconda
source activate venv
module load cuda/11.8.0

export HF_DATASETS_OFFLINE=1

# module load python/3.10-anaconda
# conda activate myenv
# # module load cuda/11.1.0
# module load cuda/11.6.1

srun python test_gpu_exist.py


    