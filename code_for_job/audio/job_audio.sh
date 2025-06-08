#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

module load python/3.10-anaconda
source activate mypenv
module load cuda/11.6.1

srun python audio_pyannote.py


