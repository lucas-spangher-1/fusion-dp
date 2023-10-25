#!/bin/bash
#SBATCH --job-name=ccnn_sweep
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --time=7:00:00
#SBATCH -p sched_mit_psfc_gpu_r8

source /etc/profile

# Load necessary modules
module load anaconda3/2022.10

source /home/software/anaconda3/2022.10/etc/profile.d/conda.sh
conda activate will-env

source /home/spangher/ccnn/venv/bin/activate

# Ignore the bogus local user packages on the host
export PYTHONUSERBASE=intentionally-disabled
# Run multiple instantiations of the WandB agent in parallel
# export WANDB_API_KEY=$(cat lucas_wandb_api)
wandb login $(cat lucas_wandb_api)

export PYTHONUSERBASE=intentionally-disabled
# Run multiple instantiations of the WandB agent in parallel
export WANDB_API_KEY=$(cat lucas_wandb_api)
export WANDB_DIR=/dev/shm/ccnn
export WANDB_CACHE_DIR=/dev/shm/ccnn-cache
export WANDB_LOGGER_DIR=/dev/shm/ccnn-log
export TRAINER_DIR=/dev/shm/ccnn-trainer
mkdir -p $WANDB_DIR $WANDB_CACHE_DIR $WANDB_LOGGER_DIR $TRAINER_DIR

wandb agent "$1"

# Deactivate your virtual environment
conda deactivate
