#!/bin/bash
#SBATCH --job-name=ccnn_sweep
#SBATCH -N 1
#SBATCH --gres=cpu:4
#SBATCH --time=3:00:00
#SBATCH -p sched_mit_psfc_r8

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

python main.py device=cpu offline=True net.no_hidden=15 net.no_blocks=3 kernel.no_hidden=4 \
  kernel.no_layers=3 test.eval_high_thresh=.5 test.eval_low_thresh=.5 test.eval_hysteris=$1 \
  dataset.params.case_number=$2

# Deactivate your virtual environment
conda deactivate
