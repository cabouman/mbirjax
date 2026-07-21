#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH -t 03:00:00
#SBATCH -J w2-inc5c
#SBATCH -o /home/buzzard/headroom/logs/w2inc5c-%j.log

set -u
source ~/load_conda_cuda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mbirjax_headroom
cd ~/PycharmProjects/mbirjax_headroom
git log --oneline -1
python -u plans/experiments/projector_kernels/w2_inc5_convergence.py
