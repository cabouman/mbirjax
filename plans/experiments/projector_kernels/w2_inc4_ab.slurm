#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH -t 02:00:00
#SBATCH -J w2-inc4
#SBATCH -o /home/buzzard/headroom/logs/w2inc4-%j.log

set -u
source ~/load_conda_cuda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mbirjax_headroom
cd ~/PycharmProjects/mbirjax_headroom
git log --oneline -1
python -u plans/experiments/projector_kernels/w2_inc4_ab.py
