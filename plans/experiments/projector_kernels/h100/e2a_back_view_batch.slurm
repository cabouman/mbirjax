#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:30:00
#SBATCH -J headroom-e2a
#SBATCH -o /home/buzzard/headroom/logs/e2a-%j.log
set -u
source ~/load_conda_cuda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mbirjax_headroom
cd ~/PycharmProjects/mbirjax_headroom
git log --oneline -1
python -u plans/experiments/projector_kernels/e2a_back_view_batch.py
echo "=== E2a done ==="
