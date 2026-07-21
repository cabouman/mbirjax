#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J fwd-guard-sweep
#SBATCH -o /home/buzzard/headroom/logs/fwd_guard_sweep-%j.log
set -u
source ~/load_conda_cuda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mbirjax_headroom
cd ~/PycharmProjects/mbirjax_headroom
git log --oneline -1
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv
python -u plans/experiments/projector_kernels/fwd_guard_sweep.py
echo "=== fwd guard sweep done ==="
