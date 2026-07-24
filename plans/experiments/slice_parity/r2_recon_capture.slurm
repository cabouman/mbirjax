#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 02:00:00
#SBATCH -J parity-recons
#SBATCH -o /home/buzzard/headroom/logs/recons-%j.log
#
# R2 follow-up: D0 vs D1 recon volume capture at 15/20/30 iterations.
#     sbatch plans/experiments/slice_parity/r2_recon_capture.slurm

set -u
source ~/load_conda_cuda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mbirjax_headroom
cd ~/PycharmProjects/mbirjax_headroom

echo "=== provenance ==="
git log --oneline -1
nvidia-smi -L

python -u plans/experiments/slice_parity/r2_recon_capture.py
echo "=== capture job done ==="
