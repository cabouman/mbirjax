#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J parity-r1
#SBATCH -o /home/buzzard/headroom/logs/r1-%j.log
#
# R1: real-data schedule protocol on Lilly ds8 (slice_parity_plan.md R1 section).
#     sbatch plans/experiments/slice_parity/parity_realdata.slurm

set -u
source ~/load_conda_cuda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mbirjax_headroom
cd ~/PycharmProjects/mbirjax_headroom

echo "=== provenance ==="
git log --oneline -1
python -c "import mbirjax, jax; print('mbirjax:', mbirjax.__file__); print('jax:', jax.__version__)"
nvidia-smi -L

python -u plans/experiments/slice_parity/parity_realdata.py
echo "=== R1 done ==="
