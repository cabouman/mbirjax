#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:00:00
#SBATCH -J headroom-e3q
#SBATCH -o /home/buzzard/headroom/logs/e3q-%j.log
#
# Queued kernel-track round: cone fwd hfan/vfan split (E1 open item) + Pallas smoke (E3
# step zero).  gpu_headroom_plan.md sequencing, Greg-approved 2026-07-12.
#     sbatch plans/experiments/projector_kernels/e3_queue.slurm

set -u
source ~/load_conda_cuda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mbirjax_headroom
cd ~/PycharmProjects/mbirjax_headroom

echo "=== provenance ==="
git log --oneline -1
python -c "import mbirjax, jax; print('mbirjax:', mbirjax.__file__); print('jax:', jax.__version__)"
nvidia-smi -L

echo "=== cone fwd hfan/vfan split (1024^3 cell) ==="
python -u plans/experiments/projector_kernels/cone_fwd_split_ab.py

echo "=== E3 step zero: Pallas smoke ==="
python -u plans/experiments/projector_kernels/e3_pallas_smoke.py
echo "=== e3q done ==="
